//! Anisotropic kernels (Yu & Turk): one covariance fit + eigensolve per
//! particle (`mc_anisotropy.wgsl`) writes an ellipsoid record that the MC
//! density pass and the screen-space renderer's splats stretch along.

use super::mc_field::SimBuffers;
use crate::gpu::bind::{entry, layout, COMPUTE};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Hard cap on anisotropic ellipsoid axis scale. Bounds the density pass
/// neighbor search radius (and the MC grid margin in app/scene.rs). 1.6 covers the
/// flat-sheet case fully (in-plane stretch kr^(1/3) ≈ 1.59 at kr = 4) while
/// keeping the gather loop footprint small; only extreme strings get capped.
pub const ANISO_MAX_STRETCH: f32 = 1.6;
/// Yu & Turk k_r: max ratio between largest and smallest covariance stddev.
const ANISO_KR: f32 = 4.0;
/// Center smoothing factor toward the weighted neighbor mean (Yu & Turk λ).
const ANISO_LAMBDA: f32 = 0.9;
/// Covariance neighborhood radius as a multiple of the sim kernel radius.
const ANISO_SUPPORT_SCALE: f32 = 2.0;
/// Center smoothing shift cap as a multiple of the sim kernel radius.
const ANISO_MAX_SHIFT_SCALE: f32 = 0.4;
/// Bytes per ParticleAniso record (3 × vec4<f32>).
const ANISO_STRIDE: u64 = 48;

/// Anisotropic kernel parameters (matches AnisoParams in mc_anisotropy.wgsl
/// and mc_density.wgsl — all scalars, no padding needed).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuAnisoParams {
    enabled: u32,
    strength: f32,
    support_radius: f32,
    h_mc: f32,
    kr: f32,
    lambda: f32,
    max_stretch: f32,
    max_shift: f32,
}

pub struct AnisotropyPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    /// Per-particle ellipsoid records (sorted-particle indexed); grows on demand
    records: wgpu::Buffer,
    capacity: u32,
    /// Cached across frames; see `invalidate`
    bind_group: Option<wgpu::BindGroup>,
}

impl AnisotropyPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let aniso_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Anisotropy Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../shaders/container_common.wgsl"),
                    include_str!("../shaders/mc_anisotropy.wgsl")
                )
                .into(),
            ),
        });

        let aniso_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC Anisotropy BGL"),
            entries: &[
                // Sorted particles (read)
                layout::storage(0, COMPUTE),
                // SPH cell_starts
                layout::storage(1, COMPUTE),
                // SPH cell_counts
                layout::storage(2, COMPUTE),
                // SPH grid params
                layout::uniform(3, COMPUTE),
                // Aniso params
                layout::uniform(4, COMPUTE),
                // Output records
                layout::storage_rw(5, COMPUTE),
                // Container geometry (walls mirror the neighbourhood)
                layout::uniform(6, COMPUTE),
            ],
        });

        let aniso_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC Anisotropy Pipeline Layout"),
            bind_group_layouts: &[&aniso_bind_group_layout],
            push_constant_ranges: &[],
        });

        let aniso_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MC Anisotropy Pipeline"),
            layout: Some(&aniso_pipeline_layout),
            module: &aniso_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let aniso_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Aniso Params"),
            contents: bytemuck::bytes_of(&GpuAnisoParams::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Per-particle ellipsoid records; grown lazily in ensure_capacity()
        let aniso_capacity: u32 = 1024;
        let aniso_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MC Aniso Records"),
            size: aniso_capacity as u64 * ANISO_STRIDE,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Self {
            pipeline: aniso_pipeline,
            bind_group_layout: aniso_bind_group_layout,
            params_buffer: aniso_params_buffer,
            records: aniso_buffer,
            capacity: aniso_capacity,
            bind_group: None,
        }
    }

    /// Update the kernel parameters. `kernel_radius` is the sim h; `h_mc`
    /// the MC density kernel radius.
    pub fn update_params(&self, queue: &wgpu::Queue, enabled: bool, strength: f32, kernel_radius: f32, h_mc: f32) {
        let params = GpuAnisoParams {
            enabled: if enabled { 1 } else { 0 },
            strength: strength.clamp(0.0, 1.0),
            support_radius: ANISO_SUPPORT_SCALE * kernel_radius,
            h_mc,
            kr: ANISO_KR,
            lambda: ANISO_LAMBDA,
            max_stretch: ANISO_MAX_STRETCH,
            max_shift: ANISO_MAX_SHIFT_SCALE * kernel_radius,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Records from the last pass, indexed by sorted particle index
    pub fn records(&self) -> &wgpu::Buffer {
        &self.records
    }

    pub fn params_buffer(&self) -> &wgpu::Buffer {
        &self.params_buffer
    }

    /// Drop the cached bind group (the sim buffers were swapped out)
    pub fn invalidate(&mut self) {
        self.bind_group = None;
    }

    /// Grow the record buffer if needed. Returns true if it was recreated:
    /// every bind group over the old one is stale (this pass's own is
    /// dropped here; the density pass's is the caller's to invalidate).
    pub fn ensure_capacity(&mut self, device: &wgpu::Device, num_particles: u32) -> bool {
        if num_particles <= self.capacity {
            return false;
        }
        let capacity = num_particles.next_power_of_two().max(1024);
        self.records = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MC Aniso Records"),
            size: capacity as u64 * ANISO_STRIDE,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.capacity = capacity;
        self.bind_group = None;
        true
    }

    /// One fit per particle. Its own compute pass, which gives an implicit
    /// barrier before any consumer of the records.
    pub fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        sim: &SimBuffers,
        container_geom_buffer: &wgpu::Buffer,
        num_particles: u32,
    ) {
        let (bind_group_layout, params_buffer, records) = (&self.bind_group_layout, &self.params_buffer, &self.records);
        let bind_group = self.bind_group.get_or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MC Anisotropy BG"),
                layout: bind_group_layout,
                entries: &[
                    entry::buffer(0, sim.sorted_particles),
                    entry::buffer(1, sim.cell_starts),
                    entry::buffer(2, sim.cell_counts),
                    entry::buffer(3, sim.grid_params),
                    entry::buffer(4, params_buffer),
                    entry::buffer(5, records),
                    entry::buffer(6, container_geom_buffer),
                ],
            })
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MC Anisotropy Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &*bind_group, &[]);
        pass.dispatch_workgroups(num_particles.div_ceil(128), 1, 1);
    }
}
