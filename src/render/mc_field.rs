//! The marching-cubes density field: its two ping-pong textures and the
//! passes that fill and smooth it (`mc_density.wgsl`, `mc_blur.wgsl`).
//! The later field stages live next door: calm_smoothing.rs, wall_bound.rs,
//! voxel_normals.rs; mc_anisotropy.rs shapes the kernels the density pass
//! splats.

use crate::gpu::bind::{entry, layout, COMPUTE};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Blur parameters for density field smoothing
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuBlurParams {
    dir_x: i32,
    dir_y: i32,
    dir_z: i32,
    radius: i32,
    grid_size: u32,
    _pad: [u32; 3],
}

/// Grid parameters for compute shaders
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuGridParams {
    pub grid_min: [f32; 3],
    pub grid_size: u32,
    pub grid_max: [f32; 3],
    pub cell_size: f32,
    pub kernel_radius: f32,
    pub iso_value: f32,
    pub num_particles: u32,
    pub max_vertices: u32,
}

/// The density field's two textures (R32Float, grid_size^3). The density
/// pass writes A; the blur and later stages ping-pong between A and B.
pub struct FieldTextures {
    pub texture_a: wgpu::Texture,
    pub view_a: wgpu::TextureView,
    pub texture_b: wgpu::Texture,
    pub view_b: wgpu::TextureView,
}

impl FieldTextures {
    pub fn new(device: &wgpu::Device, grid_size: u32) -> Self {
        // Create 3D density texture
        let density_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MC Density Field"),
            size: wgpu::Extent3d {
                width: grid_size,
                height: grid_size,
                depth_or_array_layers: grid_size,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let density_view = density_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Second density texture for ping-pong blur
        let density_texture_b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MC Density Field B"),
            size: wgpu::Extent3d {
                width: grid_size,
                height: grid_size,
                depth_or_array_layers: grid_size,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let density_view_b = density_texture_b.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture_a: density_texture,
            view_a: density_view,
            texture_b: density_texture_b,
            view_b: density_view_b,
        }
    }

    /// Field dump (--dump-field): the field in texture A or B as last
    /// generated — grid parameters + grid_size^3 f32 (x fastest). Voxel i
    /// sits at grid_min + i * cell_size (mc_generate's convention). Blocking.
    pub fn read(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        in_b: bool,
        grid_size: u32,
        grid_params_buffer: &wgpu::Buffer,
    ) -> (GpuGridParams, Vec<u8>) {
        let n = grid_size;
        let texture = if in_b { &self.texture_b } else { &self.texture_a };
        let row = n * 4;
        let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MC Field Readback"),
            size: padded as u64 * n as u64 * n as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MC Grid Params Readback"),
            size: std::mem::size_of::<GpuGridParams>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MC Field Readback"),
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(n),
                },
            },
            wgpu::Extent3d { width: n, height: n, depth_or_array_layers: n },
        );
        encoder.copy_buffer_to_buffer(grid_params_buffer, 0, &params_staging, 0, params_staging.size());
        queue.submit(Some(encoder.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        params_staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let params: GpuGridParams = *bytemuck::from_bytes(&params_staging.slice(..).get_mapped_range());
        let mut bytes = Vec::with_capacity(row as usize * n as usize * n as usize);
        {
            let data = staging.slice(..).get_mapped_range();
            for r in 0..(n * n) as usize {
                let start = r * padded as usize;
                bytes.extend_from_slice(&data[start..start + row as usize]);
            }
        }
        (params, bytes)
    }
}

/// The SPH simulation's buffers the field passes read: the grid-sorted
/// particles and the spatial hash that finds their neighbours
#[derive(Clone, Copy)]
pub struct SimBuffers<'a> {
    pub sorted_particles: &'a wgpu::Buffer,
    pub cell_starts: &'a wgpu::Buffer,
    pub cell_counts: &'a wgpu::Buffer,
    pub grid_params: &'a wgpu::Buffer,
}

impl SimBuffers<'_> {
    /// Whether these are the buffer objects `cached` was taken from
    pub fn same_as(&self, cached: &[wgpu::Buffer; 4]) -> bool {
        let [a, b, c, d] = cached;
        a == self.sorted_particles && b == self.cell_starts && c == self.cell_counts && d == self.grid_params
    }

    pub fn handles(self) -> [wgpu::Buffer; 4] {
        [
            self.sorted_particles.clone(),
            self.cell_starts.clone(),
            self.cell_counts.clone(),
            self.grid_params.clone(),
        ]
    }
}

/// Everything the density bind group reads besides the sim buffers
pub struct DensityInputs<'a> {
    pub grid_params: &'a wgpu::Buffer,
    /// Field texture A (written)
    pub field: &'a wgpu::TextureView,
    pub container_geom: &'a wgpu::Buffer,
    pub aniso_records: &'a wgpu::Buffer,
    pub aniso_params: &'a wgpu::Buffer,
}

/// Splats the particles into field texture A (SPH-grid accelerated gather)
pub struct DensityPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Cached across frames; see `invalidate`
    bind_group: Option<wgpu::BindGroup>,
}

impl DensityPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let density_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Density Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../shaders/container_common.wgsl"),
                    include_str!("../shaders/mc_density.wgsl")
                )
                .into(),
            ),
        });

        let density_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC Density BGL"),
            entries: &[
                // Sorted particles (storage buffer, bound dynamically from SPH sim)
                layout::storage(0, COMPUTE),
                // MC grid params
                layout::uniform(1, COMPUTE),
                // Density field (write)
                layout::storage_texture_3d(2, COMPUTE, wgpu::TextureFormat::R32Float),
                // Container geometry (for boundary gamma correction + clipping)
                layout::uniform(3, COMPUTE),
                // SPH cell_starts (storage buffer, from SPH spatial hash grid)
                layout::storage(4, COMPUTE),
                // SPH cell_counts (storage buffer, from SPH spatial hash grid)
                layout::storage(5, COMPUTE),
                // SPH grid params (uniform)
                layout::uniform(6, COMPUTE),
                // Anisotropic kernel records (from mc_anisotropy pass)
                layout::storage(7, COMPUTE),
                // Anisotropic kernel params (uniform)
                layout::uniform(8, COMPUTE),
            ],
        });

        let density_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC Density Pipeline Layout"),
            bind_group_layouts: &[&density_bind_group_layout],
            push_constant_ranges: &[],
        });

        let density_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MC Density Pipeline"),
            layout: Some(&density_pipeline_layout),
            module: &density_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline: density_pipeline,
            bind_group_layout: density_bind_group_layout,
            bind_group: None,
        }
    }

    /// Drop the cached bind group: call when the sim buffers are swapped out
    /// (sim rebuild), the anisotropy record buffer grows, or the field
    /// textures are recreated
    pub fn invalidate(&mut self) {
        self.bind_group = None;
    }

    pub fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        sim: &SimBuffers,
        inputs: &DensityInputs,
        grid_size: u32,
    ) {
        let bind_group_layout = &self.bind_group_layout;
        let bind_group = self.bind_group.get_or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MC Density BG"),
                layout: bind_group_layout,
                entries: &[
                    entry::buffer(0, sim.sorted_particles),
                    entry::buffer(1, inputs.grid_params),
                    entry::view(2, inputs.field),
                    entry::buffer(3, inputs.container_geom),
                    entry::buffer(4, sim.cell_starts),
                    entry::buffer(5, sim.cell_counts),
                    entry::buffer(6, sim.grid_params),
                    entry::buffer(7, inputs.aniso_records),
                    entry::buffer(8, inputs.aniso_params),
                ],
            })
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MC Density Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &*bind_group, &[]);
        let workgroups = grid_size.div_ceil(4);
        pass.dispatch_workgroups(workgroups, workgroups, workgroups);
    }
}

const BLUR_DIRECTIONS: [[i32; 3]; 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

/// Base smoothing of the field: three separable passes (X, Y, Z)
pub struct BlurPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// One per direction: X, Y, Z
    params_buffers: [wgpu::Buffer; 3],
    /// `bind_groups[dir][0]` = A -> B, `bind_groups[dir][1]` = B -> A
    bind_groups: [[wgpu::BindGroup; 2]; 3],
}

impl BlurPass {
    pub fn new(device: &wgpu::Device, grid_size: u32, field: &FieldTextures) -> Self {
        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mc_blur.wgsl").into()),
        });

        let blur_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC Blur BGL"),
            entries: &[
                // Input density field (read)
                layout::texture_3d_unfilterable(0, COMPUTE),
                // Output density field (write)
                layout::storage_texture_3d(1, COMPUTE, wgpu::TextureFormat::R32Float),
                // Blur params
                layout::uniform(2, COMPUTE),
            ],
        });

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC Blur Pipeline Layout"),
            bind_group_layouts: &[&blur_bind_group_layout],
            push_constant_ranges: &[],
        });

        let blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MC Blur Pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create 3 blur param buffers (one per direction: X, Y, Z)
        let blur_params_buffers: [wgpu::Buffer; 3] = std::array::from_fn(|i| {
            let params = GpuBlurParams {
                dir_x: BLUR_DIRECTIONS[i][0],
                dir_y: BLUR_DIRECTIONS[i][1],
                dir_z: BLUR_DIRECTIONS[i][2],
                radius: 2, // default
                grid_size,
                _pad: [0; 3],
            };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("MC Blur Params {}", ["X", "Y", "Z"][i])),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });

        let bind_groups = Self::create_bind_groups(device, &blur_bind_group_layout, &blur_params_buffers, field);

        Self {
            pipeline: blur_pipeline,
            bind_group_layout: blur_bind_group_layout,
            params_buffers: blur_params_buffers,
            bind_groups,
        }
    }

    /// 6 bind groups: [direction][a_to_b=0, b_to_a=1]
    fn create_bind_groups(
        device: &wgpu::Device,
        blur_bind_group_layout: &wgpu::BindGroupLayout,
        blur_params_buffers: &[wgpu::Buffer; 3],
        field: &FieldTextures,
    ) -> [[wgpu::BindGroup; 2]; 3] {
        std::array::from_fn(|dir| {
            [
                // a -> b
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("MC Blur BG {} A->B", ["X", "Y", "Z"][dir])),
                    layout: blur_bind_group_layout,
                    entries: &[
                        entry::view(0, &field.view_a),
                        entry::view(1, &field.view_b),
                        entry::buffer(2, &blur_params_buffers[dir]),
                    ],
                }),
                // b -> a
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("MC Blur BG {} B->A", ["X", "Y", "Z"][dir])),
                    layout: blur_bind_group_layout,
                    entries: &[
                        entry::view(0, &field.view_b),
                        entry::view(1, &field.view_a),
                        entry::buffer(2, &blur_params_buffers[dir]),
                    ],
                }),
            ]
        })
    }

    /// Rebind over recreated field textures (grid resolution change)
    pub fn rebuild_bind_groups(&mut self, device: &wgpu::Device, field: &FieldTextures) {
        self.bind_groups = Self::create_bind_groups(device, &self.bind_group_layout, &self.params_buffers, field);
    }

    pub fn update(&self, queue: &wgpu::Queue, radius: u32, grid_size: u32) {
        for (dir, buffer) in BLUR_DIRECTIONS.iter().zip(&self.params_buffers) {
            let blur_params = GpuBlurParams {
                dir_x: dir[0],
                dir_y: dir[1],
                dir_z: dir[2],
                radius: radius as i32,
                grid_size,
                _pad: [0; 3],
            };
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&blur_params));
        }
    }

    /// X (A -> B), Y (B -> A), Z (A -> B): the result ends up in texture B
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, grid_size: u32) {
        let workgroups = grid_size.div_ceil(4);
        // a_to_b = 0, b_to_a = 1: which bind group variant per pass
        let pass_sources = [0usize, 1, 0];
        for (dir, &src) in pass_sources.iter().enumerate() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MC Blur Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_groups[dir][src], &[]);
            pass.dispatch_workgroups(workgroups, workgroups, workgroups);
        }
    }
}
