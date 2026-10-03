//! Mesh extraction (`mc_generate.wgsl`): turns the final density field
//! into triangles in a storage buffer, counted by an atomic that also
//! drives the indirect draw, plus the vertex-count readback for the GUI and
//! stats.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::mc_field::FieldTextures;
use super::mc_tables::{EDGE_TABLE, TRI_TABLE};
use crate::gpu::bind::{entry, layout, COMPUTE};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Maximum vertices for the output mesh buffer.
/// Capped at 4M to avoid absurd VRAM allocation (96 MB at 24 bytes/vertex).
/// The atomic counter in the generate shader handles overflow gracefully.
pub const MAX_VERTICES: u32 = 4_000_000;

/// Vertex output from marching cubes
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct McVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Atomic counter for vertex allocation
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Counter {
    vertex_count: u32,
}

pub struct MeshPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// [reads field texture A, reads field texture B]
    bind_groups: [wgpu::BindGroup; 2],
    edge_table_buffer: wgpu::Buffer,
    tri_table_buffer: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    /// Indirect draw args; the vertex count is copied in from the counter
    indirect_buffer: wgpu::Buffer,

    // For reading back the vertex count
    counter_staging_buffer: wgpu::Buffer,
    // Some(flag) while a staging map is in flight or mapped; the flag is set
    // by the map_async callback. While Some, the staging buffer must not be
    // copied into (validation error), so encode() skips that copy.
    counter_map_done: Option<Arc<AtomicBool>>,
    // A counter copy was encoded whose result hasn't been mapped yet
    counter_copy_in_flight: bool,
    // Last read-back vertex count
    current_vertex_count: u32,
}

impl MeshPass {
    pub fn new(
        device: &wgpu::Device,
        field: &FieldTextures,
        grid_params_buffer: &wgpu::Buffer,
        voxel_normals: &wgpu::TextureView,
    ) -> Self {
        // Edge table buffer
        let edge_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Edge Table"),
            contents: bytemuck::cast_slice(&EDGE_TABLE),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Triangle table buffer (flatten 2D array)
        let tri_table_flat: Vec<i32> = TRI_TABLE.iter().flatten().copied().collect();
        let tri_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Tri Table"),
            contents: bytemuck::cast_slice(&tri_table_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Counter buffer
        let counter = Counter { vertex_count: 0 };
        let counter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Counter"),
            contents: bytemuck::bytes_of(&counter),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // Staging buffer for reading back counter
        let counter_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MC Counter Staging"),
            size: std::mem::size_of::<Counter>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Indirect draw buffer: [vertex_count, instance_count, first_vertex, first_instance]
        let indirect_data: [u32; 4] = [0, 1, 0, 0];
        let indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Indirect Draw"),
            contents: bytemuck::cast_slice(&indirect_data),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        });

        // Vertex buffer
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MC Vertices"),
            size: (MAX_VERTICES as usize * std::mem::size_of::<McVertex>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let generate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Generate Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../shaders/octahedral_common.wgsl"),
                    include_str!("../shaders/mc_generate.wgsl")
                )
                .into(),
            ),
        });

        // === Generate Pipeline ===
        let generate_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC Generate BGL"),
            entries: &[
                // Density field (read)
                layout::texture_3d_unfilterable(0, COMPUTE),
                // Grid params
                layout::uniform(1, COMPUTE),
                // Edge table
                layout::storage(2, COMPUTE),
                // Tri table
                layout::storage(3, COMPUTE),
                // Counter
                layout::storage_rw(4, COMPUTE),
                // Vertices
                layout::storage_rw(5, COMPUTE),
                // Voxel normals (octahedral, R32Uint)
                layout::texture_3d_uint(6, COMPUTE),
            ],
        });

        let generate_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC Generate Pipeline Layout"),
            bind_group_layouts: &[&generate_bind_group_layout],
            push_constant_ranges: &[],
        });

        let generate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MC Generate Pipeline"),
            layout: Some(&generate_pipeline_layout),
            module: &generate_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let tables = [&edge_table_buffer, &tri_table_buffer];
        let outputs = [&counter_buffer, &vertex_buffer];
        let bind_groups = Self::create_bind_groups(
            device, &generate_bind_group_layout, field, grid_params_buffer, tables, outputs, voxel_normals,
        );

        Self {
            pipeline: generate_pipeline,
            bind_group_layout: generate_bind_group_layout,
            bind_groups,
            edge_table_buffer,
            tri_table_buffer,
            counter_buffer,
            vertex_buffer,
            indirect_buffer,
            counter_staging_buffer,
            counter_map_done: None,
            counter_copy_in_flight: false,
            current_vertex_count: 0,
        }
    }

    fn create_bind_groups(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        field: &FieldTextures,
        grid_params_buffer: &wgpu::Buffer,
        [edge_table_buffer, tri_table_buffer]: [&wgpu::Buffer; 2],
        [counter_buffer, vertex_buffer]: [&wgpu::Buffer; 2],
        voxel_normals: &wgpu::TextureView,
    ) -> [wgpu::BindGroup; 2] {
        [(&field.view_a, "MC Generate BG"), (&field.view_b, "MC Generate BG B")].map(|(density_view, label)| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: bind_group_layout,
                entries: &[
                    entry::view(0, density_view),
                    entry::buffer(1, grid_params_buffer),
                    entry::buffer(2, edge_table_buffer),
                    entry::buffer(3, tri_table_buffer),
                    entry::buffer(4, counter_buffer),
                    entry::buffer(5, vertex_buffer),
                    entry::view(6, voxel_normals),
                ],
            })
        })
    }

    /// Rebind over recreated field textures / voxel normals (grid resolution change)
    pub fn rebuild_bind_groups(
        &mut self,
        device: &wgpu::Device,
        field: &FieldTextures,
        grid_params_buffer: &wgpu::Buffer,
        voxel_normals: &wgpu::TextureView,
    ) {
        self.bind_groups = Self::create_bind_groups(
            device,
            &self.bind_group_layout,
            field,
            grid_params_buffer,
            [&self.edge_table_buffer, &self.tri_table_buffer],
            [&self.counter_buffer, &self.vertex_buffer],
            voxel_normals,
        );
    }

    /// Mesh vertex storage buffer (allocated once; never recreated)
    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
    }

    /// GPU-driven indirect draw args (vertex count filled by `encode`)
    pub fn indirect_buffer(&self) -> &wgpu::Buffer {
        &self.indirect_buffer
    }

    /// The marching-cubes triangle table (the water shader re-triangulates cells with it)
    pub fn tri_table_buffer(&self) -> &wgpu::Buffer {
        &self.tri_table_buffer
    }

    /// Zero the vertex counter: first command of a frame's generate
    pub fn reset_counter(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.counter_buffer, 0, None);
    }

    /// Generate triangles from the field in texture A or B, then hand the
    /// count to the indirect draw and the readback
    pub fn encode(&mut self, encoder: &mut wgpu::CommandEncoder, result_in_b: bool, grid_size: u32) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MC Generate Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_groups[result_in_b as usize], &[]);
            let workgroups = grid_size.div_ceil(4);
            pass.dispatch_workgroups(workgroups, workgroups, workgroups);
        }

        // Copy counter to indirect buffer for GPU-driven draw
        encoder.copy_buffer_to_buffer(
            &self.counter_buffer,
            0,
            &self.indirect_buffer,
            0,
            std::mem::size_of::<u32>() as u64,  // Just the vertex_count
        );

        // Copy counter to staging buffer for readback (for stats display) —
        // unless the staging buffer is still mapped/pending from an earlier
        // frame, in which case skip and let the stat stay stale one frame
        if self.counter_map_done.is_none() {
            encoder.copy_buffer_to_buffer(
                &self.counter_buffer,
                0,
                &self.counter_staging_buffer,
                0,
                std::mem::size_of::<Counter>() as u64,
            );
            self.counter_copy_in_flight = true;
        }
    }

    /// Last read-back mesh vertex count (see `read_vertex_count`)
    pub fn vertex_count(&self) -> u32 {
        self.current_vertex_count
    }

    /// Read back vertex count, blocking until the GPU finishes (call after
    /// submit). Exact for the frame just submitted — automation/stats runs
    /// use this so CSV rows stay deterministic.
    pub fn read_vertex_count(&mut self, device: &wgpu::Device) {
        self.arm_counter_map();
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        self.harvest_counter_map();
    }

    /// Non-blocking variant for interactive frames: harvests a previously
    /// issued readback if the GPU has finished it, then arms the next one.
    /// The count lags a frame or two, which only affects the GUI stat —
    /// rendering uses the GPU-side indirect buffer.
    pub fn poll_vertex_count(&mut self, device: &wgpu::Device) {
        device.poll(wgpu::PollType::Poll).ok();
        self.harvest_counter_map();
        self.arm_counter_map();
    }

    /// If a counter copy was submitted and no map is outstanding, start one.
    fn arm_counter_map(&mut self) {
        if self.counter_map_done.is_some() || !self.counter_copy_in_flight {
            return;
        }
        let flag = Arc::new(AtomicBool::new(false));
        let done = flag.clone();
        self.counter_staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    done.store(true, Ordering::Release);
                }
            });
        self.counter_map_done = Some(flag);
        self.counter_copy_in_flight = false;
    }

    /// If the outstanding map has completed, read the count and unmap.
    fn harvest_counter_map(&mut self) {
        let done = self
            .counter_map_done
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Acquire));
        if !done {
            return;
        }
        {
            let data = self.counter_staging_buffer.slice(..).get_mapped_range();
            let counter: &Counter = bytemuck::from_bytes(&data);
            self.current_vertex_count = counter.vertex_count.min(MAX_VERTICES);
        }
        self.counter_staging_buffer.unmap();
        self.counter_map_done = None;
    }
}
