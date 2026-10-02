//! Container wall bound for the marching-cubes density field.
//!
//! Last field pass before mesh generation (see `mc_wall_bound.wgsl`): the
//! out-of-container sentinel the smoothing filters skipped becomes the field
//! extended across the wall, cut by a linear ramp that crosses the iso value
//! exactly on each wall plane — the mesh ends on the walls at any tilt, with
//! the free surface flat right up to them.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// SPH rest spacing as a fraction of the kernel radius (as in calm_smoothing):
/// the interior field value is the number density 1 / spacing^3.
const REST_SPACING_FACTOR: f32 = 0.6;
/// How far behind the opaque pool's wall faces the mesh's sides are put, in
/// MC cells: coplanar they would z-fight with the walls.
const POOL_OFFSET_CELLS: f32 = 1.0;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuBoundParams {
    interior: f32,
    wall_offset: f32,
    _pad: [f32; 2],
}

pub struct WallBound {
    full_size: u32,
    params_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    /// [0]: field in A, writes B; [1]: field in B, writes A
    bind_groups: [wgpu::BindGroup; 2],
}

impl WallBound {
    /// `field_a` / `field_b` are the MC density ping-pong textures
    pub fn new(
        device: &wgpu::Device,
        full_size: u32,
        field_a: &wgpu::TextureView,
        field_b: &wgpu::TextureView,
        grid_params_buffer: &wgpu::Buffer,
        container_geom_buffer: &wgpu::Buffer,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Wall Bound Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../shaders/container_common.wgsl"),
                    include_str!("../shaders/mc_wall_bound.wgsl")
                )
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MC Wall Bound"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Wall Bound Params"),
            contents: bytemuck::bytes_of(&GpuBoundParams { interior: 1.0, wall_offset: 0.0, _pad: [0.0; 2] }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let layout = pipeline.get_bind_group_layout(0);
        let bind_groups: [wgpu::BindGroup; 2] = std::array::from_fn(|i| {
            let (src, dst) = if i == 0 { (field_a, field_b) } else { (field_b, field_a) };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MC Wall Bound BG"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(dst) },
                    wgpu::BindGroupEntry { binding: 2, resource: grid_params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: container_geom_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
                ],
            })
        });

        Self { full_size, params_buffer, pipeline, bind_groups }
    }

    /// `kernel_radius` is the sim h (sets the interior field value);
    /// `cell_size` the MC voxel size; `is_pool`: opaque walls hide the sides.
    pub fn update(&self, queue: &wgpu::Queue, kernel_radius: f32, cell_size: f32, is_pool: bool) {
        let spacing = REST_SPACING_FACTOR * kernel_radius;
        let params = GpuBoundParams {
            interior: 1.0 / (spacing * spacing * spacing),
            wall_offset: if is_pool { POOL_OFFSET_CELLS * cell_size } else { 0.0 },
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Bound the field at the container walls. Returns where the result
    /// landed: true = texture B.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, field_in_b: bool) -> bool {
        let groups = self.full_size.div_ceil(4);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MC Wall Bound"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_groups[field_in_b as usize], &[]);
        pass.dispatch_workgroups(groups, groups, groups);
        !field_in_b
    }
}
