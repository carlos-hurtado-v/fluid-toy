//! Voxel normals for the marching-cubes mesh.
//!
//! One pass after the field is final (see `mc_voxel_normals.wgsl`) writes the
//! surface normal at every voxel into a texture that both mesh generation and
//! the water shader's exit normals read, so the two cannot disagree. On calm
//! water the normals are denoised against the calm gate field's gradient.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuNormalParams {
    denoise: f32,
    _pad: [f32; 3],
}

pub struct VoxelNormals {
    full_size: u32,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    params_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    /// [0]: field in A; [1]: field in B
    bind_groups: [wgpu::BindGroup; 2],
}

impl VoxelNormals {
    /// `field_a` / `field_b` are the MC density ping-pong textures;
    /// `wide_field` is the calm-smoothing gate field (half resolution).
    pub fn new(
        device: &wgpu::Device,
        full_size: u32,
        field_a: &wgpu::TextureView,
        field_b: &wgpu::TextureView,
        grid_params_buffer: &wgpu::Buffer,
        wide_field: &wgpu::TextureView,
    ) -> Self {
        // Octahedral normal, 16 + 16 bits (4 bytes a voxel: 32 MB at 200^3)
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MC Voxel Normals"),
            size: wgpu::Extent3d {
                width: full_size,
                height: full_size,
                depth_or_array_layers: full_size,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Voxel Normals Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../shaders/octahedral_common.wgsl"),
                    include_str!("../shaders/mc_voxel_normals.wgsl")
                )
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MC Voxel Normals"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Voxel Normals Params"),
            contents: bytemuck::bytes_of(&GpuNormalParams { denoise: 0.0, _pad: [0.0; 3] }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let layout = pipeline.get_bind_group_layout(0);
        let bind_groups: [wgpu::BindGroup; 2] = [field_a, field_b].map(|field| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MC Voxel Normals BG"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(field) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 2, resource: grid_params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(wide_field) },
                ],
            })
        });

        Self { full_size, _texture: texture, view, params_buffer, pipeline, bind_groups }
    }

    /// The normal texture (R32Uint, octahedral), valid after `encode`
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// `denoise_deg`: how far (degrees) a normal may differ from the wide
    /// field's and still count as noise; 0 = the field's own normals. Pass 0
    /// when calm smoothing is off — the wide field is not computed then.
    pub fn update(&self, queue: &wgpu::Queue, denoise_deg: f32) {
        let params = GpuNormalParams { denoise: denoise_deg.max(0.0).to_radians(), _pad: [0.0; 3] };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Write the normals of the final field (`field_in_b`: where it landed)
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, field_in_b: bool) {
        let groups = self.full_size.div_ceil(4);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MC Voxel Normals"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_groups[field_in_b as usize], &[]);
        pass.dispatch_workgroups(groups, groups, groups);
    }
}
