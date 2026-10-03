//! Environment backdrop of the marching-cubes passes (`mc_environment.wgsl`,
//! `fs_ground`: HDR sky or solid color, plus the projected ground and its
//! depth). Drawn twice per frame: single-sampled into the refraction
//! background, and at the water pass's sample count behind the water.

use crate::gpu::bind::{entry, layout, FRAGMENT, VERTEX_FRAGMENT};
use crate::state::GpuEnvironmentParams;
use wgpu::util::DeviceExt;

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    label: &str,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            // Writes the projected ground's depth (refraction + SSR see it)
            entry_point: Some("fs_ground"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual, // Draw at far plane
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

pub struct Backdrop {
    /// At the water pass's sample count
    pipeline: wgpu::RenderPipeline,
    /// Single-sampled, for the background pass
    pipeline_1x: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
}

impl Backdrop {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
        camera_buffer: &wgpu::Buffer,
        env_view: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
    ) -> Self {
        // Environment params buffer (background mode, color, intensity)
        let env_params = GpuEnvironmentParams::default();
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Env Params"),
            contents: bytemuck::bytes_of(&env_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let env_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Environment Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mc_environment.wgsl").into()),
        });

        let env_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC Env BGL"),
            entries: &[
                // Used in both vertex and fragment
                layout::uniform(0, VERTEX_FRAGMENT),
                layout::texture_2d(1, FRAGMENT),
                layout::sampler(2, FRAGMENT),
                layout::uniform(3, FRAGMENT),
            ],
        });

        let env_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC Env Pipeline Layout"),
            bind_group_layouts: &[&env_bind_group_layout],
            push_constant_ranges: &[],
        });

        let env_pipeline =
            create_pipeline(device, &env_pipeline_layout, &env_shader, surface_format, "MC Env Pipeline", sample_count);
        // Single-sampled env pipeline for background pass
        let env_pipeline_1x =
            create_pipeline(device, &env_pipeline_layout, &env_shader, surface_format, "MC Env Pipeline 1x", 1);

        let bind_group =
            Self::create_bind_group(device, &env_bind_group_layout, camera_buffer, env_view, env_sampler, &params_buffer);

        Self {
            pipeline: env_pipeline,
            pipeline_1x: env_pipeline_1x,
            bind_group_layout: env_bind_group_layout,
            bind_group,
            params_buffer,
        }
    }

    fn create_bind_group(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        camera_buffer: &wgpu::Buffer,
        env_view: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MC Env BG"),
            layout: bind_group_layout,
            entries: &[
                entry::buffer(0, camera_buffer),
                entry::view(1, env_view),
                entry::sampler(2, env_sampler),
                entry::buffer(3, params_buffer),
            ],
        })
    }

    /// Rebind after the environment map changed (HDR switch)
    pub fn rebuild_bind_group(
        &mut self,
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        env_view: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
    ) {
        self.bind_group = Self::create_bind_group(
            device, &self.bind_group_layout, camera_buffer, env_view, env_sampler, &self.params_buffer,
        );
    }

    /// Update environment parameters (background mode, color, intensity)
    pub fn update_params(&self, queue: &wgpu::Queue, params: &GpuEnvironmentParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Draw at the far plane, at the water pass's sample count
    pub fn draw(&self, pass: &mut wgpu::RenderPass) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Draw at the far plane, single-sampled (background pass)
    pub fn draw_1x(&self, pass: &mut wgpu::RenderPass) {
        pass.set_pipeline(&self.pipeline_1x);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
