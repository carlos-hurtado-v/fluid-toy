//! Environment backdrop for the Particles render mode: a fullscreen pass over
//! `mc_environment.wgsl` (`fs_main`: color only, since the particle pipeline
//! it shares the frame with has no depth attachment here). The MC and SS
//! renderers draw their own backdrop (mc_backdrop.rs, `fs_ground`).

use crate::gpu::bind::{entry, layout, FRAGMENT, VERTEX_FRAGMENT};
use crate::state::GpuEnvironmentParams;
use wgpu::util::DeviceExt;

pub struct EnvBackground {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
}

impl EnvBackground {
    /// `camera_buffer`: the particle renderer's camera uniform (the shader
    /// needs its inverse matrices)
    pub fn new(
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        env_view: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
        params: &GpuEnvironmentParams,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Env Background Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mc_environment.wgsl").into()),
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Env Params Buffer"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Env BG BGL"),
            entries: &[
                layout::uniform(0, VERTEX_FRAGMENT),
                layout::texture_2d(1, FRAGMENT),
                layout::sampler(2, FRAGMENT),
                layout::uniform(3, FRAGMENT),
            ],
        });

        let bind_group =
            Self::create_bind_group(device, &bind_group_layout, camera_buffer, env_view, env_sampler, &params_buffer);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Env BG Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Env BG Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::render::HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
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
            label: Some("Env BG BG"),
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

    pub fn update_params(&self, queue: &wgpu::Queue, params: &GpuEnvironmentParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Fill `target` with the backdrop (clears it first)
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Env Background Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
