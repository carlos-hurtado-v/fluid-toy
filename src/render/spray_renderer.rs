//! Spray particle renderer — billboard point sprites with additive blending

use wgpu::util::DeviceExt;

use crate::render::camera::GpuCameraParams;
use crate::state::GpuSprayRenderParams;

pub struct SprayRenderer {
    pipeline: wgpu::RenderPipeline,
    msaa_pipeline: Option<wgpu::RenderPipeline>,
    bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    render_params_buffer: wgpu::Buffer,
    max_particles: u32,
}

impl SprayRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_params: &GpuCameraParams,
        spray_buffer: &wgpu::Buffer,
        render_params: &GpuSprayRenderParams,
        msaa_sample_count: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spray Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/spray_render.wgsl").into()),
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spray Camera Buffer"),
            contents: bytemuck::bytes_of(camera_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let render_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spray Render Params Buffer"),
            contents: bytemuck::bytes_of(render_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spray Render BGL"),
            entries: &[
                // Camera
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Spray particles (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Render params
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Spray Render BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spray_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: render_params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Spray Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Alpha blend: spray partially occludes background (mist/cloud look)
        let blend_state = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Spray Render Pipeline"),
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
                    format: surface_format,
                    blend: Some(blend_state),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Transparent — don't occlude water
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // MSAA variant
        let msaa_pipeline = if msaa_sample_count > 1 {
            Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Spray Render MSAA Pipeline"),
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
                        format: surface_format,
                        blend: Some(blend_state),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: msaa_sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            }))
        } else {
            None
        };

        Self {
            pipeline,
            msaa_pipeline,
            bind_group,
            camera_buffer,
            render_params_buffer,
            max_particles: render_params.max_particles,
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, params: &GpuCameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_params(&self, queue: &wgpu::Queue, params: &GpuSprayRenderParams) {
        queue.write_buffer(&self.render_params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Render spray (single-sampled pass, e.g. MC background pass)
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        // 6 vertices per billboard quad, one instance per spray particle
        render_pass.draw(0..6, 0..self.max_particles);
    }

    /// Render using the MSAA pipeline (for MC water pass)
    pub fn render_msaa<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        let pipeline = self.msaa_pipeline.as_ref().unwrap_or(&self.pipeline);
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..6, 0..self.max_particles);
    }
}
