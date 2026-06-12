//! Spray particle renderer — billboard point sprites with additive blending

use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

use crate::render::camera::GpuCameraParams;
use crate::state::{GpuLightParams, GpuSprayRenderParams};

pub struct SprayRenderer {
    pipeline: wgpu::RenderPipeline,
    msaa_pipeline: Option<wgpu::RenderPipeline>,
    // Additive foam splats into the screen-space density field (MC mode)
    foam_density_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    render_params_buffer: wgpu::Buffer,
    light_buffer: wgpu::Buffer,
    // Water front depth for depth-aware foam splatting. Constructed with a
    // 1x1 fallback (zero depth = everything attenuated); the app rebinds the
    // MC front depth via set_depth_view() once both renderers exist and again
    // whenever the depth texture is recreated (resize, spray reset).
    depth_sampler: wgpu::Sampler,
    max_particles: u32,
}

fn build_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera_buffer: &wgpu::Buffer,
    spray_buffer: &wgpu::Buffer,
    render_params_buffer: &wgpu::Buffer,
    light_buffer: &wgpu::Buffer,
    depth_view: &wgpu::TextureView,
    depth_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Spray Render BG"),
        layout,
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: light_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(depth_sampler),
            },
        ],
    })
}

/// Format of the whitewater field target owned by the MC renderer.
/// R = surface foam (depth-gated splats, composited on the water surface),
/// G = aeration (bubbles + submerged foam, composited as in-water milkiness).
pub const FOAM_DENSITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

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

        // Sun parameters for foam shading (zeroed = sun off until first update)
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spray Light Buffer"),
            contents: bytemuck::bytes_of(&GpuLightParams::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spray Render BGL"),
            entries: &[
                // Camera (fragment too: the foam splat pass linearizes the
                // sampled water depth from the projection matrix)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                // Light params (sun direction for foam shading)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Water front depth (foam density splat pass only)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Spray Depth Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // 1x1 fallback so the bind group is valid before the MC depth arrives
        let fallback_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Spray Fallback Depth"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fallback_depth_view = fallback_depth.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = build_bind_group(
            device,
            &bind_group_layout,
            &camera_buffer,
            spray_buffer,
            &render_params_buffer,
            &light_buffer,
            &fallback_depth_view,
            &depth_sampler,
        );

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

        // === Foam density splat pipeline ===
        // Foam accumulates additively into a small R16Float buffer; the MC
        // water shader composites it as a continuous field (no per-particle
        // shading, so overlaps merge into patches instead of visible sprites).
        let foam_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spray Foam Density Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/spray_foam_density.wgsl").into(),
            ),
        });

        let additive_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let foam_density_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Spray Foam Density Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &foam_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &foam_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FOAM_DENSITY_FORMAT,
                    blend: Some(additive_blend),
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            msaa_pipeline,
            foam_density_pipeline,
            bind_group,
            bind_group_layout,
            camera_buffer,
            render_params_buffer,
            light_buffer,
            depth_sampler,
            max_particles: render_params.max_particles,
        }
    }

    /// Rebind the water front depth used by the foam density splat pass.
    /// Call after creation and whenever the depth texture or spray buffer is
    /// recreated (window resize, spray system reset).
    pub fn set_depth_view(
        &mut self,
        device: &wgpu::Device,
        spray_buffer: &wgpu::Buffer,
        depth_view: &wgpu::TextureView,
    ) {
        self.bind_group = build_bind_group(
            device,
            &self.bind_group_layout,
            &self.camera_buffer,
            spray_buffer,
            &self.render_params_buffer,
            &self.light_buffer,
            depth_view,
            &self.depth_sampler,
        );
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, params: &GpuCameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_params(&self, queue: &wgpu::Queue, params: &GpuSprayRenderParams) {
        queue.write_buffer(&self.render_params_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_light(&self, queue: &wgpu::Queue, params: &GpuLightParams) {
        queue.write_buffer(&self.light_buffer, 0, bytemuck::bytes_of(params));
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

    /// Splat foam particles additively into the screen-space density target.
    /// Always clears the target (so stale foam never lingers); only draws
    /// when `draw` is set (spray enabled).
    pub fn render_foam_density(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        draw: bool,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Foam Density Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if draw {
            pass.set_pipeline(&self.foam_density_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..6, 0..self.max_particles);
        }
    }
}
