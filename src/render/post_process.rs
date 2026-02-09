//! Post-processing renderer
//!
//! Applies post-processing effects to the rendered scene.
//! Supports: exposure, tonemapping, color grading, vignette, bloom, chromatic aberration, anamorphic streaks

use wgpu::util::DeviceExt;

use crate::state::post_process::GpuPostProcessParams;

/// GPU blur direction parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBlurParams {
    pub direction: [f32; 2],
    pub _padding: [f32; 2],
}

pub struct PostProcessRenderer {
    // Textures
    scene_texture: wgpu::Texture,
    scene_view: wgpu::TextureView,
    bloom_texture_a: wgpu::Texture,
    bloom_view_a: wgpu::TextureView,
    bloom_texture_b: wgpu::Texture,
    bloom_view_b: wgpu::TextureView,
    // Streak textures (for anamorphic effect)
    streak_texture_a: wgpu::Texture,
    streak_view_a: wgpu::TextureView,
    streak_texture_b: wgpu::Texture,
    streak_view_b: wgpu::TextureView,
    // FXAA intermediate texture (composite renders here, then FXAA to final output)
    fxaa_texture: wgpu::Texture,
    fxaa_view: wgpu::TextureView,

    // Sampler
    sampler: wgpu::Sampler,

    // Pipelines
    composite_pipeline: wgpu::RenderPipeline,
    bloom_threshold_pipeline: wgpu::RenderPipeline,
    bloom_blur_pipeline: wgpu::RenderPipeline,
    streak_blur_pipeline: wgpu::RenderPipeline,
    fxaa_pipeline: wgpu::RenderPipeline,

    // Bind groups
    composite_bind_group: wgpu::BindGroup,
    bloom_threshold_bind_group: wgpu::BindGroup,
    bloom_blur_h_bind_group: wgpu::BindGroup,
    bloom_blur_v_bind_group: wgpu::BindGroup,
    // Streak bind groups (threshold reuses bloom, just need blur)
    streak_threshold_bind_group: wgpu::BindGroup,
    streak_blur_h1_bind_group: wgpu::BindGroup,
    streak_blur_h2_bind_group: wgpu::BindGroup,
    // FXAA bind group
    fxaa_bind_group: wgpu::BindGroup,
    // AO bind group (group 1 on composite pipeline)
    ao_bind_group: wgpu::BindGroup,
    ao_bind_group_layout: wgpu::BindGroupLayout,

    // Buffers
    params_buffer: wgpu::Buffer,
    blur_h_buffer: wgpu::Buffer,
    blur_v_buffer: wgpu::Buffer,

    // Bind group layouts (needed for recreating bind groups on resize)
    bind_group_layout: wgpu::BindGroupLayout,
    fxaa_bind_group_layout: wgpu::BindGroupLayout,

    // Surface format for scene texture (must match what fluid renderers output)
    scene_format: wgpu::TextureFormat,

    width: u32,
    height: u32,
}

impl PostProcessRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        params: &GpuPostProcessParams,
    ) -> Self {
        // Create textures
        // Scene texture uses surface format to match fluid renderer output
        let (scene_texture, scene_view) = Self::create_texture(device, width, height, "Scene", surface_format);
        // Bloom textures use HDR format for better quality
        let (bloom_texture_a, bloom_view_a) = Self::create_texture(device, width / 2, height / 2, "Bloom A", wgpu::TextureFormat::Rgba16Float);
        let (bloom_texture_b, bloom_view_b) = Self::create_texture(device, width / 2, height / 2, "Bloom B", wgpu::TextureFormat::Rgba16Float);
        // Streak textures (can be lower res for performance, wider blur hides it)
        let (streak_texture_a, streak_view_a) = Self::create_texture(device, width / 4, height / 4, "Streak A", wgpu::TextureFormat::Rgba16Float);
        let (streak_texture_b, streak_view_b) = Self::create_texture(device, width / 4, height / 4, "Streak B", wgpu::TextureFormat::Rgba16Float);
        // FXAA intermediate texture (composite renders here when FXAA enabled)
        let (fxaa_texture, fxaa_view) = Self::create_texture(device, width, height, "FXAA", surface_format);

        // Sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PostProcess Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Buffers
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PostProcess Params Buffer"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let blur_h_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Blur H Buffer"),
            contents: bytemuck::bytes_of(&GpuBlurParams {
                direction: [1.0, 0.0],
                _padding: [0.0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let blur_v_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Blur V Buffer"),
            contents: bytemuck::bytes_of(&GpuBlurParams {
                direction: [0.0, 1.0],
                _padding: [0.0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PostProcess Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/post_process.wgsl").into()),
        });

        // Bind group layout for composite pass (6 bindings now)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Bind Group Layout"),
            entries: &[
                // Scene texture (binding 0)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Bloom texture (binding 1)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Streak texture (binding 2)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler (binding 3)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Params (binding 4)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Blur params (binding 5)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // AO bind group layout (group 1: just an AO texture)
        let ao_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess AO BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Create placeholder AO texture (1x1 white = no occlusion)
        let ao_placeholder = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("AO Placeholder"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Initialize to 1.0 (no occlusion)
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ao_placeholder,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &1.0f32.to_le_bytes(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let ao_placeholder_view = ao_placeholder.create_view(&wgpu::TextureViewDescriptor::default());

        let ao_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess AO BG"),
            layout: &ao_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ao_placeholder_view),
                },
            ],
        });

        // Pipeline layout (group 0 = scene/bloom/streak/params, group 1 = AO)
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, &ao_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Composite pipeline (final pass)
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PostProcess Composite Pipeline"),
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

        // Bloom threshold pipeline
        let bloom_threshold_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Threshold Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_bloom_threshold"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
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

        // Bloom blur pipeline
        let bloom_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Blur Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_bloom_blur"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
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

        // Streak blur pipeline (wide horizontal blur for anamorphic effect)
        let streak_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Streak Blur Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_streak_blur"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
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

        // FXAA shader and pipeline
        let fxaa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FXAA Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fxaa.wgsl").into()),
        });

        let fxaa_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FXAA Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let fxaa_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FXAA Pipeline Layout"),
            bind_group_layouts: &[&fxaa_bind_group_layout],
            push_constant_ranges: &[],
        });

        let fxaa_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("FXAA Pipeline"),
            layout: Some(&fxaa_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &fxaa_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fxaa_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
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

        let fxaa_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FXAA Bind Group"),
            layout: &fxaa_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&fxaa_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create bind groups
        // Composite: scene + bloom + streak
        let composite_bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &scene_view,
            &bloom_view_a,
            &streak_view_a,
            &sampler,
            &params_buffer,
            &blur_h_buffer,
        );

        // Bloom threshold: scene -> bloom_a (use bloom_b and streak_b as dummies)
        let bloom_threshold_bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &scene_view,
            &bloom_view_b,
            &streak_view_b,
            &sampler,
            &params_buffer,
            &blur_h_buffer,
        );

        // Bloom blur H: bloom_a -> bloom_b
        let bloom_blur_h_bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &bloom_view_a,
            &bloom_view_a,
            &streak_view_b,
            &sampler,
            &params_buffer,
            &blur_h_buffer,
        );

        // Bloom blur V: bloom_b -> bloom_a
        let bloom_blur_v_bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &bloom_view_b,
            &bloom_view_b,
            &streak_view_b,
            &sampler,
            &params_buffer,
            &blur_v_buffer,
        );

        // Streak threshold: scene -> streak_a (use bloom_b and streak_b as dummies)
        let streak_threshold_bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &scene_view,
            &bloom_view_b,
            &streak_view_b,
            &sampler,
            &params_buffer,
            &blur_h_buffer,
        );

        // Streak blur H pass 1: streak_a -> streak_b
        let streak_blur_h1_bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &streak_view_a,
            &bloom_view_b,
            &streak_view_a,
            &sampler,
            &params_buffer,
            &blur_h_buffer,
        );

        // Streak blur H pass 2: streak_b -> streak_a (second pass for extra width)
        let streak_blur_h2_bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &streak_view_b,
            &bloom_view_b,
            &streak_view_b,
            &sampler,
            &params_buffer,
            &blur_h_buffer,
        );

        Self {
            scene_texture,
            scene_view,
            bloom_texture_a,
            bloom_view_a,
            bloom_texture_b,
            bloom_view_b,
            streak_texture_a,
            streak_view_a,
            streak_texture_b,
            streak_view_b,
            fxaa_texture,
            fxaa_view,
            sampler,
            composite_pipeline,
            bloom_threshold_pipeline,
            bloom_blur_pipeline,
            streak_blur_pipeline,
            fxaa_pipeline,
            composite_bind_group,
            bloom_threshold_bind_group,
            bloom_blur_h_bind_group,
            bloom_blur_v_bind_group,
            streak_threshold_bind_group,
            streak_blur_h1_bind_group,
            streak_blur_h2_bind_group,
            fxaa_bind_group,
            ao_bind_group,
            ao_bind_group_layout,
            params_buffer,
            blur_h_buffer,
            blur_v_buffer,
            bind_group_layout,
            fxaa_bind_group_layout,
            scene_format: surface_format,
            width,
            height,
        }
    }

    fn create_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        label: &str,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        scene_view: &wgpu::TextureView,
        bloom_view: &wgpu::TextureView,
        streak_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        params_buffer: &wgpu::Buffer,
        blur_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(streak_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: blur_buffer.as_entire_binding(),
                },
            ],
        })
    }

    pub fn update_params(&self, queue: &wgpu::Queue, params: &GpuPostProcessParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Update the AO bind group with the GTAO output texture
    pub fn update_ao_bind_group(&mut self, device: &wgpu::Device, ao_view: &wgpu::TextureView) {
        self.ao_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess AO BG"),
            layout: &self.ao_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ao_view),
                },
            ],
        });
    }

    /// Get the scene texture view for rendering the scene to
    pub fn scene_view(&self) -> &wgpu::TextureView {
        &self.scene_view
    }

    /// Apply post-processing and render to the output view
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        bloom_enabled: bool,
        streaks_enabled: bool,
        fxaa_enabled: bool,
    ) {
        // If bloom is enabled, do bloom passes
        if bloom_enabled {
            // Pass 1: Extract bright pixels -> bloom_a
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Bloom Threshold Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.bloom_view_a,
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
                pass.set_pipeline(&self.bloom_threshold_pipeline);
                pass.set_bind_group(0, &self.bloom_threshold_bind_group, &[]);
                pass.set_bind_group(1, &self.ao_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }

            // Pass 2: Horizontal blur -> bloom_b
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Bloom Blur H Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.bloom_view_b,
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
                pass.set_pipeline(&self.bloom_blur_pipeline);
                pass.set_bind_group(0, &self.bloom_blur_h_bind_group, &[]);
                pass.set_bind_group(1, &self.ao_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }

            // Pass 3: Vertical blur -> bloom_a (final bloom result)
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Bloom Blur V Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.bloom_view_a,
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
                pass.set_pipeline(&self.bloom_blur_pipeline);
                pass.set_bind_group(0, &self.bloom_blur_v_bind_group, &[]);
                pass.set_bind_group(1, &self.ao_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // If anamorphic streaks are enabled
        if streaks_enabled {
            // Pass 1: Extract bright pixels -> streak_a
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Streak Threshold Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.streak_view_a,
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
                pass.set_pipeline(&self.bloom_threshold_pipeline);
                pass.set_bind_group(0, &self.streak_threshold_bind_group, &[]);
                pass.set_bind_group(1, &self.ao_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }

            // Pass 2: Wide horizontal blur -> streak_b
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Streak Blur H Pass 1"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.streak_view_b,
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
                pass.set_pipeline(&self.streak_blur_pipeline);
                pass.set_bind_group(0, &self.streak_blur_h1_bind_group, &[]);
                pass.set_bind_group(1, &self.ao_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }

            // Pass 3: Second wide horizontal blur -> streak_a (for extra width)
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Streak Blur H Pass 2"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.streak_view_a,
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
                pass.set_pipeline(&self.streak_blur_pipeline);
                pass.set_bind_group(0, &self.streak_blur_h2_bind_group, &[]);
                pass.set_bind_group(1, &self.ao_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // Final composite pass
        // If FXAA enabled, render to fxaa_texture; otherwise render directly to output
        let composite_target = if fxaa_enabled {
            &self.fxaa_view
        } else {
            output_view
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PostProcess Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: composite_target,
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
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &self.composite_bind_group, &[]);
            pass.set_bind_group(1, &self.ao_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // FXAA pass (if enabled)
        if fxaa_enabled {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("FXAA Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
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
            pass.set_pipeline(&self.fxaa_pipeline);
            pass.set_bind_group(0, &self.fxaa_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }

        self.width = width;
        self.height = height;

        // Recreate textures
        let (scene_texture, scene_view) = Self::create_texture(device, width, height, "Scene", self.scene_format);
        let (bloom_texture_a, bloom_view_a) = Self::create_texture(device, width / 2, height / 2, "Bloom A", wgpu::TextureFormat::Rgba16Float);
        let (bloom_texture_b, bloom_view_b) = Self::create_texture(device, width / 2, height / 2, "Bloom B", wgpu::TextureFormat::Rgba16Float);
        let (streak_texture_a, streak_view_a) = Self::create_texture(device, width / 4, height / 4, "Streak A", wgpu::TextureFormat::Rgba16Float);
        let (streak_texture_b, streak_view_b) = Self::create_texture(device, width / 4, height / 4, "Streak B", wgpu::TextureFormat::Rgba16Float);
        let (fxaa_texture, fxaa_view) = Self::create_texture(device, width, height, "FXAA", self.scene_format);

        self.scene_texture = scene_texture;
        self.scene_view = scene_view;
        self.bloom_texture_a = bloom_texture_a;
        self.bloom_view_a = bloom_view_a;
        self.bloom_texture_b = bloom_texture_b;
        self.bloom_view_b = bloom_view_b;
        self.streak_texture_a = streak_texture_a;
        self.streak_view_a = streak_view_a;
        self.streak_texture_b = streak_texture_b;
        self.streak_view_b = streak_view_b;
        self.fxaa_texture = fxaa_texture;
        self.fxaa_view = fxaa_view;

        // Recreate bind groups
        self.composite_bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.scene_view,
            &self.bloom_view_a,
            &self.streak_view_a,
            &self.sampler,
            &self.params_buffer,
            &self.blur_h_buffer,
        );

        self.bloom_threshold_bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.scene_view,
            &self.bloom_view_b,
            &self.streak_view_b,
            &self.sampler,
            &self.params_buffer,
            &self.blur_h_buffer,
        );

        self.bloom_blur_h_bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.bloom_view_a,
            &self.bloom_view_a,
            &self.streak_view_b,
            &self.sampler,
            &self.params_buffer,
            &self.blur_h_buffer,
        );

        self.bloom_blur_v_bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.bloom_view_b,
            &self.bloom_view_b,
            &self.streak_view_b,
            &self.sampler,
            &self.params_buffer,
            &self.blur_v_buffer,
        );

        self.streak_threshold_bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.scene_view,
            &self.bloom_view_b,
            &self.streak_view_b,
            &self.sampler,
            &self.params_buffer,
            &self.blur_h_buffer,
        );

        self.streak_blur_h1_bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.streak_view_a,
            &self.bloom_view_b,
            &self.streak_view_a,
            &self.sampler,
            &self.params_buffer,
            &self.blur_h_buffer,
        );

        self.streak_blur_h2_bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.streak_view_b,
            &self.bloom_view_b,
            &self.streak_view_b,
            &self.sampler,
            &self.params_buffer,
            &self.blur_h_buffer,
        );

        // Recreate FXAA bind group
        self.fxaa_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FXAA Bind Group"),
            layout: &self.fxaa_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.fxaa_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
    }
}
