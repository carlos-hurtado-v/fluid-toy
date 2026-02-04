//! Screen-space fluid renderer for realistic water rendering
//!
//! Multi-pass approach:
//! 1. Depth pass - render particles as spheres, output eye-space depth
//! 2. Blur pass - bilateral filter to smooth depth
//! 3. Thickness pass - additive rendering for absorption
//! 4. Composite pass - final water surface with Fresnel, refraction, specular

use wgpu::util::DeviceExt;

use crate::render::camera::GpuCameraParams;

/// GPU-compatible fluid rendering parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFluidParams {
    pub particle_radius: f32,
    pub screen_width: f32,
    pub screen_height: f32,
    pub near: f32,
    pub far: f32,
    pub _padding: [f32; 3],
}

/// GPU-compatible blur parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBlurParams {
    pub direction: [f32; 2],
    pub filter_radius: f32,
    pub blur_scale: f32,
    pub blur_depth_falloff: f32,
    pub screen_width: f32,
    pub screen_height: f32,
    pub _padding: f32,
}

/// GPU-compatible composite parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCompositeParams {
    pub water_color: [f32; 3],
    pub absorption: f32,
    pub specular_power: f32,
    pub fresnel_power: f32,
    pub fresnel_scale: f32,
    pub refraction_strength: f32,
    pub ambient: f32,
    pub screen_width: f32,
    pub screen_height: f32,
    pub _padding: f32,
}

pub struct FluidRenderer {
    // Textures
    depth_weight_texture: wgpu::Texture,  // RG16Float: (depth*weight, weight) for metaball blending
    depth_weight_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,         // Resolved depth
    depth_view: wgpu::TextureView,
    depth_texture_blurred: wgpu::Texture,
    depth_view_blurred: wgpu::TextureView,
    depth_texture_temp: wgpu::Texture,
    depth_view_temp: wgpu::TextureView,
    thickness_texture: wgpu::Texture,
    thickness_view: wgpu::TextureView,
    background_texture: wgpu::Texture,
    background_view: wgpu::TextureView,

    // Samplers
    sampler: wgpu::Sampler,

    // Pipelines
    depth_smooth_pipeline: wgpu::RenderPipeline,  // Metaball-style additive depth
    depth_resolve_pipeline: wgpu::RenderPipeline, // Resolve weighted depth
    blur_pipeline: wgpu::RenderPipeline,
    thickness_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,

    // Bind groups
    depth_smooth_bind_group: wgpu::BindGroup,
    depth_resolve_bind_group: wgpu::BindGroup,
    blur_bind_group_h: wgpu::BindGroup,
    blur_bind_group_v: wgpu::BindGroup,
    blur_bind_group_h2: wgpu::BindGroup,
    thickness_bind_group: wgpu::BindGroup,
    composite_bind_group: wgpu::BindGroup,

    // Uniform buffers
    camera_buffer: wgpu::Buffer,
    fluid_params_buffer: wgpu::Buffer,
    blur_params_buffer: wgpu::Buffer,
    composite_params_buffer: wgpu::Buffer,

    // Dimensions
    width: u32,
    height: u32,
}

impl FluidRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_params: &GpuCameraParams,
        width: u32,
        height: u32,
    ) -> Self {
        // Create textures
        let (depth_weight_texture, depth_weight_view) = create_rg16_texture(device, width, height, "Depth Weight");
        let (depth_texture, depth_view) = create_depth_texture(device, width, height, "Depth");
        let (depth_texture_blurred, depth_view_blurred) = create_depth_texture(device, width, height, "Depth Blurred");
        let (depth_texture_temp, depth_view_temp) = create_depth_texture(device, width, height, "Depth Temp");
        let (thickness_texture, thickness_view) = create_r16_texture(device, width, height, "Thickness");
        let (background_texture, background_view) = create_color_texture(device, width, height, surface_format, "Background");

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Fluid Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create uniform buffers
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Camera Buffer"),
            contents: bytemuck::bytes_of(camera_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let fluid_params = GpuFluidParams {
            particle_radius: 0.03,
            screen_width: width as f32,
            screen_height: height as f32,
            near: 0.1,
            far: 100.0,
            _padding: [0.0; 3],
        };
        let fluid_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Params Buffer"),
            contents: bytemuck::bytes_of(&fluid_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let blur_params = GpuBlurParams {
            direction: [1.0, 0.0],
            filter_radius: 10.0,
            blur_scale: 2.0,
            blur_depth_falloff: 100.0,
            screen_width: width as f32,
            screen_height: height as f32,
            _padding: 0.0,
        };
        let blur_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Blur Params Buffer"),
            contents: bytemuck::bytes_of(&blur_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let composite_params = GpuCompositeParams {
            water_color: [0.1, 0.4, 0.8],
            absorption: 2.0,
            specular_power: 64.0,
            fresnel_power: 4.0,
            fresnel_scale: 0.1,
            refraction_strength: 0.02,
            ambient: 0.4,
            screen_width: width as f32,
            screen_height: height as f32,
            _padding: 0.0,
        };
        let composite_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Composite Params Buffer"),
            contents: bytemuck::bytes_of(&composite_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create shaders
        let depth_smooth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Depth Smooth Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fluid_depth_smooth.wgsl").into()),
        });

        let depth_resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Depth Resolve Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fluid_depth_resolve.wgsl").into()),
        });

        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fluid_blur.wgsl").into()),
        });

        let thickness_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Thickness Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fluid_thickness.wgsl").into()),
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fluid_composite.wgsl").into()),
        });

        // Depth smooth pipeline (metaball-style additive blending)
        let depth_smooth_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Depth Smooth Bind Group Layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let depth_smooth_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Depth Smooth Pipeline Layout"),
            bind_group_layouts: &[&depth_smooth_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Key difference: additive blending for metaball-style depth accumulation
        let depth_smooth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Depth Smooth Pipeline"),
            layout: Some(&depth_smooth_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &depth_smooth_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 64, // SphParticle3D stride
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0, // position
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1, // velocity
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &depth_smooth_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rg16Float,
                    // Additive blending: accumulate (depth*weight, weight)
                    blend: Some(wgpu::BlendState {
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
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None, // No depth testing - we want all contributions
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let depth_smooth_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Depth Smooth Bind Group"),
            layout: &depth_smooth_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fluid_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Depth resolve pipeline (converts weighted depth to final depth)
        let depth_resolve_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Depth Resolve Bind Group Layout"),
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

        let depth_resolve_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Depth Resolve Pipeline Layout"),
            bind_group_layouts: &[&depth_resolve_bind_group_layout],
            push_constant_ranges: &[],
        });

        let depth_resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Depth Resolve Pipeline"),
            layout: Some(&depth_resolve_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &depth_resolve_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &depth_resolve_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
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

        let depth_resolve_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Depth Resolve Bind Group"),
            layout: &depth_resolve_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_weight_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Blur pipeline
        let blur_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blur Bind Group Layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blur Pipeline Layout"),
            bind_group_layouts: &[&blur_bind_group_layout],
            push_constant_ranges: &[],
        });

        let blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blur Pipeline"),
            layout: Some(&blur_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
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

        // Create blur bind groups for horizontal and vertical passes
        let blur_bind_group_h = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blur Bind Group H"),
            layout: &blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: blur_params_buffer.as_entire_binding(),
                },
            ],
        });

        let blur_bind_group_v = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blur Bind Group V"),
            layout: &blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_view_temp),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: blur_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Additional blur bind group for iterative smoothing (reads from already-blurred depth)
        let blur_bind_group_h2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blur Bind Group H2"),
            layout: &blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_view_blurred),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: blur_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Thickness pipeline
        let thickness_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Thickness Pipeline"),
            layout: Some(&depth_smooth_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &thickness_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &thickness_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
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

        let thickness_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Thickness Bind Group"),
            layout: &depth_smooth_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fluid_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Composite pipeline
        let composite_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Composite Bind Group Layout"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
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

        let composite_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Composite Pipeline Layout"),
            bind_group_layouts: &[&composite_bind_group_layout],
            push_constant_ranges: &[],
        });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Composite Pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
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

        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Composite Bind Group"),
            layout: &composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_view_blurred),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&thickness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&background_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: composite_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: camera_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            depth_weight_texture,
            depth_weight_view,
            depth_texture,
            depth_view,
            depth_texture_blurred,
            depth_view_blurred,
            depth_texture_temp,
            depth_view_temp,
            thickness_texture,
            thickness_view,
            background_texture,
            background_view,
            sampler,
            depth_smooth_pipeline,
            depth_resolve_pipeline,
            blur_pipeline,
            thickness_pipeline,
            composite_pipeline,
            depth_smooth_bind_group,
            depth_resolve_bind_group,
            blur_bind_group_h,
            blur_bind_group_v,
            blur_bind_group_h2,
            thickness_bind_group,
            composite_bind_group,
            camera_buffer,
            fluid_params_buffer,
            blur_params_buffer,
            composite_params_buffer,
            width,
            height,
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, params: &GpuCameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_params(&self, queue: &wgpu::Queue, particle_radius: f32) {
        // Use much larger radius for fluid rendering - particles need to overlap!
        // Typically 2-4x the simulation radius for seamless surface
        let render_radius = particle_radius * 3.0;
        let params = GpuFluidParams {
            particle_radius: render_radius,
            screen_width: self.width as f32,
            screen_height: self.height as f32,
            near: 0.1,
            far: 100.0,
            _padding: [0.0; 3],
        };
        queue.write_buffer(&self.fluid_params_buffer, 0, bytemuck::bytes_of(&params));
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        particle_buffer: &wgpu::Buffer,
        num_particles: u32,
        queue: &wgpu::Queue,
        background_color: &[f32; 3],
    ) {
        // Blur settings - much more aggressive for seamless surface
        let filter_radius = 20.0;  // Larger kernel
        let blur_scale = 3.0;      // Larger step size
        let depth_falloff = 50.0;  // Less aggressive depth weighting = more smoothing

        // Update blur direction for horizontal pass
        let blur_h = GpuBlurParams {
            direction: [1.0, 0.0],
            filter_radius,
            blur_scale,
            blur_depth_falloff: depth_falloff,
            screen_width: self.width as f32,
            screen_height: self.height as f32,
            _padding: 0.0,
        };
        queue.write_buffer(&self.blur_params_buffer, 0, bytemuck::bytes_of(&blur_h));

        // Pass 1a: Render particle depths with metaball-style additive blending
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fluid Depth Smooth Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.depth_weight_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None, // No depth testing for additive blending
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.depth_smooth_pipeline);
            pass.set_bind_group(0, &self.depth_smooth_bind_group, &[]);
            pass.set_vertex_buffer(0, particle_buffer.slice(..));
            pass.draw(0..6, 0..num_particles);
        }

        // Pass 1b: Resolve weighted depth to actual depth
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fluid Depth Resolve Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.depth_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.depth_resolve_pipeline);
            pass.set_bind_group(0, &self.depth_resolve_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 2a: Blur horizontal (depth -> temp)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fluid Blur H Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.depth_view_temp,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.blur_bind_group_h, &[]);
            pass.draw(0..3, 0..1);
        }

        // Update blur direction for vertical pass
        let blur_v = GpuBlurParams {
            direction: [0.0, 1.0],
            filter_radius,
            blur_scale,
            blur_depth_falloff: depth_falloff,
            screen_width: self.width as f32,
            screen_height: self.height as f32,
            _padding: 0.0,
        };
        queue.write_buffer(&self.blur_params_buffer, 0, bytemuck::bytes_of(&blur_v));

        // Pass 2b: Blur vertical (temp -> blurred)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fluid Blur V Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.depth_view_blurred,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.blur_bind_group_v, &[]);
            pass.draw(0..3, 0..1);
        }

        // Additional blur iterations for smoother, more cohesive surface
        for _ in 0..2 {
            // Blur H (blurred -> temp)
            queue.write_buffer(&self.blur_params_buffer, 0, bytemuck::bytes_of(&blur_h));
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Fluid Blur H Pass (iter)"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.depth_view_temp,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                pass.set_pipeline(&self.blur_pipeline);
                pass.set_bind_group(0, &self.blur_bind_group_h2, &[]); // Reads from blurred
                pass.draw(0..3, 0..1);
            }

            // Blur V (temp -> blurred)
            queue.write_buffer(&self.blur_params_buffer, 0, bytemuck::bytes_of(&blur_v));
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Fluid Blur V Pass (iter)"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.depth_view_blurred,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                pass.set_pipeline(&self.blur_pipeline);
                pass.set_bind_group(0, &self.blur_bind_group_v, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // Pass 3: Thickness
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fluid Thickness Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.thickness_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.thickness_pipeline);
            pass.set_bind_group(0, &self.thickness_bind_group, &[]);
            pass.set_vertex_buffer(0, particle_buffer.slice(..));
            pass.draw(0..6, 0..num_particles);
        }

        // Pass 4: Render background to texture
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Background Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.background_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: background_color[0] as f64,
                            g: background_color[1] as f64,
                            b: background_color[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // Just clear to background color
            drop(pass);
        }

        // Pass 5: Composite
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fluid Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: background_color[0] as f64,
                            g: background_color[1] as f64,
                            b: background_color[2] as f64,
                            a: 1.0,
                        }),
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
            pass.draw(0..3, 0..1);
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;

        // Recreate textures
        let (depth_weight_texture, depth_weight_view) = create_rg16_texture(device, width, height, "Depth Weight");
        let (depth_texture, depth_view) = create_depth_texture(device, width, height, "Depth");
        let (depth_texture_blurred, depth_view_blurred) = create_depth_texture(device, width, height, "Depth Blurred");
        let (depth_texture_temp, depth_view_temp) = create_depth_texture(device, width, height, "Depth Temp");
        let (thickness_texture, thickness_view) = create_r16_texture(device, width, height, "Thickness");
        let (background_texture, background_view) = create_color_texture(device, width, height, wgpu::TextureFormat::Bgra8UnormSrgb, "Background");

        self.depth_weight_texture = depth_weight_texture;
        self.depth_weight_view = depth_weight_view;
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        self.depth_texture_blurred = depth_texture_blurred;
        self.depth_view_blurred = depth_view_blurred;
        self.depth_texture_temp = depth_texture_temp;
        self.depth_view_temp = depth_view_temp;
        self.thickness_texture = thickness_texture;
        self.thickness_view = thickness_view;
        self.background_texture = background_texture;
        self.background_view = background_view;

        // Note: bind groups would need to be recreated here too for a complete implementation
        // For now, this is a simplified version
    }
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    // Use R16Float which is filterable (R32Float is not filterable by default)
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_r16_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_color_texture(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
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

fn create_rg16_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
