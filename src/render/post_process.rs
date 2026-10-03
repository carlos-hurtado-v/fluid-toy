//! Post-processing renderer
//!
//! Applies post-processing effects to the rendered scene.
//! Supports: exposure, tonemapping, color grading, vignette, bloom, chromatic aberration, anamorphic streaks

use wgpu::util::DeviceExt;

use crate::state::post_process::GpuPostProcessParams;

/// Streak pyramid: level 0 is 1/8 of the scene's width and 1/4 of its height
/// (fs_streak_threshold in post_process.wgsl reads 4 x 4 bilinear taps per
/// texel: keep in sync), each further level half as wide again, so the last
/// is 1/256 of the width and its texels spread a glint over most of the screen
const STREAK_BASE_DOWNSCALE_X: u32 = 8;
const STREAK_LEVELS: u32 = 6;

/// GPU blur direction parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBlurParams {
    pub direction: [f32; 2],
    pub _padding: [f32; 2],
}

struct PostProcessBindGroups {
    composite: wgpu::BindGroup,
    bloom_threshold: wgpu::BindGroup,
    bloom_blur_h: wgpu::BindGroup,
    bloom_blur_v: wgpu::BindGroup,
    streak_threshold: wgpu::BindGroup,
    streak_down: Vec<wgpu::BindGroup>,
    streak_up: Vec<wgpu::BindGroup>,
}

pub struct PostProcessRenderer {
    // Textures
    scene_texture: wgpu::Texture,
    scene_view: wgpu::TextureView,
    bloom_texture_a: wgpu::Texture,
    bloom_view_a: wgpu::TextureView,
    bloom_texture_b: wgpu::Texture,
    bloom_view_b: wgpu::TextureView,
    // Streak pyramid (anamorphic effect): `down[k]` is the bright-pass at
    // level k, `up[k]` the streak recombined from level k down to the
    // narrowest (up[0] is what the composite adds)
    streak_down: Vec<(wgpu::Texture, wgpu::TextureView)>,
    streak_up: Vec<(wgpu::Texture, wgpu::TextureView)>,
    // FXAA intermediate texture (composite renders here, then FXAA to final output)
    fxaa_texture: wgpu::Texture,
    fxaa_view: wgpu::TextureView,

    // Sampler
    sampler: wgpu::Sampler,

    // Pipelines
    composite_pipeline: wgpu::RenderPipeline,
    bloom_threshold_pipeline: wgpu::RenderPipeline,
    streak_threshold_pipeline: wgpu::RenderPipeline,
    bloom_blur_pipeline: wgpu::RenderPipeline,
    streak_down_pipeline: wgpu::RenderPipeline,
    streak_up_pipeline: wgpu::RenderPipeline,
    fxaa_pipeline: wgpu::RenderPipeline,

    // Bind groups
    composite_bind_group: wgpu::BindGroup,
    bloom_threshold_bind_group: wgpu::BindGroup,
    bloom_blur_h_bind_group: wgpu::BindGroup,
    bloom_blur_v_bind_group: wgpu::BindGroup,
    // Streak bind groups: threshold, then one per downsample (down[k] ->
    // down[k + 1]) and one per upsample (-> up[k])
    streak_threshold_bind_group: wgpu::BindGroup,
    streak_down_bind_groups: Vec<wgpu::BindGroup>,
    streak_up_bind_groups: Vec<wgpu::BindGroup>,
    // FXAA bind group
    fxaa_bind_group: wgpu::BindGroup,
    // AO bind group (group 1 on composite pipeline)
    ao_bind_group: wgpu::BindGroup,
    ao_bind_group_layout: wgpu::BindGroupLayout,
    // One bind group per GTAO ping-pong output view (see update_ao_bind_group)
    ao_bg_cache: Vec<(wgpu::TextureView, wgpu::BindGroup)>,

    // Buffers
    params_buffer: wgpu::Buffer,
    blur_h_buffer: wgpu::Buffer,
    blur_v_buffer: wgpu::Buffer,

    // Bind group layouts (needed for recreating bind groups on resize)
    bind_group_layout: wgpu::BindGroupLayout,
    fxaa_bind_group_layout: wgpu::BindGroupLayout,

    // Surface format for scene texture (must match what fluid renderers output)
    /// Display format of the composite/FXAA output
    output_format: wgpu::TextureFormat,

    width: u32,
    height: u32,
}

impl PostProcessRenderer {
    #[allow(clippy::too_many_lines)] // frozen in scripts/size_baseline.json: may shrink, not grow
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        params: &GpuPostProcessParams,
    ) -> Self {
        // Create textures
        // Scene texture is HDR: renderers write unclipped linear radiance, which
        // exposure, bloom and ACES need (an 8-bit scene clipped it at 1.0)
        let (scene_texture, scene_view) = Self::create_texture(device, width, height, "Scene", super::HDR_FORMAT);
        // Bloom textures use HDR format for better quality
        let (bloom_texture_a, bloom_view_a) = Self::create_texture(device, width / 2, height / 2, "Bloom A", wgpu::TextureFormat::Rgba16Float);
        let (bloom_texture_b, bloom_view_b) = Self::create_texture(device, width / 2, height / 2, "Bloom B", wgpu::TextureFormat::Rgba16Float);
        let (streak_down, streak_up) = Self::create_streak_chain(device, width, height);
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

        // Streak threshold pipeline (own threshold: streaks_threshold)
        let streak_threshold_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Streak Threshold Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_streak_threshold"),
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

        // Streak pyramid: halve the width of a level
        let streak_down_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Streak Down Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_streak_down"),
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

        // Streak pyramid: mix a level with the stretched level below it
        let streak_up_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Streak Up Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_streak_up"),
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

        let groups = Self::create_bind_groups(
            device,
            &bind_group_layout,
            &scene_view,
            [&bloom_view_a, &bloom_view_b],
            &streak_down,
            &streak_up,
            &sampler,
            &params_buffer,
            [&blur_h_buffer, &blur_v_buffer],
        );

        Self {
            scene_texture,
            scene_view,
            bloom_texture_a,
            bloom_view_a,
            bloom_texture_b,
            bloom_view_b,
            streak_down,
            streak_up,
            fxaa_texture,
            fxaa_view,
            sampler,
            composite_pipeline,
            bloom_threshold_pipeline,
            streak_threshold_pipeline,
            bloom_blur_pipeline,
            streak_down_pipeline,
            streak_up_pipeline,
            fxaa_pipeline,
            composite_bind_group: groups.composite,
            bloom_threshold_bind_group: groups.bloom_threshold,
            bloom_blur_h_bind_group: groups.bloom_blur_h,
            bloom_blur_v_bind_group: groups.bloom_blur_v,
            streak_threshold_bind_group: groups.streak_threshold,
            streak_down_bind_groups: groups.streak_down,
            streak_up_bind_groups: groups.streak_up,
            fxaa_bind_group,
            ao_bind_group,
            ao_bind_group_layout,
            ao_bg_cache: Vec::new(),
            params_buffer,
            blur_h_buffer,
            blur_v_buffer,
            bind_group_layout,
            fxaa_bind_group_layout,
            output_format: surface_format,
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

    /// Streak pyramid textures for a scene of the given size: (down, up).
    /// `up` has no narrowest level (that one is read straight from `down`).
    #[allow(clippy::type_complexity)]
    fn create_streak_chain(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (Vec<(wgpu::Texture, wgpu::TextureView)>, Vec<(wgpu::Texture, wgpu::TextureView)>) {
        let level = |i: u32, label: &str| {
            Self::create_texture(device, (width / STREAK_BASE_DOWNSCALE_X) >> i, height / 4, label, wgpu::TextureFormat::Rgba16Float)
        };
        let down = (0..STREAK_LEVELS).map(|i| level(i, "Streak Down")).collect();
        let up = (0..STREAK_LEVELS - 1).map(|i| level(i, "Streak Up")).collect();
        (down, up)
    }

    /// Every bind group of the bloom / streak / composite passes. All share
    /// one layout (input, bloom, streak, sampler, params, blur direction); a
    /// pass must not bind its own target, so slots it does not read repeat
    /// its input.
    #[allow(clippy::too_many_arguments)]
    fn create_bind_groups(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        scene_view: &wgpu::TextureView,
        [bloom_a, bloom_b]: [&wgpu::TextureView; 2],
        streak_down: &[(wgpu::Texture, wgpu::TextureView)],
        streak_up: &[(wgpu::Texture, wgpu::TextureView)],
        sampler: &wgpu::Sampler,
        params_buffer: &wgpu::Buffer,
        [blur_h, blur_v]: [&wgpu::Buffer; 2],
    ) -> PostProcessBindGroups {
        let group = |input: &wgpu::TextureView, bloom: &wgpu::TextureView, streak: &wgpu::TextureView, blur: &wgpu::Buffer| {
            Self::create_bind_group(device, layout, input, bloom, streak, sampler, params_buffer, blur)
        };
        let levels = streak_down.len();
        PostProcessBindGroups {
            // Composite: scene + bloom + streak
            composite: group(scene_view, bloom_a, &streak_up[0].1, blur_h),
            // Bloom threshold: scene -> bloom_a
            bloom_threshold: group(scene_view, scene_view, scene_view, blur_h),
            // Bloom blur H: bloom_a -> bloom_b, then V: bloom_b -> bloom_a
            bloom_blur_h: group(bloom_a, bloom_a, bloom_a, blur_h),
            bloom_blur_v: group(bloom_b, bloom_b, bloom_b, blur_v),
            // Streak threshold: scene -> down[0]
            streak_threshold: group(scene_view, scene_view, scene_view, blur_h),
            // Streak down: down[k] -> down[k + 1]
            streak_down: (0..levels - 1)
                .map(|k| group(&streak_down[k].1, &streak_down[k].1, &streak_down[k].1, blur_h))
                .collect(),
            // Streak up: down[k] + the level below (its `up`, or the
            // narrowest `down`) -> up[k]
            streak_up: (0..levels - 1)
                .map(|k| {
                    let below = if k + 2 == levels { &streak_down[k + 1].1 } else { &streak_up[k + 1].1 };
                    group(&streak_down[k].1, &streak_down[k].1, below, blur_h)
                })
                .collect(),
        }
    }

    #[allow(clippy::too_many_arguments)]
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

    /// Update the AO bind group with the GTAO output texture. GTAO ping-pongs
    /// between two output views, so keep a bind group per view and only
    /// create on first sight (or after a resize swaps the views out).
    pub fn update_ao_bind_group(&mut self, device: &wgpu::Device, ao_view: &wgpu::TextureView) {
        if let Some((_, bg)) = self.ao_bg_cache.iter().find(|(view, _)| view == ao_view) {
            self.ao_bind_group = bg.clone();
            return;
        }
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess AO BG"),
            layout: &self.ao_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ao_view),
                },
            ],
        });
        // Only the current ping-pong pair is ever live
        if self.ao_bg_cache.len() >= 2 {
            self.ao_bg_cache.remove(0);
        }
        self.ao_bg_cache.push((ao_view.clone(), bg.clone()));
        self.ao_bind_group = bg;
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

        // Anamorphic streaks: bright pixels -> level 0, down the pyramid,
        // then back up mixing each level with the stretched one below
        if streaks_enabled {
            let mut streak_pass = |label: &str, target: &wgpu::TextureView, pipeline: &wgpu::RenderPipeline, bind_group: &wgpu::BindGroup| {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(label),
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
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_bind_group(1, &self.ao_bind_group, &[]);
                pass.draw(0..3, 0..1);
            };
            streak_pass("Streak Threshold Pass", &self.streak_down[0].1, &self.streak_threshold_pipeline, &self.streak_threshold_bind_group);
            for (k, bind_group) in self.streak_down_bind_groups.iter().enumerate() {
                streak_pass("Streak Down Pass", &self.streak_down[k + 1].1, &self.streak_down_pipeline, bind_group);
            }
            for (k, bind_group) in self.streak_up_bind_groups.iter().enumerate().rev() {
                streak_pass("Streak Up Pass", &self.streak_up[k].1, &self.streak_up_pipeline, bind_group);
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
        let (scene_texture, scene_view) = Self::create_texture(device, width, height, "Scene", super::HDR_FORMAT);
        let (bloom_texture_a, bloom_view_a) = Self::create_texture(device, width / 2, height / 2, "Bloom A", wgpu::TextureFormat::Rgba16Float);
        let (bloom_texture_b, bloom_view_b) = Self::create_texture(device, width / 2, height / 2, "Bloom B", wgpu::TextureFormat::Rgba16Float);
        let (streak_down, streak_up) = Self::create_streak_chain(device, width, height);
        let (fxaa_texture, fxaa_view) = Self::create_texture(device, width, height, "FXAA", self.output_format);

        self.scene_texture = scene_texture;
        self.scene_view = scene_view;
        self.bloom_texture_a = bloom_texture_a;
        self.bloom_view_a = bloom_view_a;
        self.bloom_texture_b = bloom_texture_b;
        self.bloom_view_b = bloom_view_b;
        self.streak_down = streak_down;
        self.streak_up = streak_up;
        self.fxaa_texture = fxaa_texture;
        self.fxaa_view = fxaa_view;

        // Recreate bind groups
        let groups = Self::create_bind_groups(
            device,
            &self.bind_group_layout,
            &self.scene_view,
            [&self.bloom_view_a, &self.bloom_view_b],
            &self.streak_down,
            &self.streak_up,
            &self.sampler,
            &self.params_buffer,
            [&self.blur_h_buffer, &self.blur_v_buffer],
        );
        self.composite_bind_group = groups.composite;
        self.bloom_threshold_bind_group = groups.bloom_threshold;
        self.bloom_blur_h_bind_group = groups.bloom_blur_h;
        self.bloom_blur_v_bind_group = groups.bloom_blur_v;
        self.streak_threshold_bind_group = groups.streak_threshold;
        self.streak_down_bind_groups = groups.streak_down;
        self.streak_up_bind_groups = groups.streak_up;

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
