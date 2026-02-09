//! GTAO (Ground Truth Ambient Occlusion) renderer
//!
//! Implements Jimenez 2016 horizon-based AO at half resolution with
//! bilateral blur and temporal accumulation.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// GPU-compatible GTAO parameters (64 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuGtaoParams {
    pub radius: f32,
    pub falloff_start: f32,
    pub num_steps: u32,
    pub frame_index: u32,
    pub half_res: [f32; 2],
    pub inv_half_res: [f32; 2],
    pub full_res: [f32; 2],
    pub inv_full_res: [f32; 2],
    pub temporal_blend: f32,
    pub thickness: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

/// Previous frame's view*projection matrix (64 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuPrevViewProjection {
    pub matrix: [[f32; 4]; 4],
}

pub struct GtaoRenderer {
    // Half-res textures
    linear_depth_texture: wgpu::Texture,
    linear_depth_view: wgpu::TextureView,
    ao_working_texture: wgpu::Texture,
    ao_working_view: wgpu::TextureView,
    ao_result_textures: [wgpu::Texture; 2],
    ao_result_views: [wgpu::TextureView; 2],

    // Uniform buffers
    params_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    prev_vp_buffer: wgpu::Buffer,

    // Pipelines
    prefilter_pipeline: wgpu::ComputePipeline,
    gtao_pipeline: wgpu::ComputePipeline,
    blur_h_pipeline: wgpu::ComputePipeline,
    blur_v_pipeline: wgpu::ComputePipeline,
    temporal_pipeline: wgpu::ComputePipeline,

    // Bind group layouts (for recreation on resize)
    prefilter_bgl: wgpu::BindGroupLayout,
    gtao_bgl: wgpu::BindGroupLayout,
    blur_bgl: wgpu::BindGroupLayout,
    temporal_bgl: wgpu::BindGroupLayout,

    // Bind groups
    prefilter_bind_group: wgpu::BindGroup,
    gtao_bind_group: wgpu::BindGroup,
    // blur_h: ao_working → ao_result[output_idx]
    // blur_v: ao_result[output_idx] → ao_working
    // temporal: ao_working + ao_result[history_idx] → ao_result[output_idx]
    // We create bind groups for both ping-pong states
    blur_h_bind_groups: [wgpu::BindGroup; 2], // [output_idx=0, output_idx=1]
    blur_v_bind_groups: [wgpu::BindGroup; 2],
    temporal_bind_groups: [wgpu::BindGroup; 2],

    // Dimensions
    half_width: u32,
    half_height: u32,
    frame_counter: u32,
}

impl GtaoRenderer {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let half_width = (width / 2).max(1);
        let half_height = (height / 2).max(1);

        // Create textures
        let linear_depth_texture = Self::create_r32float_texture(device, half_width, half_height, "GTAO Linear Depth");
        let linear_depth_view = linear_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let ao_working_texture = Self::create_ao_texture(device, half_width, half_height, "GTAO AO Working");
        let ao_working_view = ao_working_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let ao_result_textures = [
            Self::create_ao_texture(device, half_width, half_height, "GTAO AO Result 0"),
            Self::create_ao_texture(device, half_width, half_height, "GTAO AO Result 1"),
        ];
        let ao_result_views = [
            ao_result_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            ao_result_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];

        // Create uniform buffers
        let params = GpuGtaoParams {
            radius: 0.5,
            falloff_start: 0.2,
            num_steps: 8,
            frame_index: 0,
            half_res: [half_width as f32, half_height as f32],
            inv_half_res: [1.0 / half_width as f32, 1.0 / half_height as f32],
            full_res: [width as f32, height as f32],
            inv_full_res: [1.0 / width as f32, 1.0 / height as f32],
            temporal_blend: 0.15,
            thickness: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GTAO Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GTAO Camera"),
            size: std::mem::size_of::<crate::render::GpuCameraParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let prev_vp = GpuPrevViewProjection {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let prev_vp_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GTAO Prev VP"),
            contents: bytemuck::bytes_of(&prev_vp),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GTAO Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/gtao.wgsl").into()),
        });

        // === Bind group layouts ===

        // Prefilter: depth_input(0), linear_depth_output(1), params(2), camera(3)
        let prefilter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GTAO Prefilter BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // GTAO main: only bindings actually used by gtao_main entry point
        let gtao_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GTAO Main BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Blur: only bindings used by blur_h/blur_v entry points
        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GTAO Blur BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Temporal: only bindings used by temporal_accumulate entry point
        let temporal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GTAO Temporal BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // === Pipelines ===
        let prefilter_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GTAO Prefilter Layout"),
            bind_group_layouts: &[&prefilter_bgl],
            push_constant_ranges: &[],
        });
        let prefilter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GTAO Prefilter Pipeline"),
            layout: Some(&prefilter_layout),
            module: &shader,
            entry_point: Some("prefilter_depth"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let gtao_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GTAO Main Layout"),
            bind_group_layouts: &[&gtao_bgl],
            push_constant_ranges: &[],
        });
        let gtao_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GTAO Main Pipeline"),
            layout: Some(&gtao_layout),
            module: &shader,
            entry_point: Some("gtao_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GTAO Blur Layout"),
            bind_group_layouts: &[&blur_bgl],
            push_constant_ranges: &[],
        });
        let blur_h_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GTAO Blur H Pipeline"),
            layout: Some(&blur_layout),
            module: &shader,
            entry_point: Some("blur_h"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let blur_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GTAO Blur V Pipeline"),
            layout: Some(&blur_layout),
            module: &shader,
            entry_point: Some("blur_v"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let temporal_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GTAO Temporal Layout"),
            bind_group_layouts: &[&temporal_bgl],
            push_constant_ranges: &[],
        });
        let temporal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GTAO Temporal Pipeline"),
            layout: Some(&temporal_layout),
            module: &shader,
            entry_point: Some("temporal_accumulate"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create a placeholder depth view for bind groups (will be replaced each frame)
        let placeholder_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GTAO Placeholder Depth"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let placeholder_depth_view = placeholder_depth.create_view(&wgpu::TextureViewDescriptor::default());

        // === Create bind groups ===
        let prefilter_bind_group = Self::create_prefilter_bg(
            device, &prefilter_bgl, &placeholder_depth_view,
            &linear_depth_view, &params_buffer, &camera_buffer,
        );

        let gtao_bind_group = Self::create_gtao_bg(
            device, &gtao_bgl,
            &linear_depth_view, &params_buffer, &camera_buffer, &ao_working_view,
        );

        let blur_h_bind_groups = [
            Self::create_blur_bg(device, &blur_bgl, &linear_depth_view,
                &params_buffer, &ao_working_view, &ao_result_views[0]),
            Self::create_blur_bg(device, &blur_bgl, &linear_depth_view,
                &params_buffer, &ao_working_view, &ao_result_views[1]),
        ];

        let blur_v_bind_groups = [
            Self::create_blur_bg(device, &blur_bgl, &linear_depth_view,
                &params_buffer, &ao_result_views[0], &ao_working_view),
            Self::create_blur_bg(device, &blur_bgl, &linear_depth_view,
                &params_buffer, &ao_result_views[1], &ao_working_view),
        ];

        let temporal_bind_groups = [
            Self::create_temporal_bg(device, &temporal_bgl, &linear_depth_view,
                &params_buffer, &camera_buffer, &ao_working_view,
                &ao_result_views[1], &ao_result_views[0], &prev_vp_buffer),
            Self::create_temporal_bg(device, &temporal_bgl, &linear_depth_view,
                &params_buffer, &camera_buffer, &ao_working_view,
                &ao_result_views[0], &ao_result_views[1], &prev_vp_buffer),
        ];

        Self {
            linear_depth_texture,
            linear_depth_view,
            ao_working_texture,
            ao_working_view,
            ao_result_textures,
            ao_result_views,
            params_buffer,
            camera_buffer,
            prev_vp_buffer,
            prefilter_pipeline,
            gtao_pipeline,
            blur_h_pipeline,
            blur_v_pipeline,
            temporal_pipeline,
            prefilter_bgl,
            gtao_bgl,
            blur_bgl,
            temporal_bgl,
            prefilter_bind_group,
            gtao_bind_group,
            blur_h_bind_groups,
            blur_v_bind_groups,
            temporal_bind_groups,
            half_width,
            half_height,
            frame_counter: 0,
        }
    }

    fn create_r32float_texture(device: &wgpu::Device, w: u32, h: u32, label: &str) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_ao_texture(device: &wgpu::Device, w: u32, h: u32, label: &str) -> wgpu::Texture {
        // Use R32Float because R8Unorm doesn't support STORAGE_BINDING on most GPUs
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_prefilter_bg(
        device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
        depth_view: &wgpu::TextureView, linear_depth_view: &wgpu::TextureView,
        params_buffer: &wgpu::Buffer, camera_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GTAO Prefilter BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(linear_depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: camera_buffer.as_entire_binding() },
            ],
        })
    }

    fn create_gtao_bg(
        device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
        linear_depth_view: &wgpu::TextureView,
        params_buffer: &wgpu::Buffer, camera_buffer: &wgpu::Buffer,
        ao_output_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GTAO Main BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(linear_depth_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(ao_output_view) },
            ],
        })
    }

    fn create_blur_bg(
        device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
        linear_depth_view: &wgpu::TextureView,
        params_buffer: &wgpu::Buffer,
        ao_input_view: &wgpu::TextureView, ao_output_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GTAO Blur BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(linear_depth_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(ao_input_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(ao_output_view) },
            ],
        })
    }

    fn create_temporal_bg(
        device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
        linear_depth_view: &wgpu::TextureView,
        params_buffer: &wgpu::Buffer, camera_buffer: &wgpu::Buffer,
        ao_current_view: &wgpu::TextureView,
        ao_history_view: &wgpu::TextureView, ao_output_view: &wgpu::TextureView,
        prev_vp_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GTAO Temporal BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(linear_depth_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(ao_current_view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(ao_history_view) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(ao_output_view) },
                wgpu::BindGroupEntry { binding: 10, resource: prev_vp_buffer.as_entire_binding() },
            ],
        })
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let half_width = (width / 2).max(1);
        let half_height = (height / 2).max(1);

        if half_width == self.half_width && half_height == self.half_height {
            return;
        }
        self.half_width = half_width;
        self.half_height = half_height;

        // Recreate textures
        self.linear_depth_texture = Self::create_r32float_texture(device, half_width, half_height, "GTAO Linear Depth");
        self.linear_depth_view = self.linear_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.ao_working_texture = Self::create_ao_texture(device, half_width, half_height, "GTAO AO Working");
        self.ao_working_view = self.ao_working_texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.ao_result_textures = [
            Self::create_ao_texture(device, half_width, half_height, "GTAO AO Result 0"),
            Self::create_ao_texture(device, half_width, half_height, "GTAO AO Result 1"),
        ];
        self.ao_result_views = [
            self.ao_result_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            self.ao_result_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];

        // Bind groups will be recreated in rebuild_bind_groups
    }

    /// Rebuild all bind groups (call after resize or when depth view changes)
    pub fn rebuild_bind_groups(&mut self, device: &wgpu::Device, depth_view: &wgpu::TextureView) {
        self.prefilter_bind_group = Self::create_prefilter_bg(
            device, &self.prefilter_bgl, depth_view,
            &self.linear_depth_view, &self.params_buffer, &self.camera_buffer,
        );

        self.gtao_bind_group = Self::create_gtao_bg(
            device, &self.gtao_bgl,
            &self.linear_depth_view, &self.params_buffer, &self.camera_buffer,
            &self.ao_working_view,
        );

        self.blur_h_bind_groups = [
            Self::create_blur_bg(device, &self.blur_bgl, &self.linear_depth_view,
                &self.params_buffer, &self.ao_working_view, &self.ao_result_views[0]),
            Self::create_blur_bg(device, &self.blur_bgl, &self.linear_depth_view,
                &self.params_buffer, &self.ao_working_view, &self.ao_result_views[1]),
        ];

        self.blur_v_bind_groups = [
            Self::create_blur_bg(device, &self.blur_bgl, &self.linear_depth_view,
                &self.params_buffer, &self.ao_result_views[0], &self.ao_working_view),
            Self::create_blur_bg(device, &self.blur_bgl, &self.linear_depth_view,
                &self.params_buffer, &self.ao_result_views[1], &self.ao_working_view),
        ];

        self.temporal_bind_groups = [
            Self::create_temporal_bg(device, &self.temporal_bgl, &self.linear_depth_view,
                &self.params_buffer, &self.camera_buffer, &self.ao_working_view,
                &self.ao_result_views[1], &self.ao_result_views[0], &self.prev_vp_buffer),
            Self::create_temporal_bg(device, &self.temporal_bgl, &self.linear_depth_view,
                &self.params_buffer, &self.camera_buffer, &self.ao_working_view,
                &self.ao_result_views[0], &self.ao_result_views[1], &self.prev_vp_buffer),
        ];
    }

    /// Run the full GTAO pipeline
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        camera_params: &crate::render::GpuCameraParams,
        radius: f32,
        prev_vp: &GpuPrevViewProjection,
        full_width: u32,
        full_height: u32,
    ) {
        let half_w = self.half_width;
        let half_h = self.half_height;

        // Update uniform buffers
        let params = GpuGtaoParams {
            radius,
            falloff_start: 0.2,
            num_steps: 8,
            frame_index: self.frame_counter,
            half_res: [half_w as f32, half_h as f32],
            inv_half_res: [1.0 / half_w as f32, 1.0 / half_h as f32],
            full_res: [full_width as f32, full_height as f32],
            inv_full_res: [1.0 / full_width as f32, 1.0 / full_height as f32],
            temporal_blend: 0.15,
            thickness: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera_params));
        queue.write_buffer(&self.prev_vp_buffer, 0, bytemuck::bytes_of(prev_vp));

        let wg_x = (half_w + 7) / 8;
        let wg_y = (half_h + 7) / 8;

        // Determine ping-pong indices
        let output_idx = (self.frame_counter as usize + 1) % 2;
        let _history_idx = self.frame_counter as usize % 2;

        // Pass 1: Prefilter depth
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GTAO Prefilter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prefilter_pipeline);
            pass.set_bind_group(0, &self.prefilter_bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Pass 2: GTAO main → ao_working
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GTAO Main"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gtao_pipeline);
            pass.set_bind_group(0, &self.gtao_bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Pass 3: Blur H: ao_working → ao_result[output_idx]
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GTAO Blur H"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_h_pipeline);
            pass.set_bind_group(0, &self.blur_h_bind_groups[output_idx], &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Pass 4: Blur V: ao_result[output_idx] → ao_working
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GTAO Blur V"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_v_pipeline);
            pass.set_bind_group(0, &self.blur_v_bind_groups[output_idx], &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Pass 5: Temporal: ao_working + ao_result[history_idx] → ao_result[output_idx]
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GTAO Temporal"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.temporal_pipeline);
            pass.set_bind_group(0, &self.temporal_bind_groups[output_idx], &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        self.frame_counter = self.frame_counter.wrapping_add(1);
    }

    /// Get the current frame's final AO output view
    pub fn ao_view(&self) -> &wgpu::TextureView {
        let output_idx = self.frame_counter as usize % 2;
        &self.ao_result_views[output_idx]
    }
}
