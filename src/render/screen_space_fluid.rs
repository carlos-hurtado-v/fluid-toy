//! Screen-Space Fluid Renderer
//! Renders fluid particles using screen-space techniques for photorealistic results
//!
//! Pipeline:
//! 1. Depth Pass - Render particles as sphere imposters
//! 2. Bilateral Blur - Smooth depth while preserving edges (2 passes)
//! 3. Thickness Pass - Accumulate particle contributions (additive blending)
//! 4. Composite Pass - Final water shading with normals from depth

use crate::render::camera::GpuCameraParams;
use crate::render::environment::load_embedded_environment_map;
use crate::simulation::SphParticle3D;
use wgpu::util::DeviceExt;

// Compile-time size assertions for debugging
const _: () = assert!(std::mem::size_of::<GpuCameraParams>() == 144, "GpuCameraParams must be 144 bytes");
const _: () = assert!(std::mem::size_of::<GpuWaterParams>() == 160, "GpuWaterParams must be 160 bytes");
const _: () = assert!(std::mem::size_of::<GpuBlurParams>() == 48, "GpuBlurParams must be 48 bytes (WGSL std140)");
const _: () = assert!(std::mem::size_of::<GpuFluidParams>() == 80, "GpuFluidParams must be 80 bytes");

/// GPU-compatible fluid rendering parameters (depth pass)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFluidParams {
    pub particle_radius: f32,
    pub screen_width: f32,
    pub screen_height: f32,
    pub _padding: f32,
    pub scene_rotation: [[f32; 4]; 4],  // mat4x4 for scene turntable rotation
}

/// GPU-compatible blur parameters
/// WGSL std140 layout: vec3 has 16-byte alignment, so struct is 48 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBlurParams {
    pub blur_dir: [f32; 2],           // 8 bytes @ 0
    pub depth_threshold: f32,          // 4 bytes @ 8
    pub max_filter_size: f32,          // 4 bytes @ 12
    pub projected_particle_constant: f32, // 4 bytes @ 16
    pub _pad1: [f32; 3],              // 12 bytes @ 20 (padding to align vec3)
    pub _padding: [f32; 3],           // 12 bytes @ 32 (matches shader's vec3 _padding)
    pub _pad2: f32,                   // 4 bytes @ 44 (struct size must be multiple of 16)
}
// Total: 48 bytes

/// GPU-compatible water shading parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuWaterParams {
    pub texel_size: [f32; 2],
    pub specular_power: f32,
    pub fresnel_bias: f32,
    pub inv_projection: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    // Surface detail parameters
    pub ripple_scale: f32,
    pub ripple_strength: f32,
    pub time: f32,
    pub _padding2: f32,
}
// Total: 160 bytes

pub struct ScreenSpaceFluidRenderer {
    // Textures
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    depth_buffer: wgpu::Texture,
    depth_buffer_view: wgpu::TextureView,
    blur_texture_a: wgpu::Texture,
    blur_view_a: wgpu::TextureView,
    blur_texture_b: wgpu::Texture,
    blur_view_b: wgpu::TextureView,
    thickness_texture: wgpu::Texture,
    thickness_view: wgpu::TextureView,

    // Environment map
    #[allow(dead_code)]
    env_texture: wgpu::Texture,
    env_view: wgpu::TextureView,
    env_sampler: wgpu::Sampler,

    // Buffers
    camera_buffer: wgpu::Buffer,
    fluid_params_buffer: wgpu::Buffer,
    blur_params_buffer_h: wgpu::Buffer,
    blur_params_buffer_v: wgpu::Buffer,
    flow_params_buffer: wgpu::Buffer,  // Separate params for curvature flow (different dt)
    water_params_buffer: wgpu::Buffer,

    // Pipelines
    depth_pipeline: wgpu::RenderPipeline,
    bilateral_blur_pipeline: wgpu::RenderPipeline,  // Primary smoothing - merges spheres
    curvature_flow_pipeline: wgpu::RenderPipeline,  // Polish - smooths surface tension
    thickness_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,

    // Bind groups
    depth_bind_group: wgpu::BindGroup,
    blur_bind_group_a: wgpu::BindGroup,  // depth → blur_a (first H pass)
    blur_bind_group_b: wgpu::BindGroup,  // blur_a → blur_b (V passes)
    blur_bind_group_c: wgpu::BindGroup,  // blur_b → blur_a (H passes for iter 2+)
    flow_bind_group_a: wgpu::BindGroup,  // blur_b → blur_a (curvature flow with separate dt)
    flow_bind_group_b: wgpu::BindGroup,  // blur_a → blur_b (curvature flow)
    thickness_bind_group: wgpu::BindGroup,
    composite_bind_group: wgpu::BindGroup,

    // Bind group layouts
    blur_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_group_layout: wgpu::BindGroupLayout,

    current_size: (u32, u32),
}

impl ScreenSpaceFluidRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        camera_params: &GpuCameraParams,
        width: u32,
        height: u32,
    ) -> Self {
        let width = width.max(1);
        let height = height.max(1);

        // Create textures (use R32Float for high precision depth storage)
        let (depth_texture, depth_view) = create_color_texture(device, width, height, "SS Depth");
        let (depth_buffer, depth_buffer_view) = create_depth_buffer(device, width, height);
        let (blur_texture_a, blur_view_a) = create_color_texture(device, width, height, "SS Blur A");
        let (blur_texture_b, blur_view_b) = create_color_texture(device, width, height, "SS Blur B");
        // Thickness needs blendable format for additive accumulation
        let (thickness_texture, thickness_view) = create_thickness_texture(device, width, height);

        // Load environment map (embedded at compile time)
        let (env_texture, env_view, env_sampler) = load_embedded_environment_map(device, queue)
            .expect("Failed to load environment map");

        // Create buffers - use explicit size to ensure alignment
        let camera_size = std::mem::size_of::<GpuCameraParams>() as u64;
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SS Camera Buffer"),
            size: camera_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let fluid_params = GpuFluidParams {
            // particle_radius is the actual radius; billboard is 2*radius (diameter)
            // Will be updated each frame from app state via update_params()
            particle_radius: 0.025,
            screen_width: width as f32,
            screen_height: height as f32,
            _padding: 0.0,
            scene_rotation: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let fluid_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Fluid Params Buffer"),
            contents: bytemuck::bytes_of(&fluid_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bilateral blur params - primary smoothing to merge spheres
        // HIGH depth_threshold (1.0-3.0) to "melt" spheres together
        // LARGE filter size (30-50 pixels) to reach across gaps
        let bilateral_params_h = GpuBlurParams {
            blur_dir: [1.0, 0.0],  // Horizontal
            depth_threshold: 2.5,  // Very high - merge spheres aggressively
            max_filter_size: 40.0, // Large spatial radius for wide coverage
            projected_particle_constant: 0.0,
            _pad1: [0.0; 3],
            _padding: [0.0; 3],
            _pad2: 0.0,
        };
        let blur_params_buffer_h = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Bilateral Params H"),
            contents: bytemuck::bytes_of(&bilateral_params_h),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bilateral_params_v = GpuBlurParams {
            blur_dir: [0.0, 1.0],  // Vertical
            depth_threshold: 2.5,  // Same very high value
            max_filter_size: 40.0, // Same large radius
            projected_particle_constant: 0.0,
            _pad1: [0.0; 3],
            _padding: [0.0; 3],
            _pad2: 0.0,
        };
        let blur_params_buffer_v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Bilateral Params V"),
            contents: bytemuck::bytes_of(&bilateral_params_v),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Curvature flow params - separate buffer with LOW dt for stable smoothing
        // dt in depth_threshold field (curvature flow shader reads it as dt)
        let flow_params = GpuBlurParams {
            blur_dir: [0.0, 0.0],  // Not used by curvature flow
            depth_threshold: 0.005, // Slightly higher dt for more aggressive smoothing
            max_filter_size: 0.0,
            projected_particle_constant: 0.0,
            _pad1: [0.0; 3],
            _padding: [0.0; 3],
            _pad2: 0.0,
        };
        let flow_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Flow Params"),
            contents: bytemuck::bytes_of(&flow_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Default ripple parameters (can be adjusted via update_water_params)
        let water_params = create_water_params(width, height, camera_params, 15.0, 0.3, 0.0);
        let water_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Water Params Buffer"),
            contents: bytemuck::bytes_of(&water_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Load shaders
        let depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Depth Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ss_depth.wgsl").into()),
        });

        // Bilateral blur for primary smoothing (merges spheres into blob)
        let bilateral_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Bilateral Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ss_bilateral_blur.wgsl").into()),
        });

        // Curvature flow for polish (smooths surface tension)
        let curvature_flow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Curvature Flow Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ss_curvature_flow.wgsl").into()),
        });

        let thickness_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Thickness Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ss_thickness.wgsl").into()),
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ss_composite.wgsl").into()),
        });

        // Depth bind group layout (camera + fluid params)
        let depth_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SS Depth Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<GpuCameraParams>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<GpuFluidParams>() as u64),
                    },
                    count: None,
                },
            ],
        });

        // Blur bind group layout (texture + params, no sampler - uses textureLoad)
        let blur_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SS Blur Bind Group Layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<GpuBlurParams>() as u64),
                    },
                    count: None,
                },
            ],
        });

        // Thickness bind group layout (same as depth)
        let thickness_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SS Thickness Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<GpuCameraParams>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<GpuFluidParams>() as u64),
                    },
                    count: None,
                },
            ],
        });

        // Composite bind group layout
        let composite_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SS Composite Bind Group Layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<GpuCameraParams>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<GpuWaterParams>() as u64),
                    },
                    count: None,
                },
                // Environment map texture (binding 4)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Environment map sampler (binding 5)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create pipelines
        let depth_pipeline = create_depth_pipeline(device, &depth_shader, &depth_bind_group_layout);
        let bilateral_blur_pipeline = create_blur_pipeline(device, &bilateral_blur_shader, &blur_bind_group_layout);
        let curvature_flow_pipeline = create_blur_pipeline(device, &curvature_flow_shader, &blur_bind_group_layout);
        let thickness_pipeline = create_thickness_pipeline(device, &thickness_shader, &thickness_bind_group_layout);
        let composite_pipeline = create_composite_pipeline(device, &composite_shader, &composite_bind_group_layout, surface_format);

        // Create bind groups
        let depth_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Depth Bind Group"),
            layout: &depth_bind_group_layout,
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

        let blur_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Blur Bind Group A"),
            layout: &blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: blur_params_buffer_h.as_entire_binding(),
                },
            ],
        });

        let blur_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Blur Bind Group B"),
            layout: &blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: blur_params_buffer_v.as_entire_binding(),
                },
            ],
        });

        // For iterations 2+: reads blur_b, writes blur_a (horizontal)
        let blur_bind_group_c = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Blur Bind Group C"),
            layout: &blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: blur_params_buffer_h.as_entire_binding(),
                },
            ],
        });

        // Curvature flow bind groups - use separate flow_params_buffer with low dt
        let flow_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Flow Bind Group A"),
            layout: &blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flow_params_buffer.as_entire_binding(),
                },
            ],
        });

        let flow_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Flow Bind Group B"),
            layout: &blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flow_params_buffer.as_entire_binding(),
                },
            ],
        });

        let thickness_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Thickness Bind Group"),
            layout: &thickness_bind_group_layout,
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

        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Composite Bind Group"),
            layout: &composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&thickness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: water_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&env_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&env_sampler),
                },
            ],
        });

        Self {
            depth_texture,
            depth_view,
            depth_buffer,
            depth_buffer_view,
            blur_texture_a,
            blur_view_a,
            blur_texture_b,
            blur_view_b,
            thickness_texture,
            thickness_view,
            env_texture,
            env_view,
            env_sampler,
            camera_buffer,
            fluid_params_buffer,
            blur_params_buffer_h,
            blur_params_buffer_v,
            flow_params_buffer,
            water_params_buffer,
            depth_pipeline,
            bilateral_blur_pipeline,
            curvature_flow_pipeline,
            thickness_pipeline,
            composite_pipeline,
            depth_bind_group,
            blur_bind_group_a,
            blur_bind_group_b,
            blur_bind_group_c,
            flow_bind_group_a,
            flow_bind_group_b,
            thickness_bind_group,
            composite_bind_group,
            blur_bind_group_layout,
            composite_bind_group_layout,
            current_size: (width, height),
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, params: &GpuCameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_params(
        &self,
        queue: &wgpu::Queue,
        particle_radius: f32,
        width: u32,
        height: u32,
        camera_params: &GpuCameraParams,
        scene_rotation: &[[f32; 4]; 4],
        ripple_scale: f32,
        ripple_strength: f32,
        time: f32,
    ) {
        let fluid_params = GpuFluidParams {
            particle_radius,
            screen_width: width as f32,
            screen_height: height as f32,
            _padding: 0.0,
            scene_rotation: *scene_rotation,
        };
        queue.write_buffer(&self.fluid_params_buffer, 0, bytemuck::bytes_of(&fluid_params));

        let water_params = create_water_params(width, height, camera_params, ripple_scale, ripple_strength, time);
        queue.write_buffer(&self.water_params_buffer, 0, bytemuck::bytes_of(&water_params));
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);

        if self.current_size == (width, height) {
            return;
        }

        self.current_size = (width, height);

        // Recreate textures
        let (depth_texture, depth_view) = create_color_texture(device, width, height, "SS Depth");
        let (depth_buffer, depth_buffer_view) = create_depth_buffer(device, width, height);
        let (blur_texture_a, blur_view_a) = create_color_texture(device, width, height, "SS Blur A");
        let (blur_texture_b, blur_view_b) = create_color_texture(device, width, height, "SS Blur B");
        let (thickness_texture, thickness_view) = create_thickness_texture(device, width, height);

        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        self.depth_buffer = depth_buffer;
        self.depth_buffer_view = depth_buffer_view;
        self.blur_texture_a = blur_texture_a;
        self.blur_view_a = blur_view_a;
        self.blur_texture_b = blur_texture_b;
        self.blur_view_b = blur_view_b;
        self.thickness_texture = thickness_texture;
        self.thickness_view = thickness_view;

        // Recreate bind groups
        self.blur_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Blur Bind Group A"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.blur_params_buffer_h.as_entire_binding(),
                },
            ],
        });

        self.blur_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Blur Bind Group B"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.blur_view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.blur_params_buffer_v.as_entire_binding(),
                },
            ],
        });

        self.blur_bind_group_c = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Blur Bind Group C"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.blur_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.blur_params_buffer_h.as_entire_binding(),
                },
            ],
        });

        self.flow_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Flow Bind Group A"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.blur_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.flow_params_buffer.as_entire_binding(),
                },
            ],
        });

        self.flow_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Flow Bind Group B"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.blur_view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.flow_params_buffer.as_entire_binding(),
                },
            ],
        });

        self.composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Composite Bind Group"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.blur_view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.thickness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.water_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.env_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                },
            ],
        });
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        particle_buffer: &wgpu::Buffer,
        num_particles: u32,
        background: &[f32; 3],
    ) {
        // Pass 1: Depth
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SS Depth Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.depth_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_buffer_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.depth_pipeline);
            pass.set_bind_group(0, &self.depth_bind_group, &[]);
            pass.set_vertex_buffer(0, particle_buffer.slice(..));
            pass.draw(0..6, 0..num_particles);
        }

        // PASS 2: Bilateral blur - primary smoothing to merge spheres into blob
        // Uses wide spatial filter with depth-aware weighting
        // More iterations = smoother result (6-8 is typical for fluid)
        // Iteration pattern: depth → blur_a → blur_b, then blur_b → blur_a → blur_b
        for iter in 0..8 {
            // Horizontal bilateral blur
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("SS Bilateral Blur H"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.blur_view_a,
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

                pass.set_pipeline(&self.bilateral_blur_pipeline);
                // First iteration reads from depth, subsequent iterations read from blur_b
                if iter == 0 {
                    pass.set_bind_group(0, &self.blur_bind_group_a, &[]);
                } else {
                    pass.set_bind_group(0, &self.blur_bind_group_c, &[]);
                }
                pass.draw(0..3, 0..1);
            }

            // Vertical bilateral blur
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("SS Bilateral Blur V"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.blur_view_b,
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

                pass.set_pipeline(&self.bilateral_blur_pipeline);
                pass.set_bind_group(0, &self.blur_bind_group_b, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // PASS 3: Curvature flow - polish pass to smooth surface tension
        // Uses separate flow_bind_groups with LOW dt for stable smoothing
        // More iterations = smoother, more polished surface
        for _iter in 0..10 {
            // Curvature flow pass A: blur_b → blur_a
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("SS Curvature Flow A"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.blur_view_a,
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

                pass.set_pipeline(&self.curvature_flow_pipeline);
                pass.set_bind_group(0, &self.flow_bind_group_a, &[]);  // reads blur_b with flow params
                pass.draw(0..3, 0..1);
            }

            // Curvature flow pass B: blur_a → blur_b
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("SS Curvature Flow B"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.blur_view_b,
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

                pass.set_pipeline(&self.curvature_flow_pipeline);
                pass.set_bind_group(0, &self.flow_bind_group_b, &[]);  // reads blur_a with flow params
                pass.draw(0..3, 0..1);
            }
        }

        // Pass 4: Thickness
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SS Thickness Pass"),
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

        // Pass 5: Composite
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SS Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: background[0] as f64,
                            g: background[1] as f64,
                            b: background[2] as f64,
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
}

fn create_water_params(
    width: u32,
    height: u32,
    camera_params: &GpuCameraParams,
    ripple_scale: f32,
    ripple_strength: f32,
    time: f32,
) -> GpuWaterParams {
    GpuWaterParams {
        texel_size: [1.0 / width as f32, 1.0 / height as f32],
        specular_power: 250.0,
        fresnel_bias: 0.02,
        inv_projection: invert_matrix(&camera_params.projection),
        inv_view: invert_matrix(&camera_params.view),
        ripple_scale,
        ripple_strength,
        time,
        _padding2: 0.0,
    }
}

fn invert_matrix(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Simple 4x4 matrix inversion (for orthogonal/perspective matrices)
    // This is a general-purpose inversion
    let mut inv = [[0.0f32; 4]; 4];

    inv[0][0] = m[1][1] * m[2][2] * m[3][3] - m[1][1] * m[2][3] * m[3][2] - m[2][1] * m[1][2] * m[3][3] + m[2][1] * m[1][3] * m[3][2] + m[3][1] * m[1][2] * m[2][3] - m[3][1] * m[1][3] * m[2][2];
    inv[1][0] = -m[1][0] * m[2][2] * m[3][3] + m[1][0] * m[2][3] * m[3][2] + m[2][0] * m[1][2] * m[3][3] - m[2][0] * m[1][3] * m[3][2] - m[3][0] * m[1][2] * m[2][3] + m[3][0] * m[1][3] * m[2][2];
    inv[2][0] = m[1][0] * m[2][1] * m[3][3] - m[1][0] * m[2][3] * m[3][1] - m[2][0] * m[1][1] * m[3][3] + m[2][0] * m[1][3] * m[3][1] + m[3][0] * m[1][1] * m[2][3] - m[3][0] * m[1][3] * m[2][1];
    inv[3][0] = -m[1][0] * m[2][1] * m[3][2] + m[1][0] * m[2][2] * m[3][1] + m[2][0] * m[1][1] * m[3][2] - m[2][0] * m[1][2] * m[3][1] - m[3][0] * m[1][1] * m[2][2] + m[3][0] * m[1][2] * m[2][1];

    inv[0][1] = -m[0][1] * m[2][2] * m[3][3] + m[0][1] * m[2][3] * m[3][2] + m[2][1] * m[0][2] * m[3][3] - m[2][1] * m[0][3] * m[3][2] - m[3][1] * m[0][2] * m[2][3] + m[3][1] * m[0][3] * m[2][2];
    inv[1][1] = m[0][0] * m[2][2] * m[3][3] - m[0][0] * m[2][3] * m[3][2] - m[2][0] * m[0][2] * m[3][3] + m[2][0] * m[0][3] * m[3][2] + m[3][0] * m[0][2] * m[2][3] - m[3][0] * m[0][3] * m[2][2];
    inv[2][1] = -m[0][0] * m[2][1] * m[3][3] + m[0][0] * m[2][3] * m[3][1] + m[2][0] * m[0][1] * m[3][3] - m[2][0] * m[0][3] * m[3][1] - m[3][0] * m[0][1] * m[2][3] + m[3][0] * m[0][3] * m[2][1];
    inv[3][1] = m[0][0] * m[2][1] * m[3][2] - m[0][0] * m[2][2] * m[3][1] - m[2][0] * m[0][1] * m[3][2] + m[2][0] * m[0][2] * m[3][1] + m[3][0] * m[0][1] * m[2][2] - m[3][0] * m[0][2] * m[2][1];

    inv[0][2] = m[0][1] * m[1][2] * m[3][3] - m[0][1] * m[1][3] * m[3][2] - m[1][1] * m[0][2] * m[3][3] + m[1][1] * m[0][3] * m[3][2] + m[3][1] * m[0][2] * m[1][3] - m[3][1] * m[0][3] * m[1][2];
    inv[1][2] = -m[0][0] * m[1][2] * m[3][3] + m[0][0] * m[1][3] * m[3][2] + m[1][0] * m[0][2] * m[3][3] - m[1][0] * m[0][3] * m[3][2] - m[3][0] * m[0][2] * m[1][3] + m[3][0] * m[0][3] * m[1][2];
    inv[2][2] = m[0][0] * m[1][1] * m[3][3] - m[0][0] * m[1][3] * m[3][1] - m[1][0] * m[0][1] * m[3][3] + m[1][0] * m[0][3] * m[3][1] + m[3][0] * m[0][1] * m[1][3] - m[3][0] * m[0][3] * m[1][1];
    inv[3][2] = -m[0][0] * m[1][1] * m[3][2] + m[0][0] * m[1][2] * m[3][1] + m[1][0] * m[0][1] * m[3][2] - m[1][0] * m[0][2] * m[3][1] - m[3][0] * m[0][1] * m[1][2] + m[3][0] * m[0][2] * m[1][1];

    inv[0][3] = -m[0][1] * m[1][2] * m[2][3] + m[0][1] * m[1][3] * m[2][2] + m[1][1] * m[0][2] * m[2][3] - m[1][1] * m[0][3] * m[2][2] - m[2][1] * m[0][2] * m[1][3] + m[2][1] * m[0][3] * m[1][2];
    inv[1][3] = m[0][0] * m[1][2] * m[2][3] - m[0][0] * m[1][3] * m[2][2] - m[1][0] * m[0][2] * m[2][3] + m[1][0] * m[0][3] * m[2][2] + m[2][0] * m[0][2] * m[1][3] - m[2][0] * m[0][3] * m[1][2];
    inv[2][3] = -m[0][0] * m[1][1] * m[2][3] + m[0][0] * m[1][3] * m[2][1] + m[1][0] * m[0][1] * m[2][3] - m[1][0] * m[0][3] * m[2][1] - m[2][0] * m[0][1] * m[1][3] + m[2][0] * m[0][3] * m[1][1];
    inv[3][3] = m[0][0] * m[1][1] * m[2][2] - m[0][0] * m[1][2] * m[2][1] - m[1][0] * m[0][1] * m[2][2] + m[1][0] * m[0][2] * m[2][1] + m[2][0] * m[0][1] * m[1][2] - m[2][0] * m[0][2] * m[1][1];

    let det = m[0][0] * inv[0][0] + m[0][1] * inv[1][0] + m[0][2] * inv[2][0] + m[0][3] * inv[3][0];

    if det.abs() < 1e-10 {
        return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    }

    let inv_det = 1.0 / det;
    for i in 0..4 {
        for j in 0..4 {
            inv[i][j] *= inv_det;
        }
    }

    inv
}

fn create_color_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // R32Float for high precision depth storage (eliminates quantization artifacts)
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_thickness_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("SS Thickness"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Rgba16Float for thickness - needs blending support for additive accumulation
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_depth_buffer(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("SS Depth Buffer"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_depth_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("SS Depth Pipeline Layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("SS Depth Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SphParticle3D>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 16,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R32Float,
                blend: None,
                write_mask: wgpu::ColorWrites::RED,
            })],
            compilation_options: Default::default(),
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
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn create_blur_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("SS Blur Pipeline Layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("SS Blur Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R32Float,
                blend: None,
                write_mask: wgpu::ColorWrites::RED,
            })],
            compilation_options: Default::default(),
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
    })
}

fn create_thickness_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("SS Thickness Pipeline Layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("SS Thickness Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SphParticle3D>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 16,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,  // Blendable format for additive thickness
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
    })
}

fn create_composite_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    bind_group_layout: &wgpu::BindGroupLayout,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("SS Composite Pipeline Layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("SS Composite Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
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
    })
}
