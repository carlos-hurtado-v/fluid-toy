//! Marching Cubes fluid surface renderer
//!
//! Generates a triangle mesh from particle density field using the marching cubes algorithm.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::mc_tables::{EDGE_TABLE, TRI_TABLE};
use crate::render::GpuCameraParams;

/// Grid resolution for marching cubes (cells per dimension)
const GRID_SIZE: u32 = 70;

/// Maximum vertices (5 triangles * 3 verts * grid_size^3 cells)
/// In practice, only ~10-30% of cells are active
const MAX_VERTICES: u32 = GRID_SIZE * GRID_SIZE * GRID_SIZE * 15;

/// Grid parameters for compute shaders
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuGridParams {
    pub grid_min: [f32; 3],
    pub grid_size: u32,
    pub grid_max: [f32; 3],
    pub cell_size: f32,
    pub kernel_radius: f32,
    pub iso_value: f32,
    pub num_particles: u32,
    pub _padding: f32,
}

/// Water shading parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuWaterParams {
    pub water_color: [f32; 3],
    pub specular_power: f32,
    pub fresnel_bias: f32,
    pub refraction_strength: f32,
    pub ripple_scale: f32,
    pub ripple_strength: f32,
}

impl Default for GpuWaterParams {
    fn default() -> Self {
        Self {
            water_color: [0.1, 0.4, 0.8],
            specular_power: 64.0,
            fresnel_bias: 0.02,
            refraction_strength: 0.5,
            ripple_scale: 25.0,
            ripple_strength: 0.15,
        }
    }
}

/// Vertex output from marching cubes
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct McVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Atomic counter for vertex allocation
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Counter {
    vertex_count: u32,
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MC Depth Texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
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

/// Create a depth texture that can be sampled (for back-face depth / thickness calculation)
fn create_samplable_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MC Back Depth Texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub struct MarchingCubesRenderer {
    // Density field (3D texture)
    density_texture: wgpu::Texture,
    density_view: wgpu::TextureView,

    // Depth buffers for rendering
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    // Back face depth for thickness calculation
    back_depth_texture: wgpu::Texture,
    back_depth_view: wgpu::TextureView,
    back_depth_sampler: wgpu::Sampler,

    // Buffers
    grid_params_buffer: wgpu::Buffer,
    edge_table_buffer: wgpu::Buffer,
    tri_table_buffer: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    water_params_buffer: wgpu::Buffer,

    // Pipelines
    density_pipeline: wgpu::ComputePipeline,
    generate_pipeline: wgpu::ComputePipeline,
    back_face_pipeline: wgpu::RenderPipeline,  // Renders back faces for thickness
    render_pipeline: wgpu::RenderPipeline,
    env_pipeline: wgpu::RenderPipeline,

    // Bind groups
    density_bind_group: wgpu::BindGroup,
    generate_bind_group: wgpu::BindGroup,
    back_face_bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,
    env_bind_group: wgpu::BindGroup,

    // Bind group layouts (needed for recreating bind groups on resize)
    render_bind_group_layout: wgpu::BindGroupLayout,

    // For reading back vertex count
    counter_staging_buffer: wgpu::Buffer,

    // Current vertex count (updated after generate pass)
    current_vertex_count: u32,

    // Grid bounds
    grid_min: [f32; 3],
    grid_max: [f32; 3],

    // Screen dimensions for depth buffer
    width: u32,
    height: u32,

    // Keep references for bind group recreation
    env_texture_view: wgpu::TextureView,
    env_sampler: wgpu::Sampler,
}

impl MarchingCubesRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        env_texture_view: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
        width: u32,
        height: u32,
    ) -> Self {
        // Grid bounds (matching simulation domain)
        let grid_min = [-1.0f32, -1.0, -1.0];
        let grid_max = [1.0f32, 1.0, 1.0];
        let cell_size = (grid_max[0] - grid_min[0]) / GRID_SIZE as f32;

        // Create 3D density texture
        let density_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MC Density Field"),
            size: wgpu::Extent3d {
                width: GRID_SIZE,
                height: GRID_SIZE,
                depth_or_array_layers: GRID_SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let density_view = density_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create depth texture for rendering
        let (depth_texture, depth_view) = create_depth_texture(device, width, height);

        // Create back-face depth texture for thickness calculation (samplable)
        let (back_depth_texture, back_depth_view) = create_samplable_depth_texture(device, width, height);
        let back_depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("MC Back Depth Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create buffers
        let grid_params = GpuGridParams {
            grid_min,
            grid_size: GRID_SIZE,
            grid_max,
            cell_size,
            kernel_radius: 0.1,
            iso_value: 500.0,  // Will be tuned based on rest_density
            num_particles: 0,
            _padding: 0.0,
        };
        let grid_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Grid Params"),
            contents: bytemuck::bytes_of(&grid_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Edge table buffer
        let edge_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Edge Table"),
            contents: bytemuck::cast_slice(&EDGE_TABLE),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Triangle table buffer (flatten 2D array)
        let tri_table_flat: Vec<i32> = TRI_TABLE.iter().flatten().copied().collect();
        let tri_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Tri Table"),
            contents: bytemuck::cast_slice(&tri_table_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Counter buffer
        let counter = Counter { vertex_count: 0 };
        let counter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Counter"),
            contents: bytemuck::bytes_of(&counter),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // Staging buffer for reading back counter
        let counter_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MC Counter Staging"),
            size: std::mem::size_of::<Counter>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Vertex buffer
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MC Vertices"),
            size: (MAX_VERTICES as usize * std::mem::size_of::<McVertex>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        // Camera buffer
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MC Camera"),
            size: std::mem::size_of::<GpuCameraParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Water params buffer
        let water_params = GpuWaterParams::default();
        let water_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Water Params"),
            contents: bytemuck::bytes_of(&water_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Load shaders
        let density_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Density Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mc_density.wgsl").into()),
        });

        let generate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Generate Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mc_generate.wgsl").into()),
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mc_render.wgsl").into()),
        });

        let env_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Environment Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mc_environment.wgsl").into()),
        });

        let back_depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Back Depth Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mc_back_depth.wgsl").into()),
        });

        // === Density Pipeline ===
        let density_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC Density BGL"),
            entries: &[
                // Particles (storage buffer, will be bound dynamically)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Grid params
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Density field (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });

        let density_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC Density Pipeline Layout"),
            bind_group_layouts: &[&density_bind_group_layout],
            push_constant_ranges: &[],
        });

        let density_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MC Density Pipeline"),
            layout: Some(&density_pipeline_layout),
            module: &density_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // === Generate Pipeline ===
        let generate_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC Generate BGL"),
            entries: &[
                // Density field (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // Grid params
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Edge table
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Tri table
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Counter
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Vertices
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let generate_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC Generate Pipeline Layout"),
            bind_group_layouts: &[&generate_bind_group_layout],
            push_constant_ranges: &[],
        });

        let generate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MC Generate Pipeline"),
            layout: Some(&generate_pipeline_layout),
            module: &generate_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Generate bind group
        let generate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MC Generate BG"),
            layout: &generate_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&density_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: edge_table_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tri_table_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: vertex_buffer.as_entire_binding(),
                },
            ],
        });

        // === Render Pipeline ===
        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC Render BGL"),
            entries: &[
                // Camera
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
                // Water params
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Vertices
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Environment texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Environment sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Back depth texture (for thickness)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC Render Pipeline Layout"),
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("MC Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,  // MC triangles are clockwise
                cull_mode: Some(wgpu::Face::Back),  // Cull back faces (render front only)
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
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
        });

        // === Back Face Pipeline (for thickness) ===
        let back_face_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC Back Face BGL"),
            entries: &[
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
            ],
        });

        let back_face_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC Back Face Pipeline Layout"),
            bind_group_layouts: &[&back_face_bind_group_layout],
            push_constant_ranges: &[],
        });

        let back_face_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("MC Back Face Pipeline"),
            layout: Some(&back_face_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &back_depth_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &back_depth_shader,
                entry_point: Some("fs_main"),
                targets: &[],  // Depth only, no color output
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,  // MC triangles are clockwise
                cull_mode: Some(wgpu::Face::Front),  // Cull front faces (render back only)
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
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
        });

        let back_face_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MC Back Face BG"),
            layout: &back_face_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vertex_buffer.as_entire_binding(),
                },
            ],
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MC Render BG"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: water_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(env_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(env_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&back_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&back_depth_sampler),
                },
            ],
        });

        // === Environment Pipeline ===
        let env_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC Env BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,  // Used in both vertex and fragment
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let env_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC Env Pipeline Layout"),
            bind_group_layouts: &[&env_bind_group_layout],
            push_constant_ranges: &[],
        });

        let env_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("MC Env Pipeline"),
            layout: Some(&env_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &env_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &env_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
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
                depth_compare: wgpu::CompareFunction::LessEqual,  // Draw at far plane
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let env_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MC Env BG"),
            layout: &env_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(env_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(env_sampler),
                },
            ],
        });

        // Placeholder density bind group (will be recreated with particle buffer)
        let placeholder_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Placeholder"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let density_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MC Density BG Placeholder"),
            layout: &density_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: placeholder_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&density_view),
                },
            ],
        });

        // Clone the texture view for storage (needed for resize)
        let env_texture_view_owned = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MC Env Texture Placeholder"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default());

        let env_sampler_owned = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("MC Env Sampler Owned"),
            ..Default::default()
        });

        Self {
            density_texture,
            density_view,
            depth_texture,
            depth_view,
            back_depth_texture,
            back_depth_view,
            back_depth_sampler,
            grid_params_buffer,
            edge_table_buffer,
            tri_table_buffer,
            counter_buffer,
            vertex_buffer,
            camera_buffer,
            water_params_buffer,
            density_pipeline,
            generate_pipeline,
            back_face_pipeline,
            render_pipeline,
            env_pipeline,
            density_bind_group,
            generate_bind_group,
            back_face_bind_group,
            render_bind_group,
            env_bind_group,
            render_bind_group_layout,
            counter_staging_buffer,
            current_vertex_count: 0,
            grid_min,
            grid_max,
            width,
            height,
            env_texture_view: env_texture_view_owned,
            env_sampler: env_sampler_owned,
        }
    }

    /// Create density bind group with actual particle buffer
    pub fn create_density_bind_group(&self, device: &wgpu::Device, particle_buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        let layout = self.density_pipeline.get_bind_group_layout(0);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MC Density BG"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.grid_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.density_view),
                },
            ],
        })
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, params: &GpuCameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_params(&self, queue: &wgpu::Queue, kernel_radius: f32, iso_value: f32, num_particles: u32) {
        let cell_size = (self.grid_max[0] - self.grid_min[0]) / GRID_SIZE as f32;
        let params = GpuGridParams {
            grid_min: self.grid_min,
            grid_size: GRID_SIZE,
            grid_max: self.grid_max,
            cell_size,
            kernel_radius,
            iso_value,
            num_particles,
            _padding: 0.0,
        };
        queue.write_buffer(&self.grid_params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Update grid bounds to match container dimensions
    pub fn set_bounds(&mut self, min: [f32; 3], max: [f32; 3]) {
        self.grid_min = min;
        self.grid_max = max;
    }

    /// Update water shading parameters
    pub fn update_water_params(&self, queue: &wgpu::Queue, water_color: &[f32; 3], ripple_scale: f32, ripple_strength: f32) {
        let params = GpuWaterParams {
            water_color: *water_color,
            ripple_scale,
            ripple_strength,
            ..Default::default()
        };
        queue.write_buffer(&self.water_params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Generate mesh from particles
    pub fn generate(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        particle_buffer: &wgpu::Buffer,
    ) {
        // Reset counter
        encoder.clear_buffer(&self.counter_buffer, 0, None);

        // Create density bind group with particle buffer
        let density_bind_group = self.create_density_bind_group(device, particle_buffer);

        // Pass 1: Generate density field
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MC Density Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.density_pipeline);
            pass.set_bind_group(0, &density_bind_group, &[]);
            let workgroups = (GRID_SIZE + 3) / 4;
            pass.dispatch_workgroups(workgroups, workgroups, workgroups);
        }

        // Pass 2: Generate triangles
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MC Generate Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.generate_pipeline);
            pass.set_bind_group(0, &self.generate_bind_group, &[]);
            let workgroups = (GRID_SIZE + 3) / 4;
            pass.dispatch_workgroups(workgroups, workgroups, workgroups);
        }

        // Copy counter to staging buffer for readback
        encoder.copy_buffer_to_buffer(
            &self.counter_buffer,
            0,
            &self.counter_staging_buffer,
            0,
            std::mem::size_of::<Counter>() as u64,
        );
    }

    /// Read back vertex count (call after submit, before next frame)
    pub fn read_vertex_count(&mut self, device: &wgpu::Device) {
        let slice = self.counter_staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        {
            let data = slice.get_mapped_range();
            let counter: &Counter = bytemuck::from_bytes(&data);
            self.current_vertex_count = counter.vertex_count.min(MAX_VERTICES);
        }
        self.counter_staging_buffer.unmap();
    }

    /// Render the generated mesh with environment background
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        _background_color: &[f32; 3],
    ) {
        // Pass 1: Render back faces to back_depth_texture (for thickness calculation)
        if self.current_vertex_count > 0 {
            let mut back_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("MC Back Face Pass"),
                color_attachments: &[],  // Depth only
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.back_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            back_pass.set_pipeline(&self.back_face_pipeline);
            back_pass.set_bind_group(0, &self.back_face_bind_group, &[]);
            back_pass.draw(0..self.current_vertex_count, 0..1);
        }

        // Pass 2: Render environment and front faces
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("MC Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Draw environment background first (fullscreen triangle at far plane)
        pass.set_pipeline(&self.env_pipeline);
        pass.set_bind_group(0, &self.env_bind_group, &[]);
        pass.draw(0..3, 0..1);

        // Draw water mesh on top (front faces only, with thickness from back depth)
        if self.current_vertex_count > 0 {
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.render_bind_group, &[]);
            pass.draw(0..self.current_vertex_count, 0..1);
        }
    }

    /// Resize depth buffer when window size changes
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        let (depth_texture, depth_view) = create_depth_texture(device, width, height);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
    }

    pub fn vertex_count(&self) -> u32 {
        self.current_vertex_count
    }
}
