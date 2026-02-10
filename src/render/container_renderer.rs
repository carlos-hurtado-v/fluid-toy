//! Opaque pool container renderer — lit box (floor + 4 walls, no top)

use wgpu::util::DeviceExt;

use crate::render::camera::GpuCameraParams;
use crate::state::{ContainerConfig, GpuContainerRenderParams};

/// Vertex layout matching the WGSL VertexInput (32 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ContainerVertex {
    position: [f32; 3],
    normal: [f32; 3],
    face_id: f32,
    _pad: f32,
}

pub struct ContainerRenderer {
    pipeline: wgpu::RenderPipeline,
    msaa_pipeline: wgpu::RenderPipeline,
    depth_only_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    // Cached dims to detect when mesh needs rebuild
    cached_width: f32,
    cached_depth: f32,
    cached_floor_y: f32,
    cached_height: f32,
    cached_kernel_radius: f32,
}

impl ContainerRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_params: &GpuCameraParams,
        render_params: &GpuContainerRenderParams,
        config: &ContainerConfig,
        msaa_sample_count: u32,
        kernel_radius: f32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Container Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/container.wgsl").into()),
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Container Camera Buffer"),
            contents: bytemuck::bytes_of(camera_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Container Params Buffer"),
            contents: bytemuck::bytes_of(render_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Container BGL"),
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Container BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Container Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ContainerVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { // position
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute { // normal
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute { // face_id
                    format: wgpu::VertexFormat::Float32,
                    offset: 24,
                    shader_location: 2,
                },
                wgpu::VertexAttribute { // _pad
                    format: wgpu::VertexFormat::Float32,
                    offset: 28,
                    shader_location: 3,
                },
            ],
        }];

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None, // Camera can be inside or outside
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        };

        let depth_stencil = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        // 1x pipeline (background texture pass)
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Container Pipeline 1x"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
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
            primitive,
            depth_stencil: Some(depth_stencil.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // MSAA pipeline (main water pass)
        let msaa_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Container Pipeline MSAA"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
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
            primitive,
            depth_stencil: Some(depth_stencil.clone()),
            multisample: wgpu::MultisampleState {
                count: msaa_sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Depth-only pipeline (GTAO front depth prepass, 1x, no color)
        let depth_only_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Container Depth-Only Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: None,
            primitive,
            depth_stencil: Some(depth_stencil),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Generate mesh
        let (vertices, indices) = generate_container_mesh(config, kernel_radius);
        let index_count = indices.len() as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Container VB"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Container IB"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            msaa_pipeline,
            depth_only_pipeline,
            bind_group,
            camera_buffer,
            params_buffer,
            vertex_buffer,
            index_buffer,
            index_count,
            cached_width: config.width,
            cached_depth: config.depth,
            cached_floor_y: config.floor_y,
            cached_height: config.height,
            cached_kernel_radius: kernel_radius,
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, params: &GpuCameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_params(&self, queue: &wgpu::Queue, params: &GpuContainerRenderParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Rebuild mesh if container dimensions or kernel radius changed
    pub fn maybe_rebuild_mesh(&mut self, device: &wgpu::Device, config: &ContainerConfig, kernel_radius: f32) {
        if (config.width - self.cached_width).abs() < 1e-6
            && (config.depth - self.cached_depth).abs() < 1e-6
            && (config.floor_y - self.cached_floor_y).abs() < 1e-6
            && (config.height - self.cached_height).abs() < 1e-6
            && (kernel_radius - self.cached_kernel_radius).abs() < 1e-6
        {
            return;
        }

        let (vertices, indices) = generate_container_mesh(config, kernel_radius);
        self.index_count = indices.len() as u32;

        self.vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Container VB"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        self.index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Container IB"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        self.cached_width = config.width;
        self.cached_depth = config.depth;
        self.cached_floor_y = config.floor_y;
        self.cached_height = config.height;
        self.cached_kernel_radius = kernel_radius;
    }

    /// Render with 1x sample count (for background texture pass)
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    /// Render with MSAA (for main water pass)
    pub fn render_msaa<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.msaa_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    /// Render depth-only (for GTAO front depth prepass)
    pub fn render_depth_only<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.depth_only_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

/// Generate container mesh: floor + 4 walls (no top). 20 vertices, 30 indices.
/// Positions in container-local space. Normals point inward.
///
/// The pool must fully contain the MC fluid surface:
/// - Walls at full container width + small outset (MC density extends beyond
///   particle positions by up to kernel_radius, pool must cover that)
/// - Floor extended well below floor_y (particles near floor create density
///   that spills below via the kernel radius)
/// - Height capped at 50% of container height (pool walls shouldn't tower
///   over the water — the physics ceiling is invisible)
fn generate_container_mesh(config: &ContainerConfig, kernel_radius: f32) -> (Vec<ContainerVertex>, Vec<u32>) {
    let outset = kernel_radius * 0.5;
    let hw = config.width / 2.0 + outset;
    let hd = config.depth / 2.0 + outset;
    let y0 = config.floor_y - kernel_radius * 2.0;
    let y1 = config.floor_y + config.height * 0.5;

    let mut vertices = Vec::with_capacity(20);
    let mut indices = Vec::with_capacity(30);

    // Helper to push a quad (2 triangles) with given normal and face_id
    let mut push_quad = |p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], normal: [f32; 3], face_id: f32| {
        let base = vertices.len() as u32;
        for &pos in &[p0, p1, p2, p3] {
            vertices.push(ContainerVertex {
                position: pos,
                normal,
                face_id,
                _pad: 0.0,
            });
        }
        // CCW winding: 0-1-2, 0-2-3
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    // Floor (normal up = inward for a pool)
    push_quad(
        [-hw, y0, -hd],
        [ hw, y0, -hd],
        [ hw, y0,  hd],
        [-hw, y0,  hd],
        [0.0, 1.0, 0.0],
        0.0,
    );

    // Front wall (Z = -hd, normal +Z = inward)
    push_quad(
        [-hw, y0, -hd],
        [ hw, y0, -hd],
        [ hw, y1, -hd],
        [-hw, y1, -hd],
        [0.0, 0.0, 1.0],
        1.0,
    );

    // Back wall (Z = +hd, normal -Z = inward)
    push_quad(
        [ hw, y0, hd],
        [-hw, y0, hd],
        [-hw, y1, hd],
        [ hw, y1, hd],
        [0.0, 0.0, -1.0],
        1.0,
    );

    // Left wall (X = -hw, normal +X = inward)
    push_quad(
        [-hw, y0,  hd],
        [-hw, y0, -hd],
        [-hw, y1, -hd],
        [-hw, y1,  hd],
        [1.0, 0.0, 0.0],
        1.0,
    );

    // Right wall (X = +hw, normal -X = inward)
    push_quad(
        [ hw, y0, -hd],
        [ hw, y0,  hd],
        [ hw, y1,  hd],
        [ hw, y1, -hd],
        [-1.0, 0.0, 0.0],
        1.0,
    );

    (vertices, indices)
}
