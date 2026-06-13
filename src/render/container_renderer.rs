//! Opaque pool container renderer — lit box (floor + 4 walls, no top)

use wgpu::util::DeviceExt;

use crate::render::camera::GpuCameraParams;
use crate::state::{ContainerConfig, GpuContainerGeometry, GpuShCoefficients, LightingConfig};

/// Pool material style parameters (tile pattern, lighting)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPoolStyle {
    pub tile_color: [f32; 3],
    pub tile_scale: f32,
    pub grout_color: [f32; 3],
    pub specular_strength: f32,
    pub light_dir: [f32; 3],
    pub grout_width: f32,
    /// Sun color x intensity, zeroed when the sun is disabled.
    /// Scales the pool's direct sun, specular, and caustic terms.
    pub sun_rgb: [f32; 3],
    pub ibl_strength: f32,
    /// Caustic irradiance multiplier on the floor (0 = caustics inactive)
    pub caustic_strength: f32,
    /// How strongly water shadows remove direct sun from the floor
    pub shadow_strength: f32,
    /// Contrast exponent on the caustic map (1 = physical; >1 sharpens
    /// filaments and deepens lanes around the flat-water level of 1.0)
    pub caustic_focus: f32,
    pub _pad0: f32,
}

impl GpuPoolStyle {
    pub fn from_config(
        config: &ContainerConfig,
        lighting: &LightingConfig,
        caustic_strength: f32,
        shadow_strength: f32,
        caustic_focus: f32,
    ) -> Self {
        let sun_on = if lighting.sun_enabled { 1.0 } else { 0.0 };
        let sun_rgb = [
            lighting.sun_color[0] * lighting.sun_intensity * sun_on,
            lighting.sun_color[1] * lighting.sun_intensity * sun_on,
            lighting.sun_color[2] * lighting.sun_intensity * sun_on,
        ];
        Self {
            tile_color: config.tile_color,
            tile_scale: config.tile_scale,
            grout_color: config.grout_color,
            specular_strength: config.specular_strength,
            light_dir: lighting.sun_direction_normalized(),
            grout_width: config.grout_width,
            sun_rgb,
            ibl_strength: 0.6,
            caustic_strength,
            shadow_strength,
            caustic_focus,
            _pad0: 0.0,
        }
    }
}

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
    container_geom_buffer: wgpu::Buffer,
    sh_buffer: wgpu::Buffer,
    pool_style_buffer: wgpu::Buffer,
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
        container_geom: &GpuContainerGeometry,
        pool_style: &GpuPoolStyle,
        config: &ContainerConfig,
        msaa_sample_count: u32,
        kernel_radius: f32,
        sh_coefficients: &GpuShCoefficients,
        caustic_view: &wgpu::TextureView,
        caustic_sampler: &wgpu::Sampler,
    ) -> Self {
        // Load shader (prepend container_common.wgsl)
        let container_common_wgsl = include_str!("../shaders/container_common.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Container Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}", container_common_wgsl, include_str!("../shaders/container.wgsl")).into(),
            ),
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Container Camera Buffer"),
            contents: bytemuck::bytes_of(camera_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let container_geom_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Container Geometry Buffer"),
            contents: bytemuck::bytes_of(container_geom),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Container SH Buffer"),
            contents: bytemuck::bytes_of(sh_coefficients),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pool_style_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Container Pool Style"),
            contents: bytemuck::bytes_of(pool_style),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
                    resource: container_geom_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pool_style_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(caustic_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(caustic_sampler),
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
                wgpu::VertexAttribute { // _pad (is_inner)
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

        // Generate mesh (in centered local space)
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
            container_geom_buffer,
            sh_buffer,
            pool_style_buffer,
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

    pub fn update_container_geometry(&self, queue: &wgpu::Queue, geom: &GpuContainerGeometry) {
        queue.write_buffer(&self.container_geom_buffer, 0, bytemuck::bytes_of(geom));
    }

    pub fn update_pool_style(&self, queue: &wgpu::Queue, style: &GpuPoolStyle) {
        queue.write_buffer(&self.pool_style_buffer, 0, bytemuck::bytes_of(style));
    }

    pub fn update_sh_coefficients(&self, queue: &wgpu::Queue, coeffs: &GpuShCoefficients) {
        queue.write_buffer(&self.sh_buffer, 0, bytemuck::bytes_of(coeffs));
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

    /// Pool mesh buffers for external depth-only draws (caustics light-space
    /// occluder). Rebuilt on dimension change, so re-fetch every frame.
    pub fn mesh_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer, u32) {
        (&self.vertex_buffer, &self.index_buffer, self.index_count)
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

/// Wall thickness for the opaque pool container
const WALL_THICKNESS: f32 = 0.06;

/// Generate thick-walled pool container mesh (open top).
/// Inner cavity matches the container config dimensions.
/// Outer shell is offset by WALL_THICKNESS in all directions.
/// Positions in container-local centered space (origin at container center).
/// Wall height is capped at 50% so the pool doesn't tower over the water.
fn generate_container_mesh(config: &ContainerConfig, _kernel_radius: f32) -> (Vec<ContainerVertex>, Vec<u32>) {
    // Inner dimensions in centered local space
    let hw = config.width / 2.0;
    let hd = config.depth / 2.0;
    let hh = config.height / 2.0;
    let y0 = -hh;                   // inner floor
    let y1 = -hh + config.height * 0.5; // wall height (50% of container)
    // Outer dimensions (expanded by wall thickness)
    let t = WALL_THICKNESS;
    let ohw = hw + t;
    let ohd = hd + t;
    let oy0 = y0 - t;

    let mut vertices = Vec::with_capacity(80);
    let mut indices = Vec::with_capacity(120);

    // Helper to push a quad (2 triangles) with given normal, face_id, and is_inner flag
    let mut push_quad = |p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], normal: [f32; 3], face_id: f32, is_inner: f32| {
        let base = vertices.len() as u32;
        for &pos in &[p0, p1, p2, p3] {
            vertices.push(ContainerVertex {
                position: pos,
                normal,
                face_id,
                _pad: is_inner,
            });
        }
        // CCW winding: 0-1-2, 0-2-3
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    // === Inner faces (facing into cavity) — is_inner = 1.0 ===

    // Inner floor (normal up)
    push_quad(
        [-hw, y0, -hd], [ hw, y0, -hd], [ hw, y0,  hd], [-hw, y0,  hd],
        [0.0, 1.0, 0.0], 0.0, 1.0,
    );
    // Inner front wall (Z = -hd, normal +Z)
    push_quad(
        [-hw, y0, -hd], [ hw, y0, -hd], [ hw, y1, -hd], [-hw, y1, -hd],
        [0.0, 0.0, 1.0], 1.0, 1.0,
    );
    // Inner back wall (Z = +hd, normal -Z)
    push_quad(
        [ hw, y0, hd], [-hw, y0, hd], [-hw, y1, hd], [ hw, y1, hd],
        [0.0, 0.0, -1.0], 1.0, 1.0,
    );
    // Inner left wall (X = -hw, normal +X)
    push_quad(
        [-hw, y0,  hd], [-hw, y0, -hd], [-hw, y1, -hd], [-hw, y1,  hd],
        [1.0, 0.0, 0.0], 1.0, 1.0,
    );
    // Inner right wall (X = +hw, normal -X)
    push_quad(
        [ hw, y0, -hd], [ hw, y0,  hd], [ hw, y1,  hd], [ hw, y1, -hd],
        [-1.0, 0.0, 0.0], 1.0, 1.0,
    );

    // === Outer faces (facing outward) — is_inner = 0.0 ===

    // Outer bottom (normal down)
    push_quad(
        [-ohw, oy0,  ohd], [ ohw, oy0,  ohd], [ ohw, oy0, -ohd], [-ohw, oy0, -ohd],
        [0.0, -1.0, 0.0], 0.0, 0.0,
    );
    // Outer front wall (Z = -ohd, normal -Z)
    push_quad(
        [ ohw, oy0, -ohd], [-ohw, oy0, -ohd], [-ohw, y1, -ohd], [ ohw, y1, -ohd],
        [0.0, 0.0, -1.0], 1.0, 0.0,
    );
    // Outer back wall (Z = +ohd, normal +Z)
    push_quad(
        [-ohw, oy0, ohd], [ ohw, oy0, ohd], [ ohw, y1, ohd], [-ohw, y1, ohd],
        [0.0, 0.0, 1.0], 1.0, 0.0,
    );
    // Outer left wall (X = -ohw, normal -X)
    push_quad(
        [-ohw, oy0, -ohd], [-ohw, oy0,  ohd], [-ohw, y1,  ohd], [-ohw, y1, -ohd],
        [-1.0, 0.0, 0.0], 1.0, 0.0,
    );
    // Outer right wall (X = +ohw, normal +X)
    push_quad(
        [ ohw, oy0,  ohd], [ ohw, oy0, -ohd], [ ohw, y1, -ohd], [ ohw, y1,  ohd],
        [1.0, 0.0, 0.0], 1.0, 0.0,
    );

    // === Top rim (connects inner wall top edge to outer wall top edge, normal up) — is_inner = 0.0 ===

    // Front rim
    push_quad(
        [-ohw, y1, -ohd], [ ohw, y1, -ohd], [ hw, y1, -hd], [-hw, y1, -hd],
        [0.0, 1.0, 0.0], 1.0, 0.0,
    );
    // Back rim
    push_quad(
        [ ohw, y1, ohd], [-ohw, y1, ohd], [-hw, y1, hd], [ hw, y1, hd],
        [0.0, 1.0, 0.0], 1.0, 0.0,
    );
    // Left rim
    push_quad(
        [-ohw, y1,  ohd], [-ohw, y1, -ohd], [-hw, y1, -hd], [-hw, y1,  hd],
        [0.0, 1.0, 0.0], 1.0, 0.0,
    );
    // Right rim
    push_quad(
        [ ohw, y1, -ohd], [ ohw, y1,  ohd], [ hw, y1,  hd], [ hw, y1, -hd],
        [0.0, 1.0, 0.0], 1.0, 0.0,
    );

    (vertices, indices)
}
