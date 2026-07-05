//! Rigid body renderer — procedural shapes + custom GLB mesh support

use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

use crate::render::camera::GpuCameraParams;
use crate::render::mesh_loader::{self, MeshVertex};
use crate::state::{
    GpuContainerGeometry, GpuRigidBodyRender, GpuShCoefficients, RigidBodyShape,
    MAX_RIGID_BODIES,
};

/// Scene lighting shared by all bodies — mirrors RbLightParams in
/// rigid_body.wgsl / rigid_body_mesh.wgsl.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRbLightParams {
    /// Direction toward the sun (world space, normalized)
    pub sun_dir: [f32; 3],
    /// SH ambient scale (environment intensity)
    pub ibl_strength: f32,
    /// Sun color x intensity, zeroed when the sun is disabled
    pub sun_rgb: [f32; 3],
    pub _pad0: f32,
}

/// Per-body draw info, parallel to the uploaded render params array
#[derive(Debug, Clone, Copy)]
pub struct RigidBodyDraw {
    pub shape: RigidBodyShape,
    pub vertex_count: u32,
}

/// Which pipeline family a pass needs
#[derive(Clone, Copy, PartialEq)]
enum DrawMode {
    Standard,
    Msaa,
    DepthOnly,
}

pub struct RigidBodyRenderer {
    // Procedural pipeline (Cube, Sphere, Cylinder, Torus, Propeller)
    pipeline: wgpu::RenderPipeline,
    msaa_pipeline: Option<wgpu::RenderPipeline>,

    // Depth-only pipelines (for GTAO front depth prepass, 1x, no color targets)
    depth_only_pipeline: wgpu::RenderPipeline,
    mesh_depth_only_pipeline: Option<wgpu::RenderPipeline>,

    // Shared bind group (group 0): camera + bodies + scene lighting — used by
    // both pipeline families
    bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    bodies_buffer: wgpu::Buffer,
    light_buffer: wgpu::Buffer,
    sh_buffer: wgpu::Buffer,
    container_geom_buffer: wgpu::Buffer,

    // Draw list for the current frame (index i draws instance i)
    draws: Vec<RigidBodyDraw>,

    // Mesh pipeline (Custom GLB models)
    mesh_pipeline: Option<wgpu::RenderPipeline>,
    mesh_msaa_pipeline: Option<wgpu::RenderPipeline>,
    mesh_texture_bind_group: Option<wgpu::BindGroup>, // Group 1: texture + sampler
    mesh_vertex_buffer: Option<wgpu::Buffer>,
    mesh_index_buffer: Option<wgpu::Buffer>,
    mesh_index_count: u32,
}

impl RigidBodyRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        camera_params: &GpuCameraParams,
        msaa_sample_count: u32,
    ) -> Self {
        // === Shared resources (group 0) ===
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RigidBody Camera Buffer"),
            contents: bytemuck::bytes_of(camera_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bodies_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RigidBody Params Buffer"),
            contents: bytemuck::cast_slice(&[GpuRigidBodyRender::default(); MAX_RIGID_BODIES]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RigidBody Light Buffer"),
            contents: bytemuck::bytes_of(&GpuRbLightParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RigidBody SH Buffer"),
            contents: bytemuck::bytes_of(&GpuShCoefficients::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Zeroed geometry => is_pool = 0 => rim_visibility returns fully lit
        let container_geom_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RigidBody Container Geometry"),
            contents: bytemuck::bytes_of(&GpuContainerGeometry::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_entry = |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RigidBody Group0 Layout"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                uniform_entry(2, wgpu::ShaderStages::FRAGMENT),
                uniform_entry(3, wgpu::ShaderStages::FRAGMENT),
                uniform_entry(4, wgpu::ShaderStages::FRAGMENT),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RigidBody Group0"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bodies_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: container_geom_buffer.as_entire_binding(),
                },
            ],
        });

        // Shared primitive and depth state
        let primitive_state = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
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

        let color_target = wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        };

        // === Procedural pipeline ===
        // container_common.wgsl supplies ContainerGeometry + rim_visibility
        let container_common = include_str!("../shaders/container_common.wgsl");
        let procedural_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Rigid Body Procedural Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}", container_common, include_str!("../shaders/rigid_body.wgsl")).into(),
            ),
        });

        let procedural_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RigidBody Procedural Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RigidBody Procedural Pipeline"),
            layout: Some(&procedural_layout),
            vertex: wgpu::VertexState {
                module: &procedural_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &procedural_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(color_target.clone())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: primitive_state,
            depth_stencil: Some(depth_stencil.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let msaa_pipeline = if msaa_sample_count > 1 {
            Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("RigidBody Procedural MSAA Pipeline"),
                layout: Some(&procedural_layout),
                vertex: wgpu::VertexState {
                    module: &procedural_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &procedural_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(color_target.clone())],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: primitive_state,
                depth_stencil: Some(depth_stencil.clone()),
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

        // === Depth-only procedural pipeline (for GTAO front depth prepass) ===
        let depth_only_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RigidBody Depth-Only Pipeline"),
            layout: Some(&procedural_layout),
            vertex: wgpu::VertexState {
                module: &procedural_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: None,
            primitive: primitive_state,
            depth_stencil: Some(depth_stencil.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // === Mesh pipeline (GLB models) ===
        let (mesh_pipeline, mesh_msaa_pipeline, mesh_depth_only_pipeline, mesh_texture_bind_group, mesh_vertex_buffer, mesh_index_buffer, mesh_index_count) =
            match mesh_loader::load_embedded_duck() {
                Ok(loaded_mesh) => {
                    let result = Self::create_mesh_resources(
                        device,
                        queue,
                        surface_format,
                        &bind_group_layout,
                        &loaded_mesh,
                        primitive_state,
                        &depth_stencil,
                        &color_target,
                        msaa_sample_count,
                    );
                    (
                        Some(result.0),
                        result.1,
                        result.6,
                        Some(result.2),
                        Some(result.3),
                        Some(result.4),
                        result.5,
                    )
                }
                Err(e) => {
                    log::error!("Failed to load duck.glb: {}", e);
                    (None, None, None, None, None, None, 0)
                }
            };

        Self {
            pipeline,
            msaa_pipeline,
            depth_only_pipeline,
            mesh_depth_only_pipeline,
            bind_group,
            camera_buffer,
            bodies_buffer,
            light_buffer,
            sh_buffer,
            container_geom_buffer,
            draws: Vec::new(),
            mesh_pipeline,
            mesh_msaa_pipeline,
            mesh_texture_bind_group,
            mesh_vertex_buffer,
            mesh_index_buffer,
            mesh_index_count,
        }
    }

    /// Create all GPU resources for the mesh pipeline
    fn create_mesh_resources(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _surface_format: wgpu::TextureFormat,
        group0_layout: &wgpu::BindGroupLayout,
        loaded_mesh: &mesh_loader::LoadedMesh,
        primitive_state: wgpu::PrimitiveState,
        depth_stencil: &wgpu::DepthStencilState,
        color_target: &wgpu::ColorTargetState,
        msaa_sample_count: u32,
    ) -> (
        wgpu::RenderPipeline,            // mesh pipeline
        Option<wgpu::RenderPipeline>,    // mesh MSAA pipeline
        wgpu::BindGroup,                 // texture bind group (group 1)
        wgpu::Buffer,                    // vertex buffer
        wgpu::Buffer,                    // index buffer
        u32,                             // index count
        Option<wgpu::RenderPipeline>,    // mesh depth-only pipeline
    ) {
        // Vertex + index buffers
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RigidBody Mesh Vertices"),
            contents: bytemuck::cast_slice(&loaded_mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RigidBody Mesh Indices"),
            contents: bytemuck::cast_slice(&loaded_mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Base color texture
        let tex_size = wgpu::Extent3d {
            width: loaded_mesh.texture_width.max(1),
            height: loaded_mesh.texture_height.max(1),
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RigidBody Mesh Texture"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload texture data
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &loaded_mesh.texture_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * loaded_mesh.texture_width),
                rows_per_image: Some(loaded_mesh.texture_height),
            },
            tex_size,
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("RigidBody Mesh Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Group 1 layout: texture + sampler
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RigidBody Mesh Texture Layout"),
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

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RigidBody Mesh Texture Group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Mesh shader (container_common.wgsl supplies rim_visibility)
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Rigid Body Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../shaders/container_common.wgsl"),
                    include_str!("../shaders/rigid_body_mesh.wgsl")
                )
                .into(),
            ),
        });

        // Mesh pipeline layout: group 0 (camera+body) + group 1 (texture)
        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RigidBody Mesh Pipeline Layout"),
            bind_group_layouts: &[group0_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex buffer layout for MeshVertex (48 bytes stride)
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0, // position
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1, // normal
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 24,
                    shader_location: 2, // uv
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 3, // color
                },
            ],
        };

        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RigidBody Mesh Pipeline"),
            layout: Some(&mesh_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: Some("vs_main"),
                buffers: std::slice::from_ref(&vertex_buffer_layout),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(color_target.clone())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: primitive_state,
            depth_stencil: Some(depth_stencil.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let mesh_msaa_pipeline = if msaa_sample_count > 1 {
            Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("RigidBody Mesh MSAA Pipeline"),
                layout: Some(&mesh_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &mesh_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_buffer_layout],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &mesh_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(color_target.clone())],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: primitive_state,
                depth_stencil: Some(depth_stencil.clone()),
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

        // Depth-only mesh pipeline (for GTAO front depth prepass)
        let mesh_depth_only_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RigidBody Mesh Depth-Only Layout"),
            bind_group_layouts: &[group0_layout],
            push_constant_ranges: &[],
        });
        let mesh_depth_only_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RigidBody Mesh Depth-Only Pipeline"),
            layout: Some(&mesh_depth_only_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
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
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 24,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 32,
                            shader_location: 3,
                        },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: None,
            primitive: primitive_state,
            depth_stencil: Some(depth_stencil.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        }));

        (
            mesh_pipeline,
            mesh_msaa_pipeline,
            texture_bind_group,
            vertex_buffer,
            index_buffer,
            loaded_mesh.indices.len() as u32,
            mesh_depth_only_pipeline,
        )
    }

    /// Render-side body array (position/rotation/shape/half_extent) — bound by
    /// the caustics splat pass for photon occlusion.
    pub fn bodies_buffer(&self) -> &wgpu::Buffer {
        &self.bodies_buffer
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, params: &GpuCameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_light(&self, queue: &wgpu::Queue, params: &GpuRbLightParams) {
        queue.write_buffer(&self.light_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_sh_coefficients(&self, queue: &wgpu::Queue, coeffs: &GpuShCoefficients) {
        queue.write_buffer(&self.sh_buffer, 0, bytemuck::bytes_of(coeffs));
    }

    pub fn update_container_geometry(&self, queue: &wgpu::Queue, geom: &GpuContainerGeometry) {
        queue.write_buffer(&self.container_geom_buffer, 0, bytemuck::bytes_of(geom));
    }

    /// Upload this frame's body render params + draw list. `bodies[i]` is
    /// drawn as instance i; both slices must be parallel (enabled bodies only).
    pub fn update_bodies(
        &mut self,
        queue: &wgpu::Queue,
        bodies: &[GpuRigidBodyRender],
        draws: &[RigidBodyDraw],
    ) {
        debug_assert_eq!(bodies.len(), draws.len());
        let count = bodies.len().min(MAX_RIGID_BODIES);
        if count > 0 {
            queue.write_buffer(
                &self.bodies_buffer,
                0,
                bytemuck::cast_slice(&bodies[..count]),
            );
        }
        self.draws.clear();
        self.draws.extend_from_slice(&draws[..count]);
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.draw_all(render_pass, DrawMode::Standard);
    }

    /// Render using the MSAA pipeline (for rendering inside MC's multisampled pass)
    pub fn render_msaa<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.draw_all(render_pass, DrawMode::Msaa);
    }

    /// Render depth-only (for GTAO front depth prepass, 1x, no color targets)
    pub fn render_depth_only<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.draw_all(render_pass, DrawMode::DepthOnly);
    }

    fn draw_all<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, mode: DrawMode) {
        // Procedural bodies (all shapes except Custom)
        if self.draws.iter().any(|d| d.shape != RigidBodyShape::Custom) {
            let pipeline = match mode {
                DrawMode::Standard => &self.pipeline,
                DrawMode::Msaa => self.msaa_pipeline.as_ref().unwrap_or(&self.pipeline),
                DrawMode::DepthOnly => &self.depth_only_pipeline,
            };
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            for (i, draw) in self.draws.iter().enumerate() {
                if draw.shape != RigidBodyShape::Custom {
                    render_pass.draw(0..draw.vertex_count, i as u32..i as u32 + 1);
                }
            }
        }

        // Mesh bodies (Custom GLB)
        if self.draws.iter().any(|d| d.shape == RigidBodyShape::Custom) {
            let pipeline = match mode {
                DrawMode::Standard => self.mesh_pipeline.as_ref(),
                DrawMode::Msaa => self.mesh_msaa_pipeline.as_ref().or(self.mesh_pipeline.as_ref()),
                DrawMode::DepthOnly => self.mesh_depth_only_pipeline.as_ref(),
            };
            if let (Some(pipeline), Some(vb), Some(ib), Some(tbg)) = (
                pipeline,
                &self.mesh_vertex_buffer,
                &self.mesh_index_buffer,
                &self.mesh_texture_bind_group,
            ) {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &self.bind_group, &[]);
                if mode != DrawMode::DepthOnly {
                    render_pass.set_bind_group(1, tbg, &[]);
                }
                render_pass.set_vertex_buffer(0, vb.slice(..));
                render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                for (i, draw) in self.draws.iter().enumerate() {
                    if draw.shape == RigidBodyShape::Custom {
                        render_pass.draw_indexed(0..self.mesh_index_count, 0, i as u32..i as u32 + 1);
                    }
                }
            }
        }
    }
}
