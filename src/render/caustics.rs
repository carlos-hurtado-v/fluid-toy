//! Caustics renderer — light-space forward splatting onto the pool floor.
//!
//! Pipeline per frame (between MC mesh generation and scene rendering):
//!   1. G-buffer: raster the MC water mesh from the sun (tight-fit ortho),
//!      capturing front-most world position + normal per texel.
//!   2. Splat: one photon per texel × 4 kinds (R/G/B refracted + shadow),
//!      analytically intersected with the container-local floor plane and
//!      additively splatted into the floor caustic map with a normalized
//!      gaussian kernel. Identical kernels for caustic and shadow channels
//!      make the redistribution energy-conserving by construction.
//!   3. Filter: separable gaussian blur + temporal EMA (sparkle suppression).
//!   4. Copy into a stable display texture sampled by the container shader.
//!
//! The map covers the floor rect in container-local space:
//!   u = local.x / half_width * 0.5 + 0.5, v = local.z / half_depth * 0.5 + 0.5

use wgpu::util::DeviceExt;

use crate::state::{CausticsConfig, GpuContainerGeometry};

/// Uniforms shared by the G-buffer and splat passes.
/// Layout mirrors `CausticsParams` in mc_caustics_gbuffer/splat.wgsl.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCausticsParams {
    light_view_proj: [[f32; 4]; 4],
    sun_dir: [f32; 3],
    flux_area: f32,
    ior_rgb: [f32; 3],
    sigma: f32,
    absorb_rgb: [f32; 3],
    inv_two_sigma_sq: f32,
    splat_norm: f32,
    splat_radius: f32,
    optical_density: f32,
    light_res: u32,
    time: f32,
    ripple_strength: f32,
    kinds: u32,
    _pad0: f32,
}

/// Layout mirrors `FilterParams` in mc_caustics_filter.wgsl.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFilterParams {
    dir_x: i32,
    dir_y: i32,
    sigma: f32,
    alpha: f32,
}

/// Beer-Lambert absorption coefficients — must match mc_render.wgsl
const ABSORPTION_COEFFS: [f32; 3] = [0.30, 0.08, 0.02];
/// Water IOR (matches the hardcoded value in MarchingCubesRenderer)
const WATER_IOR: f32 = 1.333;
/// Splat quads are truncated at this many sigmas
const SPLAT_TRUNCATION_SIGMAS: f32 = 2.5;
/// World-space padding around the container when fitting the light ortho
const LIGHT_FIT_PADDING: f32 = 0.08;

fn create_map_texture(
    device: &wgpu::Device,
    res: u32,
    label: &str,
    usage: wgpu::TextureUsages,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub struct CausticsRenderer {
    res: u32,
    // Light-space G-buffer
    _gbuffer_pos_texture: wgpu::Texture,
    gbuffer_pos_view: wgpu::TextureView,
    _gbuffer_nrm_texture: wgpu::Texture,
    gbuffer_nrm_view: wgpu::TextureView,
    _gbuffer_depth_texture: wgpu::Texture,
    gbuffer_depth_view: wgpu::TextureView,
    // Floor map chain: splat target (also EMA output), blur ping-pong, display
    splat_texture: wgpu::Texture,
    splat_view: wgpu::TextureView,
    _tmp_texture: wgpu::Texture,
    _tmp_view: wgpu::TextureView,
    _blurred_texture: wgpu::Texture,
    _blurred_view: wgpu::TextureView,
    display_texture: wgpu::Texture,
    display_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    // Pipelines + bind groups
    gbuffer_pipeline: wgpu::RenderPipeline,
    gbuffer_bind_group: wgpu::BindGroup,
    occluder_pipeline: wgpu::RenderPipeline,
    occluder_bind_group: wgpu::BindGroup,
    splat_pipeline: wgpu::RenderPipeline,
    splat_bind_group: wgpu::BindGroup,
    blur_pipeline: wgpu::ComputePipeline,
    blur_h_bind_group: wgpu::BindGroup,
    blur_v_bind_group: wgpu::BindGroup,
    ema_pipeline: wgpu::ComputePipeline,
    ema_bind_group: wgpu::BindGroup,
    // Uniforms
    params_buffer: wgpu::Buffer,
    container_geom_buffer: wgpu::Buffer,
    blur_h_buffer: wgpu::Buffer,
    blur_v_buffer: wgpu::Buffer,
    ema_buffer: wgpu::Buffer,
    // EMA history validity (false -> next accumulate takes the frame as-is)
    history_valid: bool,
    // Photon kinds per texel this frame (4 chromatic / 2 white), set by update()
    current_kinds: u32,
}

impl CausticsRenderer {
    pub fn new(
        device: &wgpu::Device,
        mc_vertex_buffer: &wgpu::Buffer,
        container_geom: &GpuContainerGeometry,
        config: &CausticsConfig,
    ) -> Self {
        let res = config.light_resolution.clamp(128, 1024);

        // --- Textures ---
        let (gbuffer_pos_texture, gbuffer_pos_view) = create_map_texture(
            device, res, "Caustics GBuffer Position",
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let (gbuffer_nrm_texture, gbuffer_nrm_view) = create_map_texture(
            device, res, "Caustics GBuffer Normal",
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let gbuffer_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Caustics GBuffer Depth"),
            size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let gbuffer_depth_view = gbuffer_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let (splat_texture, splat_view) = create_map_texture(
            device, res, "Caustics Splat Map",
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let (tmp_texture, tmp_view) = create_map_texture(
            device, res, "Caustics Blur Tmp",
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        );
        let (blurred_texture, blurred_view) = create_map_texture(
            device, res, "Caustics Blurred",
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        );
        let (display_texture, display_view) = create_map_texture(
            device, res, "Caustics Display Map",
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Caustics Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // --- Uniform buffers ---
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Caustics Params"),
            size: std::mem::size_of::<GpuCausticsParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let container_geom_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Caustics Container Geometry"),
            contents: bytemuck::bytes_of(container_geom),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let filter_buffer = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<GpuFilterParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let blur_h_buffer = filter_buffer("Caustics Blur H Params");
        let blur_v_buffer = filter_buffer("Caustics Blur V Params");
        let ema_buffer = filter_buffer("Caustics EMA Params");

        // --- Shaders (gbuffer + splat get container_common.wgsl prepended) ---
        let container_common = include_str!("../shaders/container_common.wgsl");
        let gbuffer_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Caustics GBuffer Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}", container_common, include_str!("../shaders/mc_caustics_gbuffer.wgsl")).into(),
            ),
        });
        let splat_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Caustics Splat Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}", container_common, include_str!("../shaders/mc_caustics_splat.wgsl")).into(),
            ),
        });
        let filter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Caustics Filter Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mc_caustics_filter.wgsl").into()),
        });

        // --- G-buffer pipeline ---
        let gbuffer_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Caustics GBuffer Pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &gbuffer_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &gbuffer_shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
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
        let gbuffer_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Caustics GBuffer BG"),
            layout: &gbuffer_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: mc_vertex_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: container_geom_buffer.as_entire_binding() },
            ],
        });

        // --- Container occluder pipeline (depth-only into the light raster) ---
        // Vertex layout matches ContainerVertex (32 bytes); only position read.
        let occluder_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Caustics Occluder Pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &gbuffer_shader,
                entry_point: Some("vs_occluder"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 32,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &gbuffer_shader,
                entry_point: Some("fs_occluder"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::empty(),
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::empty(),
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
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
        let occluder_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Caustics Occluder BG"),
            layout: &occluder_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: container_geom_buffer.as_entire_binding() },
            ],
        });

        // --- Splat pipeline (additive) ---
        let additive = wgpu::BlendState {
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
        let splat_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Caustics Splat Pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &splat_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &splat_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(additive),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let splat_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Caustics Splat BG"),
            layout: &splat_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: container_geom_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&gbuffer_pos_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&gbuffer_nrm_view) },
            ],
        });

        // --- Filter pipelines ---
        let blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Caustics Blur Pipeline"),
            layout: None,
            module: &filter_shader,
            entry_point: Some("blur"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let blur_layout = blur_pipeline.get_bind_group_layout(0);
        let blur_h_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Caustics Blur H BG"),
            layout: &blur_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&splat_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&tmp_view) },
                wgpu::BindGroupEntry { binding: 2, resource: blur_h_buffer.as_entire_binding() },
            ],
        });
        let blur_v_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Caustics Blur V BG"),
            layout: &blur_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&tmp_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&blurred_view) },
                wgpu::BindGroupEntry { binding: 2, resource: blur_v_buffer.as_entire_binding() },
            ],
        });

        let ema_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Caustics EMA Pipeline"),
            layout: None,
            module: &filter_shader,
            entry_point: Some("temporal_accumulate"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let ema_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Caustics EMA BG"),
            layout: &ema_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&blurred_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&splat_view) },
                wgpu::BindGroupEntry { binding: 2, resource: ema_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&display_view) },
            ],
        });

        Self {
            res,
            _gbuffer_pos_texture: gbuffer_pos_texture,
            gbuffer_pos_view,
            _gbuffer_nrm_texture: gbuffer_nrm_texture,
            gbuffer_nrm_view,
            _gbuffer_depth_texture: gbuffer_depth_texture,
            gbuffer_depth_view,
            splat_texture,
            splat_view,
            _tmp_texture: tmp_texture,
            _tmp_view: tmp_view,
            _blurred_texture: blurred_texture,
            _blurred_view: blurred_view,
            display_texture,
            display_view,
            sampler,
            gbuffer_pipeline,
            gbuffer_bind_group,
            occluder_pipeline,
            occluder_bind_group,
            splat_pipeline,
            splat_bind_group,
            blur_pipeline,
            blur_h_bind_group,
            blur_v_bind_group,
            ema_pipeline,
            ema_bind_group,
            params_buffer,
            container_geom_buffer,
            blur_h_buffer,
            blur_v_buffer,
            ema_buffer,
            history_valid: false,
            current_kinds: 2,
        }
    }

    pub fn display_view(&self) -> &wgpu::TextureView {
        &self.display_view
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// Drop temporal history; the next accumulated frame is taken as-is.
    /// Call when caustics were inactive this frame.
    pub fn invalidate_history(&mut self) {
        self.history_valid = false;
    }

    /// Update light camera fit + all uniforms. Call once per frame before run().
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        container_geom: &GpuContainerGeometry,
        sun_dir: [f32; 3],
        config: &CausticsConfig,
        water_clarity: f32,
        time: f32,
    ) {
        queue.write_buffer(&self.container_geom_buffer, 0, bytemuck::bytes_of(container_geom));

        // --- Light basis (orthonormal, forward = away from the sun) ---
        let forward = normalize([-sun_dir[0], -sun_dir[1], -sun_dir[2]]);
        let world_up = if forward[1].abs() > 0.99 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] };
        let right = normalize(cross(forward, world_up));
        let up = cross(right, forward);

        // --- Fit ortho extents to the container's 8 corners ---
        let center = [0.0, container_geom.center_y, 0.0];
        let (hw, hh, hd) = (
            container_geom.half_width,
            container_geom.half_height,
            container_geom.half_depth,
        );
        let rot = [
            [container_geom.forward_row0[0], container_geom.forward_row0[1], container_geom.forward_row0[2]],
            [container_geom.forward_row1[0], container_geom.forward_row1[1], container_geom.forward_row1[2]],
            [container_geom.forward_row2[0], container_geom.forward_row2[1], container_geom.forward_row2[2]],
        ];
        let mut min_e = [f32::MAX; 3];
        let mut max_e = [f32::MIN; 3];
        for ix in [-1.0f32, 1.0] {
            for iy in [-1.0f32, 1.0] {
                for iz in [-1.0f32, 1.0] {
                    let local = [ix * hw, iy * hh, iz * hd];
                    // World offset from container center (rotation only)
                    let w = [dot(rot[0], local), dot(rot[1], local), dot(rot[2], local)];
                    let e = [dot(right, w), dot(up, w), dot(forward, w)];
                    for a in 0..3 {
                        min_e[a] = min_e[a].min(e[a]);
                        max_e[a] = max_e[a].max(e[a]);
                    }
                }
            }
        }
        let pad = LIGHT_FIT_PADDING;
        let half_r = (max_e[0] - min_e[0]) * 0.5 + pad;
        let half_u = (max_e[1] - min_e[1]) * 0.5 + pad;
        let center_r = (max_e[0] + min_e[0]) * 0.5 + dot(right, center);
        let center_u = (max_e[1] + min_e[1]) * 0.5 + dot(up, center);
        let near_f = min_e[2] - pad + dot(forward, center);
        let far_f = max_e[2] + pad + dot(forward, center);
        let inv_depth = 1.0 / (far_f - near_f).max(1e-4);

        // Combined ortho view-proj, column-major:
        //   clip.x = (right . p - center_r) / half_r
        //   clip.y = (up . p - center_u) / half_u
        //   clip.z = (forward . p - near_f) / (far_f - near_f)   in [0, 1]
        let light_view_proj = [
            [right[0] / half_r, up[0] / half_u, forward[0] * inv_depth, 0.0],
            [right[1] / half_r, up[1] / half_u, forward[1] * inv_depth, 0.0],
            [right[2] / half_r, up[2] / half_u, forward[2] * inv_depth, 0.0],
            [-center_r / half_r, -center_u / half_u, -near_f * inv_depth, 1.0],
        ];

        // --- Photon energy + splat kernel ---
        let res = self.res as f32;
        let flux_area = (2.0 * half_r / res) * (2.0 * half_u / res);
        let photon_spacing = flux_area.sqrt();
        // Floor sigma at half a map texel: smaller splats alias into grain
        let map_texel = 2.0 * container_geom.half_width.max(container_geom.half_depth) / res;
        let sigma = (config.splat_size.max(0.25) * photon_spacing).max(0.5 * map_texel);
        let splat_radius = SPLAT_TRUNCATION_SIGMAS * sigma;
        let truncation = 1.0 - (-0.5 * SPLAT_TRUNCATION_SIGMAS * SPLAT_TRUNCATION_SIGMAS).exp();
        let splat_norm = 1.0 / (2.0 * std::f32::consts::PI * sigma * sigma * truncation);

        // Chromatic dispersion needs per-channel splats (4 kinds); without it
        // one white quad carries all channels at half the splat cost
        let d = config.dispersion;
        self.current_kinds = if d > 0.005 { 4 } else { 2 };

        let params = GpuCausticsParams {
            light_view_proj,
            sun_dir: normalize(sun_dir),
            flux_area,
            ior_rgb: [WATER_IOR - 0.010 * d, WATER_IOR, WATER_IOR + 0.012 * d],
            sigma,
            absorb_rgb: ABSORPTION_COEFFS,
            inv_two_sigma_sq: 1.0 / (2.0 * sigma * sigma),
            splat_norm,
            splat_radius,
            // Mirrors mc_render.wgsl optical density from clarity
            optical_density: (1.0 - water_clarity) * 2.5 + 0.05,
            light_res: self.res,
            time,
            ripple_strength: config.ripple_strength,
            kinds: self.current_kinds,
            _pad0: 0.0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let blur_h = GpuFilterParams { dir_x: 1, dir_y: 0, sigma: config.blur_sigma.max(0.05), alpha: 0.0 };
        let blur_v = GpuFilterParams { dir_x: 0, dir_y: 1, sigma: config.blur_sigma.max(0.05), alpha: 0.0 };
        let alpha = if self.history_valid {
            config.temporal_smoothing.clamp(0.0, 0.98)
        } else {
            0.0
        };
        let ema = GpuFilterParams { dir_x: 0, dir_y: 0, sigma: 1.0, alpha };
        queue.write_buffer(&self.blur_h_buffer, 0, bytemuck::bytes_of(&blur_h));
        queue.write_buffer(&self.blur_v_buffer, 0, bytemuck::bytes_of(&blur_v));
        queue.write_buffer(&self.ema_buffer, 0, bytemuck::bytes_of(&ema));
        self.history_valid = true;
    }

    /// Record all caustics passes. The MC vertex + indirect buffers must
    /// already be populated (call after MarchingCubesRenderer::generate).
    /// `container_mesh` = (vertex buffer, index buffer, index count) of the
    /// pool mesh, drawn depth-only so rim-shadowed water emits no photons.
    pub fn run(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        mc_indirect_buffer: &wgpu::Buffer,
        container_mesh: Option<(&wgpu::Buffer, &wgpu::Buffer, u32)>,
    ) {
        // Pass 1: light-space G-buffer
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Caustics GBuffer Pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.gbuffer_pos_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.gbuffer_nrm_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.gbuffer_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // Container first: its depth blocks photons from rim-shadowed water
            if let Some((vb, ib, index_count)) = container_mesh {
                pass.set_pipeline(&self.occluder_pipeline);
                pass.set_bind_group(0, &self.occluder_bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..index_count, 0, 0..1);
            }
            pass.set_pipeline(&self.gbuffer_pipeline);
            pass.set_bind_group(0, &self.gbuffer_bind_group, &[]);
            pass.draw_indirect(mc_indirect_buffer, 0);
        }

        // Pass 2: photon splatting (additive into the splat map)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Caustics Splat Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.splat_view,
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
            pass.set_pipeline(&self.splat_pipeline);
            pass.set_bind_group(0, &self.splat_bind_group, &[]);
            // (refracted kinds + shadow) x one photon per light texel
            pass.draw(0..6, 0..self.res * self.res * self.current_kinds);
        }

        // Passes 3-5: blur H, blur V, temporal EMA (writes back into splat map)
        let groups = self.res.div_ceil(8);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Caustics Filter Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.blur_h_bind_group, &[]);
            pass.dispatch_workgroups(groups, groups, 1);
            pass.set_bind_group(0, &self.blur_v_bind_group, &[]);
            pass.dispatch_workgroups(groups, groups, 1);
            pass.set_pipeline(&self.ema_pipeline);
            pass.set_bind_group(0, &self.ema_bind_group, &[]);
            pass.dispatch_workgroups(groups, groups, 1);
        }

        // Pass 6: publish into the stable display texture the container samples
        encoder.copy_texture_to_texture(
            self.splat_texture.as_image_copy(),
            self.display_texture.as_image_copy(),
            wgpu::Extent3d { width: self.res, height: self.res, depth_or_array_layers: 1 },
        );
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-5 {
        return [0.0, 1.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
