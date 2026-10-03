//! The water pass of the marching-cubes renderer: the MC mesh shaded by
//! the `mc_render` shader (refraction through the scene behind it, reflection,
//! medium, foam), drawn after the backdrop and scene objects in one pass.
//!
//! Group 0 binds what the other passes produced (G-buffers, background, SSR,
//! foam); group 1 hands the shader the density field the mesh came from, for
//! its world-space in-water test.

use super::mc_background::Background;
use super::mc_faces::FaceBuffers;
use super::mc_field::{FieldTextures, GpuGridParams};
use super::mc_probe::PixelProbe;
use super::mc_ssr::Ssr;
use crate::gpu::bind::{entry, layout, FRAGMENT, VERTEX, VERTEX_FRAGMENT};
use crate::render::GpuCameraParams;
use crate::state::{GpuContainerGeometry, GpuLightParams, GpuShCoefficients};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Water shading parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuWaterParams {
    pub water_color: [f32; 3],
    pub roughness: f32,
    pub ior: f32,
    pub refraction_strength: f32,
    pub env_intensity: f32,
    pub use_env_background: u32,
    pub background_r: f32,
    pub background_g: f32,
    pub background_b: f32,
    pub time: f32,
    pub deep_color_r: f32,
    pub deep_color_g: f32,
    pub deep_color_b: f32,
    pub ripple_strength: f32,
    pub clarity: f32,
    /// Mode slot: the SS path stores its debug-view index here, the MC path
    /// its physical-refraction flag (1 = Snell two-interface, 0 = legacy offset)
    pub _pad1: f32,
    /// Master scale on surface foam coverage response (whitewater GUI)
    pub foam_coverage: f32,
    /// Master scale on entrained-air milkiness (whitewater GUI)
    pub aeration_strength: f32,
    /// 1 = physical water medium (absorption + single scattering), 0 = legacy
    pub physical_medium: f32,
    /// Enabled rigid bodies at the front of the MC body array (refraction
    /// rays intersect them exactly; the SS path leaves it 0)
    pub body_count: u32,
    /// McDebugView::as_u32 (0 = off)
    pub debug_view: u32,
    /// rendering.mc_silhouette_exit (McSilhouetteExit::as_u32)
    pub silhouette_exit: u32,
    /// rendering.mc_front_face_exit
    pub front_exit: u32,
    /// Ground-projected backdrop (GpuEnvironmentParams): escaping refraction
    /// rays that head down land on it
    pub ground_enabled: u32,
    pub ground_y: f32,
    pub ground_capture_height: f32,
    /// rendering.mc_filtered_lookup
    pub filtered_lookup: u32,
    /// rendering.mc_volume_trace
    pub volume_trace: u32,
    /// 1 = the depth buffer holds opaque surfaces the refraction tracer does
    /// not know exactly (pool walls and floor, a Custom body) and rays are
    /// tested against it; 0 = glass tank with procedural bodies only
    pub depth_occluders: u32,
    pub _pad_g: u32,
}

impl Default for GpuWaterParams {
    fn default() -> Self {
        Self {
            water_color: [0.1, 0.4, 0.8],
            roughness: 0.03,
            ior: 1.333,
            refraction_strength: 0.15,
            env_intensity: 1.0,
            use_env_background: 1,
            background_r: 0.15,
            background_g: 0.15,
            background_b: 0.2,
            time: 0.0,
            deep_color_r: 0.01,
            deep_color_g: 0.04,
            deep_color_b: 0.1,
            ripple_strength: 0.015,
            clarity: 0.65,
            _pad1: 0.0,
            foam_coverage: 0.8,
            aeration_strength: 0.95,
            physical_medium: 1.0,
            body_count: 0,
            debug_view: 0,
            silhouette_exit: 0,
            front_exit: 0,
            ground_enabled: 0,
            ground_y: 0.0,
            ground_capture_height: 0.0,
            filtered_lookup: 0,
            volume_trace: 0,
            depth_occluders: 1,
            _pad_g: 0,
        }
    }
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

/// Create MSAA color texture (for multisampled rendering)
fn create_msaa_texture(device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32, sample_count: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MC MSAA Color Texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create multisampled depth texture
fn create_msaa_depth_texture(device: &wgpu::Device, width: u32, height: u32, sample_count: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MC MSAA Depth Texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// The water pass's own targets: MSAA color (None at 1x; the pass then draws
/// straight into the scene buffer) and scene depth at the same sample count
struct WaterTargets {
    msaa: Option<(wgpu::Texture, wgpu::TextureView)>,
    depth: (wgpu::Texture, wgpu::TextureView),
}

impl WaterTargets {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32, sample_count: u32) -> Self {
        if sample_count > 1 {
            Self {
                msaa: Some(create_msaa_texture(device, format, width, height, sample_count)),
                depth: create_msaa_depth_texture(device, width, height, sample_count),
            }
        } else {
            Self { msaa: None, depth: create_depth_texture(device, width, height) }
        }
    }
}

/// Uniform and storage buffers that several passes bind (the water pass
/// binds every one of them)
pub struct SharedBuffers {
    pub camera: wgpu::Buffer,
    pub grid_params: wgpu::Buffer,
    /// Container geometry (shared struct: geometry, rotation, physics, clip)
    pub container_geom: wgpu::Buffer,
    pub water_params: wgpu::Buffer,
    pub light_params: wgpu::Buffer,
    pub sh_coefficients: wgpu::Buffer,
    /// Render-side rigid body array (same layout as the body renderer uses),
    /// for exact ray-body hits in refraction: the depth buffer only holds the
    /// camera-facing side of a body
    pub bodies: wgpu::Buffer,
}

impl SharedBuffers {
    pub fn new(device: &wgpu::Device, grid_params: &GpuGridParams) -> Self {
        let grid_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Grid Params"),
            contents: bytemuck::bytes_of(grid_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
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
        let bodies_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Rigid Bodies"),
            contents: bytemuck::cast_slice(
                &[crate::state::GpuRigidBodyRender::default(); crate::state::MAX_RIGID_BODIES],
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Container geometry buffer (shared struct: geometry, rotation, physics, clip)
        let container_geom = GpuContainerGeometry::zeroed();
        let container_geom_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Container Geometry"),
            contents: bytemuck::bytes_of(&container_geom),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Light params buffer
        let light_params = GpuLightParams {
            sun_direction: [0.5, 0.8, 0.3],
            sun_enabled: 1,
            sun_color: [1.0, 0.95, 0.85],
            sun_intensity: 2.0,
            ambient_intensity: 1.0,
            _pad0: [0.0; 3],
            _padding: [0.0; 3],
            _pad1: 0.0,
        };
        let light_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Light Params"),
            contents: bytemuck::bytes_of(&light_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // SH coefficients buffer (144 bytes)
        let sh_coefficients = GpuShCoefficients::default();
        let sh_coefficients_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC SH Coefficients"),
            contents: bytemuck::bytes_of(&sh_coefficients),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            camera: camera_buffer,
            grid_params: grid_params_buffer,
            container_geom: container_geom_buffer,
            water_params: water_params_buffer,
            light_params: light_params_buffer,
            sh_coefficients: sh_coefficients_buffer,
            bodies: bodies_buffer,
        }
    }
}

/// Surface foam map (simulation::FoamMap): foam layer, coarse surface grid
/// (column tops), its params and the flow-map coordinates. The water shader
/// composites it on the top surface.
pub struct FoamMapViews {
    pub map_view: wgpu::TextureView,
    pub surface_view: wgpu::TextureView,
    pub params: wgpu::Buffer,
    pub coords_view: wgpu::TextureView,
}

/// What group 0 of the water pipeline binds, gathered from the passes that own it
pub struct WaterBindings<'a> {
    pub buffers: &'a SharedBuffers,
    pub vertices: &'a wgpu::Buffer,
    pub env_view: &'a wgpu::TextureView,
    pub env_sampler: &'a wgpu::Sampler,
    pub faces: &'a FaceBuffers,
    pub background: &'a Background,
    pub ssr: &'a Ssr,
    pub foam_density_view: &'a wgpu::TextureView,
    pub foam_map: &'a FoamMapViews,
    pub probe: &'a PixelProbe,
}

/// Water render bind group (binding numbers match mc_render/bindings.wgsl). Rebuilt
/// when the screen-sized views (resize), the environment texture (HDR
/// switch) or the probe buffer (enable_probe) change.
fn create_water_bind_group(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    b: &WaterBindings,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MC Render BG"),
        layout: bind_group_layout,
        entries: &[
            entry::buffer(0, &b.buffers.camera),
            entry::buffer(1, &b.buffers.water_params),
            entry::buffer(2, b.vertices),
            entry::view(3, b.env_view),
            entry::sampler(4, b.env_sampler),
            entry::view(5, b.faces.back_depth_view()),
            entry::sampler(6, b.faces.depth_sampler()),
            entry::view(7, b.background.mip_view()),
            entry::buffer(8, &b.buffers.light_params),
            entry::buffer(9, &b.buffers.sh_coefficients),
            entry::view(10, b.ssr.view()),
            entry::buffer(11, &b.buffers.container_geom),
            entry::view(12, b.foam_density_view),
            entry::view(13, b.faces.back_normal_view()),
            entry::view(14, b.background.depth_view()),
            entry::view(15, &b.foam_map.map_view),
            entry::view(16, &b.foam_map.surface_view),
            entry::buffer(17, &b.foam_map.params),
            entry::view(18, &b.foam_map.coords_view),
            entry::buffer(19, &b.buffers.bodies),
            entry::buffer(20, b.probe.buffer()),
            entry::view(21, b.faces.front_depth_view()),
            entry::view(22, b.faces.front_normal_view()),
            entry::sampler(23, b.background.sampler()),
        ],
    })
}

/// What group 1 of the water pipeline binds: the field the mesh was
/// extracted from, and what the shader needs to re-triangulate a cell
pub struct VolumeBindings<'a> {
    pub field: &'a FieldTextures,
    pub grid_params: &'a wgpu::Buffer,
    pub tri_table: &'a wgpu::Buffer,
    pub voxel_normals: &'a wgpu::TextureView,
}

/// Bind groups (density field in texture A / in texture B) that hand the water
/// shader the field the mesh was extracted from, for its world-space in-water
/// test (group 1 of the water pipeline)
fn create_volume_bind_groups(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    volume: &VolumeBindings,
) -> [wgpu::BindGroup; 2] {
    [&volume.field.view_a, &volume.field.view_b].map(|view| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MC Water Volume BG"),
            layout,
            entries: &[
                entry::view(0, view),
                entry::sampler(1, sampler),
                entry::buffer(2, volume.grid_params),
                entry::buffer(3, volume.tri_table),
                entry::view(4, volume.voxel_normals),
            ],
        })
    })
}

/// The water shader's source ("mc_render"), one file per topic under
/// shaders/mc_render/. WGSL has no modules: the parts are concatenated into
/// one module, and since module-scope declarations may be used before they
/// are declared, the order here only decides error line numbers. A new part
/// is a new file plus a line here; scripts/validate-shaders.ps1 reads this
/// list (keep it one `include_str!` per line) and fails on a file in that
/// directory that is missing from it.
const WATER_SHADER_PARTS: &[&str] = &[
    include_str!("../shaders/mc_render/bindings.wgsl"),
    include_str!("../shaders/mc_render/scene_depth.wgsl"),
    include_str!("../shaders/mc_render/bodies.wgsl"),
    include_str!("../shaders/mc_render/debug_records.wgsl"),
    include_str!("../shaders/mc_render/backdrop.wgsl"),
    include_str!("../shaders/mc_render/march.wgsl"),
    include_str!("../shaders/mc_render/walls.wgsl"),
    include_str!("../shaders/mc_render/volume.wgsl"),
    include_str!("../shaders/mc_render/trace.wgsl"),
    include_str!("../shaders/mc_render/exit.wgsl"),
    include_str!("../shaders/mc_render/refract.wgsl"),
    include_str!("../shaders/mc_render/lookup_filter.wgsl"),
    include_str!("../shaders/mc_render/foam.wgsl"),
    include_str!("../shaders/mc_render/ripple.wgsl"),
    include_str!("../shaders/mc_render/main.wgsl"),
];

/// Snippets shared with other shaders that the water shader links ahead of
/// its own parts (validate-shaders.ps1 reads this list too)
const WATER_SHADER_COMMON: &[&str] = &[
    include_str!("../shaders/container_common.wgsl"),
    include_str!("../shaders/water_common.wgsl"),
    include_str!("../shaders/noise_common.wgsl"),
    include_str!("../shaders/sh_common.wgsl"),
    include_str!("../shaders/octahedral_common.wgsl"),
    include_str!("../shaders/body_shapes_common.wgsl"),
    include_str!("../shaders/body_shading_common.wgsl"),
];

/// The water shader: the shared snippets + a pixel-probe snippet + the
/// mc_render parts. Normal rendering links no-op probe stubs: any storage
/// write in the fragment shader can cost the pass its early depth test.
fn water_render_shader(device: &wgpu::Device, probe: bool) -> wgpu::ShaderModule {
    let probe_wgsl = if probe {
        include_str!("../shaders/mc_probe_on.wgsl")
    } else {
        include_str!("../shaders/mc_probe_off.wgsl")
    };
    let mut source = String::new();
    for part in WATER_SHADER_COMMON.iter().chain([&probe_wgsl]).chain(WATER_SHADER_PARTS) {
        source.push_str(part);
        source.push('\n');
    }
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(if probe { "MC Render Shader (probe)" } else { "MC Render Shader" }),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

fn water_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    render_shader: &wgpu::ShaderModule,
    surface_format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("MC Render Pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: render_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: render_shader,
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
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

pub struct WaterPass {
    targets: WaterTargets,
    pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
    // Needed for recreating the bind group on resize / HDR switch
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    // The density field for the shader's world-space in-water test:
    // [field in texture A, field in texture B]
    volume_bind_group_layout: wgpu::BindGroupLayout,
    volume_sampler: wgpu::Sampler,
    volume_bind_groups: [wgpu::BindGroup; 2],
    /// The pixel-probe variant of the pipeline while `--probe` is on
    probe_pipeline: Option<wgpu::RenderPipeline>,
    surface_format: wgpu::TextureFormat,
    sample_count: u32,
}

impl WaterPass {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
        bindings: &WaterBindings,
        volume: &VolumeBindings,
    ) -> Self {
        let targets = WaterTargets::new(device, surface_format, width, height, sample_count);
        let render_shader = water_render_shader(device, false);

        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC Render BGL"),
            entries: &[
                // Camera
                layout::uniform(0, VERTEX_FRAGMENT),
                // Water params
                layout::uniform(1, FRAGMENT),
                // Vertices
                layout::storage(2, VERTEX),
                // Environment texture
                layout::texture_2d(3, FRAGMENT),
                // Environment sampler
                layout::sampler(4, FRAGMENT),
                // Back depth texture (for thickness)
                layout::texture_depth(5, FRAGMENT),
                // Depth sampler
                layout::sampler(6, FRAGMENT),
                // Background texture (for screen-space refraction)
                layout::texture_2d(7, FRAGMENT),
                // Light params
                layout::uniform(8, FRAGMENT),
                // SH irradiance coefficients
                layout::uniform(9, FRAGMENT),
                // SSR texture
                layout::texture_2d_unfilterable(10, FRAGMENT),
                // Container clip params
                layout::uniform(11, FRAGMENT),
                // Foam density field (half-res, splatted by SprayRenderer)
                layout::texture_2d(12, FRAGMENT),
                // Back face normals (refraction exit interface)
                layout::texture_2d_unfilterable(13, FRAGMENT),
                // Background depth (what the refracted ray lands on)
                layout::texture_depth(14, FRAGMENT),
                // Surface foam map + coarse surface grid + params
                layout::texture_2d_unfilterable(15, FRAGMENT),
                layout::texture_2d_unfilterable(16, FRAGMENT),
                layout::uniform(17, FRAGMENT),
                // Foam flow-map coordinates (advected lace pattern)
                layout::texture_2d_unfilterable(18, FRAGMENT),
                // Rigid bodies (exact refraction hits)
                layout::storage(19, FRAGMENT),
                // Front faces: nearest-surface depth + normals (pass 0a), for
                // rays leaving the water through a camera-facing surface
                layout::texture_depth(21, FRAGMENT),
                layout::texture_2d_unfilterable(22, FRAGMENT),
                // Background sampler for filtered refraction lookups
                layout::sampler(23, FRAGMENT),
                // Pixel probe records (only the --probe shader variant uses it)
                layout::storage_rw(20, FRAGMENT),
            ],
        });

        let volume_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC Water Volume BGL"),
            entries: &[
                // R32Float, trilinear (FLOAT32_FILTERABLE is required at device creation)
                layout::texture_3d(0, FRAGMENT),
                layout::sampler(1, FRAGMENT),
                layout::uniform(2, FRAGMENT),
                // Marching-cubes triangle table: the shader rebuilds the
                // mesh's triangles in a cell to read its normal there
                layout::storage(3, FRAGMENT),
                // Voxel normals (octahedral, R32Uint): the normals the mesh
                // was built from
                layout::texture_3d_uint(4, FRAGMENT),
            ],
        });
        let volume_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("MC Water Volume Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let volume_bind_groups = create_volume_bind_groups(device, &volume_bind_group_layout, &volume_sampler, volume);

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC Render Pipeline Layout"),
            bind_group_layouts: &[&render_bind_group_layout, &volume_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = water_render_pipeline(
            device,
            &render_pipeline_layout,
            &render_shader,
            surface_format,
            sample_count,
        );

        let bind_group = create_water_bind_group(device, &render_bind_group_layout, bindings);

        Self {
            targets,
            pipeline: render_pipeline,
            pipeline_layout: render_pipeline_layout,
            bind_group_layout: render_bind_group_layout,
            bind_group,
            volume_bind_group_layout,
            volume_sampler,
            volume_bind_groups,
            probe_pipeline: None,
            surface_format,
            sample_count,
        }
    }

    /// New screen-sized targets. Group 0 binds other passes' screen-sized
    /// views, so the caller rebinds it once those have resized too.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.targets = WaterTargets::new(device, self.surface_format, width, height, self.sample_count);
    }

    /// Group 0 over the current views and buffers; hand it to `set_bind_group`
    pub fn create_bind_group(&self, device: &wgpu::Device, bindings: &WaterBindings) -> wgpu::BindGroup {
        create_water_bind_group(device, &self.bind_group_layout, bindings)
    }

    pub fn set_bind_group(&mut self, bind_group: wgpu::BindGroup) {
        self.bind_group = bind_group;
    }

    /// Rebind group 1 over recreated field textures (grid resolution change)
    pub fn rebuild_volume_bind_groups(&mut self, device: &wgpu::Device, volume: &VolumeBindings) {
        self.volume_bind_groups =
            create_volume_bind_groups(device, &self.volume_bind_group_layout, &self.volume_sampler, volume);
    }

    /// Switch to the pixel-probe shader variant (records land in the buffer
    /// group 0 binds: rebind it after enabling the probe)
    pub fn enable_probe(&mut self, device: &wgpu::Device) {
        let shader = water_render_shader(device, true);
        self.probe_pipeline = Some(water_render_pipeline(
            device,
            &self.pipeline_layout,
            &shader,
            self.surface_format,
            self.sample_count,
        ));
    }

    /// Begin the pass (color + depth cleared). Uses MSAA if enabled: renders
    /// to the MSAA target and resolves into `color_view`.
    pub fn begin_pass<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
    ) -> wgpu::RenderPass<'e> {
        let (render_view, resolve_target) = if let Some((_, msaa_view)) = &self.targets.msaa {
            (msaa_view, Some(color_view))
        } else {
            (color_view, None)
        };

        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("MC Water Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: render_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.targets.depth.1,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        })
    }

    /// Draw the water mesh, reading the field from texture A or B
    pub fn draw(&self, pass: &mut wgpu::RenderPass, indirect_buffer: &wgpu::Buffer, result_in_b: bool) {
        pass.set_pipeline(self.probe_pipeline.as_ref().unwrap_or(&self.pipeline));
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_bind_group(1, &self.volume_bind_groups[result_in_b as usize], &[]);
        pass.draw_indirect(indirect_buffer, 0);
    }
}
