// Marching Cubes - Mesh Rendering
// Renders the generated triangle mesh with water shading

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    near: f32,
    far: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct WaterParams {
    water_color: vec3<f32>,
    roughness: f32,
    ior: f32,
    refraction_strength: f32,
    env_intensity: f32,
    use_env_background: u32,
    background_r: f32,
    background_g: f32,
    background_b: f32,
    time: f32,
    deep_color_r: f32,
    deep_color_g: f32,
    deep_color_b: f32,
    ripple_strength: f32,
    clarity: f32,
    // 1 = physical (Snell, two-interface) refraction, 0 = legacy UV offset
    physical_refraction: f32,
    foam_coverage: f32,
    aeration_strength: f32,
    // 1 = physical water medium (absorption + single scattering), 0 = legacy
    physical_medium: f32,
    // Enabled rigid bodies at the front of `rigid_bodies`
    body_count: u32,
    // rendering.mc_debug_view (McDebugView::as_u32, 0 = off)
    debug_view: u32,
    // rendering.mc_silhouette_exit (SILHOUETTE_* below)
    silhouette_exit: u32,
    // rendering.mc_front_face_exit (1 = the in-water trace tests front faces)
    front_exit: u32,
    // Ground-projected backdrop (mc_environment.wgsl EnvParams): the plane's
    // world height and the height above it the map was shot from
    ground_enabled: u32,
    ground_y: f32,
    ground_capture_height: f32,
    // rendering.mc_filtered_lookup (1 = the final lookup is filtered by its footprint)
    filtered_lookup: u32,
    // rendering.mc_volume_trace (1 = in-water tests read the density field)
    volume_trace: u32,
    // 1 = the depth buffer holds opaque surfaces the tracer does not know
    // exactly (pool walls and floor, a Custom body): rays are tested against
    // it. 0 = glass tank with procedural bodies only: nothing to test.
    depth_occluders: u32,
    _pad_g2: u32,
}

struct LightParams {
    sun_direction: vec3<f32>,
    sun_enabled: u32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    _pad2: f32,
    _padding: vec3<f32>,
}

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> water: WaterParams;
@group(0) @binding(2) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(3) var env_tex: texture_2d<f32>;
@group(0) @binding(4) var env_sampler: sampler;
@group(0) @binding(5) var back_depth_tex: texture_depth_2d;
@group(0) @binding(6) var depth_sampler: sampler;
@group(0) @binding(7) var background_tex: texture_2d<f32>;
@group(0) @binding(8) var<uniform> light: LightParams;
@group(0) @binding(9) var<uniform> sh_coeffs: array<vec4<f32>, 9>;
@group(0) @binding(10) var ssr_tex: texture_2d<f32>;
@group(0) @binding(12) var foam_density_tex: texture_2d<f32>;
// Exit interface for refraction: outward normal of the nearest back face (w = 1
// where one exists), written by the back-face pass alongside back_depth_tex
@group(0) @binding(13) var back_normal_tex: texture_2d<f32>;
// Depth of everything behind the water (backdrop, container, bodies)
@group(0) @binding(14) var background_depth_tex: texture_depth_2d;

// Surface foam map (foam_map.wgsl): advected 2D foam layer in container-local
// XZ, plus the coarse surface grid whose .a is each column's top fluid height
struct FoamMapParams {
    origin_x: f32,
    origin_z: f32,
    fine_cell: f32,
    coarse_cell: f32,
    fine_dim: u32,
    coarse_dim: u32,
    num_particles: u32,
    max_spray: u32,
    dt: f32,
    decay: f32,
    surface_band: f32,
    deposit_amount: f32,
    deposit_sigma: f32,
    grace_age: f32,
    blur_sigma: f32,
    flags: u32,
    flow_phase: f32,
    burst: f32,
    _pad0: f32,
    _pad1: f32,
}
@group(0) @binding(15) var foam_map_tex: texture_2d<f32>;
@group(0) @binding(16) var foam_surface_tex: texture_2d<f32>;
@group(0) @binding(17) var<uniform> foam_map: FoamMapParams;
// Flow-map coordinates: (phase A xy, phase B xy) = where the foam at this
// map texel was when that phase restarted (the lace pattern rides the flow)
@group(0) @binding(18) var foam_coords_tex: texture_2d<f32>;

@group(0) @binding(11) var<uniform> container: ContainerGeometry;

// Mirrors GpuRigidBodyRender (rigid_body.wgsl RigidBodyParams, 112 bytes)
struct RigidBodyParams {
    position: vec3<f32>,
    half_extent: f32,
    color: vec4<f32>,
    light_dir: vec3<f32>,
    shape: u32,
    rot_row0: vec4<f32>,
    rot_row1: vec4<f32>,
    rot_row2: vec4<f32>,
    prop_blades: u32,
    prop_pitch: f32,
    // 1 = the water's field is continued into this body (mc_wet_bodies,
    // field_bodies_common.wgsl): the mesh meets it, no dry film around it
    wet: f32,
    _pad1: f32,
}
@group(0) @binding(19) var<storage, read> rigid_bodies: array<RigidBodyParams>;
// Nearest surface at each pixel (water of either winding, bodies, pool walls)
// and the water front faces' world normals (raw MC winding, w = 1 on water),
// both from the front-face pass ahead of this one
@group(0) @binding(21) var front_depth_tex: texture_depth_2d;
@group(0) @binding(22) var front_normal_tex: texture_2d<f32>;
// Clamped, trilinear + anisotropic: filtered reads of background_tex's mip
// chain (resolve_lookup). Its level-0 reads elsewhere keep env_sampler.
@group(0) @binding(23) var background_sampler: sampler;

// The density field this frame's mesh was extracted from (mc_generate.wgsl),
// for the world-space in-water test. Voxel i sits at grid_min + i * cell_size
// in the mesh's frame (the convention mc_generate places vertices by).
struct McGridParams {
    grid_min: vec3<f32>,
    grid_size: u32,
    grid_max: vec3<f32>,
    cell_size: f32,
    kernel_radius: f32,
    iso_value: f32,
    num_particles: u32,
    max_vertices: u32,
}
@group(1) @binding(0) var density_tex: texture_3d<f32>;
@group(1) @binding(1) var density_sampler: sampler;
@group(1) @binding(2) var<uniform> mc_grid: McGridParams;
// Marching-cubes triangle table (256 cases x 16 edge indices, -1 terminated)
@group(1) @binding(3) var<storage, read> mc_tri_table: array<i32>;
// Normal at every voxel, octahedral (mc_voxel_normals.wgsl): what mc_generate
// built the mesh's vertex normals from
@group(1) @binding(4) var normal_tex: texture_3d<u32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @builtin(front_facing) front_facing: bool,
}

const PI: f32 = 3.14159265359;
// Ceiling on written radiance (far above display white, well inside f16)
const HDR_OUTPUT_MAX: f32 = 4096.0;
