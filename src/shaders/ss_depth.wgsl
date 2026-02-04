// Screen-Space Fluid - Depth Pass
// Renders particles as sphere imposters, outputs view-space depth

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct FluidParams {
    particle_radius: f32,
    screen_width: f32,
    screen_height: f32,
    _padding: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) view_center: vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> params: FluidParams;

// Visual radius multiplier - makes spheres overlap for smooth fluid surface
// Higher = more overlap before blur, helps merge spheres
const RADIUS_SCALE: f32 = 3.5;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) particle_pos: vec3<f32>,
    @location(1) particle_vel: vec3<f32>,
) -> VertexOutput {
    // Quad corners (billboard facing camera) - range -1 to 1 like render_3d.wgsl
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );

    let visual_radius = params.particle_radius * RADIUS_SCALE;
    let corner = vec3<f32>(corners[vertex_index] * visual_radius, 0.0);
    let uv = corners[vertex_index] * 0.5 + 0.5;

    // Transform particle to view space
    let view_center = (camera.view * vec4<f32>(particle_pos, 1.0)).xyz;

    // Billboard in view space (add corner offset)
    let view_pos = view_center + corner;

    // Project to clip space
    let clip_pos = camera.projection * vec4<f32>(view_pos, 1.0);

    var output: VertexOutput;
    output.position = clip_pos;
    output.uv = uv;
    output.view_center = view_center;
    return output;
}

struct FragmentOutput {
    @location(0) depth: f32,  // R32Float - high precision depth
    @builtin(frag_depth) frag_depth: f32,
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    // UV in [-1, 1] range
    let normalxy = input.uv * 2.0 - 1.0;
    let r2 = dot(normalxy, normalxy);

    if (r2 > 1.0) {
        discard;
    }

    // Sphere normal (in view space, pointing toward camera)
    let normalz = sqrt(1.0 - r2);
    let normal = vec3<f32>(normalxy, normalz);

    // Actual view-space position on sphere surface
    // Use scaled radius for visual overlap
    let radius = params.particle_radius * RADIUS_SCALE;
    let real_view_pos = vec4<f32>(input.view_center + normal * radius, 1.0);

    // Project to get proper depth
    let clip_pos = camera.projection * real_view_pos;

    var output: FragmentOutput;
    // Output view-space Z (negative in view space, so negate for positive depth)
    output.depth = -real_view_pos.z;
    // Hardware depth for depth testing
    output.frag_depth = clip_pos.z / clip_pos.w;
    return output;
}
