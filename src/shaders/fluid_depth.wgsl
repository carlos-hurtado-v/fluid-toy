// Fluid Depth Pass - Render particles as spheres, output eye-space depth

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
    near: f32,
    far: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) view_center: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) sphere_radius: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> params: FluidParams;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) particle_pos: vec3<f32>,
    @location(1) particle_vel: vec3<f32>,
) -> VertexOutput {
    // Generate quad vertices (2 triangles, 6 vertices)
    var quad_verts = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );

    let local_pos = quad_verts[vertex_index];

    // Transform particle center to view space
    let view_center = (camera.view * vec4<f32>(particle_pos, 1.0)).xyz;

    // Create billboard quad in view space
    let view_pos = view_center + vec3<f32>(local_pos * params.particle_radius, 0.0);

    // Project to clip space
    let clip_pos = camera.projection * vec4<f32>(view_pos, 1.0);

    var output: VertexOutput;
    output.position = clip_pos;
    output.view_center = view_center;
    output.uv = local_pos;
    output.sphere_radius = params.particle_radius;
    return output;
}

struct FragmentOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) view_depth: f32,
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    // Calculate sphere intersection
    let uv = input.uv;
    let dist_sq = dot(uv, uv);

    // Discard if outside sphere
    if (dist_sq > 1.0) {
        discard;
    }

    // Calculate z offset on sphere surface
    let z_offset = sqrt(1.0 - dist_sq) * input.sphere_radius;

    // Eye-space depth (negative because camera looks down -Z)
    let eye_depth = input.view_center.z + z_offset;

    // Convert to clip depth for depth buffer
    let clip_depth = (params.far + params.near + 2.0 * params.far * params.near / eye_depth) / (params.far - params.near);
    let ndc_depth = clip_depth * 0.5 + 0.5;

    var output: FragmentOutput;
    output.depth = ndc_depth;
    output.view_depth = -eye_depth; // Store positive depth for processing
    return output;
}
