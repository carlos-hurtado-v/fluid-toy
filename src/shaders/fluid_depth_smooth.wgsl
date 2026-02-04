// Fluid Depth Pass - Smooth metaball-style depth using additive blending
// Instead of min-depth (bumpy spheres), we compute weighted average depth

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
    var quad_verts = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );

    let local_pos = quad_verts[vertex_index];
    let view_center = (camera.view * vec4<f32>(particle_pos, 1.0)).xyz;
    let view_pos = view_center + vec3<f32>(local_pos * params.particle_radius, 0.0);
    let clip_pos = camera.projection * vec4<f32>(view_pos, 1.0);

    var output: VertexOutput;
    output.position = clip_pos;
    output.view_center = view_center;
    output.uv = local_pos;
    output.sphere_radius = params.particle_radius;
    return output;
}

// Output: RG = (weighted_depth, density_weight)
// Final depth will be computed as weighted_depth / density_weight
struct FragmentOutput {
    @location(0) depth_weight: vec2<f32>,  // (depth * weight, weight)
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    let uv = input.uv;
    let dist_sq = dot(uv, uv);

    // Discard outside the quad's circle
    if (dist_sq > 1.0) {
        discard;
    }

    // Metaball-style falloff: smooth blend at edges
    // This creates overlapping contributions that blend together
    let falloff = 1.0 - dist_sq;  // 1 at center, 0 at edge
    let weight = falloff * falloff;  // Smooth quadratic falloff

    // Calculate sphere depth (z offset on sphere surface)
    let z_offset = sqrt(1.0 - dist_sq) * input.sphere_radius;
    let eye_depth = -input.view_center.z - z_offset;  // Positive depth value

    var output: FragmentOutput;
    output.depth_weight = vec2<f32>(eye_depth * weight, weight);
    return output;
}
