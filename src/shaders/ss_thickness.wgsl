// Screen-Space Fluid - Thickness Pass
// Accumulates particle contributions for subsurface scattering / depth coloring
// Uses additive blending to sum thickness

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
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> params: FluidParams;

// Visual radius multiplier - must match ss_depth.wgsl
const RADIUS_SCALE: f32 = 3.0;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) particle_pos: vec3<f32>,
    @location(1) particle_vel: vec3<f32>,
) -> VertexOutput {
    // Generate quad vertices
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
    let view_center = camera.view * vec4<f32>(particle_pos, 1.0);

    // Create billboard quad in view space (use scaled radius for visual overlap)
    let visual_radius = params.particle_radius * RADIUS_SCALE;
    let view_pos = view_center.xyz + vec3<f32>(local_pos * visual_radius, 0.0);

    // Project to clip space
    let clip_pos = camera.projection * vec4<f32>(view_pos, 1.0);

    var output: VertexOutput;
    output.position = clip_pos;
    output.uv = local_pos;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Soft circular falloff
    let dist_sq = dot(input.uv, input.uv);
    if (dist_sq > 1.0) {
        discard;
    }

    // Thickness contribution based on sphere profile
    // Integrating through a sphere gives sqrt(1 - r^2) profile
    let thickness = sqrt(1.0 - dist_sq);

    return vec4<f32>(thickness, 0.0, 0.0, 1.0);
}
