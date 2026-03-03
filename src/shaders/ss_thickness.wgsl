// Screen-space fluid rendering — Thickness splatting pass
// Renders particles as billboard quads with hemispherical thickness profile.
// Uses additive blending, no depth test (accumulates all particles).

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

struct SsParams {
    particle_radius: f32,
    num_particles: u32,
    screen_width: f32,
    screen_height: f32,
}

struct SphParticle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    force: vec3<f32>,
    density: f32,
    near_density: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<storage, read> particles: array<SphParticle3D>;
@group(0) @binding(2) var<uniform> ss_params: SsParams;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
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
    let particle = particles[instance_index];

    let view_center = camera.view * vec4<f32>(particle.position, 1.0);
    let radius = ss_params.particle_radius * 1.05;
    let view_pos = vec3<f32>(
        view_center.x + local_pos.x * radius,
        view_center.y + local_pos.y * radius,
        view_center.z,
    );
    let clip_pos = camera.projection * vec4<f32>(view_pos, 1.0);

    var output: VertexOutput;
    output.clip_position = clip_pos;
    output.uv = local_pos;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let r_sq = dot(input.uv, input.uv);
    if (r_sq > 1.0) {
        discard;
    }

    // Hemispherical thickness profile: full sphere diameter at center, zero at edge
    // thickness = 2 * R * sqrt(1 - r²), normalized by factor for plausible accumulation
    let thickness = 2.0 * ss_params.particle_radius * sqrt(1.0 - r_sq) / 8.0;
    return vec4<f32>(thickness, 0.0, 0.0, 0.0);
}
