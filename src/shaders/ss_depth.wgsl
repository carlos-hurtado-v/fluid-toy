// Screen-space fluid rendering — Depth splatting pass
// Renders particles as billboard quads with sphere depth replacement.
// R32Float output: dome-shaped linear depth (front surface of sphere).
// Hardware depth: same dome depth in clip space (for correct occlusion).

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
    @location(1) view_center_z: f32,
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

    // Billboard in view space — no enlargement beyond radius
    let radius = ss_params.particle_radius;
    let view_pos = vec3<f32>(
        view_center.x + local_pos.x * radius,
        view_center.y + local_pos.y * radius,
        view_center.z,
    );

    let clip_pos = camera.projection * vec4<f32>(view_pos, 1.0);

    var output: VertexOutput;
    output.clip_position = clip_pos;
    output.uv = local_pos;
    output.view_center_z = view_center.z;
    return output;
}

struct FragOutput {
    @location(0) eye_depth: f32,
    @builtin(frag_depth) hw_depth: f32,
}

@fragment
fn fs_main(input: VertexOutput) -> FragOutput {
    let r_sq = dot(input.uv, input.uv);
    if (r_sq > 1.0) {
        discard;
    }

    // Sphere depth offset for hardware occlusion (dome shape)
    let dz = ss_params.particle_radius * sqrt(1.0 - r_sq);
    let view_z_hw = input.view_center_z + dz;
    let clip_z = (camera.projection[2][2] * view_z_hw + camera.projection[3][2]) / (-view_z_hw);

    // R32Float output: dome-shaped depth (sphere raycast, matching Splash).
    // The dome makes center pixels closer and edge pixels farther, creating
    // smooth depth gradients between overlapping particles. The narrow-range
    // filter then smooths these into a continuous surface.
    let linear_depth = -(input.view_center_z + dz);

    var output: FragOutput;
    output.eye_depth = linear_depth;
    output.hw_depth = clamp(clip_z, 0.0, 1.0);
    return output;
}
