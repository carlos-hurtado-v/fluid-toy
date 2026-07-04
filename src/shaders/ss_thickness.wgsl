// Screen-space fluid rendering — Thickness splatting pass
// Additive blend, no depth test: each fragment adds the exact chord length of
// the per-pixel ray through the particle's surface (sphere, or Yu & Turk
// ellipsoid when anisotropy is on — same intersection as ss_depth.wgsl).
// The CPU-computed thickness_scale (1 / splat volume fraction) keeps the
// accumulated total ≈ true water depth in world units; the ellipsoids are
// volume-normalized (axis product = 1) so the same normalization holds.

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
    thickness_scale: f32,
    aniso_enabled: u32,
    aniso_surface_scale: f32,
    _pad0: f32,
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

struct ParticleAniso {
    q0: vec4<f32>, // (Gxx, Gxy, Gxz, center.x)
    q1: vec4<f32>, // (Gyy, Gyz, Gzz, center.y)
    q2: vec4<f32>, // (center.z, reach, amplitude, 0)
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_pos: vec3<f32>,
    @location(1) @interpolate(flat) center_view: vec3<f32>,
    @location(2) @interpolate(flat) g0: vec4<f32>,
    @location(3) @interpolate(flat) g1: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<storage, read> particles: array<SphParticle3D>;
@group(0) @binding(2) var<uniform> ss_params: SsParams;
@group(0) @binding(3) var<storage, read> aniso: array<ParticleAniso>;

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

    let inv_r = 1.0 / max(ss_params.particle_radius, 1e-6);
    var center_world = particle.position;
    var g0 = vec4<f32>(inv_r, 0.0, 0.0, 1.0);
    var g1 = vec4<f32>(inv_r, 0.0, inv_r, 0.0);
    var extent = ss_params.particle_radius;

    if (ss_params.aniso_enabled != 0u) {
        let an = aniso[instance_index];
        center_world = vec3<f32>(an.q0.w, an.q1.w, an.q2.x);
        g0 = vec4<f32>(an.q0.x, an.q0.y, an.q0.z, ss_params.aniso_surface_scale);
        g1 = vec4<f32>(an.q1.x, an.q1.y, an.q1.z, 0.0);
        extent = an.q2.y * ss_params.aniso_surface_scale;
    }

    let view_center = camera.view * vec4<f32>(center_world, 1.0);
    let view_pos = vec3<f32>(
        view_center.x + local_pos.x * extent,
        view_center.y + local_pos.y * extent,
        view_center.z,
    );

    var output: VertexOutput;
    output.clip_position = camera.projection * vec4<f32>(view_pos, 1.0);
    output.view_pos = view_pos;
    output.center_view = view_center.xyz;
    output.g0 = g0;
    output.g1 = g1;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let d = normalize(input.view_pos);

    let G = mat3x3<f32>(
        vec3<f32>(input.g0.x, input.g0.y, input.g0.z),
        vec3<f32>(input.g0.y, input.g1.x, input.g1.y),
        vec3<f32>(input.g0.z, input.g1.y, input.g1.z),
    );
    let rt = transpose(mat3x3<f32>(
        camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz,
    ));

    let md = G * (rt * d);
    let me = G * (rt * input.center_view);
    let k = input.g0.w;
    let a = dot(md, md);
    let b = dot(md, me);
    let c = dot(me, me) - k * k;
    let disc = b * b - a * c;
    if (disc <= 0.0 || a <= 0.0) {
        discard;
    }

    // Chord length between the two roots of a t^2 - 2 b t + c = 0 (d is
    // normalized, so root spacing is world distance)
    let chord = 2.0 * sqrt(disc) / a;
    return vec4<f32>(chord * ss_params.thickness_scale, 0.0, 0.0, 0.0);
}
