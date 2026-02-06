// Spray particle simulation — ballistic integration with gravity and drag
// Dispatched over all spray buffer slots.

struct SprayParticle {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    lifetime: f32,
    vel_x: f32,
    vel_y: f32,
    vel_z: f32,
    max_lifetime: f32,
};

struct SprayParams {
    emission_threshold: f32,
    spray_count: u32,
    lifetime: f32,
    lifetime_variation: f32,
    drag: f32,
    speed_multiplier: f32,
    velocity_jitter: f32,
    dt: f32,
    max_particles: u32,
    num_sph_particles: u32,
    frame_count: u32,
    gravity_y: f32,
};

struct BoundsParams {
    bound_x: f32,
    bound_z: f32,
    floor_y: f32,
    ceiling_y: f32,
    wall_stiffness: f32,
    damping: f32,
    _padding1: f32,
    _padding2: f32,
    rotation_row0: vec4<f32>,
    rotation_row1: vec4<f32>,
    rotation_row2: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> spray_particles: array<SprayParticle>;
@group(0) @binding(1) var<uniform> params: SprayParams;
@group(0) @binding(2) var<uniform> bounds: BoundsParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.max_particles) {
        return;
    }

    var p = spray_particles[idx];

    // Skip dead particles
    if (p.lifetime <= 0.0) {
        return;
    }

    let dt = params.dt;

    // Gravity (only Y component)
    let gravity = vec3<f32>(0.0, params.gravity_y, 0.0);

    // Drag force: -drag * velocity
    let vel = vec3<f32>(p.vel_x, p.vel_y, p.vel_z);
    let drag_force = -params.drag * vel;

    // Update velocity
    let new_vel = vel + dt * (gravity + drag_force);

    // Update position
    p.pos_x = p.pos_x + dt * new_vel.x;
    p.pos_y = p.pos_y + dt * new_vel.y;
    p.pos_z = p.pos_z + dt * new_vel.z;

    p.vel_x = new_vel.x;
    p.vel_y = new_vel.y;
    p.vel_z = new_vel.z;

    // Decrease lifetime
    p.lifetime = p.lifetime - dt;

    // Kill spray particles that exit the container bounds
    // Transform position to container-local space
    let pos = vec3<f32>(p.pos_x, p.pos_y, p.pos_z);
    let rot_row0 = bounds.rotation_row0.xyz;
    let rot_row1 = bounds.rotation_row1.xyz;
    let rot_row2 = bounds.rotation_row2.xyz;
    let local_pos = vec3<f32>(
        dot(rot_row0, pos),
        dot(rot_row1, pos),
        dot(rot_row2, pos)
    );

    if (local_pos.x < -bounds.bound_x || local_pos.x > bounds.bound_x ||
        local_pos.y < bounds.floor_y || local_pos.y > bounds.ceiling_y ||
        local_pos.z < -bounds.bound_z || local_pos.z > bounds.bound_z) {
        p.lifetime = 0.0;
    }

    spray_particles[idx] = p;
}
