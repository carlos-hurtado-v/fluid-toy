// Particle simulation compute shader

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
}

struct SimParams {
    delta_time: f32,
    gravity: f32,
    bound_x: f32,
    bound_y: f32,
    damping: f32,
    num_particles: u32,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> particles_src: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particles_dst: array<Particle>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_particles) {
        return;
    }

    var pos = particles_src[index].pos;
    var vel = particles_src[index].vel;

    // Apply gravity
    vel.y += params.gravity * params.delta_time;

    // Update position
    pos += vel * params.delta_time;

    // Boundary collision
    if (pos.x < -params.bound_x) {
        pos.x = -params.bound_x;
        vel.x = -vel.x * params.damping;
    }
    if (pos.x > params.bound_x) {
        pos.x = params.bound_x;
        vel.x = -vel.x * params.damping;
    }
    if (pos.y < -params.bound_y) {
        pos.y = -params.bound_y;
        vel.y = -vel.y * params.damping;
    }
    if (pos.y > params.bound_y) {
        pos.y = params.bound_y;
        vel.y = -vel.y * params.damping;
    }

    // Write result
    particles_dst[index] = Particle(pos, vel);
}
