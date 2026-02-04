// SPH Integration Shader
// Updates velocity and position, handles boundary conditions

struct SphParticle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    density: f32,
    pressure: f32,
    force: vec2<f32>,
}

struct SphParams {
    kernel_radius: f32,
    kernel_radius_sq: f32,
    kernel_radius_4: f32,
    kernel_radius_5: f32,
    mass: f32,
    rest_density: f32,
    stiffness: f32,
    viscosity: f32,
    dt: f32,
    gravity: f32,
    num_particles: u32,
    _padding: u32,
}

struct BoundsParams {
    bound_x: f32,
    bound_y: f32,
    damping: f32,
    wall_stiffness: f32,
}

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SphParticle>;
@group(0) @binding(2) var<uniform> bounds: BoundsParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_particles) {
        return;
    }

    let density = particles[i].density;
    var pos = particles[i].pos;
    var vel = particles[i].vel;
    let force = particles[i].force;

    // Compute acceleration from SPH forces (pressure + viscosity)
    var accel: vec2<f32>;
    if (density > 0.0001) {
        // SPH forces are per unit volume, divide by density for acceleration
        accel = force / density;
    } else {
        accel = vec2<f32>(0.0, 0.0);
    }

    // Add gravity directly as acceleration (not divided by density!)
    accel.y += params.gravity;

    // Soft boundary forces (push particles away from walls)
    let wall_margin = params.kernel_radius * 0.5;

    if (pos.x < -bounds.bound_x + wall_margin) {
        accel.x += bounds.wall_stiffness * (-bounds.bound_x + wall_margin - pos.x);
    }
    if (pos.x > bounds.bound_x - wall_margin) {
        accel.x -= bounds.wall_stiffness * (pos.x - bounds.bound_x + wall_margin);
    }
    if (pos.y < -bounds.bound_y + wall_margin) {
        accel.y += bounds.wall_stiffness * (-bounds.bound_y + wall_margin - pos.y);
    }
    if (pos.y > bounds.bound_y - wall_margin) {
        accel.y -= bounds.wall_stiffness * (pos.y - bounds.bound_y + wall_margin);
    }

    // Semi-implicit Euler integration
    vel = vel + params.dt * accel;
    pos = pos + params.dt * vel;

    // Hard boundary clamping with damping (bounce)
    if (pos.x < -bounds.bound_x) {
        pos.x = -bounds.bound_x;
        vel.x = -vel.x * bounds.damping;
    }
    if (pos.x > bounds.bound_x) {
        pos.x = bounds.bound_x;
        vel.x = -vel.x * bounds.damping;
    }
    if (pos.y < -bounds.bound_y) {
        pos.y = -bounds.bound_y;
        vel.y = -vel.y * bounds.damping;
    }
    if (pos.y > bounds.bound_y) {
        pos.y = bounds.bound_y;
        vel.y = -vel.y * bounds.damping;
    }

    particles[i].vel = vel;
    particles[i].pos = pos;
}
