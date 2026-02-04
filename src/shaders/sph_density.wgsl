// SPH Density Computation Shader
// Computes density for each particle using Poly6 kernel

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

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SphParticle>;

const PI: f32 = 3.14159265359;

// 2D Poly6 kernel: W(r, h) = (4 / (pi * h^8)) * (h^2 - r^2)^3
fn poly6_2d(r_sq: f32) -> f32 {
    if (r_sq >= params.kernel_radius_sq) {
        return 0.0;
    }
    let scale = 4.0 / (PI * params.kernel_radius_4 * params.kernel_radius_4);
    let diff = params.kernel_radius_sq - r_sq;
    return scale * diff * diff * diff;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_particles) {
        return;
    }

    let pos_i = particles[i].pos;
    var density = 0.0;

    // O(n^2) brute force neighbor search (includes self-contribution)
    for (var j = 0u; j < params.num_particles; j = j + 1u) {
        let pos_j = particles[j].pos;
        let r = pos_i - pos_j;
        let r_sq = dot(r, r);

        if (r_sq < params.kernel_radius_sq) {
            density += params.mass * poly6_2d(r_sq);
        }
    }

    particles[i].density = density;
}
