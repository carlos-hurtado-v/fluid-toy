// SPH Force Computation Shader
// Computes pressure and viscosity forces using Spiky and Viscosity kernels

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

// 2D Spiky kernel gradient: grad W = -(30 / (pi * h^5)) * (h - r)^2 * (r_vec / r)
fn spiky_gradient_2d(r_vec: vec2<f32>, r: f32) -> vec2<f32> {
    if (r < 0.0001 || r >= params.kernel_radius) {
        return vec2<f32>(0.0, 0.0);
    }
    let scale = -30.0 / (PI * params.kernel_radius_5);
    let diff = params.kernel_radius - r;
    return scale * diff * diff * (r_vec / r);
}

// 2D Viscosity kernel Laplacian: laplacian W = (40 / (pi * h^5)) * (h - r)
fn viscosity_laplacian_2d(r: f32) -> f32 {
    if (r >= params.kernel_radius) {
        return 0.0;
    }
    let scale = 40.0 / (PI * params.kernel_radius_5);
    return scale * (params.kernel_radius - r);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_particles) {
        return;
    }

    let pos_i = particles[i].pos;
    let vel_i = particles[i].vel;
    let density_i = particles[i].density;

    // Skip if density is too low (shouldn't happen)
    if (density_i < 0.0001) {
        particles[i].force = vec2<f32>(0.0, params.gravity * params.mass);
        particles[i].pressure = 0.0;
        return;
    }

    // Compute pressure from density (clamped to non-negative for stability)
    let pressure_i = max(0.0, params.stiffness * (density_i - params.rest_density));
    particles[i].pressure = pressure_i;

    var f_pressure = vec2<f32>(0.0, 0.0);
    var f_viscosity = vec2<f32>(0.0, 0.0);

    // O(n^2) brute force neighbor search
    for (var j = 0u; j < params.num_particles; j = j + 1u) {
        if (i == j) {
            continue;
        }

        let pos_j = particles[j].pos;
        let vel_j = particles[j].vel;
        let density_j = particles[j].density;

        if (density_j < 0.0001) {
            continue;
        }

        let r_vec = pos_i - pos_j;
        let r_sq = dot(r_vec, r_vec);

        if (r_sq < params.kernel_radius_sq && r_sq > 0.000001) {
            let r = sqrt(r_sq);
            let pressure_j = max(0.0, params.stiffness * (density_j - params.rest_density));

            // Pressure force: -m * (p_i + p_j) / (2 * rho_j) * grad W
            let pressure_term = -params.mass * (pressure_i + pressure_j) / (2.0 * density_j);
            f_pressure += pressure_term * spiky_gradient_2d(r_vec, r);

            // Viscosity force: mu * m * (v_j - v_i) / rho_j * laplacian W
            let visc_term = params.viscosity * params.mass / density_j;
            f_viscosity += visc_term * (vel_j - vel_i) * viscosity_laplacian_2d(r);
        }
    }

    // Store pressure and viscosity forces (gravity applied in integration as acceleration)
    particles[i].force = f_pressure + f_viscosity;
}
