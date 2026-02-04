// SPH 3D Density Computation Shader
// Uses double density relaxation (density + nearDensity) for stability

struct SphParticle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    force: vec3<f32>,
    density: f32,
    near_density: f32,
    // WGSL automatically pads struct to 64 bytes for arrays
}

struct SphParams {
    kernel_radius: f32,
    kernel_radius_sq: f32,
    kernel_radius_pow5: f32,
    kernel_radius_pow6: f32,
    kernel_radius_pow9: f32,
    mass: f32,
    rest_density: f32,
    stiffness: f32,
    near_stiffness: f32,
    viscosity: f32,
    dt: f32,
    num_particles: u32,
}

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SphParticle3D>;

const PI: f32 = 3.14159265359;

// Standard density kernel (Poly6): W(r,h) = 315/(64*pi*h^9) * (h^2 - r^2)^3
fn density_kernel(r_sq: f32) -> f32 {
    let scale = 315.0 / (64.0 * PI * params.kernel_radius_pow9);
    let diff = params.kernel_radius_sq - r_sq;
    return scale * diff * diff * diff;
}

// Near density kernel: W_near(r,h) = 15/(pi*h^6) * (h - r)^3
fn near_density_kernel(r: f32) -> f32 {
    let scale = 15.0 / (PI * params.kernel_radius_pow6);
    let diff = params.kernel_radius - r;
    return scale * diff * diff * diff;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_particles) {
        return;
    }

    let pos_i = particles[i].position;
    var density = 0.0;
    var near_density = 0.0;

    // O(n^2) neighbor search
    for (var j = 0u; j < params.num_particles; j = j + 1u) {
        let pos_j = particles[j].position;
        let r_vec = pos_i - pos_j;
        let r_sq = dot(r_vec, r_vec);

        if (r_sq < params.kernel_radius_sq) {
            let r = sqrt(r_sq);
            density += params.mass * density_kernel(r_sq);
            near_density += params.mass * near_density_kernel(r);
        }
    }

    particles[i].density = density;
    particles[i].near_density = near_density;
}
