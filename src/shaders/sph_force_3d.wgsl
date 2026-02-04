// SPH 3D Force Computation Shader
// Uses double density relaxation with pressure + near pressure

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

// Spiky kernel gradient magnitude: 45/(pi*h^6) * (h-r)^2
fn density_kernel_gradient(r: f32) -> f32 {
    let scale = 45.0 / (PI * params.kernel_radius_pow6);
    let diff = params.kernel_radius - r;
    return scale * diff * diff;
}

// Near density kernel gradient: 45/(pi*h^5) * (h-r)^2
fn near_density_kernel_gradient(r: f32) -> f32 {
    let scale = 45.0 / (PI * params.kernel_radius_pow5);
    let diff = params.kernel_radius - r;
    return scale * diff * diff;
}

// Viscosity kernel laplacian: 45/(pi*h^6) * (h-r)
fn viscosity_kernel_laplacian(r: f32) -> f32 {
    let scale = 45.0 / (PI * params.kernel_radius_pow6);
    return scale * (params.kernel_radius - r);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_particles) {
        return;
    }

    let pos_i = particles[i].position;
    let vel_i = particles[i].velocity;

    // Use safety minimum for density to prevent numerical issues
    let density_i = max(particles[i].density, 1.0);
    let near_density_i = max(particles[i].near_density, 1.0);

    // Compute pressures
    // Regular pressure can be negative (attractive when sparse)
    let pressure_i = params.stiffness * (density_i - params.rest_density);
    // Near pressure is always positive (repulsive at close range)
    let near_pressure_i = params.near_stiffness * near_density_i;

    var f_pressure = vec3<f32>(0.0, 0.0, 0.0);
    var f_viscosity = vec3<f32>(0.0, 0.0, 0.0);

    for (var j = 0u; j < params.num_particles; j = j + 1u) {
        let pos_j = particles[j].position;
        let r_vec = pos_i - pos_j;
        let r_sq = dot(r_vec, r_vec);

        // Only process neighbors within kernel radius (and skip self)
        if (r_sq < params.kernel_radius_sq && r_sq > 1e-12) {
            let r = sqrt(r_sq);
            let dir = normalize(pos_j - pos_i);  // Direction from i to j

            // Get neighbor density with safety minimum to prevent division issues
            let density_j = max(particles[j].density, 1.0);
            let near_density_j = max(particles[j].near_density, 1.0);

            // Pressure from j
            let pressure_j = params.stiffness * (density_j - params.rest_density);
            let near_pressure_j = params.near_stiffness * near_density_j;

            // Symmetric pressure (average of both particles)
            let shared_pressure = (pressure_i + pressure_j) / 2.0;
            let near_shared_pressure = (near_pressure_i + near_pressure_j) / 2.0;

            // Pressure force
            f_pressure += -params.mass * shared_pressure * dir * density_kernel_gradient(r) / density_j;
            f_pressure += -params.mass * near_shared_pressure * dir * near_density_kernel_gradient(r) / near_density_j;

            // Viscosity force
            let vel_j = particles[j].velocity;
            let relative_vel = vel_j - vel_i;
            f_viscosity += params.mass * relative_vel * viscosity_kernel_laplacian(r) / density_j;
        }
    }

    // Apply viscosity coefficient
    f_viscosity *= params.viscosity;

    // Gravity force: density * g so that accel = force/density = g
    let f_gravity = density_i * vec3<f32>(0.0, -9.8, 0.0);

    // Total force
    particles[i].force = f_pressure + f_viscosity + f_gravity;
}
