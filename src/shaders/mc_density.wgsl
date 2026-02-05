// Marching Cubes - Density Field Generation
// Samples particle contributions onto a 3D grid

struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    density: f32,
    pressure: f32,
    force: vec3<f32>,
    _padding: f32,
}

struct GridParams {
    grid_min: vec3<f32>,
    grid_size: u32,          // Number of cells per dimension (e.g., 64)
    grid_max: vec3<f32>,
    cell_size: f32,          // Size of each cell in world units
    kernel_radius: f32,
    iso_value: f32,
    num_particles: u32,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: GridParams;
@group(0) @binding(2) var density_field: texture_storage_3d<r32float, write>;

// Poly6 kernel for density estimation
fn poly6_kernel(r_sq: f32, h: f32) -> f32 {
    let h_sq = h * h;
    if (r_sq >= h_sq) {
        return 0.0;
    }
    let diff = h_sq - r_sq;
    // 3D Poly6: 315 / (64 * pi * h^9)
    let coeff = 315.0 / (64.0 * 3.14159265359 * pow(h, 9.0));
    return coeff * diff * diff * diff;
}

// Convert grid coordinates to world position (cell center)
fn grid_to_world(grid_pos: vec3<u32>) -> vec3<f32> {
    let cell_size = params.cell_size;
    return params.grid_min + (vec3<f32>(grid_pos) + 0.5) * cell_size;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_size = params.grid_size;

    // Skip if outside grid bounds
    if (global_id.x >= grid_size || global_id.y >= grid_size || global_id.z >= grid_size) {
        return;
    }

    let world_pos = grid_to_world(global_id);
    let h = params.kernel_radius;
    let h_sq = h * h;

    // Sum density contributions from all particles
    var density = 0.0;

    for (var i = 0u; i < params.num_particles; i++) {
        let particle_pos = particles[i].position;
        let diff = world_pos - particle_pos;
        let r_sq = dot(diff, diff);

        if (r_sq < h_sq) {
            // Use kernel weight (mass is implicitly 1.0)
            density += poly6_kernel(r_sq, h);
        }
    }

    // Store density value
    textureStore(density_field, vec3<i32>(global_id), vec4<f32>(density, 0.0, 0.0, 0.0));
}
