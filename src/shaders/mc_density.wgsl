// Marching Cubes - Density Grid Generation
// Samples particle density onto a 3D grid for surface extraction

// Must match Rust SphParticle3D layout (64 bytes)
// Same struct as grid_build.wgsl - WGSL auto-pads vec3 to 16-byte alignment
struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    force: vec3<f32>,
    density: f32,
    near_density: f32,
}

struct GridParams {
    grid_min: vec3<f32>,
    cell_size: f32,
    grid_dims: vec3<u32>,
    num_particles: u32,
    smoothing_radius: f32,
    surface_threshold: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: GridParams;
@group(0) @binding(2) var<storage, read_write> density_grid: array<f32>;

// Poly6 kernel for density estimation
fn poly6_kernel(r_sq: f32, h: f32) -> f32 {
    let h_sq = h * h;
    if (r_sq >= h_sq) {
        return 0.0;
    }
    let diff = h_sq - r_sq;
    let h_9 = h_sq * h_sq * h_sq * h_sq * h;
    return 315.0 / (64.0 * 3.14159265 * h_9) * diff * diff * diff;
}

fn grid_index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.grid_dims.x + z * params.grid_dims.x * params.grid_dims.y;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Check bounds
    if (global_id.x >= params.grid_dims.x ||
        global_id.y >= params.grid_dims.y ||
        global_id.z >= params.grid_dims.z) {
        return;
    }

    // World position of this grid vertex
    let grid_pos = params.grid_min + vec3<f32>(global_id) * params.cell_size;

    let h = params.smoothing_radius;
    var density = 0.0;

    // Sum contributions from all particles
    // In production, you'd use spatial hashing here for performance
    for (var i = 0u; i < params.num_particles; i++) {
        let particle_pos = particles[i].position;
        let diff = grid_pos - particle_pos;
        let r_sq = dot(diff, diff);

        density += poly6_kernel(r_sq, h);
    }

    // Store density at this grid point
    let idx = grid_index_3d(global_id.x, global_id.y, global_id.z);
    density_grid[idx] = density;
}
