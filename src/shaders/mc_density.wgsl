// Marching Cubes - Density Field Generation
// Samples particle contributions onto a 3D grid
// Applies boundary gamma correction so the iso-surface extends to container walls

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
    max_vertices: u32,
}

struct ContainerClipParams {
    half_width: f32,
    half_depth: f32,
    half_height: f32,
    center_y: f32,
    sin_x: f32,
    cos_x: f32,
    sin_z: f32,
    cos_z: f32,
    clip_enabled: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: GridParams;
@group(0) @binding(2) var density_field: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<uniform> clip: ContainerClipParams;

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

// Transform world position to container-local space (inverse tilt)
fn world_to_local(world_pos: vec3<f32>) -> vec3<f32> {
    var p = world_pos;
    p.y -= clip.center_y;

    // Inverse Z rotation
    let x1 = p.x * clip.cos_z + p.y * clip.sin_z;
    let y1 = -p.x * clip.sin_z + p.y * clip.cos_z;

    // Inverse X rotation
    let y2 = y1 * clip.cos_x + p.z * clip.sin_x;
    let z2 = -y1 * clip.sin_x + p.z * clip.cos_x;

    return vec3<f32>(x1, y2, z2);
}

// Estimate fraction of kernel support volume inside the container.
// Same approach as the simulation density shader (boundary_gamma).
// Near a flat wall at distance d: gamma ≈ 0.5 + 0.5*(d/h).
// In corners, multiply per-wall gammas (each wall clips independently).
fn boundary_gamma(world_pos: vec3<f32>, h: f32) -> f32 {
    let local = world_to_local(world_pos);

    var gamma = 1.0;

    // 5 walls: ±X, floor Y, ±Z (no ceiling — open top)
    let dist_nx = local.x + clip.half_width;
    let dist_px = clip.half_width - local.x;
    let dist_floor = local.y + clip.half_height;
    let dist_nz = local.z + clip.half_depth;
    let dist_pz = clip.half_depth - local.z;

    if (dist_nx < h) { gamma *= 0.5 + 0.5 * clamp(dist_nx / h, 0.0, 1.0); }
    if (dist_px < h) { gamma *= 0.5 + 0.5 * clamp(dist_px / h, 0.0, 1.0); }
    if (dist_floor < h) { gamma *= 0.5 + 0.5 * clamp(dist_floor / h, 0.0, 1.0); }
    if (dist_nz < h) { gamma *= 0.5 + 0.5 * clamp(dist_nz / h, 0.0, 1.0); }
    if (dist_pz < h) { gamma *= 0.5 + 0.5 * clamp(dist_pz / h, 0.0, 1.0); }

    return max(gamma, 0.15);
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
            density += poly6_kernel(r_sq, h);
        }
    }

    // Boundary gamma correction: near container walls, the kernel support
    // extends outside the domain where there are no particles. The computed
    // density is therefore lower than it should be. Dividing by gamma
    // (the fraction of kernel volume inside the domain) restores the
    // density, pushing the MC iso-surface out to the walls.
    let gamma = boundary_gamma(world_pos, h);
    density = density / gamma;

    // Store density value
    textureStore(density_field, vec3<i32>(global_id), vec4<f32>(density, 0.0, 0.0, 0.0));
}
