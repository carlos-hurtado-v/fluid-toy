// SPH 3D Density Computation with Grid-based neighbor search

struct SphParticle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    force: vec3<f32>,
    density: f32,
    near_density: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
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
    surface_tension: f32,
    pcisph_delta: f32,
    xsph_epsilon: f32,
    _pad_st2: f32,
}

struct GridParams {
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,
    total_cells: u32,
    cell_size: f32,
    inv_cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    num_particles: u32,
    _padding: vec2<u32>,
}

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
}

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SphParticle3D>;
@group(0) @binding(2) var<storage, read_write> sorted_particles: array<SphParticle3D>;
@group(0) @binding(6) var<storage, read> sorted_index: array<u32>;
@group(0) @binding(3) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(4) var<storage, read> cell_counts: array<u32>;
@group(0) @binding(5) var<uniform> grid: GridParams;
@group(0) @binding(7) var<uniform> bounds: BoundsParams;

const PI: f32 = 3.14159265359;

fn density_kernel(r_sq: f32) -> f32 {
    let scale = 315.0 / (64.0 * PI * params.kernel_radius_pow9);
    let diff = params.kernel_radius_sq - r_sq;
    return scale * diff * diff * diff;
}

fn near_density_kernel(r: f32) -> f32 {
    let scale = 15.0 / (PI * params.kernel_radius_pow6);
    let diff = params.kernel_radius - r;
    return scale * diff * diff * diff;
}

fn position_to_cell(pos: vec3<f32>) -> vec3<i32> {
    let local_pos = pos - vec3<f32>(grid.grid_origin_x, grid.grid_origin_y, grid.grid_origin_z);
    return vec3<i32>(floor(local_pos * grid.inv_cell_size));
}

fn cell_to_index(cell: vec3<i32>) -> u32 {
    return u32(cell.x) + u32(cell.y) * grid.grid_size_x + u32(cell.z) * grid.grid_size_x * grid.grid_size_y;
}

fn is_valid_cell(cell: vec3<i32>) -> bool {
    return cell.x >= 0i && cell.x < i32(grid.grid_size_x) &&
           cell.y >= 0i && cell.y < i32(grid.grid_size_y) &&
           cell.z >= 0i && cell.z < i32(grid.grid_size_z);
}

/// Estimate the fraction of kernel support volume that lies inside the domain.
/// Near a flat wall at distance d: gamma ≈ 0.5 + 0.5*(d/h).
/// In corners multiply per-wall gammas (each wall clips independently).
fn boundary_gamma(pos: vec3<f32>) -> f32 {
    let h = params.kernel_radius;
    // Transform world position into container-local space
    let local = vec3<f32>(
        dot(bounds.rotation_row0.xyz, pos),
        dot(bounds.rotation_row1.xyz, pos),
        dot(bounds.rotation_row2.xyz, pos),
    );

    var gamma = 1.0;

    // 6 walls: ±X, floor/ceiling Y, ±Z
    let dist_nx = local.x - (-bounds.bound_x);
    let dist_px = bounds.bound_x - local.x;
    let dist_floor = local.y - bounds.floor_y;
    let dist_ceil = bounds.ceiling_y - local.y;
    let dist_nz = local.z - (-bounds.bound_z);
    let dist_pz = bounds.bound_z - local.z;

    if (dist_nx < h) { gamma *= 0.5 + 0.5 * clamp(dist_nx / h, 0.0, 1.0); }
    if (dist_px < h) { gamma *= 0.5 + 0.5 * clamp(dist_px / h, 0.0, 1.0); }
    if (dist_floor < h) { gamma *= 0.5 + 0.5 * clamp(dist_floor / h, 0.0, 1.0); }
    if (dist_ceil < h) { gamma *= 0.5 + 0.5 * clamp(dist_ceil / h, 0.0, 1.0); }
    if (dist_nz < h) { gamma *= 0.5 + 0.5 * clamp(dist_nz / h, 0.0, 1.0); }
    if (dist_pz < h) { gamma *= 0.5 + 0.5 * clamp(dist_pz / h, 0.0, 1.0); }

    return max(gamma, 0.1); // Prevent division by near-zero
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_particles) {
        return;
    }

    let pos_i = particles[i].position;
    let cell_i = position_to_cell(pos_i);

    var density = 0.0;
    var near_density = 0.0;
    var normal = vec3<f32>(0.0, 0.0, 0.0);

    // Iterate over 3x3x3 neighboring cells
    for (var dz = -1i; dz <= 1i; dz++) {
        for (var dy = -1i; dy <= 1i; dy++) {
            for (var dx = -1i; dx <= 1i; dx++) {
                let neighbor_cell = cell_i + vec3<i32>(dx, dy, dz);

                if (!is_valid_cell(neighbor_cell)) {
                    continue;
                }

                let cell_idx = cell_to_index(neighbor_cell);
                let count = cell_counts[cell_idx];
                // cell_starts contains inclusive prefix sum, so start = end - count
                let end = cell_starts[cell_idx];
                let start = end - count;

                // Iterate over particles in this cell
                for (var k = 0u; k < count; k++) {
                    let j = start + k;
                    let pos_j = sorted_particles[j].position;
                    let r_vec = pos_i - pos_j;
                    let r_sq = dot(r_vec, r_vec);

                    if (r_sq < params.kernel_radius_sq) {
                        let r = sqrt(r_sq);
                        density += params.mass * density_kernel(r_sq);
                        near_density += params.mass * near_density_kernel(r);

                        // Surface normal: unweighted poly6 gradient (proportional to ∇W)
                        // Points outward from surface (away from fluid bulk)
                        if (r_sq > 1e-12) {
                            let diff = params.kernel_radius_sq - r_sq;
                            normal += r_vec * diff * diff;
                        }
                    }
                }
            }
        }
    }

    // Boundary density correction: compensate for missing neighbors near walls.
    // gamma estimates what fraction of the kernel volume is inside the domain;
    // dividing by it restores the density that would exist with a full neighborhood.
    let gamma = boundary_gamma(pos_i);
    density = density / gamma;

    particles[i].density = density;
    particles[i].near_density = near_density;
    particles[i].normal_x = normal.x;
    particles[i].normal_y = normal.y;
    particles[i].normal_z = normal.z;

    // Also update sorted buffer so force shader can read neighbor data
    // without needing a second reorder pass
    let si = sorted_index[i];
    sorted_particles[si].density = density;
    sorted_particles[si].near_density = near_density;
    sorted_particles[si].normal_x = normal.x;
    sorted_particles[si].normal_y = normal.y;
    sorted_particles[si].normal_z = normal.z;
}
