// Marching Cubes - Density Field Generation (Grid-Accelerated)
// Uses SPH spatial hash grid for O(1) neighbor lookups per voxel
// instead of brute-force O(N) particle iteration.
// Applies boundary gamma correction so the iso-surface extends to container walls.

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

struct GridParams {
    grid_min: vec3<f32>,
    grid_size: u32,          // MC grid cells per dimension (e.g., 100)
    grid_max: vec3<f32>,
    cell_size: f32,          // MC cell size in world units
    kernel_radius: f32,
    iso_value: f32,
    num_particles: u32,
    max_vertices: u32,
}

struct SphGridParams {
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

@group(0) @binding(0) var<storage, read> sorted_particles: array<SphParticle3D>;
@group(0) @binding(1) var<uniform> params: GridParams;
@group(0) @binding(2) var density_field: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var<uniform> container: ContainerGeometry;
@group(0) @binding(4) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(5) var<storage, read> cell_counts: array<u32>;
@group(0) @binding(6) var<uniform> sph_grid: SphGridParams;

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

// Convert MC grid coordinates to world position (cell center)
fn grid_to_world(grid_pos: vec3<u32>) -> vec3<f32> {
    let cell_size = params.cell_size;
    return params.grid_min + (vec3<f32>(grid_pos) + 0.5) * cell_size;
}

// --- SPH grid helper functions ---

fn position_to_sph_cell(pos: vec3<f32>) -> vec3<i32> {
    let local = pos - vec3<f32>(sph_grid.grid_origin_x, sph_grid.grid_origin_y, sph_grid.grid_origin_z);
    return vec3<i32>(floor(local * sph_grid.inv_cell_size));
}

fn sph_cell_to_index(cell: vec3<i32>) -> u32 {
    return u32(cell.x) + u32(cell.y) * sph_grid.grid_size_x + u32(cell.z) * sph_grid.grid_size_x * sph_grid.grid_size_y;
}

fn is_valid_sph_cell(cell: vec3<i32>) -> bool {
    return cell.x >= 0 && cell.x < i32(sph_grid.grid_size_x) &&
           cell.y >= 0 && cell.y < i32(sph_grid.grid_size_y) &&
           cell.z >= 0 && cell.z < i32(sph_grid.grid_size_z);
}

// --- Boundary gamma correction ---

// Estimate fraction of kernel support volume inside the container.
fn boundary_gamma(world_pos: vec3<f32>, h: f32) -> f32 {
    let local = world_to_local(container, world_pos);

    var gamma = 1.0;

    // 5 walls: ±X, floor Y, ±Z (no ceiling — open top)
    let dist_nx = local.x + container.half_width;
    let dist_px = container.half_width - local.x;
    let dist_floor = local.y + container.half_height;
    let dist_nz = local.z + container.half_depth;
    let dist_pz = container.half_depth - local.z;

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

    // Skip if outside MC grid bounds
    if (global_id.x >= grid_size || global_id.y >= grid_size || global_id.z >= grid_size) {
        return;
    }

    let world_pos = grid_to_world(global_id);
    let h = params.kernel_radius;
    let h_sq = h * h;

    // Map this voxel's world position to SPH grid cell
    let center_cell = position_to_sph_cell(world_pos);

    // Search radius in SPH grid cells
    // MC kernel_radius may be larger than SPH cell_size (due to mc_density_radius_scale)
    let cell_radius = i32(ceil(h * sph_grid.inv_cell_size));

    // Sum density contributions from nearby particles using grid acceleration
    var density = 0.0;

    for (var dz = -cell_radius; dz <= cell_radius; dz++) {
        for (var dy = -cell_radius; dy <= cell_radius; dy++) {
            for (var dx = -cell_radius; dx <= cell_radius; dx++) {
                let neighbor_cell = center_cell + vec3<i32>(dx, dy, dz);

                if (!is_valid_sph_cell(neighbor_cell)) {
                    continue;
                }

                let cell_idx = sph_cell_to_index(neighbor_cell);
                let count = cell_counts[cell_idx];
                if (count == 0u) {
                    continue;
                }

                // cell_starts contains inclusive prefix sum, so start = end - count
                let end = cell_starts[cell_idx];
                let start = end - count;

                for (var k = 0u; k < count; k++) {
                    let j = start + k;
                    let particle_pos = sorted_particles[j].position;
                    let diff = world_pos - particle_pos;
                    let r_sq = dot(diff, diff);

                    if (r_sq < h_sq) {
                        density += poly6_kernel(r_sq, h);
                    }
                }
            }
        }
    }

    // Mark voxels outside the container with a sentinel (-1).
    // The blur shader skips sentinels, preventing density bleed across walls.
    let local = world_to_local(container, world_pos);
    if (container.clip_enabled != 0u && !is_inside_box(container, local, 0.0)) {
        textureStore(density_field, vec3<i32>(global_id), vec4<f32>(-1.0, 0.0, 0.0, 0.0));
        return;
    }

    // Boundary gamma correction: near container walls, the kernel support
    // extends outside the domain where there are no particles. Dividing by gamma
    // restores the density, pushing the MC iso-surface out to the walls.
    let gamma = boundary_gamma(world_pos, h);
    density = density / gamma;

    // Store density value
    textureStore(density_field, vec3<i32>(global_id), vec4<f32>(density, 0.0, 0.0, 0.0));
}
