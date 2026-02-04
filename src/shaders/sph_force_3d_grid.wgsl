// SPH 3D Force Computation with Grid-based neighbor search

struct SphParticle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    force: vec3<f32>,
    density: f32,
    near_density: f32,
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

struct Gravity {
    direction: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SphParticle3D>;
@group(0) @binding(2) var<storage, read> sorted_particles: array<SphParticle3D>;
@group(0) @binding(3) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(4) var<storage, read> cell_counts: array<u32>;
@group(0) @binding(5) var<uniform> grid: GridParams;
@group(0) @binding(6) var<uniform> gravity: Gravity;

const PI: f32 = 3.14159265359;

fn density_kernel_gradient(r: f32) -> f32 {
    let scale = 45.0 / (PI * params.kernel_radius_pow6);
    let diff = params.kernel_radius - r;
    return scale * diff * diff;
}

fn near_density_kernel_gradient(r: f32) -> f32 {
    let scale = 45.0 / (PI * params.kernel_radius_pow5);
    let diff = params.kernel_radius - r;
    return scale * diff * diff;
}

fn viscosity_kernel_laplacian(r: f32) -> f32 {
    let scale = 45.0 / (PI * params.kernel_radius_pow6);
    return scale * (params.kernel_radius - r);
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_particles) {
        return;
    }

    let pos_i = particles[i].position;
    let vel_i = particles[i].velocity;
    let cell_i = position_to_cell(pos_i);

    let density_i = max(particles[i].density, 1.0);
    let near_density_i = max(particles[i].near_density, 1.0);

    let pressure_i = params.stiffness * (density_i - params.rest_density);
    let near_pressure_i = params.near_stiffness * near_density_i;

    var f_pressure = vec3<f32>(0.0, 0.0, 0.0);
    var f_viscosity = vec3<f32>(0.0, 0.0, 0.0);

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

                    if (r_sq < params.kernel_radius_sq && r_sq > 1e-12) {
                        let r = sqrt(r_sq);

                        // Compute direction - use index-based fallback when too close
                        var dir: vec3<f32>;
                        let min_dist = params.kernel_radius * 0.1;  // 10% of kernel radius
                        if (r < min_dist) {
                            // Particles too close - use deterministic direction based on indices
                            // This prevents random direction due to floating point noise
                            let idx_diff = f32(i) - f32(j);
                            dir = normalize(vec3<f32>(
                                sin(idx_diff * 1.0),
                                cos(idx_diff * 2.0),
                                sin(idx_diff * 3.0)
                            ));
                        } else {
                            dir = normalize(pos_j - pos_i);
                        }

                        let density_j = max(sorted_particles[j].density, 1.0);
                        let near_density_j = max(sorted_particles[j].near_density, 1.0);

                        let pressure_j = params.stiffness * (density_j - params.rest_density);
                        let near_pressure_j = params.near_stiffness * near_density_j;

                        let shared_pressure = (pressure_i + pressure_j) / 2.0;
                        let near_shared_pressure = (near_pressure_i + near_pressure_j) / 2.0;

                        f_pressure += -params.mass * shared_pressure * dir * density_kernel_gradient(r) / density_j;
                        f_pressure += -params.mass * near_shared_pressure * dir * near_density_kernel_gradient(r) / near_density_j;

                        // Extra separation force when extremely close (prevents collapse)
                        if (r < min_dist) {
                            let separation_strength = 1000.0 * (1.0 - r / min_dist);
                            f_pressure += -separation_strength * dir;
                        }

                        let vel_j = sorted_particles[j].velocity;
                        let relative_vel = vel_j - vel_i;
                        f_viscosity += params.mass * relative_vel * viscosity_kernel_laplacian(r) / density_j;
                    }
                }
            }
        }
    }

    f_viscosity *= params.viscosity;
    let f_gravity = density_i * gravity.direction;

    particles[i].force = f_pressure + f_viscosity + f_gravity;
}
