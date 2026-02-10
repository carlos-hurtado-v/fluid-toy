// XSPH Velocity Smoothing (Monaghan 1989)
// Smooths each particle's velocity toward the local weighted average,
// damping relative jitter while preserving bulk flow.
// v_smoothed = v_i + epsilon * sum_j (m_j / rho_j) * (v_j - v_i) * W_poly6(r_ij, h)

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

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SphParticle3D>;
@group(0) @binding(2) var<storage, read> sorted_particles: array<SphParticle3D>;
@group(0) @binding(3) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(4) var<storage, read> cell_counts: array<u32>;
@group(0) @binding(5) var<uniform> grid: GridParams;

const PI: f32 = 3.14159265359;

fn poly6_kernel(r_sq: f32) -> f32 {
    let scale = 315.0 / (64.0 * PI * params.kernel_radius_pow9);
    let diff = params.kernel_radius_sq - r_sq;
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_particles) {
        return;
    }

    let epsilon = params.xsph_epsilon;
    if (epsilon <= 0.0) {
        return;
    }

    let pos_i = particles[i].position;
    let vel_i = particles[i].velocity;
    let cell_i = position_to_cell(pos_i);

    var vel_correction = vec3<f32>(0.0, 0.0, 0.0);

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
                let end = cell_starts[cell_idx];
                let start = end - count;

                for (var k = 0u; k < count; k++) {
                    let j = start + k;
                    let pos_j = sorted_particles[j].position;
                    let r_vec = pos_i - pos_j;
                    let r_sq = dot(r_vec, r_vec);

                    if (r_sq < params.kernel_radius_sq && r_sq > 1e-12) {
                        let density_j = max(sorted_particles[j].density, 1.0);
                        let vel_j = sorted_particles[j].velocity;
                        let w = poly6_kernel(r_sq);
                        vel_correction += (params.mass / density_j) * (vel_j - vel_i) * w;
                    }
                }
            }
        }
    }

    particles[i].velocity = vel_i + epsilon * vel_correction;
}
