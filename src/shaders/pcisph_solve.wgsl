// PCISPH Solve — one iteration of predictive-corrective pressure solving
// Reads from sorted_predicted_in, writes to sorted_predicted_out (ping-pong)

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
    _pad_st1: f32,
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

struct PredictedState {
    pred_pos_x: f32,
    pred_pos_y: f32,
    pred_pos_z: f32,
    pressure: f32,
    pred_vel_x: f32,
    pred_vel_y: f32,
    pred_vel_z: f32,
    pred_density: f32,
}

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read> particles: array<SphParticle3D>;
@group(0) @binding(2) var<storage, read> sorted_predicted_in: array<PredictedState>;
@group(0) @binding(3) var<storage, read_write> sorted_predicted_out: array<PredictedState>;
@group(0) @binding(4) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(5) var<storage, read> cell_counts: array<u32>;
@group(0) @binding(6) var<uniform> grid: GridParams;
@group(0) @binding(7) var<storage, read> sorted_index: array<u32>;

const PI: f32 = 3.14159265359;

// Poly6 kernel for density (same as density shader)
fn density_kernel(r_sq: f32) -> f32 {
    let scale = 315.0 / (64.0 * PI * params.kernel_radius_pow9);
    let diff = params.kernel_radius_sq - r_sq;
    return scale * diff * diff * diff;
}

// Spiky gradient for pressure force
fn pressure_kernel_gradient(r: f32) -> f32 {
    let scale = 45.0 / (PI * params.kernel_radius_pow6);
    let diff = params.kernel_radius - r;
    return scale * diff * diff;
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

    let si = sorted_index[i];
    let self_pred = sorted_predicted_in[si];

    let pred_pos = vec3<f32>(self_pred.pred_pos_x, self_pred.pred_pos_y, self_pred.pred_pos_z);
    let old_pressure = self_pred.pressure;

    // Use CURRENT positions for cell lookup (grid was built on current positions)
    let pos_i = particles[i].position;
    let cell_i = position_to_cell(pos_i);

    // Accumulate predicted density and pressure acceleration
    var pred_density = 0.0;
    var a_pressure = vec3<f32>(0.0, 0.0, 0.0);

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
                    let j = start + k; // sorted index of neighbor
                    let neighbor_pred = sorted_predicted_in[j];
                    let neighbor_pos = vec3<f32>(
                        neighbor_pred.pred_pos_x,
                        neighbor_pred.pred_pos_y,
                        neighbor_pred.pred_pos_z,
                    );

                    let r_vec = pred_pos - neighbor_pos;
                    let r_sq = dot(r_vec, r_vec);

                    if (r_sq < params.kernel_radius_sq) {
                        // Density includes self-contribution (r=0 is valid for poly6)
                        pred_density += params.mass * density_kernel(r_sq);

                        // Pressure gradient excludes self (direction undefined at r=0)
                        if (r_sq > 1e-12) {
                            let r = sqrt(r_sq);

                            // Monaghan symmetric pressure acceleration (spiky gradient)
                            let neighbor_density = max(neighbor_pred.pred_density, 1.0);
                            let neighbor_pressure = neighbor_pred.pressure;
                            let self_density = max(self_pred.pred_density, 1.0);

                            // dir points from self toward neighbor (same convention as force shader)
                            let dir = -r_vec / r;
                            a_pressure += -params.mass * (
                                old_pressure / (self_density * self_density) +
                                neighbor_pressure / (neighbor_density * neighbor_density)
                            ) * pressure_kernel_gradient(r) * dir;
                        }
                    }
                }
            }
        }
    }

    // Update pressure: only correct compression (density > rest)
    let density_error = max(0.0, pred_density - params.rest_density);
    let new_pressure = old_pressure + params.pcisph_delta * density_error;

    // Recompute predicted state from scratch using original velocity + total acceleration
    let original_vel = particles[i].velocity;
    let original_pos = particles[i].position;
    let a_np = particles[i].force; // non-pressure acceleration

    let a_total = a_np + a_pressure;
    let v_star = original_vel + params.dt * a_total;
    let x_star = original_pos + params.dt * v_star;

    sorted_predicted_out[si] = PredictedState(
        x_star.x, x_star.y, x_star.z,
        new_pressure,
        v_star.x, v_star.y, v_star.z,
        pred_density,
    );
}
