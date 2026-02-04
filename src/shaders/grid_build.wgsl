// Build grid: count particles per cell and store particle's cell index

struct SphParticle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    force: vec3<f32>,
    density: f32,
    near_density: f32,
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

@group(0) @binding(0) var<storage, read> particles: array<SphParticle3D>;
@group(0) @binding(1) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> particle_cell_indices: array<u32>;
@group(0) @binding(3) var<uniform> grid: GridParams;

fn position_to_cell(pos: vec3<f32>) -> vec3<i32> {
    let local_pos = pos - vec3<f32>(grid.grid_origin_x, grid.grid_origin_y, grid.grid_origin_z);
    return vec3<i32>(floor(local_pos * grid.inv_cell_size));
}

fn cell_to_index(cell: vec3<i32>) -> u32 {
    // Clamp to valid range
    let cx = clamp(cell.x, 0i, i32(grid.grid_size_x) - 1i);
    let cy = clamp(cell.y, 0i, i32(grid.grid_size_y) - 1i);
    let cz = clamp(cell.z, 0i, i32(grid.grid_size_z) - 1i);
    return u32(cx) + u32(cy) * grid.grid_size_x + u32(cz) * grid.grid_size_x * grid.grid_size_y;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= grid.num_particles) {
        return;
    }

    let pos = particles[i].position;
    let cell = position_to_cell(pos);
    let cell_idx = cell_to_index(cell);

    // Store which cell this particle belongs to
    particle_cell_indices[i] = cell_idx;

    // Increment count for this cell (atomic)
    atomicAdd(&cell_counts[cell_idx], 1u);
}
