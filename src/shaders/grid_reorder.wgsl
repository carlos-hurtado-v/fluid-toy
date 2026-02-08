// Reorder particles by cell for cache-friendly neighbor access

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

@group(0) @binding(0) var<storage, read> particles_in: array<SphParticle3D>;
@group(0) @binding(1) var<storage, read_write> particles_out: array<SphParticle3D>;
@group(0) @binding(2) var<storage, read_write> particle_cell_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_offsets: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(5) var<storage, read> cell_counts: array<u32>;
@group(0) @binding(6) var<uniform> grid: GridParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= grid.num_particles) {
        return;
    }

    let cell_idx = particle_cell_indices[i];
    // cell_starts contains inclusive prefix sum (end of range), compute start
    let end = cell_starts[cell_idx];
    let count = cell_counts[cell_idx];
    let cell_start = end - count;

    // Atomically get the next slot in this cell
    let slot = atomicAdd(&cell_offsets[cell_idx], 1u);
    let dest_idx = cell_start + slot;

    // Copy particle to sorted position
    particles_out[dest_idx] = particles_in[i];

    // Store mapping so density shader can write to both buffers
    particle_cell_indices[i] = dest_idx;
}
