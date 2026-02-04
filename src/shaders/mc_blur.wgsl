// Gaussian blur for density grid smoothing
// Runs after density sampling, before mesh generation

struct GridParams {
    grid_min: vec3<f32>,
    cell_size: f32,
    grid_dims: vec3<u32>,
    num_particles: u32,
    smoothing_radius: f32,
    surface_threshold: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> density_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> density_out: array<f32>;
@group(0) @binding(2) var<uniform> params: GridParams;

fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.grid_dims.x + z * params.grid_dims.x * params.grid_dims.y;
}

fn get_density(x: i32, y: i32, z: i32) -> f32 {
    // Clamp to grid bounds
    let cx = clamp(x, 0, i32(params.grid_dims.x) - 1);
    let cy = clamp(y, 0, i32(params.grid_dims.y) - 1);
    let cz = clamp(z, 0, i32(params.grid_dims.z) - 1);
    return density_in[grid_index(u32(cx), u32(cy), u32(cz))];
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.grid_dims.x ||
        global_id.y >= params.grid_dims.y ||
        global_id.z >= params.grid_dims.z) {
        return;
    }

    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let z = i32(global_id.z);

    // 3x3x3 Gaussian blur kernel (separable approximation)
    // Weights: center=0.5, face-adjacent=0.0625, edge-adjacent=0.03125, corner=0.015625
    var sum = get_density(x, y, z) * 0.125; // Center weight

    // 6 face neighbors
    sum += (get_density(x-1, y, z) + get_density(x+1, y, z) +
            get_density(x, y-1, z) + get_density(x, y+1, z) +
            get_density(x, y, z-1) + get_density(x, y, z+1)) * 0.0625;

    // 12 edge neighbors
    sum += (get_density(x-1, y-1, z) + get_density(x+1, y-1, z) +
            get_density(x-1, y+1, z) + get_density(x+1, y+1, z) +
            get_density(x-1, y, z-1) + get_density(x+1, y, z-1) +
            get_density(x-1, y, z+1) + get_density(x+1, y, z+1) +
            get_density(x, y-1, z-1) + get_density(x, y+1, z-1) +
            get_density(x, y-1, z+1) + get_density(x, y+1, z+1)) * 0.03125;

    // 8 corner neighbors
    sum += (get_density(x-1, y-1, z-1) + get_density(x+1, y-1, z-1) +
            get_density(x-1, y+1, z-1) + get_density(x+1, y+1, z-1) +
            get_density(x-1, y-1, z+1) + get_density(x+1, y-1, z+1) +
            get_density(x-1, y+1, z+1) + get_density(x+1, y+1, z+1)) * 0.015625;

    let idx = grid_index(global_id.x, global_id.y, global_id.z);
    density_out[idx] = sum;
}
