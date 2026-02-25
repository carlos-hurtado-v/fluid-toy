// Marching Cubes - Separable 3D Gaussian Blur
// Run 3 times (X, Y, Z) to blur the density field for smoother surfaces

struct BlurParams {
    dir_x: i32,
    dir_y: i32,
    dir_z: i32,
    radius: i32,
    grid_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var input_field: texture_3d<f32>;
@group(0) @binding(1) var output_field: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: BlurParams;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_size = params.grid_size;
    if (global_id.x >= grid_size || global_id.y >= grid_size || global_id.z >= grid_size) {
        return;
    }

    let pos = vec3<i32>(global_id);

    // Propagate sentinel: outside-container voxels stay outside through all blur passes
    let center_val = textureLoad(input_field, pos, 0).r;
    if (center_val < 0.0) {
        textureStore(output_field, pos, vec4<f32>(-1.0, 0.0, 0.0, 0.0));
        return;
    }

    let dir = vec3<i32>(params.dir_x, params.dir_y, params.dir_z);
    let r = params.radius;
    let grid_max = i32(grid_size) - 1;

    // Triangle filter (approximates Gaussian, cheap to compute)
    // Skip sentinel values (< 0) — these are outside-container voxels.
    // This prevents blur from bleeding zeros across container walls.
    var sum = 0.0;
    var weight_sum = 0.0;

    for (var i = -r; i <= r; i++) {
        let sample_pos = pos + dir * i;
        let clamped = clamp(sample_pos, vec3<i32>(0), vec3<i32>(grid_max));

        let val = textureLoad(input_field, clamped, 0).r;
        if (val < 0.0) {
            continue;  // Skip outside-container samples
        }

        let w = f32(r + 1 - abs(i));
        sum += val * w;
        weight_sum += w;
    }

    if (weight_sum <= 0.0) {
        textureStore(output_field, pos, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    textureStore(output_field, pos, vec4<f32>(sum / weight_sum, 0.0, 0.0, 0.0));
}
