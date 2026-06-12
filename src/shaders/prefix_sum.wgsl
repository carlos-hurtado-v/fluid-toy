// Prefix sum (scan) - single pass of Hillis-Steele algorithm
// Requires multiple dispatches with increasing offset values

struct PrefixSumParams {
    count: u32,
    offset: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: PrefixSumParams;

// 256 = CELL_WORKGROUP_SIZE in sph_3d_grid.rs (cell counts scale with 1/h³;
// keeps the 1D dispatch under the 65,535-workgroup limit at small kernel radii)
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.count) {
        return;
    }

    if (i >= params.offset) {
        output[i] = input[i] + input[i - params.offset];
    } else {
        output[i] = input[i];
    }
}
