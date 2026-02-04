// Convert inclusive prefix sum to exclusive by shifting right
// Output[0] = 0, Output[i] = Input[i-1] for i > 0

struct PrefixSumParams {
    count: u32,
    offset: u32,  // unused here but kept for struct compatibility
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: PrefixSumParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.count) {
        return;
    }

    if (i == 0u) {
        output[i] = 0u;
    } else {
        output[i] = input[i - 1u];
    }
}
