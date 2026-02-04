// Resolve weighted depth to final depth
// Input: texture with (depth * weight, weight)
// Output: depth = weighted_depth / weight

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var depth_weight_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );

    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv = uvs[vertex_index];
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) f32 {
    let depth_weight = textureSample(depth_weight_texture, texture_sampler, input.uv).rg;

    let weighted_depth = depth_weight.r;
    let weight = depth_weight.g;

    // If no contribution, return 0 (no fluid)
    if (weight < 0.001) {
        return 0.0;
    }

    // Resolve to actual depth
    return weighted_depth / weight;
}
