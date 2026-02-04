// Fluid Depth Blur - Curvature-flow inspired filter for smooth fluid surface
// This is more aggressive than bilateral blur to create a seamless liquid look

struct BlurParams {
    direction: vec2<f32>,  // (1,0) for horizontal, (0,1) for vertical
    filter_radius: f32,
    blur_scale: f32,
    blur_depth_falloff: f32,
    screen_width: f32,
    screen_height: f32,
    _padding: f32,
}

@group(0) @binding(0) var depth_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_sampler: sampler;
@group(0) @binding(2) var<uniform> params: BlurParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-screen triangle
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
    let center_depth = textureSample(depth_texture, depth_sampler, input.uv).r;

    // Skip if no particle (depth is 0 or very large)
    if (center_depth <= 0.001 || center_depth > 100.0) {
        return 0.0;
    }

    let pixel_size = vec2<f32>(1.0 / params.screen_width, 1.0 / params.screen_height);
    let blur_dir = params.direction * pixel_size * params.blur_scale;

    var sum = 0.0;
    var weight_sum = 0.0;

    // Use a larger, more aggressive Gaussian blur
    // The key to seamless fluid is smoothing across particle boundaries
    let filter_radius = i32(params.filter_radius);
    let sigma = params.filter_radius * 0.5;
    let sigma_sq = sigma * sigma;

    for (var i = -filter_radius; i <= filter_radius; i++) {
        let offset = blur_dir * f32(i);
        let sample_uv = input.uv + offset;
        let sample_depth = textureSample(depth_texture, depth_sampler, sample_uv).r;

        // Skip completely empty areas but be more lenient
        if (sample_depth <= 0.001) {
            continue;
        }

        // Gaussian spatial weight
        let dist_sq = f32(i * i);
        let spatial_weight = exp(-dist_sq / (2.0 * sigma_sq));

        // Depth similarity weight - but make it gentler to allow more blending
        // This is the key: we want to blur ACROSS particle boundaries
        let depth_diff = abs(sample_depth - center_depth);
        let depth_threshold = center_depth * 0.15; // Allow 15% depth variation
        let depth_weight = exp(-depth_diff * depth_diff / (depth_threshold * depth_threshold + 0.001));

        // Combine weights - favor spatial smoothing
        let weight = spatial_weight * (0.3 + 0.7 * depth_weight);

        sum += sample_depth * weight;
        weight_sum += weight;
    }

    if (weight_sum > 0.001) {
        return sum / weight_sum;
    }
    return center_depth;
}
