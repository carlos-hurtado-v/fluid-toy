// Screen-Space Fluid - Bilateral Blur
// Smooths depth while preserving edges (depth discontinuities)
// Uses textureLoad for precise pixel access

struct BlurParams {
    blur_dir: vec2<f32>,           // (1,0) for horizontal, (0,1) for vertical
    depth_threshold: f32,          // Controls edge preservation
    max_filter_size: f32,          // Maximum blur radius in pixels
    projected_particle_constant: f32, // Controls depth-dependent blur size
    _padding: vec3<f32>,
}

@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: BlurParams;

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

    let pos = positions[vertex_index];

    var output: VertexOutput;
    output.position = vec4<f32>(pos, 0.0, 1.0);
    output.uv = pos * 0.5 + 0.5;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(depth_tex));
    let iuv = vec2<i32>(input.uv * tex_size);

    let depth = textureLoad(depth_tex, iuv, 0).r;

    // Background check (depth = 0 or very large means no particle)
    if (depth <= 0.0 || depth >= 1e4) {
        return vec4<f32>(depth, 0.0, 0.0, 1.0);
    }

    // Adaptive filter size based on depth (closer = larger blur)
    let filter_size = min(
        i32(params.max_filter_size),
        i32(ceil(params.projected_particle_constant / depth))
    );

    // Gaussian sigma based on filter size
    let sigma = f32(filter_size) / 3.0;
    let two_sigma_sq = 2.0 * sigma * sigma;

    // Depth sigma for bilateral weighting
    let sigma_depth = params.depth_threshold / 3.0;
    let two_sigma_depth_sq = 2.0 * sigma_depth * sigma_depth;

    var sum = 0.0;
    var weight_sum = 0.0;

    for (var x = -filter_size; x <= filter_size; x++) {
        let offset = vec2<i32>(vec2<f32>(f32(x)) * params.blur_dir);
        let sample_coord = iuv + offset;

        // Bounds check
        if (sample_coord.x < 0 || sample_coord.x >= i32(tex_size.x) ||
            sample_coord.y < 0 || sample_coord.y >= i32(tex_size.y)) {
            continue;
        }

        let sampled_depth = textureLoad(depth_tex, sample_coord, 0).r;

        // Skip background
        if (sampled_depth <= 0.0 || sampled_depth >= 1e4) {
            continue;
        }

        // Spatial weight (Gaussian)
        let r_sq = f32(x * x);
        let spatial_weight = exp(-r_sq / two_sigma_sq);

        // Range weight (depth similarity)
        let depth_diff = sampled_depth - depth;
        let range_weight = exp(-depth_diff * depth_diff / two_sigma_depth_sq);

        let weight = spatial_weight * range_weight;
        sum += sampled_depth * weight;
        weight_sum += weight;
    }

    var final_depth = depth;
    if (weight_sum > 0.0) {
        final_depth = sum / weight_sum;
    }

    return vec4<f32>(final_depth, 0.0, 0.0, 1.0);
}
