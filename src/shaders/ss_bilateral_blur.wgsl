// Screen-Space Fluid - Bilateral Blur
// Wide spatial blur that respects depth discontinuities
// This merges nearby spheres while preserving fluid silhouette against background

// Must match GpuBlurParams in screen_space_fluid.rs (48 bytes)
struct BlurParams {
    blur_dir: vec2<f32>,              // 8 bytes @ 0
    depth_threshold: f32,              // 4 bytes @ 8 - Depth sigma
    max_filter_size: f32,              // 4 bytes @ 12 - Spatial blur radius in pixels
    projected_particle_constant: f32,  // 4 bytes @ 16 - Not used
    _pad1_a: f32,                      // 4 bytes @ 20
    _pad1_b: f32,                      // 4 bytes @ 24
    _pad1_c: f32,                      // 4 bytes @ 28
    _padding_a: f32,                   // 4 bytes @ 32
    _padding_b: f32,                   // 4 bytes @ 36
    _padding_c: f32,                   // 4 bytes @ 40
    _pad2: f32,                        // 4 bytes @ 44
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
    let tex_size = vec2<i32>(textureDimensions(depth_tex));
    let iuv = vec2<i32>(input.uv * vec2<f32>(tex_size));

    let center_depth = textureLoad(depth_tex, iuv, 0).r;

    // Background check - don't blur background pixels
    if (center_depth <= 0.0 || center_depth >= 1e4) {
        return vec4<f32>(center_depth, 0.0, 0.0, 1.0);
    }

    let radius = i32(params.max_filter_size);
    let blur_dir = vec2<i32>(params.blur_dir);

    var sum: f32 = 0.0;
    var weight_sum: f32 = 0.0;

    // Bilateral blur: spatial gaussian + depth-based weight
    let sigma_spatial = params.max_filter_size / 2.0;
    let sigma_depth = params.depth_threshold;

    for (var i: i32 = -radius; i <= radius; i++) {
        let sample_coord = iuv + blur_dir * i;

        // Bounds check
        if (sample_coord.x < 0 || sample_coord.x >= tex_size.x ||
            sample_coord.y < 0 || sample_coord.y >= tex_size.y) {
            continue;
        }

        let sample_depth = textureLoad(depth_tex, sample_coord, 0).r;

        // Skip background samples
        if (sample_depth <= 0.0 || sample_depth >= 1e4) {
            continue;
        }

        // Spatial weight (Gaussian)
        let dist = f32(i);
        let spatial_weight = exp(-(dist * dist) / (2.0 * sigma_spatial * sigma_spatial));

        // Depth weight (bilateral term) - reject samples with large depth difference
        let depth_diff = abs(sample_depth - center_depth);
        let depth_weight = exp(-(depth_diff * depth_diff) / (2.0 * sigma_depth * sigma_depth));

        // Combined weight
        let weight = spatial_weight * depth_weight;

        sum += sample_depth * weight;
        weight_sum += weight;
    }

    // Normalize
    var result = center_depth;
    if (weight_sum > 0.001) {
        result = sum / weight_sum;
    }

    return vec4<f32>(result, 0.0, 0.0, 1.0);
}
