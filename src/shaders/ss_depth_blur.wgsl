// Screen-space fluid rendering — Post-filter Depth Blur
// Small isotropic Gaussian blur applied AFTER the narrow-range filter.
// The narrow-range filter is nonlinear and not truly separable — the H-then-V
// processing leaves directional artifacts (vertical streaks). This 2D blur
// smooths them out. Small radius (4-5px) is enough.
// Skips empty pixels (depth <= 0) and only averages valid neighbors.

struct BlurParams {
    screen_width: u32,
    screen_height: u32,
    radius: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: BlurParams;
@group(0) @binding(1) var input_depth: texture_2d<f32>;
@group(0) @binding(2) var output_depth: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.screen_width || y >= params.screen_height) { return; }

    let coord = vec2<i32>(i32(x), i32(y));
    let z_c = textureLoad(input_depth, coord, 0).r;

    if (z_c <= 0.0) {
        textureStore(output_depth, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let r = i32(params.radius);
    let sigma = max(f32(r) / 2.0, 1.0);
    let two_sigma_sq = 2.0 * sigma * sigma;

    var sum = 0.0;
    var wsum = 0.0;

    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            let sc = clamp(
                vec2<i32>(i32(x) + dx, i32(y) + dy),
                vec2<i32>(0),
                vec2<i32>(i32(params.screen_width) - 1, i32(params.screen_height) - 1)
            );
            let z = textureLoad(input_depth, sc, 0).r;
            if (z <= 0.0) { continue; }

            let fd = vec2<f32>(f32(dx), f32(dy));
            let w = exp(-dot(fd, fd) / two_sigma_sq);
            sum += z * w;
            wsum += w;
        }
    }

    if (wsum > 0.0) {
        textureStore(output_depth, coord, vec4<f32>(sum / wsum, 0.0, 0.0, 0.0));
    } else {
        textureStore(output_depth, coord, vec4<f32>(z_c, 0.0, 0.0, 0.0));
    }
}
