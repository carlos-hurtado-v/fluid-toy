// Screen-space fluid rendering — Thickness Gaussian Blur
// Simple separable Gaussian blur on the thickness buffer.
// No edge-awareness needed since thickness is naturally smooth.
// Two dispatches: horizontal (direction=0) then vertical (direction=1).

struct BlurParams {
    screen_width: u32,
    screen_height: u32,
    radius: u32,         // blur kernel radius (default 10)
    direction: u32,      // 0 = horizontal, 1 = vertical
}

@group(0) @binding(0) var<uniform> params: BlurParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= params.screen_width || y >= params.screen_height) {
        return;
    }

    let coord = vec2<i32>(i32(x), i32(y));
    let r = i32(params.radius);
    let sigma = f32(r) / 3.0;
    let two_sigma_sq = 2.0 * sigma * sigma;

    var sum = 0.0;
    var wsum = 0.0;

    for (var i = -r; i <= r; i++) {
        var sample_coord: vec2<i32>;
        if (params.direction == 0u) {
            sample_coord = vec2<i32>(i32(x) + i, i32(y));
        } else {
            sample_coord = vec2<i32>(i32(x), i32(y) + i);
        }

        sample_coord = clamp(sample_coord, vec2<i32>(0), vec2<i32>(i32(params.screen_width) - 1, i32(params.screen_height) - 1));

        let val = textureLoad(input_tex, sample_coord, 0).r;
        let fi = f32(i);
        let w = exp(-fi * fi / two_sigma_sq);

        sum += val * w;
        wsum += w;
    }

    let result = sum / max(wsum, 0.001);
    textureStore(output_tex, coord, vec4<f32>(result, 0.0, 0.0, 0.0));
}
