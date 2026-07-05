// Caustics - Atlas Filtering
// Separable gaussian blur (band-limits splat noise) + temporal EMA
// (suppresses frame-to-frame sparkle from the changing MC mesh).
// All passes operate on the RGBA16F caustic atlas — a vertical strip of
// per-face tiles (floor tile_res tall, then four wall tiles tile_res/2) —
// with RGB = caustic irradiance, A = water shadow coverage.

struct FilterParams {
    // Blur direction (1,0) or (0,1)
    dir_x: i32,
    dir_y: i32,
    // Gaussian sigma in map texels
    sigma: f32,
    // EMA history weight (0 = take current frame entirely)
    alpha: f32,
    // Floor tile size in texels (wall rows are tile_res/2)
    tile_res: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> filter_params: FilterParams;
@group(0) @binding(3) var history: texture_2d<f32>;

// Vertical bounds [start, end) of the atlas tile row containing y, so the
// vertical blur never mixes one face into another
fn row_bounds(y: i32) -> vec2<i32> {
    let res = i32(filter_params.tile_res);
    if (y < res) {
        return vec2<i32>(0, res);
    }
    let half = res / 2;
    let k = (y - res) / half;
    let start = res + k * half;
    return vec2<i32>(start, start + half);
}

@compute @workgroup_size(8, 8)
fn blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let coord = vec2<i32>(gid.xy);
    let dir = vec2<i32>(filter_params.dir_x, filter_params.dir_y);
    let rb = row_bounds(coord.y);

    // Radius tracks sigma so the slider has effect across its whole range
    let radius = clamp(i32(ceil(filter_params.sigma * 2.5)), 1, 6);
    let inv_two_sigma_sq = 1.0 / max(2.0 * filter_params.sigma * filter_params.sigma, 1e-4);
    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var i = -radius; i <= radius; i++) {
        let w = exp(-f32(i * i) * inv_two_sigma_sq);
        var tap = coord + dir * i;
        tap.x = clamp(tap.x, 0, i32(dims.x) - 1);
        tap.y = clamp(tap.y, rb.x, rb.y - 1);
        sum += textureLoad(src, tap, 0) * w;
        weight_sum += w;
    }
    textureStore(dst, coord, sum / weight_sum);
}

@compute @workgroup_size(8, 8)
fn temporal_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let coord = vec2<i32>(gid.xy);
    let current = textureLoad(src, coord, 0);
    let prev = textureLoad(history, coord, 0);
    textureStore(dst, coord, mix(current, prev, filter_params.alpha));
}
