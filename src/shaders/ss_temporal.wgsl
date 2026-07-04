// Screen-space fluid rendering — Temporal depth accumulation
// Validity-gated EMA on the narrow-range-filtered eye depth, blended in
// place (each thread touches only its own texel of the read_write depth).
// Runs only while the camera is static (the CPU compares view matrices and
// skips the dispatch during motion, refreshing history so the blend
// re-engages the frame the camera stops). Static camera means history and
// current share pixel coordinates — no reprojection needed. The tolerance
// gate rejects genuinely moving surfaces (rising splashes, disocclusions),
// so smoothing applies to shimmer, not to motion.

struct TemporalParams {
    screen_width: u32,
    screen_height: u32,
    // EMA weight on accepted history
    history_weight: f32,
    // World-space acceptance tolerance (scaled to the particle radius)
    depth_tolerance: f32,
}

@group(0) @binding(0) var<uniform> params: TemporalParams;
@group(0) @binding(2) var depth_tex: texture_storage_2d<r32float, read_write>;
@group(0) @binding(3) var history_tex: texture_2d<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.screen_width || gid.y >= params.screen_height) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let d_cur = textureLoad(depth_tex, coord).r;

    // Background stays background — no history creep across the silhouette
    if (d_cur <= 0.0) {
        return;
    }

    let d_hist = textureLoad(history_tex, coord, 0).r;
    if (d_hist <= 0.0 || abs(d_hist - d_cur) >= params.depth_tolerance) {
        return;
    }

    let blended = mix(d_cur, d_hist, params.history_weight);
    textureStore(depth_tex, coord, vec4<f32>(blended, 0.0, 0.0, 0.0));
}
