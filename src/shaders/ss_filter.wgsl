// Screen-space fluid rendering — Narrow-Range Depth Filter
// Truong & Yuksel, i3D 2018 — matching Splash reference implementation.
//
// 1D mode: separable bilateral Gaussian with adaptive thresholds.
//   Samples outward symmetrically. Close outliers reject both sides,
//   far outliers clamp to center+mu, in-range samples expand the threshold window.
// 2D mode: diamond-pattern bilateral with filterSize=2 for post-1D refinement.
//
// Background pixels (depth <= 0) are treated as far outliers (clamped to center+mu),
// NOT as close outliers. This creates smooth depth falloff at water-air boundaries.

struct Params {
    projected_particle_constant: f32,
    max_filter_size: f32,
    mu: f32,
    depth_threshold: f32,
    screen_width: u32,
    screen_height: u32,
    blur_2d: u32,
    direction: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input_depth: texture_2d<f32>;
@group(0) @binding(2) var output_depth: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (gid.x >= params.screen_width || gid.y >= params.screen_height) { return; }

    let coord = vec2<i32>(x, y);
    let depth = textureLoad(input_depth, coord, 0).r;

    // Background: pass through
    if (depth <= 0.0) {
        textureStore(output_depth, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Adaptive filter size: inversely proportional to depth (closer = larger filter)
    let filterSize = min(i32(params.max_filter_size), i32(ceil(params.projected_particle_constant / depth)));
    if (filterSize <= 0) {
        textureStore(output_depth, coord, vec4<f32>(depth, 0.0, 0.0, 0.0));
        return;
    }

    let sigma = f32(filterSize) / 2.0;
    let sigma_sq_inv = 1.0 / (2.0 * sigma * sigma);

    let higherDepthBound = depth + params.mu;

    var sum = depth;
    var wsum = 1.0;

    let w_max = i32(params.screen_width) - 1;
    let h_max = i32(params.screen_height) - 1;

    if (params.blur_2d == 0u) {
        // ── 1D separable mode ─────────────────────────────────────────
        let dir = select(vec2<i32>(0, 1), vec2<i32>(1, 0), params.direction == 0u);

        var sum2 = vec2<f32>(0.0);
        var wsum2 = vec2<f32>(0.0);
        var threshLowX = depth - params.depth_threshold;
        var threshHighX = depth + params.depth_threshold;
        var threshLowY = depth - params.depth_threshold;
        var threshHighY = depth + params.depth_threshold;

        for (var r = 1; r <= filterSize; r++) {
            let gaussW = exp(-f32(r * r) * sigma_sq_inv);

            let coordX = clamp(coord - dir * r, vec2<i32>(0), vec2<i32>(w_max, h_max));
            let coordY = clamp(coord + dir * r, vec2<i32>(0), vec2<i32>(w_max, h_max));

            var sampledX = textureLoad(input_depth, coordX, 0).r;
            var sampledY = textureLoad(input_depth, coordY, 0).r;

            var w = vec2<f32>(gaussW);

            // ── X side (negative direction) ──
            if (sampledX <= 0.0) {
                // Background: treat as far outlier (smooth boundary falloff)
                sampledX = higherDepthBound;
            } else if (sampledX < threshLowX) {
                // Close outlier: different surface in front → reject BOTH sides
                w.x = 0.0;
                w.y = 0.0;
            } else if (sampledX > threshHighX) {
                // Far outlier: different surface behind → clamp
                sampledX = higherDepthBound;
            } else {
                // In range: expand threshold window
                threshLowX = min(threshLowX, sampledX - params.depth_threshold);
                threshHighX = max(threshHighX, sampledX + params.depth_threshold);
            }

            // ── Y side (positive direction) ──
            if (sampledY <= 0.0) {
                sampledY = higherDepthBound;
            } else if (sampledY < threshLowY) {
                w.x = 0.0;
                w.y = 0.0;
            } else if (sampledY > threshHighY) {
                sampledY = higherDepthBound;
            } else {
                threshLowY = min(threshLowY, sampledY - params.depth_threshold);
                threshHighY = max(threshHighY, sampledY + params.depth_threshold);
            }

            sum2 += vec2<f32>(sampledX, sampledY) * w;
            wsum2 += w;
        }
        sum += sum2.x + sum2.y;
        wsum += wsum2.x + wsum2.y;
    } else {
        // ── 2D diamond-pattern mode ───────────────────────────────────
        // Small radius (2) for post-1D refinement. Samples in a diamond
        // pattern to smooth directional artifacts from separable passes.
        let filterSize2D = 2;
        var threshLow = depth - params.depth_threshold;
        var threshHigh = depth + params.depth_threshold;

        var sum4 = vec4<f32>(0.0);
        var wsum4 = vec4<f32>(0.0);

        for (var r = 1; r <= filterSize2D; r++) {
            for (var i = 0; i < 2 * r; i++) {
                let gaussW = exp((-f32(r * r) + f32((r - i) * (r - i))) * sigma_sq_inv);

                // Diamond offsets matching Splash exactly:
                // X: center - (r, r-i),  Y: center + (r, r-i)
                // Z: center - (r-i, r),  W: center + (r-i, r)
                let cX = clamp(coord - vec2<i32>(r, r - i), vec2<i32>(0), vec2<i32>(w_max, h_max));
                let cY = clamp(coord + vec2<i32>(r, r - i), vec2<i32>(0), vec2<i32>(w_max, h_max));
                let cZ = clamp(coord - vec2<i32>(r - i, r), vec2<i32>(0), vec2<i32>(w_max, h_max));
                let cW = clamp(coord + vec2<i32>(r - i, r), vec2<i32>(0), vec2<i32>(w_max, h_max));

                var sX = textureLoad(input_depth, cX, 0).r;
                var sY = textureLoad(input_depth, cY, 0).r;
                var sZ = textureLoad(input_depth, cZ, 0).r;
                var sW = textureLoad(input_depth, cW, 0).r;

                var w = vec4<f32>(gaussW);

                // X check
                if (sX <= 0.0) {
                    sX = higherDepthBound;
                } else if (sX < threshLow) {
                    w.x = 0.0; w.y = 0.0;
                } else if (sX > threshHigh) {
                    sX = higherDepthBound;
                } else {
                    threshLow = min(threshLow, sX - params.depth_threshold);
                    threshHigh = max(threshHigh, sX + params.depth_threshold);
                }

                // Y check
                if (sY <= 0.0) {
                    sY = higherDepthBound;
                } else if (sY < threshLow) {
                    w.x = 0.0; w.y = 0.0;
                } else if (sY > threshHigh) {
                    sY = higherDepthBound;
                } else {
                    threshLow = min(threshLow, sY - params.depth_threshold);
                    threshHigh = max(threshHigh, sY + params.depth_threshold);
                }

                // Z check
                if (sZ <= 0.0) {
                    sZ = higherDepthBound;
                } else if (sZ < threshLow) {
                    w.z = 0.0; w.w = 0.0;
                } else if (sZ > threshHigh) {
                    sZ = higherDepthBound;
                } else {
                    threshLow = min(threshLow, sZ - params.depth_threshold);
                    threshHigh = max(threshHigh, sZ + params.depth_threshold);
                }

                // W check
                if (sW <= 0.0) {
                    sW = higherDepthBound;
                } else if (sW < threshLow) {
                    w.z = 0.0; w.w = 0.0;
                } else if (sW > threshHigh) {
                    sW = higherDepthBound;
                } else {
                    threshLow = min(threshLow, sW - params.depth_threshold);
                    threshHigh = max(threshHigh, sW + params.depth_threshold);
                }

                sum4 += vec4<f32>(sX, sY, sZ, sW) * w;
                wsum4 += w;
            }
        }
        sum += sum4.x + sum4.y + sum4.z + sum4.w;
        wsum += wsum4.x + wsum4.y + wsum4.z + wsum4.w;
    }

    textureStore(output_depth, coord, vec4<f32>(sum / wsum, 0.0, 0.0, 0.0));
}
