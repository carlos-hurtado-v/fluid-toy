// GTAO (Ground Truth Ambient Occlusion) - Jimenez 2016
// All compute shaders in one file with multiple entry points.
// Operates at half resolution for performance.
// Uses 4 slices per pixel with per-pixel hash noise for smooth results.

struct GtaoParams {
    radius: f32,
    falloff_start: f32,
    num_steps: u32,
    frame_index: u32,
    half_res: vec2<f32>,
    inv_half_res: vec2<f32>,
    full_res: vec2<f32>,
    inv_full_res: vec2<f32>,
    temporal_blend: f32,
    thickness: f32,
    _pad0: f32,
    _pad1: f32,
}

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    near: f32,
    far: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// ============================================================
// Entry point 1: Prefilter depth (full-res hardware depth → half-res linear depth)
// ============================================================

@group(0) @binding(0) var depth_input: texture_depth_2d;
@group(0) @binding(1) var linear_depth_output: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: GtaoParams;
@group(0) @binding(3) var<uniform> camera: CameraParams;

fn linearize_depth(d: f32) -> f32 {
    // GTAO depth linearization for standard depth [0, 1].
    let near = camera.near;
    let far = camera.far;
    return (near * far) / max(far - d * (far - near), 0.0001);
}

@compute @workgroup_size(8, 8)
fn prefilter_depth(@builtin(global_invocation_id) id: vec3<u32>) {
    let half_size = vec2<u32>(params.half_res);
    if (id.x >= half_size.x || id.y >= half_size.y) {
        return;
    }

    // Sample 2x2 block from full-res depth (average for stability).
    let base = id.xy * 2u;
    let d0 = textureLoad(depth_input, base, 0);
    let d1 = textureLoad(depth_input, base + vec2<u32>(1u, 0u), 0);
    let d2 = textureLoad(depth_input, base + vec2<u32>(0u, 1u), 0);
    let d3 = textureLoad(depth_input, base + vec2<u32>(1u, 1u), 0);

    let l0 = linearize_depth(d0);
    let l1 = linearize_depth(d1);
    let l2 = linearize_depth(d2);
    let l3 = linearize_depth(d3);

    let linear_z = 0.25 * (l0 + l1 + l2 + l3);

    textureStore(linear_depth_output, id.xy, vec4<f32>(linear_z, 0.0, 0.0, 0.0));
}

// ============================================================
// Entry point 2: GTAO main pass (multi-slice)
// ============================================================

@group(0) @binding(4) var linear_depth_input: texture_2d<f32>;
@group(0) @binding(5) var ao_output: texture_storage_2d<r32float, write>;

fn reconstruct_view_pos(uv: vec2<f32>, linear_z: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let view_xy = vec2<f32>(
        ndc.x * linear_z / camera.projection[0][0],
        ndc.y * linear_z / camera.projection[1][1]
    );
    return vec3<f32>(view_xy, linear_z);
}

fn clamp_coord(coord: vec2<i32>) -> vec2<i32> {
    let half_size = vec2<i32>(params.half_res);
    return clamp(coord, vec2<i32>(0), half_size - vec2<i32>(1));
}

fn load_linear_depth(coord: vec2<i32>) -> f32 {
    return textureLoad(linear_depth_input, clamp_coord(coord), 0).r;
}

fn calculate_edges(center_z: f32, left_z: f32, right_z: f32, top_z: f32, bottom_z: f32) -> vec4<f32> {
    var edges = vec4<f32>(left_z, right_z, top_z, bottom_z) - center_z;
    let slope_lr = (edges.y - edges.x) * 0.5;
    let slope_tb = (edges.w - edges.z) * 0.5;
    let edges_slope_adjusted = edges + vec4<f32>(slope_lr, -slope_lr, slope_tb, -slope_tb);
    edges = min(abs(edges), abs(edges_slope_adjusted));
    return clamp(1.25 - edges / max(center_z * 0.011, 0.0001), vec4<f32>(0.0), vec4<f32>(1.0));
}

fn reconstruct_normal(
    edges_lrtb: vec4<f32>,
    center_pos: vec3<f32>,
    pos_l: vec3<f32>,
    pos_r: vec3<f32>,
    pos_u: vec3<f32>,
    pos_d: vec3<f32>,
) -> vec3<f32> {
    let accepted = clamp(
        vec4<f32>(
            edges_lrtb.x * edges_lrtb.z,
            edges_lrtb.z * edges_lrtb.y,
            edges_lrtb.y * edges_lrtb.w,
            edges_lrtb.w * edges_lrtb.x
        ) + vec4<f32>(0.01),
        vec4<f32>(0.0),
        vec4<f32>(1.0)
    );

    let l = normalize(pos_l - center_pos);
    let r = normalize(pos_r - center_pos);
    let u = normalize(pos_u - center_pos);
    let d = normalize(pos_d - center_pos);

    let n =
        accepted.x * cross(l, u) +
        accepted.y * cross(u, r) +
        accepted.z * cross(r, d) +
        accepted.w * cross(d, l);

    let n_len = length(n);
    if (n_len < 0.0001) {
        return vec3<f32>(0.0);
    }
    return n / n_len;
}

const PI: f32 = 3.14159265359;
const PI_HALF: f32 = 1.57079632679;
const NUM_SLICES: u32 = 4u;

// Per-pixel spatial hash for noise — avoids visible tiling patterns
fn spatial_hash(pos: vec2<u32>, frame: u32) -> f32 {
    var h = pos.x * 73856093u ^ pos.y * 19349663u ^ frame * 83492791u;
    // Wang hash finalization for good distribution
    h = (h ^ 61u) ^ (h >> 16u);
    h = h + (h << 3u);
    h = h ^ (h >> 4u);
    h = h * 0x27d4eb2du;
    h = h ^ (h >> 15u);
    return f32(h & 0xFFFFu) / 65536.0;
}

@compute @workgroup_size(8, 8)
fn gtao_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let half_size = vec2<u32>(params.half_res);
    if (id.x >= half_size.x || id.y >= half_size.y) {
        return;
    }

    let coord = vec2<i32>(id.xy);
    let uv = (vec2<f32>(id.xy) + 0.5) * params.inv_half_res;

    let center_depth = load_linear_depth(coord);

    // Skip sky pixels
    if (center_depth >= camera.far * 0.99) {
        textureStore(ao_output, id.xy, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    let center_pos = reconstruct_view_pos(uv, center_depth);
    let view_vec = normalize(-center_pos);

    // Reconstruct normal from depth via cross-product of neighbors
    let depth_l = load_linear_depth(coord + vec2<i32>(-1, 0));
    let depth_r = load_linear_depth(coord + vec2<i32>(1, 0));
    let depth_u = load_linear_depth(coord + vec2<i32>(0, -1));
    let depth_d = load_linear_depth(coord + vec2<i32>(0, 1));

    let pos_l = reconstruct_view_pos(uv + vec2<f32>(-params.inv_half_res.x, 0.0), depth_l);
    let pos_r = reconstruct_view_pos(uv + vec2<f32>(params.inv_half_res.x, 0.0), depth_r);
    let pos_u = reconstruct_view_pos(uv + vec2<f32>(0.0, -params.inv_half_res.y), depth_u);
    let pos_d = reconstruct_view_pos(uv + vec2<f32>(0.0, params.inv_half_res.y), depth_d);

    let edges_lrtb = calculate_edges(center_depth, depth_l, depth_r, depth_u, depth_d);
    var normal = reconstruct_normal(edges_lrtb, center_pos, pos_l, pos_r, pos_u, pos_d);
    if (length(normal) < 0.0001) {
        textureStore(ao_output, id.xy, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    // Keep the normal in the visible hemisphere to avoid horizon inversion.
    if (dot(normal, view_vec) < 0.0) {
        normal = -normal;
    }

    // Per-pixel random rotation offset (changes each frame for temporal smoothing)
    let noise = spatial_hash(id.xy, params.frame_index);

    // Project world-space radius to screen pixels at this depth.
    let proj_scale = params.half_res.y * 0.5 * camera.projection[1][1];
    let radius_pixels = min(params.radius * proj_scale / center_depth, 64.0);
    if (radius_pixels < 1.0) {
        textureStore(ao_output, id.xy, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    let falloff_start = params.falloff_start;
    let falloff_range = max(1.0 - falloff_start, 0.001);
    let num_steps = max(params.num_steps, 1u);

    // GTAO visibility integrated across multiple horizon slices.
    var visibility_sum: f32 = 0.0;

    for (var slice = 0u; slice < NUM_SLICES; slice++) {
        // Slice angle as in Jimenez et al.
        let angle = (f32(slice) + noise) * PI / f32(NUM_SLICES);
        let cos_phi = cos(angle);
        let sin_phi = sin(angle);
        // UV Y increases downward, while view-space Y increases upward.
        // Keep sampling direction consistent with the view-space direction basis.
        let slice_dir = vec2<f32>(cos_phi, -sin_phi);

        let direction_vec = vec3<f32>(cos_phi, sin_phi, 0.0);
        let ortho_direction_vec = direction_vec - dot(direction_vec, view_vec) * view_vec;
        if (length(ortho_direction_vec) < 0.0001) {
            visibility_sum += 1.0;
            continue;
        }

        let axis_vec = normalize(cross(ortho_direction_vec, view_vec));
        let projected_normal_vec = normal - axis_vec * dot(normal, axis_vec);
        var projected_normal_len = length(projected_normal_vec);
        if (projected_normal_len < 0.0001) {
            visibility_sum += 1.0;
            continue;
        }
        // XeGTAO uses a slight bias toward 1 to avoid over-darkening on high slopes.
        projected_normal_len = mix(projected_normal_len, 1.0, 0.05);

        let sign_norm = select(-1.0, 1.0, dot(ortho_direction_vec, projected_normal_vec) >= 0.0);
        let cos_norm = clamp(dot(projected_normal_vec, view_vec) / projected_normal_len, 0.0, 1.0);
        let n = sign_norm * acos(cos_norm);

        let low_horizon_cos0 = cos(n + PI_HALF);
        let low_horizon_cos1 = cos(n - PI_HALF);
        var horizon_cos0 = low_horizon_cos0;
        var horizon_cos1 = low_horizon_cos1;

        for (var step = 1u; step <= num_steps; step++) {
            let t = (f32(step) - 0.5 + 0.5 * noise) / f32(num_steps);
            let s = t * t;
            let offset = max(s * radius_pixels, 1.0) * slice_dir;

            let uv_pos = uv + offset * params.inv_half_res;
            if (uv_pos.x >= 0.0 && uv_pos.x < 1.0 && uv_pos.y >= 0.0 && uv_pos.y < 1.0) {
                let sc = vec2<i32>(uv_pos * params.half_res);
                let sd = load_linear_depth(sc);
                let sp = reconstruct_view_pos(uv_pos, sd);
                let sample_delta = sp - center_pos;
                let sample_dist = length(sample_delta);
                if (sample_dist > 0.0001) {
                    var shc = dot(sample_delta / sample_dist, view_vec);
                    let dist_frac = sample_dist / max(params.radius, 0.0001);
                    let dist_weight = clamp(1.0 - max(dist_frac - falloff_start, 0.0) / falloff_range, 0.0, 1.0);
                    shc = mix(low_horizon_cos0, shc, dist_weight);
                    horizon_cos0 = max(horizon_cos0, shc);
                }
            }

            let uv_neg = uv - offset * params.inv_half_res;
            if (uv_neg.x >= 0.0 && uv_neg.x < 1.0 && uv_neg.y >= 0.0 && uv_neg.y < 1.0) {
                let sc = vec2<i32>(uv_neg * params.half_res);
                let sd = load_linear_depth(sc);
                let sp = reconstruct_view_pos(uv_neg, sd);
                let sample_delta = sp - center_pos;
                let sample_dist = length(sample_delta);
                if (sample_dist > 0.0001) {
                    var shc = dot(sample_delta / sample_dist, view_vec);
                    let dist_frac = sample_dist / max(params.radius, 0.0001);
                    let dist_weight = clamp(1.0 - max(dist_frac - falloff_start, 0.0) / falloff_range, 0.0, 1.0);
                    shc = mix(low_horizon_cos1, shc, dist_weight);
                    horizon_cos1 = max(horizon_cos1, shc);
                }
            }
        }

        // GTAO arc integration (Jimenez 2016, Eq. 7/8-style formulation).
        let h0 = -acos(clamp(horizon_cos1, -1.0, 1.0));
        let h1 =  acos(clamp(horizon_cos0, -1.0, 1.0));
        let iarc0 = (cos_norm + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) * 0.25;
        let iarc1 = (cos_norm + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) * 0.25;
        let local_visibility = clamp(projected_normal_len * (iarc0 + iarc1), 0.0, 1.0);
        visibility_sum += local_visibility;
    }

    var ao = visibility_sum / f32(NUM_SLICES);
    ao = clamp(ao, 0.0, 1.0);

    textureStore(ao_output, id.xy, vec4<f32>(ao, 0.0, 0.0, 0.0));
}

// ============================================================
// Entry point 3 & 4: Bilateral blur (H and V) — 9-tap
// ============================================================

@group(0) @binding(6) var ao_blur_input: texture_2d<f32>;
@group(0) @binding(7) var ao_blur_output: texture_storage_2d<r32float, write>;

fn bilateral_weight(center_depth: f32, sample_depth: f32) -> f32 {
    let diff = abs(center_depth - sample_depth);
    let threshold = center_depth * 0.05;
    return exp(-(diff * diff) / max(threshold * threshold, 0.0001));
}

@compute @workgroup_size(8, 8)
fn blur_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let half_size = vec2<u32>(params.half_res);
    if (id.x >= half_size.x || id.y >= half_size.y) {
        return;
    }

    let coord = vec2<i32>(id.xy);
    let center_depth = load_linear_depth(coord);

    // 9-tap Gaussian kernel (sigma ≈ 2)
    let offsets = array<i32, 9>(-4, -3, -2, -1, 0, 1, 2, 3, 4);
    let weights = array<f32, 9>(
        0.028, 0.066, 0.122, 0.176, 0.216, 0.176, 0.122, 0.066, 0.028
    );

    var ao_sum: f32 = 0.0;
    var weight_sum: f32 = 0.0;

    for (var i = 0; i < 9; i++) {
        let sc = clamp(coord + vec2<i32>(offsets[i], 0), vec2<i32>(0), vec2<i32>(half_size) - 1);
        let sd = load_linear_depth(sc);
        let sa = textureLoad(ao_blur_input, sc, 0).r;
        let bw = bilateral_weight(center_depth, sd);
        let w = weights[i] * bw;
        ao_sum += sa * w;
        weight_sum += w;
    }

    textureStore(ao_blur_output, id.xy, vec4<f32>(ao_sum / max(weight_sum, 0.0001), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn blur_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let half_size = vec2<u32>(params.half_res);
    if (id.x >= half_size.x || id.y >= half_size.y) {
        return;
    }

    let coord = vec2<i32>(id.xy);
    let center_depth = load_linear_depth(coord);

    let offsets = array<i32, 9>(-4, -3, -2, -1, 0, 1, 2, 3, 4);
    let weights = array<f32, 9>(
        0.028, 0.066, 0.122, 0.176, 0.216, 0.176, 0.122, 0.066, 0.028
    );

    var ao_sum: f32 = 0.0;
    var weight_sum: f32 = 0.0;

    for (var i = 0; i < 9; i++) {
        let sc = clamp(coord + vec2<i32>(0, offsets[i]), vec2<i32>(0), vec2<i32>(half_size) - 1);
        let sd = load_linear_depth(sc);
        let sa = textureLoad(ao_blur_input, sc, 0).r;
        let bw = bilateral_weight(center_depth, sd);
        let w = weights[i] * bw;
        ao_sum += sa * w;
        weight_sum += w;
    }

    textureStore(ao_blur_output, id.xy, vec4<f32>(ao_sum / max(weight_sum, 0.0001), 0.0, 0.0, 0.0));
}

// ============================================================
// Entry point 5: Temporal accumulation
// ============================================================

struct PrevViewProjection {
    matrix: mat4x4<f32>,
}

@group(0) @binding(8) var ao_history: texture_2d<f32>;
@group(0) @binding(9) var ao_temporal_output: texture_storage_2d<r32float, write>;
@group(0) @binding(10) var<uniform> prev_vp: PrevViewProjection;

@compute @workgroup_size(8, 8)
fn temporal_accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let half_size = vec2<u32>(params.half_res);
    if (id.x >= half_size.x || id.y >= half_size.y) {
        return;
    }

    let coord = vec2<i32>(id.xy);
    let uv = (vec2<f32>(id.xy) + 0.5) * params.inv_half_res;

    let current_ao = textureLoad(ao_blur_input, coord, 0).r;
    let center_depth = load_linear_depth(coord);

    // Skip sky
    if (center_depth >= camera.far * 0.99) {
        textureStore(ao_temporal_output, id.xy, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    // Reproject to previous frame
    let view_pos = reconstruct_view_pos(uv, center_depth);
    let world_pos = camera.inv_view * vec4<f32>(view_pos, 1.0);
    let prev_clip = prev_vp.matrix * world_pos;
    let prev_ndc = prev_clip.xy / prev_clip.w;
    let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);

    var blend_factor = params.temporal_blend;
    if (params.frame_index < 2u) {
        blend_factor = 1.0;
    }
    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0) {
        blend_factor = 1.0;
    }

    let prev_coord = vec2<i32>(prev_uv * params.half_res);
    let clamped_prev = clamp(prev_coord, vec2<i32>(0), vec2<i32>(half_size) - 1);
    var history_ao = textureLoad(ao_history, clamped_prev, 0).r;

    // Neighborhood clamping (3x3 min/max)
    var ao_min: f32 = 1.0;
    var ao_max: f32 = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nc = clamp(coord + vec2<i32>(dx, dy), vec2<i32>(0), vec2<i32>(half_size) - 1);
            let nao = textureLoad(ao_blur_input, nc, 0).r;
            ao_min = min(ao_min, nao);
            ao_max = max(ao_max, nao);
        }
    }
    history_ao = clamp(history_ao, ao_min, ao_max);

    // No previous-depth history texture is available in this path, so avoid
    // invalid disocclusion tests against current-frame depth at reprojected UV.

    let result = mix(history_ao, current_ao, blend_factor);
    textureStore(ao_temporal_output, id.xy, vec4<f32>(result, 0.0, 0.0, 0.0));
}
