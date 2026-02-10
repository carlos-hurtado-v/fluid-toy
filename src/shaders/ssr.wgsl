// Screen-Space Reflections (SSR) compute shader
// Ray-marches in screen space against the background depth buffer
// using water surface normals to find reflections of scene geometry.

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

struct SsrParams {
    max_distance: f32,
    thickness: f32,
    enabled: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var front_depth: texture_depth_2d;
@group(0) @binding(2) var bg_depth: texture_depth_2d;
@group(0) @binding(3) var bg_color: texture_2d<f32>;
@group(0) @binding(4) var depth_samp: sampler;
@group(0) @binding(5) var color_samp: sampler;
@group(0) @binding(6) var<uniform> ssr_params: SsrParams;
@group(0) @binding(7) var ssr_output: texture_storage_2d<rgba16float, write>;

const MAX_STEPS: u32 = 48u;
const BINARY_STEPS: u32 = 6u;

// Reconstruct view-space position from screen UV + depth buffer value
fn view_pos_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // UV [0,1] → NDC [-1,1], flip Y (UV Y-down → clip Y-up)
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let clip = vec4<f32>(ndc.x, -ndc.y, ndc.z, 1.0);
    let view_h = camera.inv_projection * clip;
    return view_h.xyz / view_h.w;
}

// Project view-space position → screen UV + NDC z
fn project_view_to_screen(pos: vec3<f32>) -> vec3<f32> {
    let clip = camera.projection * vec4<f32>(pos, 1.0);
    if (clip.w <= 0.0) {
        return vec3<f32>(-1.0, -1.0, -1.0); // behind camera
    }
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    return vec3<f32>(uv, ndc.z);
}

// Linearize depth buffer value to view-space distance
fn linearize_depth(d: f32) -> f32 {
    return camera.near * camera.far / (camera.far - d * (camera.far - camera.near));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(front_depth);
    let pixel = vec2<i32>(gid.xy);

    // Out-of-bounds or disabled → write zero
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    if (ssr_params.enabled == 0u) {
        textureStore(ssr_output, pixel, vec4<f32>(0.0));
        return;
    }

    let screen_size = vec2<f32>(f32(dims.x), f32(dims.y));
    let uv = (vec2<f32>(gid.xy) + 0.5) / screen_size;

    // Sample water front depth
    let water_depth = textureLoad(front_depth, pixel, 0);

    // Skip if no water at this pixel
    if (water_depth >= 0.9999) {
        textureStore(ssr_output, pixel, vec4<f32>(0.0));
        return;
    }

    // Reconstruct view-space position of water surface
    let view_pos = view_pos_from_depth(uv, water_depth);

    // Reconstruct view-space normal from screen-space depth derivatives
    let depth_r = textureLoad(front_depth, min(pixel + vec2<i32>(1, 0), vec2<i32>(dims) - 1), 0);
    let depth_l = textureLoad(front_depth, max(pixel - vec2<i32>(1, 0), vec2<i32>(0)), 0);
    let depth_u = textureLoad(front_depth, max(pixel - vec2<i32>(0, 1), vec2<i32>(0)), 0);
    let depth_d = textureLoad(front_depth, min(pixel + vec2<i32>(0, 1), vec2<i32>(dims) - 1), 0);

    let texel = vec2<f32>(1.0) / screen_size;
    let pos_r = view_pos_from_depth(uv + vec2<f32>(texel.x, 0.0), depth_r);
    let pos_l = view_pos_from_depth(uv - vec2<f32>(texel.x, 0.0), depth_l);
    let pos_u = view_pos_from_depth(uv - vec2<f32>(0.0, texel.y), depth_u);
    let pos_d = view_pos_from_depth(uv + vec2<f32>(0.0, texel.y), depth_d);

    // Pick smallest derivative per axis to avoid depth discontinuity edges
    var ddx_v = pos_r - view_pos;
    let ddx_l = view_pos - pos_l;
    if (abs(ddx_l.z) < abs(ddx_v.z)) { ddx_v = ddx_l; }
    var ddy_v = pos_d - view_pos;
    let ddy_u = view_pos - pos_u;
    if (abs(ddy_u.z) < abs(ddy_v.z)) { ddy_v = ddy_u; }

    let normal_view = normalize(cross(ddy_v, ddx_v));

    // View direction: from camera origin to surface point (view space)
    let view_dir = normalize(view_pos);

    // Reflect
    let reflect_dir = normalize(reflect(view_dir, normal_view));

    // === Screen-space ray march ===
    // Choose a view-space end point along the reflected ray.
    // If the ray goes toward the camera (positive z), clip to near plane.
    let max_dist = ssr_params.max_distance;
    var ray_len = max_dist;
    if (reflect_dir.z > 0.0) {
        // Ray heads toward camera; clip to where it reaches near plane
        let t_near = (-camera.near - view_pos.z) / reflect_dir.z;
        if (t_near <= 0.0) {
            textureStore(ssr_output, pixel, vec4<f32>(0.0));
            return;
        }
        ray_len = min(ray_len, t_near * 0.95); // slight margin
    }

    // Small offset to avoid self-intersection
    let ray_start = view_pos + reflect_dir * 0.01;
    let ray_end = view_pos + reflect_dir * ray_len;

    // Project both endpoints to screen space
    let p0 = project_view_to_screen(ray_start);
    let p1 = project_view_to_screen(ray_end);

    // If either projects behind camera, bail
    if (p0.x < -0.5 || p1.x < -0.5) {
        textureStore(ssr_output, pixel, vec4<f32>(0.0));
        return;
    }

    // March in screen space: interpolate UV and 1/z for perspective-correct depth
    let start_px = p0.xy * screen_size;
    let end_px = p1.xy * screen_size;
    let delta_px = end_px - start_px;
    let px_len = max(abs(delta_px.x), abs(delta_px.y));

    if (px_len < 1.0) {
        textureStore(ssr_output, pixel, vec4<f32>(0.0));
        return;
    }

    // Number of steps = pixel distance, capped
    let step_count = min(MAX_STEPS, u32(ceil(px_len)));
    let inv_steps = 1.0 / f32(step_count);

    // For perspective-correct interpolation: interpolate 1/z and UV/z
    let z0 = 1.0 / max(linearize_depth(p0.z), 0.001);
    let z1 = 1.0 / max(linearize_depth(p1.z), 0.001);
    let uv0_over_z = p0.xy * z0;
    let uv1_over_z = p1.xy * z1;

    let thickness = ssr_params.thickness;

    var hit = false;
    var hit_uv = vec2<f32>(0.0);
    var hit_frac = 0.0;
    var prev_diff = 0.0;

    for (var i = 1u; i <= step_count; i++) {
        let t = f32(i) * inv_steps;

        // Perspective-correct interpolation of UV and depth
        let inv_z = mix(z0, z1, t);
        let interp_uv = mix(uv0_over_z, uv1_over_z, t) / inv_z;
        let ray_linear = 1.0 / inv_z;

        // Check screen bounds
        if (interp_uv.x < 0.0 || interp_uv.x > 1.0 || interp_uv.y < 0.0 || interp_uv.y > 1.0) {
            break;
        }

        // Sample background depth
        let sample_depth_raw = textureSampleLevel(bg_depth, depth_samp, interp_uv, 0);
        let scene_linear = linearize_depth(sample_depth_raw);

        let depth_diff = ray_linear - scene_linear;

        // Hit: ray crossed behind scene surface, within thickness
        if (depth_diff > 0.0 && depth_diff < thickness) {
            hit = true;
            hit_uv = interp_uv;
            hit_frac = t;
            break;
        }

        prev_diff = depth_diff;
    }

    // Binary refinement: narrow down the exact intersection point
    if (hit) {
        var lo = max(hit_frac - inv_steps, 0.0);
        var hi = hit_frac;

        for (var j = 0u; j < BINARY_STEPS; j++) {
            let mid = (lo + hi) * 0.5;

            let inv_z = mix(z0, z1, mid);
            let interp_uv_mid = mix(uv0_over_z, uv1_over_z, mid) / inv_z;
            let ray_linear = 1.0 / inv_z;

            if (interp_uv_mid.x < 0.0 || interp_uv_mid.x > 1.0 || interp_uv_mid.y < 0.0 || interp_uv_mid.y > 1.0) {
                hi = mid;
                continue;
            }

            let sd = textureSampleLevel(bg_depth, depth_samp, interp_uv_mid, 0);
            let scene_linear = linearize_depth(sd);
            let depth_diff = ray_linear - scene_linear;

            if (depth_diff > 0.0 && depth_diff < thickness) {
                hi = mid;
                hit_uv = interp_uv_mid;
            } else {
                lo = mid;
            }
        }
    }

    if (!hit) {
        textureStore(ssr_output, pixel, vec4<f32>(0.0));
        return;
    }

    // Sample background color at hit point
    let reflection = textureSampleLevel(bg_color, color_samp, hit_uv, 0.0).rgb;

    // Confidence: fade at screen edges and for long rays
    var confidence = 1.0;

    let edge_fade = 0.1;
    confidence *= smoothstep(0.0, edge_fade, hit_uv.x) * smoothstep(1.0, 1.0 - edge_fade, hit_uv.x);
    confidence *= smoothstep(0.0, edge_fade, hit_uv.y) * smoothstep(1.0, 1.0 - edge_fade, hit_uv.y);

    // Fade for distant reflections
    confidence *= 1.0 - smoothstep(0.5, 1.0, hit_frac);

    confidence = clamp(confidence, 0.0, 1.0);

    textureStore(ssr_output, pixel, vec4<f32>(reflection, confidence));
}
