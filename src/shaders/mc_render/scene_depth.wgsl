// Water shader (mc_render), part: depth-buffer reads, smooth G-buffer normals

// Linearize depth from depth buffer (reverse-Z or standard)
fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

// === Physical screen-space refraction ===
// Raw depth at or past this is the environment backdrop (drawn at 0.9999) or
// nothing at all: whatever lies behind is infinitely far.
const BACKDROP_DEPTH: f32 = 0.9998;

fn texel_at(uv: vec2<f32>, dims: vec2<u32>) -> vec2<i32> {
    return clamp(vec2<i32>(uv * vec2<f32>(dims)), vec2<i32>(0), vec2<i32>(dims) - 1);
}

fn background_depth_at(uv: vec2<f32>) -> f32 {
    return textureLoad(background_depth_tex, texel_at(uv, textureDimensions(background_depth_tex)), 0);
}

// Neighbouring texels whose raw depths differ by more than this fraction of
// (1 - depth), roughly the relative distance, belong to different surfaces
const DEPTH_EDGE_REL: f32 = 0.05;

// Depth at a screen point for ray-march crossing tests, bilinear between texel
// centres. Raw depth of a plane is affine in screen space, so floors, walls and
// the ground come back exact. With per-texel depth every crossing a march finds
// snaps to the texel grid, and a refraction that magnifies repeats the same
// lookup on neighbouring pixel rows: a staircase that beats against the pixel
// grid as banding. Across a silhouette, interpolating would invent a surface
// in between, so there it falls back to the nearest texel.
fn depth_smooth(tex: texture_depth_2d, uv: vec2<f32>) -> f32 {
    let dims = vec2<i32>(textureDimensions(tex));
    let p = uv * vec2<f32>(dims) - 0.5;
    let f = fract(p);
    let i0 = clamp(vec2<i32>(floor(p)), vec2<i32>(0), dims - 1);
    let i1 = clamp(vec2<i32>(floor(p)) + 1, vec2<i32>(0), dims - 1);
    let d00 = textureLoad(tex, i0, 0);
    let d10 = textureLoad(tex, vec2<i32>(i1.x, i0.y), 0);
    let d01 = textureLoad(tex, vec2<i32>(i0.x, i1.y), 0);
    let d11 = textureLoad(tex, i1, 0);
    let d_min = min(min(d00, d10), min(d01, d11));
    let d_max = max(max(d00, d10), max(d01, d11));
    if (d_max - d_min > DEPTH_EDGE_REL * max(1.0 - d_min, 1e-6)) {
        return textureLoad(tex, texel_at(uv, vec2<u32>(dims)), 0);
    }
    return mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
}

// Back-face normal at a screen point, bilinear between texel centres over the
// texels that hold one (w = 1 if any did). Per-texel normals snapped every
// exit a trace finds to the texel grid, which a reflection that magnifies the
// surface stretched into stripes.
fn back_normal_smooth(uv: vec2<f32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(back_normal_tex));
    let p = uv * vec2<f32>(dims) - 0.5;
    let f = fract(p);
    let i0 = clamp(vec2<i32>(floor(p)), vec2<i32>(0), dims - 1);
    let i1 = clamp(vec2<i32>(floor(p)) + 1, vec2<i32>(0), dims - 1);
    let n00 = textureLoad(back_normal_tex, i0, 0);
    let n10 = textureLoad(back_normal_tex, vec2<i32>(i1.x, i0.y), 0);
    let n01 = textureLoad(back_normal_tex, vec2<i32>(i0.x, i1.y), 0);
    let n11 = textureLoad(back_normal_tex, i1, 0);
    let w00 = (1.0 - f.x) * (1.0 - f.y) * step(0.5, n00.w);
    let w10 = f.x * (1.0 - f.y) * step(0.5, n10.w);
    let w01 = (1.0 - f.x) * f.y * step(0.5, n01.w);
    let w11 = f.x * f.y * step(0.5, n11.w);
    let n = n00.xyz * w00 + n10.xyz * w10 + n01.xyz * w01 + n11.xyz * w11;
    if (w00 + w10 + w01 + w11 <= 0.0 || dot(n, n) < 1e-8) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(normalize(n), 1.0);
}

// Front-face normal at a screen point, bilinear between texel centres over the
// texels that hold one (w = 1 if any did). The front G-buffer keeps the raw MC
// winding, so each texel is first turned outward: a surface seen from the
// camera's side faces against the camera ray.
fn front_normal_smooth(uv: vec2<f32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(front_normal_tex));
    let p = uv * vec2<f32>(dims) - 0.5;
    let f = fract(p);
    let i0 = clamp(vec2<i32>(floor(p)), vec2<i32>(0), dims - 1);
    let i1 = clamp(vec2<i32>(floor(p)) + 1, vec2<i32>(0), dims - 1);
    let ray = screen_to_world(uv, 0.5) - camera.camera_pos;
    var texels = array<vec2<i32>, 4>(i0, vec2<i32>(i1.x, i0.y), vec2<i32>(i0.x, i1.y), i1);
    var weights = array<f32, 4>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    var n = vec3<f32>(0.0);
    var w_sum = 0.0;
    for (var i = 0; i < 4; i++) {
        let t = textureLoad(front_normal_tex, texels[i], 0);
        if (t.w > 0.5) {
            n += select(t.xyz, -t.xyz, dot(t.xyz, ray) > 0.0) * weights[i];
            w_sum += weights[i];
        }
    }
    if (w_sum <= 0.0 || dot(n, n) < 1e-8) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(normalize(n), 1.0);
}

// How close to the depth-buffer surface at its pixel a point must be to count
// as touching it, relative to its distance from the camera ((1 - raw depth)
// is ~ 1 / view distance)
const SURFACE_CONTACT_REL: f32 = 0.05;

// Is screen point q (uv, raw depth) behind the opaque surface seen at its
// pixel, i.e. has a march reached it? The depth buffer has no thickness, so
// for a body this would also catch rays passing BEHIND it (as the camera sees
// it) and land them on its silhouette edge: whole regions of a reflection or
// refraction then sampled those edge pixels (dithered stripes). The
// procedural shapes are intersected exactly instead (ray_body_hit), so their
// pixels are skipped here. Floors, walls and the ground are solid: behind is
// a hit.
fn behind_background(q: vec3<f32>) -> bool {
    let bg = depth_smooth(background_depth_tex, q.xy);
    prb_bg = bg;
    return q.z >= bg && (water.body_count == 0u || !on_analytic_body(screen_to_world(q.xy, bg)));
}

// World point seen at screen uv with raw hardware depth. Goes through the real
// inverse matrices, so it holds whatever depth convention the projection uses.
fn screen_to_world(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let view = camera.inv_projection * ndc;
    return (camera.inv_view * vec4<f32>(view.xyz / view.w, 1.0)).xyz;
}
