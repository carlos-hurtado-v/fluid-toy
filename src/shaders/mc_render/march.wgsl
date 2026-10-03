// Water shader (mc_render), part: screen-space march onto the background depth

// March a ray from `origin` (in or leaving the water) until it passes
// behind the background depth buffer, then bisect to the crossing. Linear
// steps reach several times the straight-line distance to the surface first
// seen at `uv0`: a refracted ray skimming a floor lands far behind that first
// guess, which a fixed-point iteration overshoots toward the horizon.
// Returns xy = screen uv of the surface reached, z = 1 if the ray reached one
// (0: no crossing, xy = uv0).
const MARCH_STEPS: i32 = 20;
const MARCH_REFINE: i32 = 8;
const MARCH_REACH: f32 = 4.0;

// Screen uv (y down) and raw depth of a world point
fn screen_point(p: vec3<f32>) -> vec3<f32> {
    let clip = camera.projection * camera.view * vec4<f32>(p, 1.0);
    let w = max(clip.w, 1e-4);
    return vec3<f32>(clip.x / w * 0.5 + 0.5, 0.5 - clip.y / w * 0.5, clip.z / w);
}

// Distance along the ray at which the last march_to_background found its
// surface (-1: none)
var<private> march_dist: f32 = -1.0;
// In-water path of the refracted ray where it is known better than from the
// depth buffers at the pixel (-1: not set). The medium otherwise measures to
// the nearer of back face and opaque surface along the VIEW ray, which is
// wrong where the refracted ray misses the body the view ray sees: that
// stretch kept the body's short path and stood out as a tinted ghost of it.
var<private> refracted_path: f32 = -1.0;
var<private> missed_inside: bool = false;

fn march_to_background(origin: vec3<f32>, dir: vec3<f32>, uv0: vec2<f32>, depth0: f32) -> vec3<f32> {
    march_dist = -1.0;
    var reach = MARCH_REACH * distance(origin, screen_to_world(uv0, depth0)) + 0.1;
    // A body on the way ends the ray exactly where the depth buffer can't see
    let t_body = ray_body_hit(origin, dir, reach);
    if (t_body > 0.0) {
        reach = t_body;
    }
    probe_event(PRB_MARCH_BEGIN, origin, dir, reach);
    var lo = 0.0;
    var hi = -1.0;
    for (var k = 1; k <= MARCH_STEPS; k++) {
        let s = reach * f32(k) / f32(MARCH_STEPS);
        let q = screen_point(origin + dir * s);
        if (any(q.xy < vec2<f32>(0.0)) || any(q.xy > vec2<f32>(1.0))) {
            probe_event(PRB_MARCH, q, vec3<f32>(-1.0, -1.0, 0.0), s);
            break;  // left the screen
        }
        let behind = behind_background(q);
        probe_event(PRB_MARCH, q, vec3<f32>(prb_bg, f32(behind), 0.0), s);
        if (behind) {
            hi = s;
            break;
        }
        lo = s;
    }
    if (hi < 0.0 && t_body > 0.0) {
        dbg_body = true;
        let body_uv = screen_point(origin + dir * t_body).xy;
        march_dist = t_body;
        probe_event(PRB_MARCH_END, vec3<f32>(body_uv, 1.0), vec3<f32>(lo, hi, t_body), 0.0);
        return vec3<f32>(body_uv, 1.0);
    }
    if (hi < 0.0) {
        probe_event(PRB_MARCH_END, vec3<f32>(uv0, 0.0), vec3<f32>(lo, hi, t_body), 0.0);
        return vec3<f32>(uv0, 0.0);
    }
    for (var k = 0; k < MARCH_REFINE; k++) {
        let mid = 0.5 * (lo + hi);
        let q = screen_point(origin + dir * mid);
        let behind = behind_background(q);
        probe_event(PRB_MARCH_REFINE, q, vec3<f32>(prb_bg, f32(behind), 0.0), mid);
        if (behind) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let hit_uv = screen_point(origin + dir * hi).xy;
    march_dist = hi;
    probe_event(PRB_MARCH_END, vec3<f32>(hit_uv, 1.0), vec3<f32>(lo, hi, t_body), 0.0);
    return vec3<f32>(hit_uv, 1.0);
}
