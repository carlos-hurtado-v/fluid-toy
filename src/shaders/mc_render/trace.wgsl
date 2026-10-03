// Water shader (mc_render), part: in-water ray tracing (screen-space sample test, both tracers)

// What a point inside the water would run into, at screen point q (uv, raw
// depth): 2 = an opaque surface (floor or ground, a body, pool walls), 1 = it's
// out of the water, 0 = still in it. In-water is judged against the
// nearest back face behind q's pixel, which ends the first stretch of water
// the camera sees there. With no front depth to check against, that only
// holds for points on rays heading away from the camera: true of every ray
// refracted in through the tank's walls or free surface, and of its bounces
// off walls, floor and surface (each keeps the component that leads away).
// Rays bent by drops and crests can break it.
// rendering.mc_front_face_exit: how far in front of the nearest surface at its
// pixel a sample must lie to count as out of the water, relative to
// (1 - depth) (~ relative distance). Rays start ON the front surface, and the
// bilinear depth of a curved surface is not exact between texel centres.
const FRONT_EXIT_REL: f32 = 0.002;

// Returns (kind, back depth read at q or -1). Kinds: 0 in the water, 1 out
// (behind the nearest back face, or no water behind the pixel at all), 2 an
// opaque surface, 3 out through a surface facing the camera (front exits on).
fn trace_event(q: vec3<f32>) -> vec2<f32> {
    let bg = depth_smooth(background_depth_tex, q.xy);
    prb_bg = bg;
    prb_front = -1.0;
    if (q.z >= bg) {
        // Hidden behind a sphere or box: ray_body_hit owns hits on it, and the
        // back face at this pixel is the water in front of the body (where the
        // camera sees the body through it), not the water the sample is in.
        // Read as "out of the water", rays passing behind a floating body
        // exited there with that surface's normal and scattered.
        if (water.body_count > 0u && on_analytic_body(screen_to_world(q.xy, bg))) {
            return vec2<f32>(0.0, -1.0);
        }
        return vec2<f32>(2.0, -1.0);
    }
    // In front of the nearest surface at its pixel: between the camera and
    // everything there, so not in the water whatever that surface is (water
    // of either winding, a body, a pool wall). The back-face test alone sees
    // a ray rising through a free surface viewed from above only where the
    // sample leaves the water's outline on screen, and takes the normal of
    // whatever back face is there (the far rim).
    if (water.front_exit != 0u) {
        let front = depth_smooth(front_depth_tex, q.xy);
        prb_front = front;
        if (front - q.z > FRONT_EXIT_REL * max(1.0 - front, 1e-6)) {
            return vec2<f32>(3.0, -1.0);
        }
    }
    let back = depth_smooth(back_depth_tex, q.xy);
    if (back >= 1.0 || q.z > back) {
        // Well behind the film of water in front of a sphere or box (the MC
        // surface stops short of a body, so that film's outline is a little
        // wider than the body's own): hidden like a sample behind the body
        // itself. Read as out of the water, rays passing behind a body just
        // outside its outline exited through the film with its normal.
        // Just behind the film is a real way out (a thin sheet on the body)
        if (back < 1.0 && water.body_count > 0u
            && q.z - back > SILHOUETTE_BEHIND_REL * (1.0 - back)
            && analytic_body_distance(screen_to_world(q.xy, back)) < BODY_WET_GAP) {
            return vec2<f32>(0.0, -1.0);
        }
        return vec2<f32>(1.0, back);
    }
    return vec2<f32>(0.0, back);
}

struct TraceEvent {
    // trace_event kind of the first event (0 = none: the ray reaches max_dist)
    kind: f32,
    // Last distance still in the water, and first distance at the event
    dist_in: f32,
    dist: f32,
    // Screen uv at the event
    uv: vec2<f32>,
}

// rendering.mc_silhouette_exit: what an out-of-water verdict that comes from a
// back-face silhouette means. The in-water test sees only the nearest back
// face at a sample's pixel, so a ray that slips BEHIND a nearer layer of water
// surface (as the camera sees it: a crater wall, a crest) reads as out of the
// water without crossing anything. Bisecting a real crossing closes onto a
// continuous back face; across a silhouette the back depth jumps and stays
// jumped. As an exit (old behaviour) its normal is blended across the two
// unrelated layers and neighbouring pixels flip between outcomes: stripes,
// strongest in mirrors (snap_001). Continuing behind the layer is physically
// right there (the ray is measured to be in water behind a crater), but its
// switch still follows the near layer's texel-precision outline, which a
// mirror magnifies into stair steps (snap_003): opt-in until both outcomes can
// be blended across that outline.
const SILHOUETTE_EXIT: u32 = 0u;       // treat it as an exit (old, default)
const SILHOUETTE_CONTINUE: u32 = 1u;   // behind the nearer layer counts as in the water

// How far behind the nearer layer (relative to (1 - depth), ~ relative
// distance) the out-of-water sample must lie for the step to count as a slip
// behind it. A back face seen nearly edge-on also jumps between texels
// (depth_smooth falls back to the nearest texel on it), but a ray crossing it
// stays within a few % of it: snap_003 ~2% (a real exit), snap_001's crater
// wall 16%+ (behind it by 0.56 m and more).
const SILHOUETTE_BEHIND_REL: f32 = 0.05;

// Is the step from an in-water sample (back depth back_in) to an out-of-water
// one (back_out, sample depth z_out) a slip behind a nearer back-face layer?
fn is_silhouette(back_in: f32, back_out: f32, z_out: f32) -> bool {
    return back_in > 0.0 && back_out >= 0.0
        && back_out < back_in - DEPTH_EDGE_REL * max(1.0 - back_in, 1e-6)
        && z_out - back_out > SILHOUETTE_BEHIND_REL * max(1.0 - back_out, 1e-6);
}

// A ray that passes out of sight behind a sphere or box (trace_event reads its
// hidden samples as in the water: nothing can be tested there) may be about to
// leave through the free surface: skimming under it, the gap to the back face
// closes steadily and reaches zero somewhere behind the body. Left to the
// coarse samples, the last one still in sight decided: just through the
// surface = an exit there, a hair short = no exit at all, on to the far wall.
// With samples ~20 cm apart that verdict flipped in steps across the image
// (snap_004: a stair-stepped edge between a mirror and the view straight
// through). Instead the gap and its closing rate are measured at the body's
// outline, the same place for every pixel, and the crossing is put where the
// gap runs out, if that is still behind the body.
// Stretch before the outline over which the closing rate is measured, as a
// fraction of the distance travelled, and its bounds (m)
const HIDDEN_RATE_SPAN: f32 = 0.2;
const HIDDEN_RATE_SPAN_MIN: f32 = 0.05;
const HIDDEN_RATE_SPAN_MAX: f32 = 0.3;
// The exit normal is read this far (texels) before the outline along the
// ray's screen track: right at it, the bilinear footprint takes in the back
// face in front of the body (the wet film around it)
const HIDDEN_NORMAL_BACKOFF: f32 = 2.0;

// Distance (m) along the view ray from screen point q (uv, raw depth) to the
// back face seen at its pixel (raw depth `back`): > 0 in front of it
fn back_face_gap(q: vec3<f32>, back: f32) -> f32 {
    return distance(screen_to_world(q.xy, back), camera.camera_pos)
        - distance(screen_to_world(q.xy, q.z), camera.camera_pos);
}

// The crossing behind a body for a ray last seen in the water at distance
// s_edge (screen point q_edge, back depth back_edge), just before its outline:
// kind 1 with the estimated distance, or kind 0 if the gap is not closing or
// outlasts the stretch out of sight.
fn hidden_crossing(
    origin: vec3<f32>,
    dir: vec3<f32>,
    s_edge: f32,
    q_edge: vec3<f32>,
    back_edge: f32,
    max_dist: f32,
) -> TraceEvent {
    let none = TraceEvent(0.0, 0.0, 0.0, vec2<f32>(0.0));
    let gap = back_face_gap(q_edge, back_edge);
    // (a ray out of sight within its first few cm has no track to judge by)
    let span = clamp(HIDDEN_RATE_SPAN * s_edge, HIDDEN_RATE_SPAN_MIN, HIDDEN_RATE_SPAN_MAX);
    if (s_edge < 2.0 * span) {
        probe_event(PRB_HIDDEN, vec3<f32>(s_edge, gap, 0.0), vec3<f32>(-1.0, 0.0, 0.0), 0.0);
        return none;
    }
    let qb = screen_point(origin + dir * (s_edge - span));
    let eb = trace_event(qb);
    probe_event4(PRB_TRACE_REFINE, qb, vec3<f32>(prb_bg, eb.y, eb.x), s_edge - span, vec4<f32>(prb_front, 0.0, 0.0, 0.0));
    if (gap <= 0.0 || eb.x > 0.5 || eb.y < 0.0) {
        probe_event(PRB_HIDDEN, vec3<f32>(s_edge, gap, 0.0), vec3<f32>(-1.0, 0.0, 0.0), 0.0);
        return none;
    }
    let rate = (back_face_gap(qb, eb.y) - gap) / span;
    if (rate <= 0.0) {
        // Diving away from the surface
        probe_event(PRB_HIDDEN, vec3<f32>(s_edge, gap, rate), vec3<f32>(-1.0, 0.0, 0.0), 0.0);
        return none;
    }
    let s_cross = s_edge + gap / rate;
    // Only where the crossing itself is out of sight: past the body, what is
    // seen decides
    var taken = false;
    if (s_cross < max_dist) {
        let qc = screen_point(origin + dir * s_cross);
        if (all(qc.xy >= vec2<f32>(0.0)) && all(qc.xy <= vec2<f32>(1.0))) {
            let ec = trace_event(qc);
            probe_event4(PRB_TRACE_REFINE, qc, vec3<f32>(prb_bg, ec.y, ec.x), s_cross, vec4<f32>(prb_front, 0.0, 0.0, 0.0));
            taken = ec.x < 0.5 && ec.y < 0.0;
        }
    }
    let dims = vec2<f32>(textureDimensions(back_normal_tex));
    let track = (q_edge.xy - qb.xy) * dims;
    var normal_uv = q_edge.xy;
    if (dot(track, track) > 1e-6) {
        normal_uv -= normalize(track) * HIDDEN_NORMAL_BACKOFF / dims;
    }
    probe_event(PRB_HIDDEN, vec3<f32>(s_edge, gap, rate), vec3<f32>(s_cross, normal_uv), select(select(0.0, 2.0, s_cross >= max_dist), 1.0, taken));
    if (!taken) {
        return none;
    }
    return TraceEvent(1.0, max(s_cross - 0.001, s_edge), s_cross, normal_uv);
}

// trace_in_water against the density field (rendering.mc_volume_trace): same
// sample spacing, but the verdict at each sample is the field's. A crossing
// is closed in on by bisection, then placed where the field between the last
// two samples reaches the iso value, so it moves smoothly from pixel to pixel.
fn trace_in_volume(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> TraceEvent {
    var lo = 0.0;
    var d_lo = 1.0;
    var entered = false;
    for (var k = 1; k <= TRACE_STEPS; k++) {
        let f = f32(k) / f32(TRACE_STEPS);
        let s = max_dist * f * f;
        let p = origin + dir * s;
        let ev = volume_event(p);
        probe_event(PRB_VTRACE, p, vec3<f32>(ev.y, prb_bg, ev.x), s);
        if (ev.x < 0.5) {
            entered = true;
            lo = s;
            d_lo = ev.y;
            continue;
        }
        if (ev.x < 1.5 && !entered && s < VOLUME_ENTRY_SLACK) {
            continue;
        }
        var hi = s;
        var d_hi = ev.y;
        var kind = ev.x;
        for (var r = 0; r < TRACE_REFINE; r++) {
            let mid = 0.5 * (lo + hi);
            let pm = origin + dir * mid;
            let e = volume_event(pm);
            probe_event(PRB_VTRACE_REFINE, pm, vec3<f32>(e.y, prb_bg, e.x), mid);
            if (e.x > 0.5) {
                hi = mid;
                d_hi = e.y;
                kind = e.x;
            } else {
                lo = mid;
                d_lo = e.y;
            }
        }
        if (kind < 1.5 && d_lo > d_hi) {
            hi = lo + (hi - lo) * clamp((d_lo - 1.0) / (d_lo - d_hi), 0.0, 1.0);
        }
        let event_uv = screen_point(origin + dir * hi).xy;
        probe_event(PRB_TRACE_END, vec3<f32>(kind, lo, hi), vec3<f32>(event_uv, 0.0), max_dist);
        return TraceEvent(kind, lo, hi, event_uv);
    }
    probe_event(PRB_TRACE_END, vec3<f32>(0.0, max_dist, max_dist), vec3<f32>(0.0), max_dist);
    return TraceEvent(0.0, max_dist, max_dist, vec2<f32>(0.0));
}

// First event along a ray inside the water, within max_dist. Samples are
// spaced quadratically: ~1 cm apart at the start, where thin drops and crests
// need them, coarse toward the far walls.
fn trace_in_water(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> TraceEvent {
    if (water.volume_trace != 0u) {
        return trace_in_volume(origin, dir, max_dist);
    }
    var lo = 0.0;
    // Back depth at the last in-water sample (-1: none yet)
    var back_lo = -1.0;
    // Last sample in the water with its back face in sight (seen_s < 0: none
    // yet), and whether the sample before this one was hidden behind a body
    var seen_s = -1.0;
    var seen_q = vec3<f32>(0.0);
    var seen_back = -1.0;
    var was_hidden = false;
    // SILHOUETTE_CONTINUE: back depths nearer than this belong to a layer the
    // ray has slipped behind; samples hidden by it count as in the water
    var hidden_below = 0.0;
    var k = 1;
    loop {
        if (k > TRACE_STEPS) {
            break;
        }
        let f = f32(k) / f32(TRACE_STEPS);
        var s = max_dist * f * f;
        k += 1;
        var q = screen_point(origin + dir * s);
        if (any(q.xy < vec2<f32>(0.0)) || any(q.xy > vec2<f32>(1.0))) {
            probe_event(PRB_TRACE, q, vec3<f32>(-1.0, -1.0, -1.0), s);
            break;
        }
        var ev = trace_event(q);
        if (ev.x > 0.5 && ev.x < 1.5 && ev.y >= 0.0 && ev.y < hidden_below) {
            ev.x = 0.0;
        }
        probe_event4(PRB_TRACE, q, vec3<f32>(prb_bg, ev.y, ev.x), s, vec4<f32>(prb_front, 0.0, 0.0, 0.0));
        // Hidden behind a body: in the water, but with no back face read
        if (ev.x < 0.5 && ev.y < 0.0 && !was_hidden && seen_s >= 0.0) {
            // Going out of sight: close in on the body's outline, then judge
            // the stretch behind it from there (hidden_crossing)
            var o_hi = s;
            var early = false;
            for (var r = 0; r < TRACE_REFINE; r++) {
                let mid = 0.5 * (seen_s + o_hi);
                let qm = screen_point(origin + dir * mid);
                var e = trace_event(qm);
                if (e.x > 0.5 && e.x < 1.5 && e.y >= 0.0 && e.y < hidden_below) {
                    e.x = 0.0;
                }
                probe_event4(PRB_TRACE_REFINE, qm, vec3<f32>(prb_bg, e.y, e.x), mid, vec4<f32>(prb_front, 0.0, 0.0, 0.0));
                if (e.x > 0.5) {
                    // An event in sight before the outline: this is the
                    // sample the bisection below starts from
                    s = mid;
                    q = qm;
                    ev = e;
                    early = true;
                    break;
                }
                if (e.y < 0.0) {
                    o_hi = mid;
                } else {
                    seen_s = mid;
                    seen_q = qm;
                    seen_back = e.y;
                    lo = mid;
                    back_lo = e.y;
                }
            }
            if (!early && seen_q.z <= seen_back) {
                let est = hidden_crossing(origin, dir, seen_s, seen_q, seen_back, max_dist);
                if (est.kind > 0.5) {
                    probe_event(PRB_TRACE_END, vec3<f32>(est.kind, est.dist_in, est.dist), vec3<f32>(est.uv, 0.0), max_dist);
                    return est;
                }
            }
        }
        was_hidden = ev.x < 0.5 && ev.y < 0.0;
        if (ev.x < 0.5) {
            lo = s;
            back_lo = ev.y;
            if (ev.y >= 0.0 && q.z <= ev.y) {
                seen_s = s;
                seen_q = q;
                seen_back = ev.y;
            }
            continue;
        }
        var hi = s;
        var kind = ev.x;
        var back_hi = ev.y;
        var z_hi = q.z;
        for (var r = 0; r < TRACE_REFINE; r++) {
            let mid = 0.5 * (lo + hi);
            let qm = screen_point(origin + dir * mid);
            var e = trace_event(qm);
            if (e.x > 0.5 && e.x < 1.5 && e.y >= 0.0 && e.y < hidden_below) {
                e.x = 0.0;
            }
            probe_event4(PRB_TRACE_REFINE, qm, vec3<f32>(prb_bg, e.y, e.x), mid, vec4<f32>(prb_front, 0.0, 0.0, 0.0));
            if (e.x > 0.5) {
                hi = mid;
                kind = e.x;
                back_hi = e.y;
                z_hi = qm.z;
            } else {
                lo = mid;
                back_lo = e.y;
            }
        }
        let silhouette = kind < 1.5 && is_silhouette(back_lo, back_hi, z_hi);
        if (silhouette) {
            let action = select(0.0, 1.0, water.silhouette_exit == SILHOUETTE_CONTINUE);
            probe_event(PRB_SILHOUETTE, vec3<f32>(back_lo, back_hi, f32(water.silhouette_exit)), vec3<f32>(lo, hi, action), 0.0);
            if (water.silhouette_exit == SILHOUETTE_CONTINUE) {
                // Not a way out: carry on behind the nearer layer
                hidden_below = back_lo - DEPTH_EDGE_REL * max(1.0 - back_lo, 1e-6);
                lo = s;
                continue;
            }
        }
        let event_uv = screen_point(origin + dir * hi).xy;
        probe_event(PRB_TRACE_END, vec3<f32>(kind, lo, hi), vec3<f32>(event_uv, 0.0), max_dist);
        return TraceEvent(kind, lo, hi, event_uv);
    }
    probe_event(PRB_TRACE_END, vec3<f32>(0.0, max_dist, max_dist), vec3<f32>(0.0), max_dist);
    return TraceEvent(0.0, max_dist, max_dist, vec2<f32>(0.0));
}
