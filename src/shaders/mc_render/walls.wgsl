// Water shader (mc_render), part: container walls as optical interfaces

// === Container walls as optical interfaces ===
// In a wireframe (glass-less) tank the water's sides and bottom ARE the
// container walls, like water against glass: flat planes. The mesh's sides lie
// on those planes (mc_wall_bound.wgsl cuts the field with them), but its
// vertex normals there come from field differences and lean near the edges;
// a slab with leaning normals is a weak prism, and seen at eye level that is
// enough to bend the horizon down onto the ground. Fragments on a wall use
// the plane.
// Distance beyond the clip margin that still counts as "on the wall" (m)
const WALL_SNAP_TOLERANCE: f32 = 0.02;
const WALL_SNAP_MIN_COS: f32 = 0.7;
// Total internal reflection inside a thin body (drop, crest) has no
// meaningful next interface in screen space: only follow it in bulk water
const TIR_MIN_BODY: f32 = 0.1;
const TIR_MAX_BOUNCES: i32 = 3;
// Ray tracing inside the water, in screen space (see trace_in_water)
const TRACE_STEPS: i32 = 16;
const TRACE_REFINE: i32 = 6;

// Outward plane normal (local, .w = 1) of the wall or floor this point lies on
// with a matching outward MC normal, else .w = 0. Never the open top.
fn wall_plane(local: vec3<f32>, n_local: vec3<f32>) -> vec4<f32> {
    if (container.is_pool != 0u) {
        return vec4<f32>(0.0);
    }
    let h = vec3<f32>(container.half_width, container.half_height, container.half_depth);
    let tol = container.clip_margin + WALL_SNAP_TOLERANCE;
    var best = vec4<f32>(0.0);
    var best_dist = tol;
    let planes = array<vec3<f32>, 5>(
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, -1.0),
        vec3<f32>(0.0, -1.0, 0.0),
    );
    for (var i = 0; i < 5; i++) {
        let pn = planes[i];
        let dist = abs(dot(local, pn) - dot(h, abs(pn)));
        if (dist < best_dist && dot(n_local, pn) > WALL_SNAP_MIN_COS) {
            best_dist = dist;
            best = vec4<f32>(pn, 1.0);
        }
    }
    return best;
}

// Ray from inside the container box to its walls/floor (local space; the
// top is open): xyz = outward normal of the plane hit, w = distance
fn box_interior_exit(o: vec3<f32>, d: vec3<f32>) -> vec4<f32> {
    let h = vec3<f32>(container.half_width, container.half_height, container.half_depth);
    var best = vec4<f32>(0.0, 0.0, 0.0, 1e6);
    if (abs(d.x) > 1e-5) {
        let t = (sign(d.x) * h.x - o.x) / d.x;
        if (t < best.w) { best = vec4<f32>(sign(d.x), 0.0, 0.0, t); }
    }
    if (abs(d.z) > 1e-5) {
        let t = (sign(d.z) * h.z - o.z) / d.z;
        if (t < best.w) { best = vec4<f32>(0.0, 0.0, sign(d.z), t); }
    }
    if (d.y < -1e-5) {
        let t = (-h.y - o.y) / d.y;
        if (t < best.w) { best = vec4<f32>(0.0, -1.0, 0.0, t); }
    }
    best.w = max(best.w, 0.0);
    return best;
}
