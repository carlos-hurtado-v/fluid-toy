// Water shader (mc_render), part: rigid bodies as exact scene geometry (intersection, shading, the wet film)

// === Rigid bodies as exact scene geometry ===
// The depth buffer holds only the camera-facing side of a body, and the
// background image only what the camera sees of it. A refracted or mirrored
// ray that meets a body anywhere else (its far side, its underside in a floor
// mirror, a part hidden behind another body, off screen) finds neither. Read
// off the screen anyway, such a hit showed the colour of whatever the camera
// sees at the hit point's pixel: the body's lit side with its highlight
// combed across the mirror image, or the body in front of it.

// Water wets a body, but where the field is not continued into it
// (rendering.mc_wet_bodies off) the MC surface stops about a particle radius
// short of it. A ray leaving the water this close in front of such a body
// hits the body: refracting into that air film painted ragged fringes around
// submerged bodies (m)
const BODY_WET_GAP: f32 = 0.05;
// How far a ray that has left the water may travel to a body (m)
const BODY_MAX_REACH: f32 = 50.0;
// Distance from an exactly intersected body within which a depth-buffer
// surface point is taken to be that body (tessellation + depth precision) (m)
const BODY_SURFACE_EPS: f32 = 0.02;

struct BodyHit {
    // Distance along the ray, or < 0: no body
    t: f32,
    // Outward surface normal at the hit (world)
    normal: vec3<f32>,
    // Which entry of rigid_bodies
    index: u32,
}

const NO_BODY_HIT: BodyHit = BodyHit(-1.0, vec3<f32>(0.0), 0u);

// World <-> body-local directions (the rotation rows map world -> local)
fn body_to_local(body: RigidBodyParams, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(body.rot_row0.xyz, v), dot(body.rot_row1.xyz, v), dot(body.rot_row2.xyz, v));
}

fn body_to_world(body: RigidBodyParams, v: vec3<f32>) -> vec3<f32> {
    return body.rot_row0.xyz * v.x + body.rot_row1.xyz * v.y + body.rot_row2.xyz * v.z;
}

// The nearest enabled body the ray meets within max_dist (`dir` unit length).
// A ray starting inside a body is on it; Custom bodies are never hit.
fn ray_body_hit(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> BodyHit {
    var best = NO_BODY_HIT;
    let n = min(water.body_count, 8u);
    for (var i = 0u; i < n; i++) {
        let body = rigid_bodies[i];
        // The shape's bounding sphere first: outside it and heading away,
        // or passing it by
        let rel = origin - body.position;
        let bound = SHAPE_BOUND * body.half_extent;
        let b = dot(rel, dir);
        let c = dot(rel, rel) - bound * bound;
        if (c > 0.0 && (b > 0.0 || b * b < c)) {
            continue;
        }
        let local = body_to_local(body, rel);
        var hit = ray_shape_hit(
            body.shape, body.half_extent, body.prop_blades, body.prop_pitch, local, body_to_local(body, dir),
        );
        // A ray that starts inside the body has arrived: the water's mesh
        // runs into bodies (mc_wet_bodies), and where a curved body's faceted
        // mesh lies a hair inside its true surface, water fragments that are
        // already inside the shape show along the contact line. Ignoring the
        // body there let those rays through it: a dotted line of sky along
        // every waterline.
        if (hit.w < 0.0 && c <= 0.0
            && body_shape_sdf(body.shape, body.half_extent, body.prop_blades, body.prop_pitch, local) < 0.0) {
            hit = vec4<f32>(
                body_shape_gradient(body.shape, body.half_extent, body.prop_blades, body.prop_pitch, local), 1e-4,
            );
        }
        if (hit.w > 0.0 && hit.w <= max_dist && (best.t < 0.0 || hit.w < best.t)) {
            best = BodyHit(hit.w, body_to_world(body, hit.xyz), i);
        }
    }
    return best;
}

// Radiance a ray (origin, dir) receives from the body it hits: the body
// renderer's shading at the hit point, seen along the ray
fn body_shade(origin: vec3<f32>, dir: vec3<f32>, hit: BodyHit) -> vec3<f32> {
    let body = rigid_bodies[hit.index];
    var sun_rgb = vec3<f32>(0.0);
    if (light.sun_enabled == 1u) {
        sun_rgb = light.sun_color * light.sun_intensity;
    }
    return shade_body_lit(
        body.color.rgb, hit.normal, -dir, origin + dir * hit.t, light.sun_direction, sun_rgb, water.env_intensity,
    );
}

// body_shade for a refraction ray: the lookup record is cleared (nothing on
// screen was read) and the route ends on a body
fn body_radiance(origin: vec3<f32>, dir: vec3<f32>, hit: BodyHit) -> vec3<f32> {
    dbg_end = DBG_END_BODY;
    look_kind = LOOK_NONE;
    probe_event(PRB_BODY, origin + dir * hit.t, hit.normal, f32(hit.index));
    return body_shade(origin, dir, hit);
}

// Signed distance from a world point to the nearest body that ray_body_hit
// handles (negative inside). Exact up to BODY_WET_GAP outside a body's
// bounding sphere, a large value beyond (callers only ask "this close?").
fn analytic_body_distance(p: vec3<f32>) -> f32 {
    var best = 1e6;
    let n = min(water.body_count, 8u);
    for (var i = 0u; i < n; i++) {
        let body = rigid_bodies[i];
        let rel = p - body.position;
        let bound = SHAPE_BOUND * body.half_extent + BODY_WET_GAP;
        if (dot(rel, rel) > bound * bound) {
            continue;
        }
        best = min(best, body_shape_sdf(
            body.shape, body.half_extent, body.prop_blades, body.prop_pitch, body_to_local(body, rel),
        ));
    }
    return best;
}

// Is this world point on the surface of a body that ray_body_hit handles?
fn on_analytic_body(p: vec3<f32>) -> bool {
    return abs(analytic_body_distance(p)) < BODY_SURFACE_EPS;
}

// Does the mesh leave a dry film around this body? Not where the field is
// continued into it: the mesh then reaches the body, and the film rules
// would only misread the air next to its dry part as water.
fn body_has_film(index: u32) -> bool {
    return rigid_bodies[index].wet < 0.5;
}

// p moved out of the dry film the mesh leaves around a body, to the film's
// outer edge straight out from the body. Water wets a body: the film is water
// wherever the water next to it is.
fn body_film_push(p: vec3<f32>) -> vec3<f32> {
    var q = p;
    let n = min(water.body_count, 8u);
    for (var i = 0u; i < n; i++) {
        let body = rigid_bodies[i];
        let c = q - body.position;
        let bound = SHAPE_BOUND * body.half_extent + BODY_WET_GAP;
        if (body.wet > 0.5 || dot(c, c) > bound * bound) {
            continue;
        }
        if (body.shape == SHAPE_SPHERE) {
            let dist = length(c);
            if (dist - body.half_extent < BODY_WET_GAP && dist > 1e-5) {
                q = body.position + c * ((body.half_extent + BODY_WET_GAP) / dist);
            }
        } else if (body.shape == SHAPE_CUBE) {
            let lp = body_to_local(body, c);
            let d = abs(lp) - vec3<f32>(body.half_extent);
            let outside = max(d, vec3<f32>(0.0));
            let sdf = length(outside) + min(max(d.x, max(d.y, d.z)), 0.0);
            if (sdf < BODY_WET_GAP) {
                // Outward direction: away from the nearest point of the box
                var dir_l = outside * sign(lp);
                if (dot(dir_l, dir_l) < 1e-10) {
                    // Inside: through the nearest face
                    if (d.x >= d.y && d.x >= d.z) {
                        dir_l = vec3<f32>(sign(lp.x), 0.0, 0.0);
                    } else if (d.y >= d.z) {
                        dir_l = vec3<f32>(0.0, sign(lp.y), 0.0);
                    } else {
                        dir_l = vec3<f32>(0.0, 0.0, sign(lp.z));
                    }
                }
                q = body.position + body_to_world(body, lp + normalize(dir_l) * (BODY_WET_GAP - sdf));
            }
        } else if (body.shape != SHAPE_CUSTOM) {
            // Any other shape: out along its distance field's gradient
            let lp = body_to_local(body, c);
            let sdf = body_shape_sdf(body.shape, body.half_extent, body.prop_blades, body.prop_pitch, lp);
            if (sdf < BODY_WET_GAP) {
                let outward = body_shape_gradient(body.shape, body.half_extent, body.prop_blades, body.prop_pitch, lp);
                q += body_to_world(body, outward) * (BODY_WET_GAP - sdf);
            }
        }
    }
    return q;
}
