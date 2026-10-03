// Rigid bodies as the marching-cubes field passes see them (mc_density,
// mc_wall_bound): solid boundaries the water's field has to end ON, like the
// container walls, rather than water/air edges.
//
// Without this a body is a hole in the field: the smoothing filters treat its
// outline as a free surface and round the water off toward it.
//
// Uses the including shader's `field_bodies: array<FieldBody>` binding by
// name; body_shapes_common.wgsl must be prepended before this file. Only
// procedural shapes with `wet` set take part: a Custom (mesh) body has no
// distance field here, and rendering.mc_wet_bodies = false clears the flag.

// Mirrors GpuRigidBodyRender (112 bytes); `wet` rides in its first pad
struct FieldBody {
    position: vec3<f32>,
    half_extent: f32,
    color: vec4<f32>,
    light_dir: vec3<f32>,
    shape: u32,
    rot_row0: vec4<f32>,
    rot_row1: vec4<f32>,
    rot_row2: vec4<f32>,
    prop_blades: u32,
    prop_pitch: f32,
    wet: f32,
    _pad1: f32,
}

// Slots in the body array (unused ones have half_extent 0)
const FIELD_BODY_SLOTS: u32 = 8u;
// Width of the band around a body that belongs to it as far as the field is
// concerned, in units of the field's kernel radius.
const FIELD_BODY_BAND: f32 = 0.4;

fn field_body_active(body: FieldBody) -> bool {
    return body.half_extent > 0.0 && body.wet > 0.5 && body.shape != SHAPE_CUSTOM;
}

fn field_body_local(body: FieldBody, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(body.rot_row0.xyz, v), dot(body.rot_row1.xyz, v), dot(body.rot_row2.xyz, v));
}

// (signed distance to the nearest wetted body within `reach` of its surface,
// its slot), or (1e6, 0) if p is further than that from all of them
fn field_body_distance(p: vec3<f32>, reach: f32) -> vec2<f32> {
    var best = vec2<f32>(1e6, 0.0);
    for (var i = 0u; i < FIELD_BODY_SLOTS; i++) {
        let body = field_bodies[i];
        if (!field_body_active(body)) {
            continue;
        }
        let rel = p - body.position;
        let bound = SHAPE_BOUND * body.half_extent + reach;
        if (dot(rel, rel) > bound * bound) {
            continue;
        }
        let sdf = body_shape_sdf(
            body.shape, body.half_extent, body.prop_blades, body.prop_pitch, field_body_local(body, rel),
        );
        if (sdf < best.x) {
            best = vec2<f32>(sdf, f32(i));
        }
    }
    return best;
}

// Steps the push below may take to leave a body sideways, and their least
// length in units of half_extent: together 2.4 half-extents (a point over
// the middle of a cube's top has a half-extent to cross; a blade tip is 1.0
// from a propeller's axis)
const FIELD_BODY_PUSH_STEPS: i32 = 16;
const FIELD_BODY_STRIDE: f32 = 0.15;

// p moved SIDEWAYS out of body `slot` (where its distance is `sdf`) until it
// is `clearance` outside the surface: at the same height (gravity is world
// -Y), toward the nearest side where the surface has one there, else away
// from the body's centre. The water inside a body and its band then takes
// the level the water has next to the body.
// Returns p where no clearance is found within the steps.
fn field_body_push(p: vec3<f32>, slot: u32, sdf: f32, clearance: f32) -> vec3<f32> {
    let body = field_bodies[slot];
    let g = body_shape_gradient(
        body.shape, body.half_extent, body.prop_blades, body.prop_pitch, field_body_local(body, p - body.position),
    );
    let outward = body.rot_row0.xyz * g.x + body.rot_row1.xyz * g.y + body.rot_row2.xyz * g.z;
    var side = vec3<f32>(outward.x, 0.0, outward.z);
    if (dot(side, side) < 0.09) {
        // A top or an underside: no side in sight, head away from the centre
        side = vec3<f32>(p.x - body.position.x, 0.0, p.z - body.position.z);
        if (dot(side, side) < 1e-8) {
            side = vec3<f32>(1.0, 0.0, 0.0);
        }
    }
    let dir = normalize(side);
    var q = p;
    var d = sdf;
    // The distance cannot grow faster than the step, so stepping the
    // shortfall never overshoots; along a flat face it does not grow at all,
    // hence the floor on the step (the steps together span the whole body)
    let stride = FIELD_BODY_STRIDE * body.half_extent;
    for (var i = 0; i < FIELD_BODY_PUSH_STEPS; i++) {
        q += dir * max(clearance - d, stride);
        d = body_shape_sdf(
            body.shape, body.half_extent, body.prop_blades, body.prop_pitch, field_body_local(body, q - body.position),
        );
        if (d >= clearance) {
            // Back to the clearance (same argument: still at least that far)
            return q - dir * (d - clearance);
        }
    }
    return p;
}
