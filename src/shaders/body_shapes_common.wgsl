// Rigid body shapes in body-local space: signed distance and exact ray
// intersection. Pure functions of (shape, half_extent, blades, pitch): no
// bindings, no structs, so the SPH compute shaders (through
// rigid_body_common.wgsl) and the water shader (mc_render/bodies.wgsl) link
// the same definitions. Prepended at module creation, like every snippet.
//
// The SDFs have CPU copies in simulation/body_shapes.rs (body-body contact):
// change both, or bodies and fluid disagree on a surface. Propeller
// proportions must also match state/rigid_body.rs PROP_* and the mesh
// generator in rigid_body.wgsl.
//
// The Custom (voxel SDF) shape is not here: only the integrate shader has
// its texture. Its distance reads as "far" and no ray hits it.

const SHAPE_CUBE: u32 = 0u;
const SHAPE_SPHERE: u32 = 1u;
const SHAPE_CYLINDER: u32 = 2u;
const SHAPE_TORUS: u32 = 3u;
const SHAPE_CUSTOM: u32 = 4u;
const SHAPE_PROPELLER: u32 = 5u;

const RB_TWO_PI: f32 = 6.28318530718;

// Torus tube radius, and the radius of the sphere that holds any shape, in
// units of half_extent (a cube's corner is sqrt(3) away)
const TORUS_TUBE: f32 = 0.3;
const SHAPE_BOUND: f32 = 1.75;

// Propeller proportions in units of half_extent (spin axis = body-local Y)
const PROP_HUB_RADIUS: f32 = 0.25;
const PROP_HUB_HALF_HEIGHT: f32 = 0.30;
const PROP_BLADE_CENTER: f32 = 0.55;
const PROP_BLADE_HALF: vec3<f32> = vec3<f32>(0.44, 0.18, 0.045);

// p in the frame of the propeller blade at angle `turn` around Y: the blade
// is then the box PROP_BLADE_HALF centred PROP_BLADE_CENTER along +X.
// Linear, so it maps directions too.
fn propeller_blade_frame(p: vec3<f32>, turn: f32, pitch: f32) -> vec3<f32> {
    let cs = cos(turn);
    let sn = sin(turn);
    let q = vec3<f32>(cs * p.x + sn * p.z, p.y, -sn * p.x + cs * p.z);
    // Un-pitch around the radial (X) axis
    let cp = cos(pitch);
    let sp = sin(pitch);
    return vec3<f32>(q.x, cp * q.y + sp * q.z, -sp * q.y + cp * q.z);
}

// A blade-frame direction back in the body frame (inverse of the above)
fn propeller_blade_unframe(v: vec3<f32>, turn: f32, pitch: f32) -> vec3<f32> {
    let cp = cos(pitch);
    let sp = sin(pitch);
    let q = vec3<f32>(v.x, cp * v.y - sp * v.z, sp * v.y + cp * v.z);
    let cs = cos(turn);
    let sn = sin(turn);
    return vec3<f32>(cs * q.x - sn * q.z, q.y, sn * q.x + cs * q.z);
}

// Hub cylinder + N pitched blades via angular domain repetition
fn propeller_sdf(p: vec3<f32>, he: f32, blades: u32, pitch: f32) -> f32 {
    let hub = max(length(p.xz) - PROP_HUB_RADIUS * he, abs(p.y) - PROP_HUB_HALF_HEIGHT * he);

    // Snap to the nearest blade sector and rotate that blade onto +X
    let sector = RB_TWO_PI / f32(blades);
    let snapped = round(atan2(p.z, p.x) / sector) * sector;
    let v = propeller_blade_frame(p, snapped, pitch);
    // Exact box SDF for the blade
    let d = abs(v - vec3<f32>(PROP_BLADE_CENTER * he, 0.0, 0.0)) - PROP_BLADE_HALF * he;
    let blade = length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);

    return min(hub, blade);
}

// Signed distance from body-local point p to a shape's surface (negative
// inside). Custom (voxel) bodies return "far".
fn body_shape_sdf(shape: u32, he: f32, blades: u32, pitch: f32, p: vec3<f32>) -> f32 {
    switch (shape) {
        case SHAPE_SPHERE: {
            return length(p) - he;
        }
        case SHAPE_CYLINDER: {
            let d = vec2<f32>(length(p.xz) - he, abs(p.y) - he);
            return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
        }
        case SHAPE_TORUS: {
            let q = vec2<f32>(length(p.xz) - he, p.y);
            return length(q) - he * TORUS_TUBE;
        }
        case SHAPE_PROPELLER: {
            return propeller_sdf(p, he, max(blades, 1u), pitch);
        }
        case SHAPE_CUSTOM: {
            return 1e9;
        }
        default: {
            // Cube: exact box SDF
            let d = abs(p) - vec3<f32>(he);
            return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
        }
    }
}

// Outward unit normal of the shape's distance field at body-local p, by
// central differences (exact gradients exist per shape, but this is used off
// the hot path: film pushes, penetrating predictions)
fn body_shape_gradient(shape: u32, he: f32, blades: u32, pitch: f32, p: vec3<f32>) -> vec3<f32> {
    let eps = max(0.01 * he, 1e-4);
    let ex = vec3<f32>(eps, 0.0, 0.0);
    let ey = vec3<f32>(0.0, eps, 0.0);
    let ez = vec3<f32>(0.0, 0.0, eps);
    let grad = vec3<f32>(
        body_shape_sdf(shape, he, blades, pitch, p + ex) - body_shape_sdf(shape, he, blades, pitch, p - ex),
        body_shape_sdf(shape, he, blades, pitch, p + ey) - body_shape_sdf(shape, he, blades, pitch, p - ey),
        body_shape_sdf(shape, he, blades, pitch, p + ez) - body_shape_sdf(shape, he, blades, pitch, p - ez),
    );
    let len = length(grad);
    if (len < 1e-6) {
        return vec3<f32>(0.0);
    }
    return grad / len;
}

// === Ray intersection ===
// xyz = outward normal at the hit (body-local), w = distance along the ray,
// or w < 0: no hit. `rd` must be unit length. A ray starting inside a shape
// does not hit it.
const SHAPE_MISS: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, -1.0);

fn ray_sphere(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> vec4<f32> {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - radius * radius;
    let disc = b * b - c;
    if (c <= 0.0 || disc < 0.0) {
        return SHAPE_MISS;
    }
    let t = -b - sqrt(disc);
    if (t <= 0.0) {
        return SHAPE_MISS;
    }
    return vec4<f32>((ro + rd * t) / radius, t);
}

// Box of half sizes `half` centred on `center`
fn ray_box(ro_in: vec3<f32>, rd: vec3<f32>, center: vec3<f32>, half: vec3<f32>) -> vec4<f32> {
    let ro = ro_in - center;
    let safe_rd = select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));
    let inv = vec3<f32>(1.0) / safe_rd;
    let t1 = (-half - ro) * inv;
    let t2 = (half - ro) * inv;
    let near = min(t1, t2);
    let t_near = max(max(near.x, near.y), near.z);
    let t_far = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (t_near <= 0.0 || t_near > t_far) {
        return SHAPE_MISS;
    }
    // The slab entered last is the face hit; its normal faces the ray
    var n = vec3<f32>(0.0, 0.0, -sign(safe_rd.z));
    if (near.x >= near.y && near.x >= near.z) {
        n = vec3<f32>(-sign(safe_rd.x), 0.0, 0.0);
    } else if (near.y >= near.z) {
        n = vec3<f32>(0.0, -sign(safe_rd.y), 0.0);
    }
    return vec4<f32>(n, t_near);
}

// Capped cylinder around Y: radius `radius`, half height `half_height`
fn ray_cylinder(ro: vec3<f32>, rd: vec3<f32>, radius: f32, half_height: f32) -> vec4<f32> {
    var t0 = -1e30;
    var t1 = 1e30;
    var n = vec3<f32>(0.0);
    // Barrel: the infinite cylinder's interval
    let a = dot(rd.xz, rd.xz);
    let b = dot(ro.xz, rd.xz);
    let c = dot(ro.xz, ro.xz) - radius * radius;
    if (a > 1e-10) {
        let disc = b * b - a * c;
        if (disc < 0.0) {
            return SHAPE_MISS;
        }
        let root = sqrt(disc);
        t0 = (-b - root) / a;
        t1 = (-b + root) / a;
        let hit = ro.xz + rd.xz * t0;
        n = vec3<f32>(hit.x, 0.0, hit.y) / radius;
    } else if (c > 0.0) {
        return SHAPE_MISS;  // parallel to the axis, outside the barrel
    }
    // Caps: the slab's interval
    if (abs(rd.y) > 1e-10) {
        let ta = (-half_height - ro.y) / rd.y;
        let tb = (half_height - ro.y) / rd.y;
        let slab_near = min(ta, tb);
        if (slab_near > t0) {
            t0 = slab_near;
            n = vec3<f32>(0.0, -sign(rd.y), 0.0);
        }
        t1 = min(t1, max(ta, tb));
    } else if (abs(ro.y) > half_height) {
        return SHAPE_MISS;
    }
    if (t0 <= 0.0 || t0 > t1) {
        return SHAPE_MISS;
    }
    return vec4<f32>(n, t0);
}

// Torus around Y (ring radius `ring`, tube radius `tube`): sphere tracing
// between the ray's entry into and exit from the bounding sphere. The
// distance field is exact, so steps never overshoot; a ray skimming the tube
// can run out of steps and then counts as a miss.
const TORUS_TRACE_STEPS: i32 = 40;

fn ray_torus(ro: vec3<f32>, rd: vec3<f32>, ring: f32, tube: f32) -> vec4<f32> {
    let bound = ring + tube;
    let b = dot(ro, rd);
    let c = dot(ro, ro) - bound * bound;
    let disc = b * b - c;
    if (disc < 0.0) {
        return SHAPE_MISS;
    }
    let root = sqrt(disc);
    let t_exit = -b + root;
    if (t_exit <= 0.0) {
        return SHAPE_MISS;
    }
    var t = max(-b - root, 0.0);
    let tolerance = 1e-4 * ring;
    for (var i = 0; i < TORUS_TRACE_STEPS; i++) {
        let p = ro + rd * t;
        let q = vec2<f32>(length(p.xz) - ring, p.y);
        let d = length(q) - tube;
        if (d < tolerance) {
            if (t <= 0.0) {
                return SHAPE_MISS;  // started inside the tube
            }
            let radial = p.xz / max(length(p.xz), 1e-6);
            return vec4<f32>(normalize(vec3<f32>(radial.x * q.x, q.y, radial.y * q.x)), t);
        }
        t += d;
        if (t > t_exit) {
            return SHAPE_MISS;
        }
    }
    return SHAPE_MISS;
}

// Propeller: the nearer of the hub cylinder and each blade's box
fn ray_propeller(ro: vec3<f32>, rd: vec3<f32>, he: f32, blades: u32, pitch: f32) -> vec4<f32> {
    var best = ray_cylinder(ro, rd, PROP_HUB_RADIUS * he, PROP_HUB_HALF_HEIGHT * he);
    let sector = RB_TWO_PI / f32(blades);
    let center = vec3<f32>(PROP_BLADE_CENTER * he, 0.0, 0.0);
    for (var k = 0u; k < blades; k++) {
        let turn = f32(k) * sector;
        let hit = ray_box(
            propeller_blade_frame(ro, turn, pitch), propeller_blade_frame(rd, turn, pitch), center, PROP_BLADE_HALF * he,
        );
        if (hit.w > 0.0 && (best.w < 0.0 || hit.w < best.w)) {
            best = vec4<f32>(propeller_blade_unframe(hit.xyz, turn, pitch), hit.w);
        }
    }
    return best;
}

// First hit of the body-local ray (ro, unit rd) on a shape's surface: xyz =
// outward normal (body-local), w = distance, or w < 0. Custom never hits.
fn ray_shape_hit(shape: u32, he: f32, blades: u32, pitch: f32, ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    switch (shape) {
        case SHAPE_SPHERE: {
            return ray_sphere(ro, rd, he);
        }
        case SHAPE_CYLINDER: {
            return ray_cylinder(ro, rd, he, he);
        }
        case SHAPE_TORUS: {
            return ray_torus(ro, rd, he, he * TORUS_TUBE);
        }
        case SHAPE_PROPELLER: {
            return ray_propeller(ro, rd, he, max(blades, 1u), pitch);
        }
        case SHAPE_CUSTOM: {
            return SHAPE_MISS;
        }
        default: {
            return ray_box(ro, rd, vec3<f32>(0.0), vec3<f32>(he));
        }
    }
}
