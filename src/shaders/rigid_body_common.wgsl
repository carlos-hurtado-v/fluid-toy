// Shared rigid body definitions for SPH compute shaders (integrate, PCISPH
// predict/solve): structs, motion constants, and the boundary-aware pressure
// helpers. The shapes themselves (constants, SDFs) are body_shapes_common.wgsl.
//
// Conventions:
// - container_common.wgsl and body_shapes_common.wgsl are always concatenated
//   BEFORE this file (the helpers below use ContainerGeometry +
//   world_to_local/local_to_world, and body_shape_sdf).
// - Each consumer shader declares its own @group/@binding for
//   `container: ContainerGeometry` and `rigid_bodies: RigidBodies` (WGSL
//   module-scope declarations are order-independent, so the helper functions
//   here may reference them).
// - The Custom (voxel SDF) shape is handled only by the integrate shader,
//   which owns the 3D texture binding; the analytic helpers here skip it.

const MOTION_STATIC: u32 = 0u;
const MOTION_KINEMATIC: u32 = 1u;
const MOTION_DYNAMIC: u32 = 2u;

const MAX_RIGID_BODIES: u32 = 8u;

struct RigidBody {
    position: vec3<f32>,
    half_extent: f32,
    velocity: vec3<f32>,
    is_active: u32,
    stiffness: f32,
    shape: u32,
    motion: u32,
    prop_blades: u32,
    angular_velocity: vec3<f32>,
    prop_pitch: f32,
    rot_row0: vec4<f32>,
    rot_row1: vec4<f32>,
    rot_row2: vec4<f32>,
}

struct RigidBodies {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    bodies: array<RigidBody>,
}

// Analytic body SDF in body-local space (body_shapes_common.wgsl). Custom
// (voxel) bodies return "far": only the integrate shader has the SDF texture
// to resolve them.
fn rb_analytic_sdf(body: RigidBody, p: vec3<f32>) -> f32 {
    return body_shape_sdf(body.shape, body.half_extent, body.prop_blades, body.prop_pitch, p);
}

// Project a predicted position out of the container box and all active
// analytic bodies. Position-only: the caller's predicted velocity is left
// untouched (wall/body velocity response stays the integrate shader's job).
// This keeps the predicted DENSITY consistent with what integration will
// actually allow, so the pressure solve stops fighting phantom compression
// at boundaries (the tilted-tank slosh pump) and, together with the boundary
// density term, resists real compression against obstacles.
fn boundary_clamp_predicted(pos: vec3<f32>) -> vec3<f32> {
    // Container: per-axis clamp in local space
    var lp = world_to_local(container, pos);
    let half = vec3<f32>(container.half_width, container.half_height, container.half_depth);
    lp = clamp(lp, -half, half);
    var p = local_to_world(container, lp);

    // Bodies: push out along the SDF gradient
    for (var b = 0u; b < min(rigid_bodies.count, MAX_RIGID_BODIES); b++) {
        let body = rigid_bodies.bodies[b];
        if (body.is_active == 0u || body.shape == SHAPE_CUSTOM) {
            continue;
        }
        let rel = p - body.position;
        let bound = body.half_extent * 1.75;
        if (dot(rel, rel) > bound * bound) {
            continue;
        }
        let local = vec3<f32>(
            dot(body.rot_row0.xyz, rel),
            dot(body.rot_row1.xyz, rel),
            dot(body.rot_row2.xyz, rel),
        );
        let sdf = rb_analytic_sdf(body, local);
        if (sdf < 0.0) {
            // Gradient by central differences (penetrating predictions are rare)
            let ln = body_shape_gradient(body.shape, body.half_extent, body.prop_blades, body.prop_pitch, local);
            if (dot(ln, ln) > 0.5) {
                // Local -> world (transpose multiply)
                let wn = vec3<f32>(
                    body.rot_row0.x * ln.x + body.rot_row1.x * ln.y + body.rot_row2.x * ln.z,
                    body.rot_row0.y * ln.x + body.rot_row1.y * ln.y + body.rot_row2.y * ln.z,
                    body.rot_row0.z * ln.x + body.rot_row1.z * ln.y + body.rot_row2.z * ln.z,
                );
                p += wn * (-sdf);
            }
        }
    }
    return p;
}

// Fraction of the poly6 kernel support cut off by a flat boundary at
// normalized distance q = d/h: the analytic half-space integral
// f(q) = 0.5 - (315/256) * (q - 4/3 q^3 + 6/5 q^5 - 4/7 q^7 + 1/9 q^9).
// f(0) = 0.5 (touching the surface), f(q >= 1) = 0 (out of range).
fn wall_density_deficit(q_in: f32) -> f32 {
    let q = clamp(q_in, 0.0, 1.0);
    let q2 = q * q;
    let poly = q * (1.0 + q2 * (-4.0 / 3.0 + q2 * (6.0 / 5.0 + q2 * (-4.0 / 7.0 + q2 * (1.0 / 9.0)))));
    return 0.5 - 1.23046875 * poly;
}

// Total boundary density fraction at a (clamped) position: sum of half-space
// deficits from nearby container faces and analytic bodies, treating solids
// as filled with rest-density fluid. Multiplied by rest_density and the
// boundary_density strength in the pressure solve. Capped: overlapping
// boundaries (corners, wedges) double-count the shared region.
fn boundary_density_fraction(pos: vec3<f32>, h: f32) -> f32 {
    var deficit = 0.0;

    let lp = world_to_local(container, pos);
    deficit += wall_density_deficit((container.half_width - lp.x) / h);
    deficit += wall_density_deficit((container.half_width + lp.x) / h);
    deficit += wall_density_deficit((container.half_height - lp.y) / h);
    deficit += wall_density_deficit((container.half_height + lp.y) / h);
    deficit += wall_density_deficit((container.half_depth - lp.z) / h);
    deficit += wall_density_deficit((container.half_depth + lp.z) / h);

    for (var b = 0u; b < min(rigid_bodies.count, MAX_RIGID_BODIES); b++) {
        let body = rigid_bodies.bodies[b];
        if (body.is_active == 0u || body.shape == SHAPE_CUSTOM) {
            continue;
        }
        let rel = pos - body.position;
        let bound = body.half_extent * 1.75 + h;
        if (dot(rel, rel) > bound * bound) {
            continue;
        }
        let local = vec3<f32>(
            dot(body.rot_row0.xyz, rel),
            dot(body.rot_row1.xyz, rel),
            dot(body.rot_row2.xyz, rel),
        );
        deficit += wall_density_deficit(rb_analytic_sdf(body, local) / h);
    }

    return min(deficit, 0.85);
}
