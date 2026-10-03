//! Rigid body contact: collisions between bodies, resolved on the CPU once
//! per frame after `integrate_rigid_body` has moved every Dynamic body.
//!
//! Detection is shape-agnostic (`body_shapes.rs`): each body's probes are
//! tested against the other body's signed distance function, both ways, so
//! every pair of shapes collides, a sphere exactly. The response is a small
//! sequential-impulse solver: a normal impulse with restitution, Coulomb
//! friction, torque through the scalar inertia the fluid coupling already
//! uses, and a position push that separates what still overlaps. Static and
//! Kinematic bodies are immovable; a Kinematic body's spin enters through its
//! surface velocity, so a propeller bats a ball away.
//!
//! The container is part of the solve for bodies that touch another body:
//! its faces enter as contacts with an immovable partner, so a body squeezed
//! between a wall and a neighbour pushes the neighbour instead of leaving.
//!
//! Stateless and deterministic (fixed pair order, no warm start): body poses
//! stay fully described by the config a snapshot saves.

use super::body_shapes::{add, cross, dot, length, scale, sub, Collider, V3};
use crate::render::mesh_loader::SdfData;
use crate::state::{
    clamp_rigid_body_to_container, container_face_gaps, ContainerConfig, RigidBodyConfig,
    RigidBodyMotion, MAX_RIGID_BODIES,
};

/// Bounce of a body off another: the container clamp's 0.3
const RESTITUTION: f32 = 0.3;
/// Impact speed (m/s) below which a contact does not bounce
const BOUNCE_SPEED: f32 = 0.3;
const FRICTION: f32 = 0.3;
/// Overlap left in place, so a resting contact is found again next frame
const SLOP: f32 = 0.001;
/// Speed (m/s) at which overlapping bodies are pushed apart: bodies that
/// start inside each other (two bodies added at the default position) ease
/// apart over a few frames instead of teleporting
const MAX_PUSH_SPEED: f32 = 2.0;
/// A body whose inner sphere is buried deeper than this fraction of its
/// radius is separated along the inner sphere's contact alone: inside
/// another body, surface points find faces on opposite sides and their
/// normals fight
const DEEP_OVERLAP: f32 = 0.25;
/// A body within this distance (m) of a container face is resting on it
const WALL_TOUCH: f32 = 0.001;
/// Which way two bodies with the same centre separate (lower index toward
/// +X). Horizontal: stacked vertically, one would balance on the other.
const FALLBACK_AXIS: V3 = [1.0, 0.0, 0.0];
/// Solver sweeps over all contacts, at most. Without a warm start a body
/// resting on many points needs them: at 8 an offset cube on a cube kept a
/// residual spin every frame, crept sideways and fell off within 5 s.
const VELOCITY_ITERATIONS: usize = 32;
const POSITION_ITERATIONS: usize = 8;
/// A sweep that changed no contact speed (m/s) / moved no body (m) by more
/// than this ends the iteration
const CONVERGED_SPEED: f32 = 1e-6;
const CONVERGED_SHIFT: f32 = 1e-6;

/// A body in the solver: inverse mass and inertia (0 = immovable), the
/// velocity it arrived with, the velocities being solved, and the position
/// correction so far
struct Solid {
    inv_mass: f32,
    inv_inertia: f32,
    arrival: V3,
    velocity: V3,
    spin: V3,
    shift: V3,
}

impl Solid {
    fn new(body: &RigidBodyConfig, arrival: V3, fluid_density: f32) -> Self {
        let mass = body.mass(fluid_density);
        let inertia = body.shape.moment_of_inertia(mass, body.half_extent);
        let movable = is_dynamic(body) && mass > 0.0 && inertia > 0.0;
        Self {
            inv_mass: if movable { 1.0 / mass } else { 0.0 },
            inv_inertia: if movable { 1.0 / inertia } else { 0.0 },
            arrival,
            velocity: body.velocity,
            spin: body.angular_velocity,
            shift: [0.0; 3],
        }
    }

    /// The container: immovable and at rest
    fn fixed() -> Self {
        Self {
            inv_mass: 0.0,
            inv_inertia: 0.0,
            arrival: [0.0; 3],
            velocity: [0.0; 3],
            spin: [0.0; 3],
            shift: [0.0; 3],
        }
    }

    fn point_velocity(&self, r: V3) -> V3 {
        add(self.velocity, cross(self.spin, r))
    }

    fn apply_impulse(&mut self, impulse: V3, r: V3) {
        self.velocity = add(self.velocity, scale(impulse, self.inv_mass));
        self.spin = add(self.spin, scale(cross(r, impulse), self.inv_inertia));
    }
}

/// One contact point between solids `a` and `b` (`a < b`; `b` may be the
/// container)
struct Contact {
    a: usize,
    b: usize,
    /// Unit normal from `b` toward `a`: the way `a` is pushed
    normal: V3,
    /// Contact point relative to each centre
    ra: V3,
    rb: V3,
    depth: f32,
    friction: f32,
    tangents: [V3; 2],
    /// Normal speed the contact must leave with (restitution)
    target_speed: f32,
    /// Separation the position solve owes this contact
    push: f32,
    normal_impulse: f32,
    tangent_impulse: [f32; 2],
}

impl Contact {
    fn new(a: usize, b: usize, normal: V3, ra: V3, rb: V3, depth: f32, friction: f32) -> Self {
        // Any two directions that complete the normal to a frame
        let seed = if normal[0].abs() < 0.6 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let side = cross(normal, seed);
        let t0 = scale(side, 1.0 / length(side));
        Self {
            a,
            b,
            normal,
            ra,
            rb,
            depth,
            friction,
            tangents: [t0, cross(normal, t0)],
            target_speed: 0.0,
            push: 0.0,
            normal_impulse: 0.0,
            tangent_impulse: [0.0; 2],
        }
    }

    /// Speed of `a` relative to `b` at the contact point, along `dir`
    fn speed(&self, a: &Solid, b: &Solid, dir: V3) -> f32 {
        dot(sub(a.point_velocity(self.ra), b.point_velocity(self.rb)), dir)
    }

    /// Relative speed gained along `dir` per unit of impulse along it
    fn response(&self, a: &Solid, b: &Solid, dir: V3) -> f32 {
        let (ca, cb) = (cross(self.ra, dir), cross(self.rb, dir));
        a.inv_mass + b.inv_mass + a.inv_inertia * dot(ca, ca) + b.inv_inertia * dot(cb, cb)
    }

    fn apply(&self, a: &mut Solid, b: &mut Solid, dir: V3, impulse: f32) {
        a.apply_impulse(scale(dir, impulse), self.ra);
        b.apply_impulse(scale(dir, -impulse), self.rb);
    }

    /// One sequential-impulse step: totals are clamped, not increments, so
    /// later iterations may take back what earlier ones overdid. Returns the
    /// largest change of contact speed it made (m/s).
    fn solve_velocity(&mut self, a: &mut Solid, b: &mut Solid) -> f32 {
        let response = self.response(a, b, self.normal);
        if response <= 0.0 {
            return 0.0;
        }
        let correction = (self.target_speed - self.speed(a, b, self.normal)) / response;
        let total = (self.normal_impulse + correction).max(0.0);
        let mut changed = (total - self.normal_impulse).abs() * response;
        self.apply(a, b, self.normal, total - self.normal_impulse);
        self.normal_impulse = total;

        let limit = self.friction * self.normal_impulse;
        for t in 0..2 {
            let tangent = self.tangents[t];
            let response = self.response(a, b, tangent);
            if response <= 0.0 {
                continue;
            }
            let correction = -self.speed(a, b, tangent) / response;
            let total = (self.tangent_impulse[t] + correction).clamp(-limit, limit);
            changed = changed.max((total - self.tangent_impulse[t]).abs() * response);
            self.apply(a, b, tangent, total - self.tangent_impulse[t]);
            self.tangent_impulse[t] = total;
        }
        changed
    }

    /// Move the pair apart by what this contact is still owed, split by
    /// inverse mass. Never pulls: another contact may have moved them
    /// further. Returns the distance moved.
    fn solve_position(&self, a: &mut Solid, b: &mut Solid) -> f32 {
        let weight = a.inv_mass + b.inv_mass;
        if weight <= 0.0 {
            return 0.0;
        }
        let owed = self.push - dot(sub(a.shift, b.shift), self.normal);
        if owed <= 0.0 {
            return 0.0;
        }
        a.shift = add(a.shift, scale(self.normal, owed * a.inv_mass / weight));
        b.shift = sub(b.shift, scale(self.normal, owed * b.inv_mass / weight));
        owed
    }
}

fn is_dynamic(body: &RigidBodyConfig) -> bool {
    body.enabled && body.motion == RigidBodyMotion::Dynamic
}

/// Two distinct elements of a slice, mutably (`a < b`)
fn pair_mut<T>(items: &mut [T], a: usize, b: usize) -> (&mut T, &mut T) {
    let (head, tail) = items.split_at_mut(b);
    (&mut head[a], &mut tail[0])
}

/// One probe of a body reaching inside the other body
struct Hit {
    point: V3,
    /// Unit normal out of the body whose SDF was probed
    normal: V3,
    depth: f32,
    /// By an inner sphere that stands in for flat faces (the sphere inscribed
    /// in a cube, cylinder or hub), not by a point or by a round body's own
    /// surface
    stand_in: bool,
}

/// Where `prober`'s probes reach inside `field`. `fallback` is the normal
/// where the field has none.
fn probe_hits(prober: &Collider, field: &Collider, fallback: V3) -> Vec<Hit> {
    let mut hits = Vec::new();
    let mut inner = Vec::new();
    let mut buried = false;
    for probe in &prober.probes {
        if length(sub(probe.center, field.position)) > field.bound_radius + probe.radius {
            continue;
        }
        let (distance, normal) = field.distance_and_normal(probe.center);
        let gap = distance - probe.radius;
        if gap >= 0.0 {
            continue;
        }
        let normal = normal.unwrap_or(fallback);
        // Midway between the probe's deepest point and the field's surface
        let point = sub(probe.center, scale(normal, probe.radius + 0.5 * gap));
        let is_inner = probe.radius > 0.0;
        buried |= is_inner && -gap > DEEP_OVERLAP * probe.radius;
        inner.push(is_inner);
        hits.push(Hit { point, normal, depth: -gap, stand_in: is_inner && !prober.is_round() });
    }
    if buried {
        // Deep inside: the inner spheres alone, and nothing outranks them
        let mut keep = inner.iter();
        hits.retain(|_| *keep.next().unwrap());
        for hit in &mut hits {
            hit.stand_in = false;
        }
    }
    hits
}

/// Contacts between bodies `i < j`
fn collide(i: usize, a: &Collider, j: usize, b: &Collider, contacts: &mut Vec<Contact>) {
    if length(sub(a.position, b.position)) > a.bound_radius + b.bound_radius {
        return;
    }
    // A sphere is probed by nobody: its own probe gives the exact contact,
    // and the other body's points inside it would add tilted copies of it.
    let mut hits = Vec::new();
    // a's probes in b: the normal out of b is the way a is pushed
    if a.is_sphere() || !b.is_sphere() {
        hits = probe_hits(a, b, FALLBACK_AXIS);
    }
    // b's probes in a: the normal out of a pushes b, so a goes the other way
    if !a.is_sphere() {
        for mut hit in probe_hits(b, a, scale(FALLBACK_AXIS, -1.0)) {
            hit.normal = scale(hit.normal, -1.0);
            hits.push(hit);
        }
    }
    // An inscribed sphere is how two flat faces find each other when no edge
    // or corner is inside (two equal cubes stacked squarely). Where a point
    // is in at least as deep, that point is the contact: the sphere would
    // add the normal of the other body's face, and between two faces a
    // fraction of a degree apart that pushed a resting cube sideways,
    // 0.7 mm/s, for as long as it rested.
    let deepest_point =
        hits.iter().filter(|hit| !hit.stand_in).map(|hit| hit.depth).fold(0.0, f32::max);
    for hit in hits {
        if hit.stand_in && hit.depth <= deepest_point {
            continue;
        }
        let (ra, rb) = (sub(hit.point, a.position), sub(hit.point, b.position));
        contacts.push(Contact::new(i, j, hit.normal, ra, rb, hit.depth, FRICTION));
    }
}

/// Resolve the contacts between the bodies at their current poses: corrects
/// the velocities, spins and positions of Dynamic bodies in place.
/// `arrival` holds each body's velocity before this frame's forces were
/// integrated (what a bounce is measured from), `dt` is the frame's simulated
/// time, `fluid_density` the SPH rest density (body mass is relative to it),
/// `voxels` the mesh SDF of Custom bodies.
pub fn resolve_body_contacts(
    bodies: &mut [RigidBodyConfig],
    arrival: &[V3],
    container: &ContainerConfig,
    dt: f32,
    fluid_density: f32,
    voxels: Option<&SdfData>,
) {
    let count = bodies.len().min(MAX_RIGID_BODIES);
    let bodies = &mut bodies[..count];
    if !bodies.iter().any(is_dynamic) {
        return;
    }

    let colliders: Vec<Option<Collider>> = bodies
        .iter()
        .map(|body| body.enabled.then(|| Collider::new(body, voxels)))
        .collect();
    let mut contacts = Vec::new();
    for i in 0..count {
        for j in i + 1..count {
            if !is_dynamic(&bodies[i]) && !is_dynamic(&bodies[j]) {
                continue;
            }
            if let (Some(a), Some(b)) = (&colliders[i], &colliders[j]) {
                collide(i, a, j, b, &mut contacts);
            }
        }
    }
    if contacts.is_empty() {
        return;
    }
    if std::env::var_os("RB_DEBUG").is_some() {
        let deepest = contacts.iter().map(|c| c.depth).fold(0.0, f32::max);
        eprintln!("rb contacts={} deepest={:.4}", contacts.len(), deepest);
    }

    // Solver bodies; the container is the extra last one
    let world = count;
    let mut solids: Vec<Solid> = bodies
        .iter()
        .enumerate()
        .map(|(i, body)| {
            Solid::new(body, arrival.get(i).copied().unwrap_or(body.velocity), fluid_density)
        })
        .collect();
    solids.push(Solid::fixed());

    let max_push = MAX_PUSH_SPEED * dt;
    let mut touching = [false; MAX_RIGID_BODIES];
    for contact in &mut contacts {
        touching[contact.a] = true;
        touching[contact.b] = true;
        // Bounce off the speed the bodies arrived with, not the speed this
        // frame's forces added: a body pressed onto another (gravity; under
        // water buoyancy gives a light body 1 m/s per frame and more) is
        // resting on it, not hitting it again every frame
        let (a, b) = (&solids[contact.a], &solids[contact.b]);
        let arriving = sub(
            add(a.arrival, cross(a.spin, contact.ra)),
            add(b.arrival, cross(b.spin, contact.rb)),
        );
        let impact = -dot(arriving, contact.normal);
        if impact > BOUNCE_SPEED {
            contact.target_speed = RESTITUTION * impact;
        }
        contact.push = (contact.depth - SLOP).clamp(0.0, max_push);
    }
    // The container faces under every body in contact: no bounce (the clamp
    // in integrate_rigid_body already did that), no friction, no torque
    for (i, body) in bodies.iter().enumerate() {
        if !touching[i] || !is_dynamic(body) {
            continue;
        }
        for (normal, gap) in container_face_gaps(body, container) {
            if gap < WALL_TOUCH {
                let mut contact = Contact::new(i, world, normal, [0.0; 3], [0.0; 3], -gap, 0.0);
                contact.push = contact.depth.max(0.0);
                contacts.push(contact);
            }
        }
    }

    // Sweep until a pass changes nothing: one or two passes for a single
    // contact, many for a body resting on several
    for _ in 0..VELOCITY_ITERATIONS {
        let mut changed = 0.0f32;
        for contact in &mut contacts {
            let (a, b) = pair_mut(&mut solids, contact.a, contact.b);
            changed = changed.max(contact.solve_velocity(a, b));
        }
        if changed < CONVERGED_SPEED {
            break;
        }
    }
    for _ in 0..POSITION_ITERATIONS {
        let mut moved = 0.0f32;
        for contact in &contacts {
            let (a, b) = pair_mut(&mut solids, contact.a, contact.b);
            moved = moved.max(contact.solve_position(a, b));
        }
        if moved < CONVERGED_SHIFT {
            break;
        }
    }

    for (i, body) in bodies.iter_mut().enumerate() {
        if !touching[i] || !is_dynamic(body) {
            continue;
        }
        body.velocity = solids[i].velocity;
        body.angular_velocity = solids[i].spin;
        body.position = add(body.position, solids[i].shift);
        // Backstop: the wall contacts above leave nothing for it to do
        // unless the iterations ran out
        clamp_rigid_body_to_container(body, container, true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{quat_from_euler_deg, quat_mul, quat_normalize, RigidBodyShape};
    use RigidBodyMotion::{Dynamic, Kinematic, Static};
    use RigidBodyShape::{Cube, Sphere, Torus};

    const DT: f32 = 1.0 / 60.0;
    const DENSITY: f32 = 1000.0;
    /// Default half_extent: sphere radius, cube half side
    const R: f32 = 0.15;

    fn body(shape: RigidBodyShape, motion: RigidBodyMotion, position: V3) -> RigidBodyConfig {
        RigidBodyConfig { shape, motion, position, ..Default::default() }
    }

    /// Contacts of bodies that arrived with the velocities they have
    fn resolve(bodies: &mut [RigidBodyConfig]) {
        let arrival: Vec<V3> = bodies.iter().map(|b| b.velocity).collect();
        resolve_body_contacts(bodies, &arrival, &ContainerConfig::default(), DT, DENSITY, None);
    }

    /// One frame in a vacuum: gravity, motion (as integrate_rigid_body moves
    /// and turns a body), contacts
    fn step(bodies: &mut [RigidBodyConfig], gravity: f32) {
        let arrival: Vec<V3> = bodies.iter().map(|b| b.velocity).collect();
        for body in bodies.iter_mut().filter(|b| b.motion == Dynamic) {
            body.velocity[1] -= gravity * DT;
            body.position = add(body.position, scale(body.velocity, DT));
            let (w, q) = (body.angular_velocity, body.orientation);
            let turn = quat_mul([w[0], w[1], w[2], 0.0], q);
            body.orientation = quat_normalize([0, 1, 2, 3].map(|i| q[i] + 0.5 * DT * turn[i]));
        }
        resolve_body_contacts(bodies, &arrival, &ContainerConfig::default(), DT, DENSITY, None);
    }

    fn apart(bodies: &[RigidBodyConfig]) -> V3 {
        sub(bodies[0].position, bodies[1].position)
    }

    /// The reported bug: two bodies added at the default position
    #[test]
    fn coincident_spheres_ease_apart() {
        let mut bodies = [body(Sphere, Dynamic, [0.0; 3]), body(Sphere, Dynamic, [0.0; 3])];
        step(&mut bodies, 0.0);
        assert!((length(apart(&bodies)) - MAX_PUSH_SPEED * DT).abs() < 1e-5);
        for _ in 0..30 {
            step(&mut bodies, 0.0);
        }
        let offset = apart(&bodies);
        assert!(offset[0] > 2.0 * R - 2.0 * SLOP && offset[0] < 2.0 * R + 1e-4, "{offset:?}");
        assert!(offset[1].abs() < 1e-5 && offset[2].abs() < 1e-5);
        assert!(length(bodies[0].velocity) < 1e-5 && length(bodies[1].velocity) < 1e-5);
    }

    #[test]
    fn coincident_cubes_separate_without_drifting() {
        let mut bodies = [body(Cube, Dynamic, [0.0; 3]), body(Cube, Dynamic, [0.0; 3])];
        for _ in 0..40 {
            step(&mut bodies, 0.0);
        }
        let offset = apart(&bodies);
        assert!(offset[0] > 2.0 * R - 2.0 * SLOP && offset[0] < 2.0 * R + 1e-4, "{offset:?}");
        assert!(offset[1].abs() < 1e-4 && offset[2].abs() < 1e-4, "{offset:?}");
    }

    #[test]
    fn head_on_spheres_bounce_and_keep_momentum() {
        let mut bodies = [body(Sphere, Dynamic, [-0.14, 0.0, 0.0]), body(Sphere, Dynamic, [0.14, 0.0, 0.0])];
        bodies[0].velocity = [1.0, 0.0, 0.0];
        bodies[1].velocity = [-1.0, 0.0, 0.0];
        bodies[1].relative_density = 2.0;
        let mass = [bodies[0].mass(DENSITY), bodies[1].mass(DENSITY)];
        let momentum = |b: &[RigidBodyConfig]| mass[0] * b[0].velocity[0] + mass[1] * b[1].velocity[0];
        let before = momentum(&bodies);
        resolve(&mut bodies);
        assert!((momentum(&bodies) - before).abs() < 1e-3 * before.abs());
        // Closing at 2 m/s, parting at 0.3 of that
        assert!((bodies[1].velocity[0] - bodies[0].velocity[0] - 2.0 * RESTITUTION).abs() < 1e-4);
        assert!(bodies[0].angular_velocity == [0.0; 3]);
    }

    #[test]
    fn sphere_comes_to_rest_on_a_static_cube() {
        let mut bodies = [body(Cube, Static, [0.0, -0.5, 0.0]), body(Sphere, Dynamic, [0.02, -0.15, 0.01])];
        for _ in 0..300 {
            step(&mut bodies, 9.81);
        }
        // Cube top at -0.35: the sphere's centre rests one radius above
        assert!((bodies[1].position[1] + 0.2).abs() < 0.004, "{:?}", bodies[1].position);
        assert!(length(bodies[1].velocity) < 1e-3);
        assert!(bodies[0].position == [0.0, -0.5, 0.0]);
    }

    /// Resting on many points needs the solver to converge: at 8 sweeps this
    /// cube crept sideways and fell off
    #[test]
    fn offset_cube_stays_on_a_static_cube() {
        let start = [0.1, -0.15, 0.05];
        let mut bodies = [body(Cube, Static, [0.0, -0.5, 0.0]), body(Cube, Dynamic, start)];
        for _ in 0..600 {
            step(&mut bodies, 9.81);
        }
        let p = bodies[1].position;
        assert!((p[0] - start[0]).abs() < 1e-3 && (p[2] - start[2]).abs() < 1e-3, "{p:?}");
        assert!((p[1] + 0.2).abs() < 0.004, "{p:?}");
    }

    /// A non-convex field: a ball too wide for a torus's hole sits in it
    #[test]
    fn ball_rests_in_a_torus() {
        let mut bodies = [body(Torus, Static, [0.0, -0.5, 0.0]), body(Sphere, Dynamic, [0.0, -0.2, 0.0])];
        for _ in 0..300 {
            step(&mut bodies, 9.81);
        }
        // Ball centre to the tube's centre circle (radius 0.15): 0.15 + 0.045,
        // so 0.1246 above the torus's plane
        let height = bodies[1].position[1] + 0.5;
        assert!((height - 0.1246).abs() < 0.004, "{height}");
    }

    /// A cube set down a fraction of a degree off level stays that way (the
    /// position push does not turn bodies), so its faces never quite match
    /// the faces under it: no contact may carry the wrong face's normal
    #[test]
    fn slightly_tilted_cube_does_not_creep() {
        let start = [0.068, -0.148, 0.036];
        let mut bodies = [body(Cube, Static, [0.0, -0.6, 0.0]), body(Cube, Dynamic, start)];
        bodies[0].half_extent = 0.3;
        bodies[1].orientation = quat_from_euler_deg([0.5, 7.0, 0.3]);
        for _ in 0..600 {
            step(&mut bodies, 9.81);
        }
        let p = bodies[1].position;
        assert!((p[0] - start[0]).abs() < 1e-4 && (p[2] - start[2]).abs() < 1e-4, "{p:?}");
    }

    /// A light ball held under a ceiling by buoyancy gains more speed per
    /// frame than a bounce needs: it must rest there, not hammer
    #[test]
    fn pressed_ball_rests_without_bouncing() {
        let mut bodies = [body(Cube, Static, [0.0, 0.3, 0.0]), body(Sphere, Dynamic, [0.01, -0.05, 0.0])];
        let mut heights = Vec::new();
        for _ in 0..300 {
            // 60 m/s^2 upward: 1 m/s per frame
            step(&mut bodies, -60.0);
            heights.push(bodies[1].position[1]);
        }
        let settled = &heights[200..];
        let (low, high) = settled.iter().fold((f32::MAX, f32::MIN), |(l, h), &y| (l.min(y), h.max(y)));
        assert!(high - low < 1e-5, "{low} .. {high}");
        // Cube bottom at 0.15: the ball's centre one radius below
        assert!((high - 0.0).abs() < 0.004, "{high}");
    }

    #[test]
    fn wall_stays_solid_under_a_push() {
        let wall = -ContainerConfig::default().half_width();
        let mut bodies = [
            body(Sphere, Dynamic, [wall + R, 0.0, 0.0]),
            body(Sphere, Dynamic, [wall + 3.0 * R - 0.02, 0.0, 0.0]),
        ];
        resolve(&mut bodies);
        assert!((bodies[0].position[0] - (wall + R)).abs() < 1e-4, "{:?}", bodies[0].position);
        assert!((apart(&bodies)[0] + 2.0 * R - SLOP).abs() < 1e-3);
    }

    #[test]
    fn kinematic_body_bats_without_moving() {
        let mut bodies = [body(Cube, Kinematic, [0.0; 3]), body(Sphere, Dynamic, [0.28, 0.0, 0.0])];
        bodies[0].angular_velocity = [0.0, 5.0, 0.0];
        bodies[1].velocity = [-1.0, 0.0, 0.0];
        resolve(&mut bodies);
        assert!(bodies[0].position == [0.0; 3] && bodies[0].velocity == [0.0; 3]);
        assert!(bodies[1].velocity[0] > 0.0, "{:?}", bodies[1].velocity);
        // The face under the contact moves toward -Z: friction drags the ball along
        assert!(bodies[1].velocity[2] < -0.01, "{:?}", bodies[1].velocity);
    }
}
