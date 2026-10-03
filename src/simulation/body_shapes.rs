//! CPU collision shapes for the rigid body contact pass (`body_contact.rs`).
//!
//! Every body is seen two ways: as a signed distance function (the same
//! surface the fluid shaders push particles out of) and as a set of probes,
//! spheres and points fixed in the body. Two bodies touch where a probe of
//! one reaches into the SDF of the other, so any pair of shapes collides
//! through the same code.
//!
//! What a probe set holds, and why. A probe's contact takes the OTHER body's
//! surface normal at the probe, which is the true contact normal only where
//! the probe is the feature that touches first:
//! - inner spheres (a sphere is one; a torus is a ring of them; the sphere
//!   inscribed in a cube or cylinder): exact against any SDF, and what lets
//!   two flat faces meet;
//! - points on sharp features (corners, edges, rims) and on curved surfaces
//!   (cylinder barrel, the mesh), which do touch first.
//!
//! Not points inside a flat face: against a curved body they sit beside the
//! real contact and read a tilted normal (a ball resting on a cube was
//! pushed sideways by the face points around it). What reaches into a face
//! is the other body's own probe.
//!
//! Keep the SDFs in sync with `body_shape_sdf` / `propeller_sdf` in
//! `shaders/body_shapes_common.wgsl` and the `PROP_*` constants in
//! `state/rigid_body.rs`: the fluid and the other bodies must agree on where
//! a body's surface is.

use crate::render::mesh_loader::SdfData;
use crate::state::{
    quat_to_rotation_rows, RigidBodyConfig, RigidBodyShape, PROP_BLADE_CENTER, PROP_BLADE_HALF,
    PROP_HUB_HALF_HEIGHT, PROP_HUB_RADIUS,
};

pub type V3 = [f32; 3];

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn length(a: V3) -> f32 {
    dot(a, a).sqrt()
}

/// Bounding radius in units of half_extent (1.75 covers the cube corner at
/// sqrt(3), like the integrate shader's early-out)
const BOUND_SCALE: f32 = 1.75;
/// A gradient shorter than this (a true SDF has length 1) has no direction:
/// the point sits on the shape's centre or medial axis
const MIN_GRADIENT: f32 = 1e-3;
/// Points per cube edge, corners included. Two edges crossing between
/// points go unnoticed until they overlap by half a spacing (half_extent / 10).
const CUBE_GRID: usize = 11;
/// Probes around a cylinder ring / along the torus circle
const RING_PROBES: usize = 24;
const TORUS_PROBES: usize = 32;
/// Torus tube radius in units of half_extent
const TORUS_MINOR: f32 = 0.3;

/// A sphere fixed in a body, world space. Radius 0 = a point on the surface.
#[derive(Clone, Copy, Debug)]
pub struct Probe {
    pub center: V3,
    pub radius: f32,
}

/// One body as the contact pass sees it at its current pose
pub struct Collider<'a> {
    shape: RigidBodyShape,
    half_extent: f32,
    blades: u32,
    pitch: f32,
    voxels: Option<&'a SdfData>,
    /// World -> body-local rotation rows (`quat_to_rotation_rows`)
    rows: [V3; 3],
    pub position: V3,
    pub bound_radius: f32,
    pub probes: Vec<Probe>,
}

impl<'a> Collider<'a> {
    /// `voxels` is the mesh SDF shared by Custom bodies; without it a Custom
    /// body collides as a sphere (as its volume and inertia already assume)
    pub fn new(body: &RigidBodyConfig, voxels: Option<&'a SdfData>) -> Self {
        let rows4 = quat_to_rotation_rows(body.orientation);
        let shape = match (body.shape, voxels) {
            (RigidBodyShape::Custom, None) => RigidBodyShape::Sphere,
            (shape, _) => shape,
        };
        let bound_scale = if shape == RigidBodyShape::Sphere { 1.0 } else { BOUND_SCALE };
        let mut collider = Self {
            shape,
            half_extent: body.half_extent,
            blades: body.prop_blades.max(1),
            pitch: body.prop_pitch_deg.to_radians(),
            voxels,
            rows: [
                [rows4[0][0], rows4[0][1], rows4[0][2]],
                [rows4[1][0], rows4[1][1], rows4[1][2]],
                [rows4[2][0], rows4[2][1], rows4[2][2]],
            ],
            position: body.position,
            bound_radius: body.half_extent * bound_scale,
            probes: Vec::new(),
        };
        collider.probes = collider
            .local_probes()
            .into_iter()
            .map(|p| Probe { center: collider.to_world(p.center), radius: p.radius })
            .collect();
        collider
    }

    /// A sphere's single probe is exact against any SDF: it needs no help
    /// from the other body's probes
    pub fn is_sphere(&self) -> bool {
        self.shape == RigidBodyShape::Sphere
    }

    /// The inner spheres ARE the body (sphere, torus), rather than standing
    /// in for the flat faces around them (cube, cylinder, hub, mesh)
    pub fn is_round(&self) -> bool {
        matches!(self.shape, RigidBodyShape::Sphere | RigidBodyShape::Torus)
    }

    /// World point -> body-local
    fn to_local(&self, world: V3) -> V3 {
        let rel = sub(world, self.position);
        [dot(self.rows[0], rel), dot(self.rows[1], rel), dot(self.rows[2], rel)]
    }

    /// Body-local direction -> world (transpose multiply, as the shaders do)
    fn rotate_to_world(&self, local: V3) -> V3 {
        let r = &self.rows;
        [
            r[0][0] * local[0] + r[1][0] * local[1] + r[2][0] * local[2],
            r[0][1] * local[0] + r[1][1] * local[1] + r[2][1] * local[2],
            r[0][2] * local[0] + r[1][2] * local[1] + r[2][2] * local[2],
        ]
    }

    fn to_world(&self, local: V3) -> V3 {
        add(self.position, self.rotate_to_world(local))
    }

    /// Signed distance from a world point to the body's surface (negative
    /// inside) and the outward unit normal there. The normal is `None` where
    /// the distance field has no direction (the centre of a sphere or cube).
    pub fn distance_and_normal(&self, world: V3) -> (f32, Option<V3>) {
        let p = self.to_local(world);
        let distance = self.local_sdf(p);
        let gradient = if self.shape == RigidBodyShape::Sphere {
            // Exact, and well defined down to much smaller offsets than a
            // finite difference
            let len = length(p);
            if len > 1e-6 { scale(p, 1.0 / len) } else { [0.0; 3] }
        } else {
            // The voxel field is piecewise linear: difference over one voxel
            let eps = match self.shape {
                RigidBodyShape::Custom => self.half_extent / 16.0,
                _ => (0.01 * self.half_extent).max(1e-4),
            };
            let mut g = [0.0f32; 3];
            for (axis, slot) in g.iter_mut().enumerate() {
                let mut hi = p;
                let mut lo = p;
                hi[axis] += eps;
                lo[axis] -= eps;
                *slot = (self.local_sdf(hi) - self.local_sdf(lo)) / (2.0 * eps);
            }
            g
        };
        let len = length(gradient);
        if len < MIN_GRADIENT {
            return (distance, None);
        }
        (distance, Some(self.rotate_to_world(scale(gradient, 1.0 / len))))
    }

    /// Signed distance in body-local space
    fn local_sdf(&self, p: V3) -> f32 {
        let he = self.half_extent;
        match self.shape {
            RigidBodyShape::Sphere => length(p) - he,
            RigidBodyShape::Cylinder => {
                let radial = p[0].hypot(p[2]) - he;
                let cap = p[1].abs() - he;
                radial.max(cap).min(0.0) + radial.max(0.0).hypot(cap.max(0.0))
            }
            RigidBodyShape::Torus => (p[0].hypot(p[2]) - he).hypot(p[1]) - TORUS_MINOR * he,
            RigidBodyShape::Propeller => propeller_sdf(p, he, self.blades, self.pitch),
            RigidBodyShape::Custom => match self.voxels {
                // The voxel field is in units of half_extent
                Some(voxels) => voxels.sample(scale(p, 1.0 / he)) * he,
                None => length(p) - he,
            },
            RigidBodyShape::Cube => box_sdf(p, [he; 3]),
        }
    }

    /// The probe set in body-local space
    fn local_probes(&self) -> Vec<Probe> {
        let he = self.half_extent;
        let point = |center: V3| Probe { center, radius: 0.0 };
        let core = |radius: f32| Probe { center: [0.0; 3], radius };
        match self.shape {
            RigidBodyShape::Sphere => vec![core(he)],
            RigidBodyShape::Cube => {
                // Inscribed sphere + the twelve edges: grid points that lie
                // on at least two faces
                let mut probes = vec![core(he)];
                let last = CUBE_GRID - 1;
                let at = |i: usize| he * (2.0 * i as f32 / last as f32 - 1.0);
                for i in 0..CUBE_GRID {
                    for j in 0..CUBE_GRID {
                        for k in 0..CUBE_GRID {
                            let faces = [i, j, k].iter().filter(|&&n| n == 0 || n == last).count();
                            if faces >= 2 {
                                probes.push(point([at(i), at(j), at(k)]));
                            }
                        }
                    }
                }
                probes
            }
            RigidBodyShape::Cylinder => {
                // Inscribed sphere (it touches both caps) + five barrel
                // rings, the outer two being the rims
                let mut probes = vec![core(he)];
                for level in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
                    probes.extend(ring(he, level * he, RING_PROBES).map(point));
                }
                probes
            }
            RigidBodyShape::Torus => {
                // The tube is a sphere swept along the ring: sample the sweep
                ring(he, 0.0, TORUS_PROBES)
                    .map(|center| Probe { center, radius: TORUS_MINOR * he })
                    .collect()
            }
            RigidBodyShape::Propeller => propeller_probes(he, self.blades, self.pitch),
            RigidBodyShape::Custom => {
                let Some(voxels) = self.voxels else { return vec![core(he)] };
                let mut probes: Vec<Probe> =
                    voxels.surface_points.iter().map(|&p| point(scale(p, he))).collect();
                let center_depth = -voxels.sample([0.0; 3]) * he;
                if center_depth > 0.0 {
                    probes.push(core(center_depth));
                }
                probes
            }
        }
    }
}

/// Exact box SDF
fn box_sdf(p: V3, half: V3) -> f32 {
    let d = [p[0].abs() - half[0], p[1].abs() - half[1], p[2].abs() - half[2]];
    let outside = length([d[0].max(0.0), d[1].max(0.0), d[2].max(0.0)]);
    outside + d[0].max(d[1]).max(d[2]).min(0.0)
}

/// Hub cylinder + N pitched blades via angular domain repetition
/// (`propeller_sdf` in body_shapes_common.wgsl)
fn propeller_sdf(p: V3, he: f32, blades: u32, pitch: f32) -> f32 {
    let hub = (p[0].hypot(p[2]) - PROP_HUB_RADIUS * he).max(p[1].abs() - PROP_HUB_HALF_HEIGHT * he);

    // Snap to the nearest blade sector and rotate that blade onto +X
    let sector = std::f32::consts::TAU / blades as f32;
    let snapped = (p[2].atan2(p[0]) / sector).round() * sector;
    let (sn, cs) = snapped.sin_cos();
    let q = [cs * p[0] + sn * p[2], p[1], -sn * p[0] + cs * p[2]];
    // Un-pitch around the radial (X) axis
    let (sp, cp) = pitch.sin_cos();
    let v = [q[0] - PROP_BLADE_CENTER * he, cp * q[1] + sp * q[2], -sp * q[1] + cp * q[2]];

    hub.min(box_sdf(v, scale(PROP_BLADE_HALF, he)))
}

/// Hub: inscribed sphere + both rims. Blades: the outline of the two large
/// faces of each (the inverse of the transform in `propeller_sdf`).
fn propeller_probes(he: f32, blades: u32, pitch: f32) -> Vec<Probe> {
    let point = |center: V3| Probe { center, radius: 0.0 };
    let hub_radius = PROP_HUB_RADIUS * he;
    let hub_half = PROP_HUB_HALF_HEIGHT * he;
    let mut probes = vec![Probe { center: [0.0; 3], radius: hub_radius.min(hub_half) }];
    for rim in [-hub_half, hub_half] {
        probes.extend(ring(hub_radius, rim, RING_PROBES / 2).map(point));
    }

    const ALONG: usize = 7;
    const ACROSS: usize = 3;
    let (sp, cp) = pitch.sin_cos();
    let sector = std::f32::consts::TAU / blades as f32;
    for blade in 0..blades {
        let (sn, cs) = (blade as f32 * sector).sin_cos();
        for i in 0..ALONG {
            for j in 0..ACROSS {
                let on_outline = i == 0 || i == ALONG - 1 || j == 0 || j == ACROSS - 1;
                if !on_outline {
                    continue;
                }
                for side in [-1.0f32, 1.0] {
                    // Blade frame: radial, chord, thickness
                    let x = PROP_BLADE_CENTER
                        + PROP_BLADE_HALF[0] * (2.0 * i as f32 / (ALONG - 1) as f32 - 1.0);
                    let y = PROP_BLADE_HALF[1] * (2.0 * j as f32 / (ACROSS - 1) as f32 - 1.0);
                    let z = PROP_BLADE_HALF[2] * side;
                    // Pitch, then rotate the blade into its sector
                    let q = [x, cp * y - sp * z, sp * y + cp * z];
                    probes.push(point(scale([cs * q[0] - sn * q[2], q[1], sn * q[0] + cs * q[2]], he)));
                }
            }
        }
    }
    probes
}

/// `count` points on the circle of `radius` around the Y axis at height `y`
fn ring(radius: f32, y: f32, count: usize) -> impl Iterator<Item = V3> {
    (0..count).map(move |i| {
        let (s, c) = (i as f32 * std::f32::consts::TAU / count as f32).sin_cos();
        [radius * c, y, radius * s]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{quat_from_euler_deg, quat_rotate_vec, RigidBodyMotion};

    fn body(shape: RigidBodyShape) -> RigidBodyConfig {
        RigidBodyConfig {
            shape,
            motion: RigidBodyMotion::Dynamic,
            position: [0.3, -0.2, 0.1],
            orientation: quat_from_euler_deg([20.0, 30.0, 40.0]),
            ..Default::default()
        }
    }

    /// Surface probes lie on the SDF's zero set and inner spheres are
    /// inscribed, for every analytic shape at a tilted pose: checks each
    /// probe generator against its SDF, and to_world against to_local
    #[test]
    fn probes_match_their_sdf() {
        use RigidBodyShape::*;
        for shape in [Cube, Sphere, Cylinder, Torus, Propeller] {
            let body = body(shape);
            let collider = Collider::new(&body, None);
            let tolerance = 1e-4 * body.half_extent;
            assert!(!collider.probes.is_empty());
            for probe in &collider.probes {
                let (distance, _) = collider.distance_and_normal(probe.center);
                // Blade roots sit inside the hub: surface points may be
                // interior there, never outside
                let surface_error = if shape == Propeller && probe.radius == 0.0 {
                    (distance + probe.radius).max(0.0)
                } else {
                    (distance + probe.radius).abs()
                };
                assert!(surface_error < tolerance, "{shape:?}: probe {probe:?} at distance {distance}");
                assert!(length(sub(probe.center, body.position)) <= collider.bound_radius);
            }
        }
    }

    /// The collider's frame is the renderer's: a cube corner carried to world
    /// space by the body quaternion is on the collider's surface, and the
    /// mirrored rotation (the transposed convention) is not
    #[test]
    fn frame_matches_body_orientation() {
        let mut body = body(RigidBodyShape::Cube);
        body.orientation = quat_from_euler_deg([0.0, 0.0, 30.0]);
        let he = body.half_extent;
        let collider = Collider::new(&body, None);
        let corner = add(body.position, quat_rotate_vec(body.orientation, [he, he, 0.0]));
        assert!(collider.distance_and_normal(corner).0.abs() < 1e-5);
        let mirrored = quat_from_euler_deg([0.0, 0.0, -30.0]);
        let wrong = add(body.position, quat_rotate_vec(mirrored, [he, he, 0.0]));
        assert!(collider.distance_and_normal(wrong).0 > 0.3 * he);
    }

    /// Normals point out of the body, in world space
    #[test]
    fn normal_points_outward() {
        let body = body(RigidBodyShape::Cylinder);
        let collider = Collider::new(&body, None);
        let axis = quat_rotate_vec(body.orientation, [0.0, 1.0, 0.0]);
        let above = add(body.position, scale(axis, 1.5 * body.half_extent));
        let (distance, normal) = collider.distance_and_normal(above);
        assert!((distance - 0.5 * body.half_extent).abs() < 1e-5);
        assert!(dot(normal.unwrap(), axis) > 0.999);
        // The centre has no nearest face: callers need their own direction
        assert!(collider.distance_and_normal(body.position).1.is_none());
    }
}
