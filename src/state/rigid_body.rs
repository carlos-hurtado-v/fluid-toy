//! Rigid body configuration, quaternion helpers, CPU integration, and GPU types
//!
//! Bodies come in three motion types:
//! - `Static` — immovable obstacle: pose from GUI (position + euler), zero
//!   velocity, ignores fluid forces, exempt from container clamp (may be
//!   embedded in walls/floor to build scenes).
//! - `Kinematic` — scripted motion: spins at `spin_rpm` around the body-local
//!   Y axis (as oriented by the euler base rotation). Ignores fluid forces
//!   but drives the fluid through its surface velocity (v + omega x r in the
//!   integrate shader).
//! - `Dynamic` — full physics: fluid reaction forces + gravity, CPU
//!   integration, container collision.

use super::simulation::ContainerConfig;

/// Maximum number of rigid bodies (GPU buffer capacity)
pub const MAX_RIGID_BODIES: usize = 8;

/// Rigid body shape types (repr matches GPU constants)
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RigidBodyShape {
    Cube = 0,
    Sphere = 1,
    Cylinder = 2,
    Torus = 3,
    Custom = 4,
    Propeller = 5,
}

/// Motion type (repr matches GPU constants)
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RigidBodyMotion {
    Static = 0,
    Kinematic = 1,
    Dynamic = 2,
}

// Propeller proportions in units of half_extent (body-local, spin axis = Y).
// Mesh generation (rigid_body.wgsl) and the SDF (sph_integrate_3d.wgsl) must
// stay in sync with these.
pub const PROP_HUB_RADIUS: f32 = 0.25;
pub const PROP_HUB_HALF_HEIGHT: f32 = 0.30;
pub const PROP_BLADE_CENTER: f32 = 0.55;
pub const PROP_BLADE_HALF: [f32; 3] = [0.44, 0.18, 0.045]; // radial, chord, thickness

impl RigidBodyShape {
    /// Number of vertices for the procedural mesh (propeller: hub only —
    /// blades add 36 per blade, see `RigidBodyConfig::render_vertex_count`)
    pub fn vertex_count(self) -> u32 {
        match self {
            RigidBodyShape::Cube => 36,        // 6 faces × 2 tri × 3 verts
            RigidBodyShape::Sphere => 3072,    // 32 slices × 16 stacks × 6
            RigidBodyShape::Cylinder => 384,   // 32 segments: barrel(192) + 2 caps(192)
            RigidBodyShape::Torus => 3072,     // 32 major × 16 minor × 6
            RigidBodyShape::Custom => 0,       // Uses index buffer, not vertex_count
            RigidBodyShape::Propeller => 384,  // hub cylinder; + blades × 36
        }
    }

    /// Volume of the shape given half_extent
    pub fn volume(self, half_extent: f32) -> f32 {
        let he = half_extent;
        match self {
            RigidBodyShape::Cube => {
                let side = 2.0 * he;
                side * side * side
            }
            RigidBodyShape::Sphere => {
                (4.0 / 3.0) * std::f32::consts::PI * he * he * he
            }
            RigidBodyShape::Cylinder => {
                // radius=he, height=2*he
                std::f32::consts::PI * he * he * (2.0 * he)
            }
            RigidBodyShape::Torus => {
                // major=he, minor=0.3*he
                let small_r = he * 0.3;
                2.0 * std::f32::consts::PI * std::f32::consts::PI * he * small_r * small_r
            }
            RigidBodyShape::Custom => {
                // Approximate as sphere
                (4.0 / 3.0) * std::f32::consts::PI * he * he * he
            }
            RigidBodyShape::Propeller => {
                // Hub cylinder + 3 blade boxes (blade count barely matters here)
                let hub = std::f32::consts::PI
                    * (PROP_HUB_RADIUS * he).powi(2)
                    * (2.0 * PROP_HUB_HALF_HEIGHT * he);
                let blade = 8.0
                    * PROP_BLADE_HALF[0] * PROP_BLADE_HALF[1] * PROP_BLADE_HALF[2]
                    * he * he * he;
                hub + 3.0 * blade
            }
        }
    }

    /// Surface area of the shape (drives the expected wetted-shell particle
    /// count for the submerged-fraction estimate)
    pub fn surface_area(self, half_extent: f32, prop_blades: u32) -> f32 {
        let he2 = half_extent * half_extent;
        let pi = std::f32::consts::PI;
        match self {
            RigidBodyShape::Cube => 24.0 * he2,
            RigidBodyShape::Sphere => 4.0 * pi * he2,
            // barrel 2πr·h (r=he, h=2he) + two caps 2πr²
            RigidBodyShape::Cylinder => 6.0 * pi * he2,
            // 4π²·R·r with R=he, r=0.3he
            RigidBodyShape::Torus => 1.2 * pi * pi * he2,
            // Approximate as sphere
            RigidBodyShape::Custom => 4.0 * pi * he2,
            RigidBodyShape::Propeller => {
                let hub = 2.0 * pi * PROP_HUB_RADIUS * (2.0 * PROP_HUB_HALF_HEIGHT)
                    + 2.0 * pi * PROP_HUB_RADIUS * PROP_HUB_RADIUS;
                let blade_faces = 2.0 * (2.0 * PROP_BLADE_HALF[0]) * (2.0 * PROP_BLADE_HALF[1]);
                (hub + blade_faces * prop_blades.max(1) as f32) * he2
            }
        }
    }

    /// Moment of inertia for a solid body of given mass and half_extent
    pub fn moment_of_inertia(self, mass: f32, half_extent: f32) -> f32 {
        match self {
            RigidBodyShape::Cube => {
                let side = 2.0 * half_extent;
                (1.0 / 6.0) * mass * side * side
            }
            RigidBodyShape::Sphere => {
                (2.0 / 5.0) * mass * half_extent * half_extent
            }
            RigidBodyShape::Cylinder => {
                // Approximate: average of axial and transverse
                let r = half_extent;
                let h = 2.0 * half_extent;
                (1.0 / 12.0) * mass * (3.0 * r * r + h * h)
            }
            RigidBodyShape::Torus => {
                let big_r = half_extent;
                let small_r = half_extent * 0.3;
                mass * (big_r * big_r + 0.75 * small_r * small_r)
            }
            RigidBodyShape::Custom => {
                // Approximate as sphere
                (2.0 / 5.0) * mass * half_extent * half_extent
            }
            RigidBodyShape::Propeller => {
                // Mass concentrated in blades reaching half_extent
                0.5 * mass * half_extent * half_extent
            }
        }
    }
}

/// Rigid body configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RigidBodyConfig {
    /// Whether the rigid body is active in the scene
    pub enabled: bool,
    /// Motion type: static obstacle, scripted kinematic, or full physics
    pub motion: RigidBodyMotion,
    /// Shape type
    pub shape: RigidBodyShape,
    /// Position in world space
    pub position: [f32; 3],
    /// Linear velocity (Dynamic only; Static/Kinematic force zero)
    pub velocity: [f32; 3],
    /// Live orientation quaternion [x, y, z, w]. For Static/Kinematic this is
    /// derived from `euler_deg` (+ spin) each frame; for Dynamic it evolves.
    pub orientation: [f32; 4],
    /// Angular velocity (world space, radians/sec). Dynamic: integrated from
    /// fluid torque. Kinematic: set from spin_rpm. Static: zero.
    pub angular_velocity: [f32; 3],
    /// Base orientation as euler angles in degrees (X pitch, Y yaw, Z roll).
    /// Drives Static/Kinematic pose; ignored while Dynamic.
    pub euler_deg: [f32; 3],
    /// Kinematic spin rate (RPM) around the body-local Y axis
    pub spin_rpm: f32,
    /// Propeller shape: blade count
    pub prop_blades: u32,
    /// Propeller shape: blade pitch in degrees (0 = flat paddle)
    pub prop_pitch_deg: f32,
    /// Half-extent (radius for sphere/cylinder/torus/propeller, half side for cube)
    pub half_extent: f32,
    /// Body density relative to the fluid rest density (specific gravity):
    /// 1.0 = neutral buoyancy, < 1 floats, > 1 sinks. Relative semantics stay
    /// correct when kernel_radius retunes the SPH rest density (~104k at the
    /// 2026-07 defaults — absolute values drifted badly when h changed).
    pub relative_density: f32,
    /// Render color (RGB)
    pub color: [f32; 3],
    /// Accumulated kinematic spin phase (radians, runtime only)
    #[serde(skip)]
    pub spin_angle: f32,
}

impl Default for RigidBodyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            motion: RigidBodyMotion::Static,
            shape: RigidBodyShape::Cube,
            position: [0.0, 0.2, 0.0],
            velocity: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],  // Identity quaternion
            angular_velocity: [0.0; 3],
            euler_deg: [0.0; 3],
            spin_rpm: 120.0,
            prop_blades: 3,
            prop_pitch_deg: 35.0,
            half_extent: 0.15,
            relative_density: 0.5,  // Half the fluid density → floats half-submerged-ish
            color: [0.9, 0.7, 0.2],  // Yellow/gold
            spin_angle: 0.0,
        }
    }
}

impl RigidBodyConfig {
    /// Base orientation quaternion from the euler sliders
    pub fn base_quat(&self) -> [f32; 4] {
        quat_from_euler_deg(self.euler_deg)
    }

    /// Kinematic spin rate in radians/sec
    pub fn spin_rate(&self) -> f32 {
        self.spin_rpm * std::f32::consts::TAU / 60.0
    }

    /// Recompute pose/velocities for Static and Kinematic bodies (applies
    /// euler slider edits and the current spin phase). No-op for Dynamic.
    pub fn refresh_pose(&mut self) {
        match self.motion {
            RigidBodyMotion::Static => {
                self.orientation = self.base_quat();
                self.velocity = [0.0; 3];
                self.angular_velocity = [0.0; 3];
            }
            RigidBodyMotion::Kinematic => {
                let base = self.base_quat();
                let spin = quat_axis_angle([0.0, 1.0, 0.0], self.spin_angle);
                // Spin around the body-local Y axis: right-multiply
                self.orientation = quat_normalize(quat_mul(base, spin));
                self.velocity = [0.0; 3];
                // World-space spin axis = base-rotated local Y (spin does not
                // move its own axis)
                let axis = quat_rotate_vec(base, [0.0, 1.0, 0.0]);
                let rate = self.spin_rate();
                self.angular_velocity = [axis[0] * rate, axis[1] * rate, axis[2] * rate];
            }
            RigidBodyMotion::Dynamic => {}
        }
    }

    /// Advance the kinematic spin phase by dt and refresh the pose
    pub fn advance_kinematic(&mut self, dt: f32) {
        if self.motion == RigidBodyMotion::Kinematic {
            self.spin_angle =
                (self.spin_angle + self.spin_rate() * dt).rem_euclid(std::f32::consts::TAU);
            self.refresh_pose();
        }
    }

    /// Vertex count for the procedural renderer (propeller depends on blades)
    pub fn render_vertex_count(&self) -> u32 {
        match self.shape {
            RigidBodyShape::Propeller => {
                RigidBodyShape::Propeller.vertex_count() + self.prop_blades * 36
            }
            s => s.vertex_count(),
        }
    }

    pub fn to_gpu_rigid_body(&self, wall_stiffness: f32) -> GpuRigidBody {
        let rows = quat_to_rotation_rows(self.orientation);
        GpuRigidBody {
            position: self.position,
            half_extent: self.half_extent,
            velocity: self.velocity,
            is_active: if self.enabled { 1 } else { 0 },
            stiffness: wall_stiffness,
            shape: self.shape as u32,
            motion: self.motion as u32,
            prop_blades: self.prop_blades.max(1),
            angular_velocity: self.angular_velocity,
            prop_pitch: self.prop_pitch_deg.to_radians(),
            rot_row0: rows[0],
            rot_row1: rows[1],
            rot_row2: rows[2],
        }
    }

    pub fn to_gpu_render(&self, light_dir: [f32; 3]) -> GpuRigidBodyRender {
        let rows = quat_to_rotation_rows(self.orientation);
        GpuRigidBodyRender {
            position: self.position,
            half_extent: self.half_extent,
            color: [self.color[0], self.color[1], self.color[2], 1.0],
            light_dir,
            shape: self.shape as u32,
            rot_row0: rows[0],
            rot_row1: rows[1],
            rot_row2: rows[2],
            prop_blades: self.prop_blades.max(1),
            prop_pitch: self.prop_pitch_deg.to_radians(),
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

// --- Quaternion helpers ---

/// Normalize a quaternion [x, y, z, w]
pub fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]).sqrt();
    if len < 1e-10 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    [q[0]/len, q[1]/len, q[2]/len, q[3]/len]
}

/// Quaternion multiplication: a * b (Hamilton product)
pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ]
}

/// Quaternion from axis (normalized) + angle in radians
pub fn quat_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let half = angle * 0.5;
    let s = half.sin();
    [axis[0] * s, axis[1] * s, axis[2] * s, half.cos()]
}

/// Quaternion from euler angles in degrees (X pitch, Y yaw, Z roll),
/// applied as yaw * pitch * roll
pub fn quat_from_euler_deg(euler_deg: [f32; 3]) -> [f32; 4] {
    let qx = quat_axis_angle([1.0, 0.0, 0.0], euler_deg[0].to_radians());
    let qy = quat_axis_angle([0.0, 1.0, 0.0], euler_deg[1].to_radians());
    let qz = quat_axis_angle([0.0, 0.0, 1.0], euler_deg[2].to_radians());
    quat_normalize(quat_mul(quat_mul(qy, qx), qz))
}

/// Rotate a vector by a quaternion (local → world)
pub fn quat_rotate_vec(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    // v' = q * (v, 0) * q^-1, expanded
    let [x, y, z, w] = q;
    let (vx, vy, vz) = (v[0], v[1], v[2]);
    // t = 2 * cross(q.xyz, v)
    let tx = 2.0 * (y * vz - z * vy);
    let ty = 2.0 * (z * vx - x * vz);
    let tz = 2.0 * (x * vy - y * vx);
    // v' = v + w * t + cross(q.xyz, t)
    [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]
}

/// Convert quaternion to 3 rotation matrix rows (world→local, i.e. R_quat transposed).
/// Matches the container bounds convention used in the integrate shader.
pub fn quat_to_rotation_rows(q: [f32; 4]) -> [[f32; 4]; 3] {
    let [x, y, z, w] = q;
    let xx = x*x; let yy = y*y; let zz = z*z;
    let xy = x*y; let xz = x*z; let yz = y*z;
    let wx = w*x; let wy = w*y; let wz = w*z;

    // R_quat (local→world):
    //   [1-2(yy+zz),  2(xy-wz),  2(xz+wy)]
    //   [2(xy+wz),  1-2(xx+zz),  2(yz-wx)]
    //   [2(xz-wy),  2(yz+wx),  1-2(xx+yy)]
    //
    // We store R_quat^T (world→local) rows = R_quat columns:
    [
        [1.0-2.0*(yy+zz), 2.0*(xy+wz), 2.0*(xz-wy), 0.0],
        [2.0*(xy-wz), 1.0-2.0*(xx+zz), 2.0*(yz+wx), 0.0],
        [2.0*(xz+wy), 2.0*(yz-wx), 1.0-2.0*(xx+yy), 0.0],
    ]
}

// --- GPU structs ---

/// GPU-compatible rigid body parameters for integrate shader (112 bytes;
/// array element in the bodies storage buffer)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRigidBody {
    pub position: [f32; 3],         // 12 bytes
    pub half_extent: f32,           // 4 bytes  → 16
    pub velocity: [f32; 3],         // 12 bytes
    pub is_active: u32,             // 4 bytes  → 32
    pub stiffness: f32,             // 4 bytes
    pub shape: u32,                 // 4 bytes
    pub motion: u32,                // 4 bytes
    pub prop_blades: u32,           // 4 bytes  → 48
    pub angular_velocity: [f32; 3], // 12 bytes
    pub prop_pitch: f32,            // 4 bytes  → 64 (radians)
    pub rot_row0: [f32; 4],         // 16 bytes → 80
    pub rot_row1: [f32; 4],         // 16 bytes → 96
    pub rot_row2: [f32; 4],         // 16 bytes → 112
}

impl Default for GpuRigidBody {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            half_extent: 0.15,
            velocity: [0.0; 3],
            is_active: 0,
            stiffness: 200.0,
            shape: 0,
            motion: 0,
            prop_blades: 3,
            angular_velocity: [0.0; 3],
            prop_pitch: 0.0,
            rot_row0: [1.0, 0.0, 0.0, 0.0],
            rot_row1: [0.0, 1.0, 0.0, 0.0],
            rot_row2: [0.0, 0.0, 1.0, 0.0],
        }
    }
}

/// Header for the bodies storage buffer (16 bytes, followed by the body array)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct GpuRigidBodiesHeader {
    pub count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// GPU rigid body force accumulator (48 bytes, atomic i32 on GPU side).
/// Penalty (static contact) and damping (velocity drag) reactions are kept
/// separate so integration can attenuate the static component by submersion.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct GpuRigidBodyAccum {
    pub penalty_x: i32,     // fixed-point × 1000
    pub penalty_y: i32,
    pub penalty_z: i32,
    pub contact_count: u32,
    pub damping_x: i32,     // fixed-point × 1000
    pub damping_y: i32,
    pub damping_z: i32,
    pub _pad0: u32,
    pub torque_x: i32,      // fixed-point × 1000
    pub torque_y: i32,
    pub torque_z: i32,
    pub _pad1: u32,
}


/// GPU rigid body rendering parameters (112 bytes; array element in the
/// render bodies storage buffer)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRigidBodyRender {
    pub position: [f32; 3],     // 12 bytes
    pub half_extent: f32,       // 4 bytes  → 16
    pub color: [f32; 4],        // 16 bytes → 32
    pub light_dir: [f32; 3],    // 12 bytes
    pub shape: u32,             // 4 bytes  → 48
    pub rot_row0: [f32; 4],     // 16 bytes → 64
    pub rot_row1: [f32; 4],     // 16 bytes → 80
    pub rot_row2: [f32; 4],     // 16 bytes → 96
    pub prop_blades: u32,       // 4 bytes
    pub prop_pitch: f32,        // 4 bytes (radians)
    pub _pad0: f32,             // 4 bytes
    pub _pad1: f32,             // 4 bytes  → 112
}

impl Default for GpuRigidBodyRender {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            half_extent: 0.15,
            color: [1.0; 4],
            light_dir: [0.4, 0.8, 0.3],
            shape: 0,
            rot_row0: [1.0, 0.0, 0.0, 0.0],
            rot_row1: [0.0, 1.0, 0.0, 0.0],
            rot_row2: [0.0, 0.0, 1.0, 0.0],
            prop_blades: 3,
            prop_pitch: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

// --- CPU rigid body integration ---

/// Integrate rigid body physics on CPU: forces → velocity → position, with container collision.
/// Called once per frame per Dynamic body after SPH simulation has accumulated reaction forces.
/// `fluid_density` is the SPH rest density (SphConfig::rest_density()) — body
/// mass is relative_density × fluid_density × volume. `kernel_radius` and
/// `particles_per_volume` (rest_density / particle mass) size the expected
/// wetted-shell contact count for the submerged-fraction estimate.
pub fn integrate_rigid_body(
    rigid_body: &mut RigidBodyConfig,
    container: &ContainerConfig,
    delta_time: f32,
    num_substeps: u32,
    gravity: [f32; 3],
    fluid_density: f32,
    kernel_radius: f32,
    particles_per_volume: f32,
    accum: &GpuRigidBodyAccum,
) {
    let he = rigid_body.half_extent;
    let volume = rigid_body.shape.volume(he);
    let body_mass = rigid_body.relative_density.max(0.01) * fluid_density * volume;
    let total_dt = num_substeps as f32 * delta_time;

    if body_mass <= 0.0 {
        return;
    }

    // Submerged fraction from the wetted contact shell: contact_count counts
    // particles within the 0.7h penalty layer, accumulated over all substeps.
    // Fully submerged -> the whole surface is wetted -> fraction 1.
    let shell_particles = rigid_body.shape.surface_area(he, rigid_body.prop_blades)
        * (0.7 * kernel_radius)
        * particles_per_volume;
    let expected = (num_substeps as f32 * shell_particles).max(1.0);
    let submerged = (accum.contact_count as f32 / expected).clamp(0.0, 1.0);
    if std::env::var_os("RB_DEBUG").is_some() {
        eprintln!(
            "rb sub={:.3} count={} expected={:.0} y={:.3}",
            submerged, accum.contact_count, expected, rigid_body.position[1]
        );
    }

    // Analytic Archimedes buoyancy: a = -g * (rho_fluid * V_submerged) / m
    //                                 = -g * submerged / relative_density.
    // The gap-penalty reaction cannot express displaced-volume buoyancy (its
    // shell force is symmetric once submerged), so without this term dense
    // bodies ride the contact cushion and never sink.
    let buoyancy = submerged / rigid_body.relative_density.max(0.01);

    // Contact reactions fade with submersion: a fully-wetted body is
    // supported by buoyancy + analytic drag, not by contact forces
    // (sustained shell support is the trampoline artifact that held dense
    // bodies in mid-water). The damping channel keeps 30% when submerged so
    // flow still entrains bodies (propeller wash shoving a ball around);
    // the static penalty channel fades out completely.
    let pen_scale = 1.0 - submerged;
    let damp_scale = 1.0 - 0.7 * submerged;
    let reaction = [
        accum.penalty_x as f32 / 1000.0 * pen_scale + accum.damping_x as f32 / 1000.0 * damp_scale,
        accum.penalty_y as f32 / 1000.0 * pen_scale + accum.damping_y as f32 / 1000.0 * damp_scale,
        accum.penalty_z as f32 / 1000.0 * pen_scale + accum.damping_z as f32 / 1000.0 * damp_scale,
    ];

    // Reaction-induced velocity change (clamped to prevent explosions with light bodies)
    let mut dv = [0.0f32; 3];
    for i in 0..3 {
        dv[i] = delta_time * reaction[i] / body_mass;
    }
    let max_dv = total_dt * 200.0; // Match SPH particle accel clamp
    let dv_mag = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
    if dv_mag > max_dv {
        let scale = max_dv / dv_mag;
        for d in &mut dv { *d *= scale; }
    }
    // Hydrodynamic drag, scaled by submersion: quadratic bluff-body term
    // (mean projected area of a convex body = surface_area / 4, Cauchy) sets
    // a physical terminal velocity; a light linear term settles bobbing.
    // Multiplicative decay cannot reverse the velocity.
    let speed = (rigid_body.velocity[0].powi(2)
        + rigid_body.velocity[1].powi(2)
        + rigid_body.velocity[2].powi(2))
    .sqrt();
    let projected_area = rigid_body.shape.surface_area(he, rigid_body.prop_blades) * 0.25;
    let quad_coeff = 0.4 * fluid_density * projected_area / body_mass; // 1/2 C_d(0.8) rho A / m
    let drag = (1.0 - (quad_coeff * speed + 1.5) * submerged * total_dt).max(0.0);
    for i in 0..3 {
        rigid_body.velocity[i] += dv[i] + total_dt * gravity[i] * (1.0 - buoyancy);
        rigid_body.velocity[i] *= 0.995 * drag;
    }
    for i in 0..3 {
        rigid_body.position[i] += total_dt * rigid_body.velocity[i];
    }

    // Angular dynamics: torque → angular acceleration → angular velocity → quaternion
    let torque = [
        accum.torque_x as f32 / 1000.0,
        accum.torque_y as f32 / 1000.0,
        accum.torque_z as f32 / 1000.0,
    ];
    let inertia = rigid_body.shape.moment_of_inertia(body_mass, he);

    if inertia > 0.0 {
        let mut dw = [0.0f32; 3];
        for i in 0..3 {
            dw[i] = delta_time * torque[i] / inertia;
        }
        let max_dw = total_dt * 50.0; // Clamp angular accel for light bodies
        let dw_mag = (dw[0] * dw[0] + dw[1] * dw[1] + dw[2] * dw[2]).sqrt();
        if dw_mag > max_dw {
            let scale = max_dw / dw_mag;
            for d in &mut dw { *d *= scale; }
        }
        for (w, &d) in rigid_body.angular_velocity.iter_mut().zip(&dw) {
            *w += d;
            *w *= 0.98; // Angular damping
        }

        // Quaternion integration: q += 0.5 * dt * [ω, 0] * q
        let av = rigid_body.angular_velocity;
        let omega_quat = [av[0], av[1], av[2], 0.0];
        let q = rigid_body.orientation;
        let q_dot = quat_mul(omega_quat, q);
        rigid_body.orientation = quat_normalize([
            q[0] + 0.5 * total_dt * q_dot[0],
            q[1] + 0.5 * total_dt * q_dot[1],
            q[2] + 0.5 * total_dt * q_dot[2],
            q[3] + 0.5 * total_dt * q_dot[3],
        ]);
    }

    // Container collision with proper rotated AABB
    clamp_rigid_body_to_container(rigid_body, container, true);
}

/// Compute per-axis AABB half-extents of the rotated rigid body in container-local space.
/// For a cube/cylinder, accounts for rotation so corners don't poke through walls.
/// For a sphere, the extent is uniform regardless of rotation.
fn rotated_aabb_half_extents(
    shape: RigidBodyShape,
    half_extent: f32,
    orientation: [f32; 4],
    container_rot: [[f32; 3]; 3],
) -> [f32; 3] {
    let he = half_extent;

    if shape == RigidBodyShape::Sphere {
        // Sphere: rotationally symmetric, no AABB inflation needed
        return [he, he, he];
    }

    // Body-local half-extents per axis (before rotation)
    let local_he = match shape {
        RigidBodyShape::Torus => {
            // major=he, minor=0.3*he → bounding box [1.3*he, 0.3*he, 1.3*he]
            let r_minor = 0.3 * he;
            [he + r_minor, r_minor, he + r_minor]
        }
        RigidBodyShape::Propeller => {
            let radial = (PROP_BLADE_CENTER + PROP_BLADE_HALF[0]) * he;
            [radial, (PROP_HUB_HALF_HEIGHT + PROP_BLADE_HALF[1]) * he, radial]
        }
        // Cube, Cylinder, Custom: all fit in [-he, he]^3
        _ => [he, he, he],
    };

    // Body rotation matrix rows (stored as R^T, i.e. world→body)
    let br = quat_to_rotation_rows(orientation);

    // Combined M = C * R (body-local → container-local)
    // M[i][j] = dot(container_rot[i], body_rot_row[j])
    // because R (local→world) = (stored R^T)^T, so R[k][j] = br[j][k]
    let mut aabb = [0.0f32; 3];
    for i in 0..3 {
        let mut sum = 0.0f32;
        for j in 0..3 {
            let m_ij = container_rot[i][0] * br[j][0]
                     + container_rot[i][1] * br[j][1]
                     + container_rot[i][2] * br[j][2];
            sum += m_ij.abs() * local_he[j];
        }
        aabb[i] = sum;
    }

    aabb
}

/// Clamp rigid body position (and optionally velocity) to container bounds.
/// Uses the rotated AABB so corners of cubes etc. don't poke through walls.
pub fn clamp_rigid_body_to_container(
    rigid_body: &mut RigidBodyConfig,
    container: &ContainerConfig,
    bounce_velocity: bool,
) {
    let (forward, inverse) = container.rotation_matrices();
    // Inverse R^T rows (world → container local), truncated to 3-component
    let inv = [
        [inverse[0][0], inverse[0][1], inverse[0][2]],
        [inverse[1][0], inverse[1][1], inverse[1][2]],
        [inverse[2][0], inverse[2][1], inverse[2][2]],
    ];
    // Forward R rows (container local → world), truncated to 3-component
    let fwd = [
        [forward[0][0], forward[0][1], forward[0][2]],
        [forward[1][0], forward[1][1], forward[1][2]],
        [forward[2][0], forward[2][1], forward[2][2]],
    ];

    // Per-axis AABB half-extents in container-local space
    let aabb = rotated_aabb_half_extents(
        rigid_body.shape,
        rigid_body.half_extent,
        rigid_body.orientation,
        inv,
    );

    let pos = rigid_body.position;
    let vel = rigid_body.velocity;
    let center_y = container.floor_y + container.height / 2.0;

    // Transform center to container-local centered space (subtract center_y, apply R^T)
    let cy = [pos[0], pos[1] - center_y, pos[2]];
    let mut lp = [
        inv[0][0]*cy[0] + inv[0][1]*cy[1] + inv[0][2]*cy[2],
        inv[1][0]*cy[0] + inv[1][1]*cy[1] + inv[1][2]*cy[2],
        inv[2][0]*cy[0] + inv[2][1]*cy[1] + inv[2][2]*cy[2],
    ];
    let mut lv = [
        inv[0][0]*vel[0] + inv[0][1]*vel[1] + inv[0][2]*vel[2],
        inv[1][0]*vel[0] + inv[1][1]*vel[1] + inv[1][2]*vel[2],
        inv[2][0]*vel[0] + inv[2][1]*vel[1] + inv[2][2]*vel[2],
    ];

    let hw = container.half_width();
    let hd = container.half_depth();
    let hh = container.height / 2.0;

    // Clamp per-axis using the rotated AABB extents (symmetric half-extent bounds)
    if lp[0] - aabb[0] < -hw { lp[0] = -hw + aabb[0]; if bounce_velocity { lv[0] =  lv[0].abs() * 0.3; } }
    if lp[0] + aabb[0] >  hw { lp[0] =  hw - aabb[0]; if bounce_velocity { lv[0] = -lv[0].abs() * 0.3; } }
    if lp[1] - aabb[1] < -hh { lp[1] = -hh + aabb[1]; if bounce_velocity { lv[1] =  lv[1].abs() * 0.3; } }
    if lp[1] + aabb[1] >  hh { lp[1] =  hh - aabb[1]; if bounce_velocity { lv[1] = -lv[1].abs() * 0.3; } }
    if lp[2] - aabb[2] < -hd { lp[2] = -hd + aabb[2]; if bounce_velocity { lv[2] =  lv[2].abs() * 0.3; } }
    if lp[2] + aabb[2] >  hd { lp[2] =  hd - aabb[2]; if bounce_velocity { lv[2] = -lv[2].abs() * 0.3; } }

    // Transform back to world space (apply R, add center_y)
    rigid_body.position = [
        fwd[0][0]*lp[0] + fwd[0][1]*lp[1] + fwd[0][2]*lp[2],
        fwd[1][0]*lp[0] + fwd[1][1]*lp[1] + fwd[1][2]*lp[2] + center_y,
        fwd[2][0]*lp[0] + fwd[2][1]*lp[1] + fwd[2][2]*lp[2],
    ];
    if bounce_velocity {
        rigid_body.velocity = [
            fwd[0][0]*lv[0] + fwd[0][1]*lv[1] + fwd[0][2]*lv[2],
            fwd[1][0]*lv[0] + fwd[1][1]*lv[1] + fwd[1][2]*lv[2],
            fwd[2][0]*lv[0] + fwd[2][1]*lv[1] + fwd[2][2]*lv[2],
        ];
    }
}
