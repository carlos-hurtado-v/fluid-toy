//! Rigid body configuration, quaternion helpers, CPU integration, and GPU types

use super::simulation::ContainerConfig;

/// Rigid body shape types (repr matches GPU constants)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigidBodyShape {
    Cube = 0,
    Sphere = 1,
    Cylinder = 2,
    Torus = 3,
    Custom = 4,
}

impl RigidBodyShape {
    /// Number of vertices for the procedural mesh
    pub fn vertex_count(self) -> u32 {
        match self {
            RigidBodyShape::Cube => 36,        // 6 faces × 2 tri × 3 verts
            RigidBodyShape::Sphere => 3072,    // 32 slices × 16 stacks × 6
            RigidBodyShape::Cylinder => 384,   // 32 segments: barrel(192) + 2 caps(192)
            RigidBodyShape::Torus => 3072,     // 32 major × 16 minor × 6
            RigidBodyShape::Custom => 0,       // Uses index buffer, not vertex_count
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
        }
    }
}

/// Rigid body configuration
#[derive(Debug, Clone)]
pub struct RigidBodyConfig {
    /// Whether the rigid body is active in the scene
    pub enabled: bool,
    /// Whether the body is held (user-positioned) or simulated
    pub held: bool,
    /// Shape type
    pub shape: RigidBodyShape,
    /// Position in world space
    pub position: [f32; 3],
    /// Linear velocity
    pub velocity: [f32; 3],
    /// Orientation quaternion [x, y, z, w]
    pub orientation: [f32; 4],
    /// Angular velocity (world space, radians/sec)
    pub angular_velocity: [f32; 3],
    /// Half-extent (radius for sphere/cylinder/torus, half side for cube)
    pub half_extent: f32,
    /// Body density (compared to fluid rest_density; < rest_density → floats)
    pub density: f32,
    /// Render color (RGB)
    pub color: [f32; 3],
}

impl Default for RigidBodyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            held: true,
            shape: RigidBodyShape::Cube,
            position: [0.0, 0.2, 0.0],
            velocity: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],  // Identity quaternion
            angular_velocity: [0.0; 3],
            half_extent: 0.15,
            density: 300.0,  // Lighter than default rest_density (6000) → floats
            color: [0.9, 0.7, 0.2],  // Yellow/gold
        }
    }
}

impl RigidBodyConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
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
            _pad1: 0.0,
            _pad2: 0.0,
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

/// GPU-compatible rigid body parameters for integrate shader (96 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRigidBody {
    pub position: [f32; 3],     // 12 bytes
    pub half_extent: f32,       // 4 bytes  → 16
    pub velocity: [f32; 3],     // 12 bytes
    pub is_active: u32,         // 4 bytes  → 32
    pub stiffness: f32,         // 4 bytes
    pub shape: u32,             // 4 bytes
    pub _pad1: f32,             // 4 bytes
    pub _pad2: f32,             // 4 bytes  → 48
    pub rot_row0: [f32; 4],     // 16 bytes → 64
    pub rot_row1: [f32; 4],     // 16 bytes → 80
    pub rot_row2: [f32; 4],     // 16 bytes → 96
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
            _pad1: 0.0,
            _pad2: 0.0,
            rot_row0: [1.0, 0.0, 0.0, 0.0],
            rot_row1: [0.0, 1.0, 0.0, 0.0],
            rot_row2: [0.0, 0.0, 1.0, 0.0],
        }
    }
}

/// GPU rigid body force accumulator (32 bytes, atomic i32 on GPU side)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRigidBodyAccum {
    pub force_x: i32,       // fixed-point × 1000
    pub force_y: i32,
    pub force_z: i32,
    pub contact_count: u32,
    pub torque_x: i32,      // fixed-point × 1000
    pub torque_y: i32,
    pub torque_z: i32,
    pub _pad: u32,
}

impl Default for GpuRigidBodyAccum {
    fn default() -> Self {
        Self {
            force_x: 0,
            force_y: 0,
            force_z: 0,
            contact_count: 0,
            torque_x: 0,
            torque_y: 0,
            torque_z: 0,
            _pad: 0,
        }
    }
}

/// GPU rigid body rendering parameters (96 bytes)
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
}

// --- CPU rigid body integration ---

/// Integrate rigid body physics on CPU: forces → velocity → position, with container collision.
/// Called once per frame after SPH simulation has accumulated reaction forces.
pub fn integrate_rigid_body(
    rigid_body: &mut RigidBodyConfig,
    container: &ContainerConfig,
    delta_time: f32,
    num_substeps: u32,
    gravity: [f32; 3],
    accum: &GpuRigidBodyAccum,
) {
    let reaction = [
        accum.force_x as f32 / 1000.0,
        accum.force_y as f32 / 1000.0,
        accum.force_z as f32 / 1000.0,
    ];

    let he = rigid_body.half_extent;
    let volume = rigid_body.shape.volume(he);
    let body_mass = rigid_body.density * volume;
    let total_dt = num_substeps as f32 * delta_time;

    if body_mass <= 0.0 {
        return;
    }

    // Reaction-induced velocity change (clamped to prevent explosions with light bodies)
    let mut dv = [0.0f32; 3];
    for i in 0..3 {
        dv[i] = delta_time * reaction[i] / body_mass;
    }
    let max_dv = total_dt * 200.0; // Match SPH particle accel clamp
    let dv_mag = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
    if dv_mag > max_dv {
        let scale = max_dv / dv_mag;
        for i in 0..3 { dv[i] *= scale; }
    }
    for i in 0..3 {
        rigid_body.velocity[i] += dv[i] + total_dt * gravity[i];
        rigid_body.velocity[i] *= 0.995; // Light damping
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
            for i in 0..3 { dw[i] *= scale; }
        }
        for i in 0..3 {
            rigid_body.angular_velocity[i] += dw[i];
            rigid_body.angular_velocity[i] *= 0.98; // Angular damping
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

    // Container collision in tilted local space
    // (same rotation matrix as particle boundary handling)
    let (sin_x, cos_x) = container.tilt_x.sin_cos();
    let (sin_z, cos_z) = container.tilt_z.sin_cos();
    // Rotation rows (world → container local): Rz * Rx
    let r0 = [cos_z, -sin_z * cos_x, sin_z * sin_x];
    let r1 = [sin_z,  cos_z * cos_x, -cos_z * sin_x];
    let r2 = [0.0,    sin_x,           cos_x];

    let rhe = rigid_body.half_extent;
    let pos = rigid_body.position;
    let vel = rigid_body.velocity;

    // Transform to local space
    let mut lp = [
        r0[0]*pos[0] + r0[1]*pos[1] + r0[2]*pos[2],
        r1[0]*pos[0] + r1[1]*pos[1] + r1[2]*pos[2],
        r2[0]*pos[0] + r2[1]*pos[1] + r2[2]*pos[2],
    ];
    let mut lv = [
        r0[0]*vel[0] + r0[1]*vel[1] + r0[2]*vel[2],
        r1[0]*vel[0] + r1[1]*vel[1] + r1[2]*vel[2],
        r2[0]*vel[0] + r2[1]*vel[1] + r2[2]*vel[2],
    ];

    let hw = container.half_width();
    let hd = container.half_depth();
    let floor_y = container.floor_y;
    let ceil_y = container.ceiling_y();

    // Clamp in local space
    if lp[0] - rhe < -hw  { lp[0] = -hw + rhe;  lv[0] =  lv[0].abs() * 0.3; }
    if lp[0] + rhe >  hw  { lp[0] =  hw - rhe;  lv[0] = -lv[0].abs() * 0.3; }
    if lp[1] - rhe < floor_y { lp[1] = floor_y + rhe; lv[1] =  lv[1].abs() * 0.3; }
    if lp[1] + rhe > ceil_y  { lp[1] = ceil_y - rhe;  lv[1] = -lv[1].abs() * 0.3; }
    if lp[2] - rhe < -hd  { lp[2] = -hd + rhe;  lv[2] =  lv[2].abs() * 0.3; }
    if lp[2] + rhe >  hd  { lp[2] =  hd - rhe;  lv[2] = -lv[2].abs() * 0.3; }

    // Transform back to world space (multiply by R^T)
    rigid_body.position = [
        r0[0]*lp[0] + r1[0]*lp[1] + r2[0]*lp[2],
        r0[1]*lp[0] + r1[1]*lp[1] + r2[1]*lp[2],
        r0[2]*lp[0] + r1[2]*lp[1] + r2[2]*lp[2],
    ];
    rigid_body.velocity = [
        r0[0]*lv[0] + r1[0]*lv[1] + r2[0]*lv[2],
        r0[1]*lv[0] + r1[1]*lv[1] + r2[1]*lv[2],
        r0[2]*lv[0] + r1[2]*lv[1] + r2[2]*lv[2],
    ];
}
