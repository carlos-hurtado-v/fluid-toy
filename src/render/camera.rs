//! Camera module - orbit camera with view/projection matrix generation

/// Orbit camera for 3D viewing
#[derive(Debug, Clone)]
pub struct Camera {
    /// Distance from target point
    pub distance: f32,
    /// Horizontal rotation angle (radians)
    pub yaw: f32,
    /// Vertical rotation angle (radians), clamped to avoid gimbal lock
    pub pitch: f32,
    /// Point the camera looks at
    pub target: [f32; 3],
    /// Field of view in radians
    pub fov: f32,
    /// Aspect ratio (width / height)
    pub aspect: f32,
    /// Near clipping plane
    pub near: f32,
    /// Far clipping plane
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            distance: 3.0,
            yaw: 0.0,
            pitch: 0.4,  // Slight downward angle
            target: [0.0, 0.0, 0.0],
            fov: std::f32::consts::FRAC_PI_4, // 45 degrees
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Camera {
    /// Compute camera position from orbit parameters
    pub fn position(&self) -> [f32; 3] {
        let cos_pitch = self.pitch.cos();
        let sin_pitch = self.pitch.sin();
        let cos_yaw = self.yaw.cos();
        let sin_yaw = self.yaw.sin();

        [
            self.target[0] + self.distance * cos_pitch * sin_yaw,
            self.target[1] + self.distance * sin_pitch,
            self.target[2] + self.distance * cos_pitch * cos_yaw,
        ]
    }

    /// Generate view matrix (camera transform)
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        let eye = self.position();
        let target = self.target;

        // Build look-at matrix
        look_at(eye, target, [0.0, 1.0, 0.0])
    }

    /// Generate perspective projection matrix
    pub fn projection_matrix(&self) -> [[f32; 4]; 4] {
        perspective(self.fov, self.aspect, self.near, self.far)
    }

    /// Rotate camera by delta angles (from mouse drag)
    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch += delta_pitch;

        // Clamp pitch to avoid flipping
        let max_pitch = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-max_pitch, max_pitch);
    }

    /// Zoom camera (change distance)
    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta).clamp(0.5, 20.0);
    }

    /// Reset camera to default position
    pub fn reset(&mut self) {
        self.distance = 3.0;
        self.yaw = 0.0;
        self.pitch = 0.4;
        self.target = [0.0, 0.0, 0.0];
    }

    /// Update aspect ratio (call on window resize)
    pub fn set_aspect(&mut self, width: f32, height: f32) {
        self.aspect = width / height;
    }
}

/// Create a look-at view matrix
fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    // Forward vector (from target to eye, camera looks down -Z)
    let f = normalize([
        target[0] - eye[0],
        target[1] - eye[1],
        target[2] - eye[2],
    ]);

    // Right vector
    let r = normalize(cross(f, up));

    // Recalculated up vector
    let u = cross(r, f);

    // View matrix: rotation * translation
    [
        [r[0], u[0], -f[0], 0.0],
        [r[1], u[1], -f[1], 0.0],
        [r[2], u[2], -f[2], 0.0],
        [-dot(r, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

/// Create a perspective projection matrix
fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov / 2.0).tan();
    let nf = 1.0 / (near - far);

    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * nf, -1.0],
        [0.0, 0.0, 2.0 * far * near * nf, 0.0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 0.00001 {
        return [0.0, 0.0, 1.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// GPU-compatible camera uniform data
/// Layout matches WGSL std140: mat4x4 requires 16-byte alignment
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCameraParams {
    pub view: [[f32; 4]; 4],           // 64 bytes, offset 0
    pub projection: [[f32; 4]; 4],     // 64 bytes, offset 64
    pub inv_view: [[f32; 4]; 4],       // 64 bytes, offset 128
    pub inv_projection: [[f32; 4]; 4], // 64 bytes, offset 192
    pub camera_pos: [f32; 3],          // 12 bytes
    pub near: f32,                     // 4 bytes
    pub far: f32,                      // 4 bytes
    pub _padding: [f32; 3],            // 12 bytes (align to 16)
}
// Total: 288 bytes

/// Invert a 4x4 matrix (for view/projection inverse)
fn invert_matrix(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Using cofactor expansion for 4x4 matrix inversion
    let m00 = m[0][0]; let m01 = m[0][1]; let m02 = m[0][2]; let m03 = m[0][3];
    let m10 = m[1][0]; let m11 = m[1][1]; let m12 = m[1][2]; let m13 = m[1][3];
    let m20 = m[2][0]; let m21 = m[2][1]; let m22 = m[2][2]; let m23 = m[2][3];
    let m30 = m[3][0]; let m31 = m[3][1]; let m32 = m[3][2]; let m33 = m[3][3];

    let c00 = m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31);
    let c01 = -(m10 * (m22 * m33 - m23 * m32) - m12 * (m20 * m33 - m23 * m30) + m13 * (m20 * m32 - m22 * m30));
    let c02 = m10 * (m21 * m33 - m23 * m31) - m11 * (m20 * m33 - m23 * m30) + m13 * (m20 * m31 - m21 * m30);
    let c03 = -(m10 * (m21 * m32 - m22 * m31) - m11 * (m20 * m32 - m22 * m30) + m12 * (m20 * m31 - m21 * m30));

    let c10 = -(m01 * (m22 * m33 - m23 * m32) - m02 * (m21 * m33 - m23 * m31) + m03 * (m21 * m32 - m22 * m31));
    let c11 = m00 * (m22 * m33 - m23 * m32) - m02 * (m20 * m33 - m23 * m30) + m03 * (m20 * m32 - m22 * m30);
    let c12 = -(m00 * (m21 * m33 - m23 * m31) - m01 * (m20 * m33 - m23 * m30) + m03 * (m20 * m31 - m21 * m30));
    let c13 = m00 * (m21 * m32 - m22 * m31) - m01 * (m20 * m32 - m22 * m30) + m02 * (m20 * m31 - m21 * m30);

    let c20 = m01 * (m12 * m33 - m13 * m32) - m02 * (m11 * m33 - m13 * m31) + m03 * (m11 * m32 - m12 * m31);
    let c21 = -(m00 * (m12 * m33 - m13 * m32) - m02 * (m10 * m33 - m13 * m30) + m03 * (m10 * m32 - m12 * m30));
    let c22 = m00 * (m11 * m33 - m13 * m31) - m01 * (m10 * m33 - m13 * m30) + m03 * (m10 * m31 - m11 * m30);
    let c23 = -(m00 * (m11 * m32 - m12 * m31) - m01 * (m10 * m32 - m12 * m30) + m02 * (m10 * m31 - m11 * m30));

    let c30 = -(m01 * (m12 * m23 - m13 * m22) - m02 * (m11 * m23 - m13 * m21) + m03 * (m11 * m22 - m12 * m21));
    let c31 = m00 * (m12 * m23 - m13 * m22) - m02 * (m10 * m23 - m13 * m20) + m03 * (m10 * m22 - m12 * m20);
    let c32 = -(m00 * (m11 * m23 - m13 * m21) - m01 * (m10 * m23 - m13 * m20) + m03 * (m10 * m21 - m11 * m20));
    let c33 = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20);

    let det = m00 * c00 + m01 * c01 + m02 * c02 + m03 * c03;
    let inv_det = if det.abs() < 1e-10 { 1.0 } else { 1.0 / det };

    [
        [c00 * inv_det, c10 * inv_det, c20 * inv_det, c30 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det, c31 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det, c32 * inv_det],
        [c03 * inv_det, c13 * inv_det, c23 * inv_det, c33 * inv_det],
    ]
}

impl Camera {
    /// Convert to GPU-compatible uniform struct
    pub fn to_gpu_params(&self) -> GpuCameraParams {
        let pos = self.position();
        let view = self.view_matrix();
        let projection = self.projection_matrix();
        GpuCameraParams {
            view,
            projection,
            inv_view: invert_matrix(view),
            inv_projection: invert_matrix(projection),
            camera_pos: pos,
            near: self.near,
            far: self.far,
            _padding: [0.0; 3],
        }
    }

    /// Convert screen coordinates to world-space ray using geometric approach
    /// Returns (ray_origin, ray_direction)
    pub fn screen_to_ray(&self, screen_x: f32, screen_y: f32, screen_width: f32, screen_height: f32) -> ([f32; 3], [f32; 3]) {
        let eye = self.position();

        // Camera basis vectors
        let forward = normalize([
            self.target[0] - eye[0],
            self.target[1] - eye[1],
            self.target[2] - eye[2],
        ]);
        let right = normalize(cross(forward, [0.0, 1.0, 0.0]));
        let up = cross(right, forward);

        // Convert screen coords to normalized device coordinates [-1, 1]
        let ndc_x = (2.0 * screen_x / screen_width) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / screen_height); // Flip Y

        // Calculate ray direction based on FOV and aspect ratio
        let half_height = (self.fov / 2.0).tan();
        let half_width = half_height * self.aspect;

        // Ray direction in world space
        let dir = normalize([
            forward[0] + right[0] * ndc_x * half_width + up[0] * ndc_y * half_height,
            forward[1] + right[1] * ndc_x * half_width + up[1] * ndc_y * half_height,
            forward[2] + right[2] * ndc_x * half_width + up[2] * ndc_y * half_height,
        ]);

        (eye, dir)
    }

    /// Intersect ray with horizontal plane at given Y height
    /// Returns intersection point or None if ray is parallel to plane
    pub fn ray_plane_intersection(&self, ray_origin: [f32; 3], ray_dir: [f32; 3], plane_y: f32) -> Option<[f32; 3]> {
        // Plane normal is (0, 1, 0) for horizontal plane
        // t = (plane_y - origin.y) / dir.y
        if ray_dir[1].abs() < 0.0001 {
            return None; // Ray is parallel to plane
        }

        let t = (plane_y - ray_origin[1]) / ray_dir[1];
        if t < 0.0 {
            return None; // Intersection is behind camera
        }

        Some([
            ray_origin[0] + t * ray_dir[0],
            plane_y,
            ray_origin[2] + t * ray_dir[2],
        ])
    }
}
