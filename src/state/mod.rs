//! Central state management - single source of truth for the application

pub mod post_process;

pub use post_process::{GpuPostProcessParams, PostProcessConfig};

/// Fluid render mode selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidRenderMode {
    /// Simple particle spheres (fast, debug-friendly)
    Particles,
    /// Screen-space fluid rendering (photorealistic)
    ScreenSpace,
    /// Marching cubes mesh generation (true surface)
    MarchingCubes,
}

impl Default for FluidRenderMode {
    fn default() -> Self {
        Self::ScreenSpace
    }
}

/// Complete application state - GUI binds to this
#[derive(Debug, Clone)]
pub struct AppState {
    pub simulation: SimulationConfig,
    pub container: ContainerConfig,
    pub sph: SphConfig,
    pub rendering: RenderConfig,
    pub post_process: PostProcessConfig,
    pub camera: CameraConfig,
    pub runtime: RuntimeState,
}

/// Simulation parameters - physics configuration
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Time step per frame (seconds)
    pub delta_time: f32,
    /// Gravity acceleration magnitude
    pub gravity: f32,
    /// Energy retained on bounce (0 = no bounce, 1 = perfect elastic)
    pub damping: f32,
    /// Whether simulation is running
    pub paused: bool,
    /// Maximum particle capacity (requires reset to change)
    pub max_particles: u32,
    /// Initial particle cube dimension (N×N×N particles on reset)
    pub initial_cube_size: u32,
}

/// Container configuration - defines the fluid container
#[derive(Debug, Clone)]
pub struct ContainerConfig {
    /// Container width (X axis, centered at 0)
    pub width: f32,
    /// Container depth (Z axis, centered at 0)
    pub depth: f32,
    /// Floor Y position (bottom of container)
    pub floor_y: f32,
    /// Container height (extends upward from floor_y)
    pub height: f32,
    /// Container tilt around X axis (radians) - tilts forward/back
    pub tilt_x: f32,
    /// Container tilt around Z axis (radians) - tilts left/right
    pub tilt_z: f32,
}

/// SPH physics configuration
#[derive(Debug, Clone)]
pub struct SphConfig {
    /// Kernel support radius
    pub kernel_radius: f32,
    /// Target rest density
    pub rest_density: f32,
    /// Pressure stiffness coefficient
    pub stiffness: f32,
    /// Near pressure stiffness (prevents particle collapse)
    pub near_stiffness: f32,
    /// Viscosity coefficient
    pub viscosity: f32,
    /// Particle mass
    pub mass: f32,
    /// Wall repulsion stiffness
    pub wall_stiffness: f32,
}

/// Rendering configuration
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Particle radius in normalized coordinates
    pub particle_radius: f32,
    /// Base color (RGB, 0-1)
    pub particle_color: [f32; 3],
    /// Color particles by velocity
    pub color_by_velocity: bool,
    /// Background color (RGB, 0-1)
    pub background_color: [f32; 3],
    /// Rendering mode (particles or marching cubes)
    pub render_mode: FluidRenderMode,
    /// Scene rotation matrix (3x3, rotates the fluid in world space)
    /// Camera and environment stays fixed, scene rotates
    pub scene_rotation: [[f32; 3]; 3],
    /// Ripple scale - frequency of surface detail ripples (higher = more dense)
    pub ripple_scale: f32,
    /// Ripple strength - how much ripples perturb surface normals
    pub ripple_strength: f32,
}

/// Camera configuration for 3D viewing
#[derive(Debug, Clone)]
pub struct CameraConfig {
    /// Distance from target
    pub distance: f32,
    /// Horizontal rotation angle (radians)
    pub yaw: f32,
    /// Vertical rotation angle (radians)
    pub pitch: f32,
    /// Look-at target point
    pub target: [f32; 3],
    /// Field of view (radians)
    pub fov: f32,
}

/// Runtime state - changes during execution
#[derive(Debug, Clone)]
pub struct RuntimeState {
    /// Number of particles
    pub particle_count: u32,
    /// Frames per second
    pub fps: f32,
    /// Simulation time elapsed
    pub time_elapsed: f32,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            simulation: SimulationConfig::default(),
            container: ContainerConfig::default(),
            sph: SphConfig::default(),
            rendering: RenderConfig::default(),
            post_process: PostProcessConfig::default(),
            camera: CameraConfig::default(),
            runtime: RuntimeState::default(),
        }
    }
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            delta_time: 0.006,
            gravity: 9.8,
            damping: 0.6,        // Energy retained on wall bounce
            paused: false,
            max_particles: 50_000,
            initial_cube_size: 20, // 20×20×20 = 8000 particles
        }
    }
}

impl Default for ContainerConfig {
    fn default() -> Self {
        Self {
            width: 1.8,          // Full X dimension (-0.9 to +0.9)
            depth: 1.8,          // Full Z dimension (-0.9 to +0.9)
            floor_y: -0.9,       // Floor at bottom
            height: 1.8,         // Extends to +0.9
            tilt_x: 0.0,         // No tilt
            tilt_z: 0.0,
        }
    }
}

impl ContainerConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }

    /// Get the ceiling Y position
    pub fn ceiling_y(&self) -> f32 {
        self.floor_y + self.height
    }

    /// Convert to half-extents for GPU (legacy format)
    pub fn half_width(&self) -> f32 {
        self.width / 2.0
    }

    pub fn half_depth(&self) -> f32 {
        self.depth / 2.0
    }

    /// Compute axis-aligned bounding box of the tilted container
    /// Returns (min, max) corners in world space
    pub fn tilted_aabb(&self) -> ([f32; 3], [f32; 3]) {
        let hw = self.half_width();
        let hd = self.half_depth();
        let y0 = self.floor_y;
        let y1 = self.ceiling_y();

        // 8 corners of the untilted container (centered at origin for rotation)
        let center_y = (y0 + y1) / 2.0;
        let half_h = (y1 - y0) / 2.0;

        let corners = [
            [-hw, -half_h, -hd],
            [ hw, -half_h, -hd],
            [-hw,  half_h, -hd],
            [ hw,  half_h, -hd],
            [-hw, -half_h,  hd],
            [ hw, -half_h,  hd],
            [-hw,  half_h,  hd],
            [ hw,  half_h,  hd],
        ];

        // Rotation matrices for tilt
        let cx = self.tilt_x.cos();
        let sx = self.tilt_x.sin();
        let cz = self.tilt_z.cos();
        let sz = self.tilt_z.sin();

        let mut min = [f32::MAX, f32::MAX, f32::MAX];
        let mut max = [f32::MIN, f32::MIN, f32::MIN];

        for corner in &corners {
            // Rotate around X axis first
            let y1 = corner[1] * cx - corner[2] * sx;
            let z1 = corner[1] * sx + corner[2] * cx;

            // Then rotate around Z axis
            let x2 = corner[0] * cz - y1 * sz;
            let y2 = corner[0] * sz + y1 * cz;
            let z2 = z1;

            // Translate back (add center_y to Y)
            let world = [x2, y2 + center_y, z2];

            // Update AABB
            for i in 0..3 {
                min[i] = min[i].min(world[i]);
                max[i] = max[i].max(world[i]);
            }
        }

        (min, max)
    }
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            distance: 3.5,
            yaw: 0.5,
            pitch: 0.4,
            target: [0.0, -0.3, 0.0],  // Look slightly below center where fluid pools
            fov: std::f32::consts::FRAC_PI_4, // 45 degrees
        }
    }
}

impl CameraConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            particle_radius: 0.02,
            particle_color: [0.2, 0.4, 0.9],
            color_by_velocity: true,
            background_color: [0.02, 0.02, 0.05],
            render_mode: FluidRenderMode::ScreenSpace,
            scene_rotation: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            ripple_scale: 15.0,
            ripple_strength: 0.3,
        }
    }
}

impl RenderConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }

    /// Calculate the visual margin for boundary compensation
    /// Screen-space rendering expands particles significantly (4.5x), particles mode does not
    pub fn visual_margin(&self) -> f32 {
        match self.render_mode {
            FluidRenderMode::ScreenSpace => self.particle_radius * 4.5,
            FluidRenderMode::MarchingCubes => self.particle_radius * 4.5, // Similar to screen-space
            FluidRenderMode::Particles => self.particle_radius,
        }
    }

    /// Apply trackball rotation from mouse delta
    /// delta_x rotates around Y axis (vertical screen axis)
    /// delta_y rotates around X axis (horizontal screen axis)
    pub fn rotate_scene(&mut self, delta_x: f32, delta_y: f32) {
        let sensitivity = 0.01;
        let angle_y = -delta_x * sensitivity;
        let angle_x = -delta_y * sensitivity;

        // Rotation around Y axis
        let rot_y = [
            [angle_y.cos(), 0.0, angle_y.sin()],
            [0.0, 1.0, 0.0],
            [-angle_y.sin(), 0.0, angle_y.cos()],
        ];

        // Rotation around X axis
        let rot_x = [
            [1.0, 0.0, 0.0],
            [0.0, angle_x.cos(), -angle_x.sin()],
            [0.0, angle_x.sin(), angle_x.cos()],
        ];

        // Combine: new_rotation = rot_y * rot_x * current_rotation
        let temp = mat3_mul(&rot_x, &self.scene_rotation);
        self.scene_rotation = mat3_mul(&rot_y, &temp);
    }

    /// Reset scene rotation to identity
    pub fn reset_scene_rotation(&mut self) {
        self.scene_rotation = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
    }

    /// Convert scene rotation to 4x4 matrix for GPU
    pub fn scene_rotation_matrix_4x4(&self) -> [[f32; 4]; 4] {
        let r = &self.scene_rotation;
        [
            [r[0][0], r[0][1], r[0][2], 0.0],
            [r[1][0], r[1][1], r[1][2], 0.0],
            [r[2][0], r[2][1], r[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}

/// Multiply two 3x3 matrices
fn mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    result
}

impl Default for SphConfig {
    fn default() -> Self {
        // Tuned values for realistic water-like behavior
        Self {
            kernel_radius: 0.08,
            rest_density: 6000.0,
            stiffness: 15.0,           // Fast pressure response for proper wave propagation
            near_stiffness: 5.0,
            viscosity: 20.0,           // Low enough for waves to persist
            mass: 1.0,
            wall_stiffness: 16000.0,
        }
    }
}

impl SphConfig {
    /// Convert to GPU-compatible uniform struct
    pub fn to_gpu_params(&self, num_particles: u32, dt: f32, gravity: f32) -> GpuSphParams {
        let h = self.kernel_radius;
        GpuSphParams {
            kernel_radius: h,
            kernel_radius_sq: h * h,
            kernel_radius_4: h * h * h * h,
            kernel_radius_5: h * h * h * h * h,
            mass: self.mass,
            rest_density: self.rest_density,
            stiffness: self.stiffness,
            viscosity: self.viscosity,
            dt,
            gravity,
            num_particles,
            _padding: 0,
        }
    }

    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            particle_count: 0,
            fps: 0.0,
            time_elapsed: 0.0,
        }
    }
}

impl SimulationConfig {
    /// Convert to GPU-compatible uniform struct (legacy 2D - bounds now in ContainerConfig)
    pub fn to_gpu_params(&self, num_particles: u32) -> GpuSimParams {
        GpuSimParams {
            delta_time: self.delta_time,
            gravity: self.gravity,
            bound_x: 0.9,  // Legacy - use ContainerConfig for 3D
            bound_y: 0.9,  // Legacy - use ContainerConfig for 3D
            damping: self.damping,
            num_particles,
            _padding: [0.0; 2],
        }
    }
}

impl RenderConfig {
    /// Convert to GPU-compatible uniform struct
    pub fn to_gpu_params(&self) -> GpuRenderParams {
        GpuRenderParams {
            particle_radius: self.particle_radius,
            color_by_velocity: if self.color_by_velocity { 1 } else { 0 },
            _padding1: [0; 2],
            particle_color: [
                self.particle_color[0],
                self.particle_color[1],
                self.particle_color[2],
                1.0,
            ],
            background_color: [
                self.background_color[0],
                self.background_color[1],
                self.background_color[2],
                1.0,
            ],
        }
    }
}

/// GPU-compatible simulation parameters (matches WGSL struct layout)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSimParams {
    pub delta_time: f32,
    pub gravity: f32,
    pub bound_x: f32,
    pub bound_y: f32,
    pub damping: f32,
    pub num_particles: u32,
    pub _padding: [f32; 2],
}

/// GPU-compatible render parameters (matches WGSL struct layout)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRenderParams {
    pub particle_radius: f32,
    pub color_by_velocity: u32,
    pub _padding1: [u32; 2],
    pub particle_color: [f32; 4],
    pub background_color: [f32; 4],
}

/// GPU-compatible SPH parameters (matches WGSL struct layout)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSphParams {
    pub kernel_radius: f32,
    pub kernel_radius_sq: f32,
    pub kernel_radius_4: f32,
    pub kernel_radius_5: f32,
    pub mass: f32,
    pub rest_density: f32,
    pub stiffness: f32,
    pub viscosity: f32,
    pub dt: f32,
    pub gravity: f32,
    pub num_particles: u32,
    pub _padding: u32,
}

/// GPU-compatible boundary parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBoundsParams {
    pub bound_x: f32,
    pub bound_y: f32,
    pub damping: f32,
    pub wall_stiffness: f32,
}

/// GPU-compatible 3D SPH parameters (matches WGSL struct layout)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSphParams3D {
    pub kernel_radius: f32,
    pub kernel_radius_sq: f32,
    pub kernel_radius_pow5: f32,
    pub kernel_radius_pow6: f32,
    pub kernel_radius_pow9: f32,
    pub mass: f32,
    pub rest_density: f32,
    pub stiffness: f32,
    pub near_stiffness: f32,
    pub viscosity: f32,
    pub dt: f32,
    pub num_particles: u32,
}

/// GPU-compatible 3D boundary parameters (matches WGSL struct layout)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBoundsParams3D {
    pub bound_x: f32,        // Half-width (symmetric: -bound_x to +bound_x)
    pub bound_z: f32,        // Half-depth (symmetric: -bound_z to +bound_z)
    pub floor_y: f32,        // Floor Y position
    pub ceiling_y: f32,      // Ceiling Y position
    pub wall_stiffness: f32,
    pub _padding: [f32; 3],  // Padding for 16-byte alignment
    // Rotation matrix for container orientation (3x3 stored as 3 vec4s for alignment)
    pub rotation_row0: [f32; 4],  // First row + padding
    pub rotation_row1: [f32; 4],  // Second row + padding
    pub rotation_row2: [f32; 4],  // Third row + padding
}

/// GPU-compatible mouse force parameters (matches WGSL struct layout)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMouseForce {
    pub position: [f32; 3],   // 3D position of force
    pub radius: f32,          // Radius of effect
    pub strength: f32,        // Force strength (negative = attract, positive = repel)
    pub is_active: u32,       // 1 if active, 0 if not
    pub _padding: [f32; 2],   // Padding for 16-byte alignment
}

impl Default for GpuMouseForce {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            radius: 0.3,
            strength: 15.0,
            is_active: 0,
            _padding: [0.0; 2],
        }
    }
}

/// GPU-compatible gravity parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGravity {
    pub direction: [f32; 3],
    pub _padding: f32,
}

impl SimulationConfig {
    /// Gravity always points down - container rotates, not gravity
    pub fn gravity_vector(&self) -> [f32; 3] {
        [0.0, -self.gravity, 0.0]
    }

    /// Convert to GPU gravity struct
    pub fn to_gpu_gravity(&self) -> GpuGravity {
        GpuGravity {
            direction: self.gravity_vector(),
            _padding: 0.0,
        }
    }
}

impl SphConfig {
    /// Convert to 3D GPU-compatible uniform struct
    pub fn to_gpu_params_3d(&self, num_particles: u32, dt: f32) -> GpuSphParams3D {
        let h = self.kernel_radius;
        GpuSphParams3D {
            kernel_radius: h,
            kernel_radius_sq: h * h,
            kernel_radius_pow5: h.powi(5),
            kernel_radius_pow6: h.powi(6),
            kernel_radius_pow9: h.powi(9),
            mass: self.mass,
            rest_density: self.rest_density,
            stiffness: self.stiffness,
            near_stiffness: self.near_stiffness,
            viscosity: self.viscosity,
            dt,
            num_particles,
        }
    }
}

impl ContainerConfig {
    /// Convert to 3D GPU-compatible bounds struct with rotation
    ///
    /// `particle_visual_radius` is used to shrink bounds so the rendered fluid
    /// surface stays within the visual wireframe (particle centers are constrained
    /// inside bounds minus this margin)
    pub fn to_gpu_bounds_3d(&self, wall_stiffness: f32, particle_visual_radius: f32) -> GpuBoundsParams3D {
        // Compute rotation matrix from tilt angles
        // Rotation around X (tilt_x) then around Z (tilt_z)
        let (sin_x, cos_x) = self.tilt_x.sin_cos();
        let (sin_z, cos_z) = self.tilt_z.sin_cos();

        // Combined rotation matrix: Rz * Rx
        // This rotates the container, so we use the transpose (inverse) to transform
        // particle positions INTO container space
        let rotation_row0 = [cos_z, -sin_z * cos_x, sin_z * sin_x, 0.0];
        let rotation_row1 = [sin_z, cos_z * cos_x, -cos_z * sin_x, 0.0];
        let rotation_row2 = [0.0, sin_x, cos_x, 0.0];

        // Shrink physics bounds by visual radius so rendered surface fits inside wireframe
        let margin = particle_visual_radius;

        GpuBoundsParams3D {
            bound_x: (self.half_width() - margin).max(0.1),
            bound_z: (self.half_depth() - margin).max(0.1),
            floor_y: self.floor_y + margin,
            ceiling_y: (self.ceiling_y() - margin).max(self.floor_y + margin + 0.1),
            wall_stiffness,
            _padding: [0.0; 3],
            rotation_row0,
            rotation_row1,
            rotation_row2,
        }
    }
}
