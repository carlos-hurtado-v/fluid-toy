//! Central state management - single source of truth for the application

/// Fluid render mode selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidRenderMode {
    /// Simple particle spheres (fast, debug-friendly)
    Particles,
    /// Marching cubes surface reconstruction (photorealistic)
    MarchingCubes,
}

impl Default for FluidRenderMode {
    fn default() -> Self {
        Self::MarchingCubes
    }
}

/// Complete application state - GUI binds to this
#[derive(Debug, Clone)]
pub struct AppState {
    pub simulation: SimulationConfig,
    pub sph: SphConfig,
    pub rendering: RenderConfig,
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
    /// Container tilt around X axis (radians) - tilts forward/back
    pub tilt_x: f32,
    /// Container tilt around Z axis (radians) - tilts left/right
    pub tilt_z: f32,
    /// Boundary extents (particles stay within [-bound, bound])
    pub bounds: (f32, f32),
    /// Z-axis boundary extent (for 3D mode)
    pub bounds_z: f32,
    /// Energy retained on bounce (0 = no bounce, 1 = perfect elastic)
    pub damping: f32,
    /// Whether simulation is running
    pub paused: bool,
    /// Maximum particle capacity (requires reset to change)
    pub max_particles: u32,
    /// Initial particle cube dimension (N×N×N particles on reset)
    pub initial_cube_size: u32,
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
            sph: SphConfig::default(),
            rendering: RenderConfig::default(),
            camera: CameraConfig::default(),
            runtime: RuntimeState::default(),
        }
    }
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            delta_time: 0.006,   // Reference: 0.006
            gravity: 9.8,        // Magnitude (positive)
            tilt_x: 0.0,         // No tilt
            tilt_z: 0.0,
            bounds: (0.9, 0.9),
            bounds_z: 0.9,
            damping: 0.3,
            paused: false,       // Start running
            max_particles: 50_000,
            initial_cube_size: 20, // 20×20×20 = 8000 particles
        }
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
            particle_radius: 0.025,  // Smaller to see individual particles
            particle_color: [0.2, 0.4, 0.9],
            color_by_velocity: true,
            background_color: [0.02, 0.02, 0.05],
            render_mode: FluidRenderMode::MarchingCubes,  // Use water rendering by default
        }
    }
}

impl Default for SphConfig {
    fn default() -> Self {
        // Tuned values for water-like behavior
        Self {
            kernel_radius: 0.07,
            rest_density: 5000.0,      // Lower = particles stack instead of spreading
            stiffness: 3.0,            // Low so gravity dominates over pressure
            near_stiffness: 0.5,       // Reduced close-range repulsion
            viscosity: 50.0,           // Higher = more energy dissipation, less bouncing
            mass: 1.0,
            wall_stiffness: 8000.0,
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
    /// Convert to GPU-compatible uniform struct
    pub fn to_gpu_params(&self, num_particles: u32) -> GpuSimParams {
        GpuSimParams {
            delta_time: self.delta_time,
            gravity: self.gravity,
            bound_x: self.bounds.0,
            bound_y: self.bounds.1,
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
    pub bound_x: f32,
    pub bound_y: f32,
    pub bound_z: f32,
    pub wall_stiffness: f32,
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

impl SimulationConfig {
    /// Convert to 3D GPU-compatible bounds struct with rotation
    pub fn to_gpu_bounds_3d(&self, wall_stiffness: f32) -> GpuBoundsParams3D {
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

        GpuBoundsParams3D {
            bound_x: self.bounds.0,
            bound_y: self.bounds.1,
            bound_z: self.bounds_z,
            wall_stiffness,
            rotation_row0,
            rotation_row1,
            rotation_row2,
        }
    }
}
