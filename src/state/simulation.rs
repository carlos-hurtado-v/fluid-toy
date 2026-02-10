//! Simulation physics configuration — SPH, container, and time stepping

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
    /// Number of simulation substeps per frame (each runs at full dt)
    pub substeps: u32,
    /// Number of PCISPH pressure solver iterations per substep
    pub pcisph_iterations: u32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            delta_time: 0.0080,
            gravity: 9.8,
            damping: 0.55,        // Energy retained on wall bounce
            paused: false,
            max_particles: 50_000,
            initial_cube_size: 20, // 20×20×20 = 8000 particles
            substeps: 2,
            pcisph_iterations: 4,
        }
    }
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
    /// Container tilt around X axis (radians) - current smoothed value
    pub tilt_x: f32,
    /// Container tilt around Z axis (radians) - current smoothed value
    pub tilt_z: f32,
    /// Target tilt around X axis (radians) - set by GUI slider
    pub tilt_x_target: f32,
    /// Target tilt around Z axis (radians) - set by GUI slider
    pub tilt_z_target: f32,
}

impl Default for ContainerConfig {
    fn default() -> Self {
        Self {
            width: 1.8,          // Full X dimension (-0.9 to +0.9)
            depth: 1.8,          // Full Z dimension (-0.9 to +0.9)
            floor_y: -0.9,       // Floor at bottom
            height: 1.8,         // Extends to +0.9
            tilt_x: 0.0,
            tilt_z: 0.0,
            tilt_x_target: 0.0,
            tilt_z_target: 0.0,
        }
    }
}

impl ContainerConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }

    /// Smoothly interpolate current tilt toward target (call once per frame)
    pub fn update_tilt(&mut self, dt: f32) {
        // Exponential smoothing: ~90% of the way in 0.15 seconds
        let rate = 15.0 * dt;
        let t = rate.min(1.0);
        self.tilt_x += (self.tilt_x_target - self.tilt_x) * t;
        self.tilt_z += (self.tilt_z_target - self.tilt_z) * t;
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

    /// Convert to 3D GPU-compatible bounds struct with rotation
    ///
    /// `particle_visual_radius` is used to shrink bounds so the rendered fluid
    /// surface stays within the visual wireframe (particle centers are constrained
    /// inside bounds minus this margin)
    pub fn to_gpu_bounds_3d(&self, wall_stiffness: f32, damping: f32, particle_visual_radius: f32) -> GpuBoundsParams3D {
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
            damping,
            _padding: [0.0; 2],
            rotation_row0,
            rotation_row1,
            rotation_row2,
        }
    }
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
    /// Surface tension (cohesion between particles)
    pub surface_tension: f32,
    /// Wall repulsion stiffness
    pub wall_stiffness: f32,
}

impl Default for SphConfig {
    fn default() -> Self {
        // Tuned values for realistic water-like behavior
        Self {
            kernel_radius: 0.08,
            rest_density: 8000.0,
            stiffness: 35.0,
            near_stiffness: 0.40,
            viscosity: 0.75,
            mass: 1.0,
            surface_tension: 0.10,    // Akinci 2013 surface tension coefficient
            wall_stiffness: 250.0,
        }
    }
}

impl SphConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }

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
            surface_tension: self.surface_tension,
            pcisph_delta: compute_pcisph_delta(h, self.mass, self.rest_density, dt),
            _padding_st: [0.0; 2],
        }
    }
}

// --- GPU structs ---

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
    pub surface_tension: f32,
    pub pcisph_delta: f32,
    pub _padding_st: [f32; 2],
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
    pub damping: f32,         // Restitution coefficient for boundary bounce (0=inelastic, 1=elastic)
    pub _padding: [f32; 2],  // Padding for 16-byte alignment
    // Rotation matrix for container orientation (3x3 stored as 3 vec4s for alignment)
    pub rotation_row0: [f32; 4],  // First row + padding
    pub rotation_row1: [f32; 4],  // Second row + padding
    pub rotation_row2: [f32; 4],  // Third row + padding
}

/// GPU-compatible gravity parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGravity {
    pub direction: [f32; 3],
    pub _padding: f32,
}

// --- Helper functions ---

/// Precompute PCISPH pressure correction factor (δ) from kernel properties.
/// Uses a regular cubic lattice prototype to numerically estimate the gradient sums.
fn compute_pcisph_delta(h: f32, mass: f32, rest_density: f32, dt: f32) -> f32 {
    let spacing = h * 0.6; // matches particle init spacing
    let cells = (h / spacing).ceil() as i32 + 1;
    let mut sum_grad = [0.0f32; 3];
    let mut sum_grad_sq = 0.0f32;
    let pi = std::f32::consts::PI;
    let h6 = h.powi(6);

    for dz in -cells..=cells {
        for dy in -cells..=cells {
            for dx in -cells..=cells {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let r_vec = [
                    dx as f32 * spacing,
                    dy as f32 * spacing,
                    dz as f32 * spacing,
                ];
                let r_sq: f32 = r_vec.iter().map(|v| v * v).sum();
                if r_sq < h * h && r_sq > 1e-12 {
                    let r = r_sq.sqrt();
                    // Spiky gradient magnitude: 45/(π h⁶) * (h-r)²
                    let grad_mag = 45.0 / (pi * h6) * (h - r) * (h - r);
                    let grad = [
                        grad_mag * r_vec[0] / r,
                        grad_mag * r_vec[1] / r,
                        grad_mag * r_vec[2] / r,
                    ];
                    sum_grad[0] += grad[0];
                    sum_grad[1] += grad[1];
                    sum_grad[2] += grad[2];
                    sum_grad_sq += grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];
                }
            }
        }
    }

    let sum_grad_dot: f32 = sum_grad.iter().map(|v| v * v).sum();
    let beta = dt * dt * mass * mass / (rest_density * rest_density);
    let denom = beta * (sum_grad_dot + sum_grad_sq);
    // Under-relaxation: theoretical delta assumes regular lattice + linear response,
    // which overshoots for real particle distributions. 0.2 gives smooth convergence
    // over 4-6 iterations without compounding over-correction.
    let omega = 0.2;
    let delta = if denom.abs() > 1e-10 {
        omega / denom
    } else {
        0.0
    };
    log::debug!("PCISPH delta={delta:.6} (omega={omega}, h={h}, m={mass}, rho0={rest_density}, dt={dt}, sum_grad_sq={sum_grad_sq:.2})");
    delta
}
