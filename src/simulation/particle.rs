//! Particle data structure for simulation

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
}

impl Particle {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            position: [x, y],
            velocity: [0.0, 0.0],
        }
    }

    pub fn with_velocity(x: f32, y: f32, vx: f32, vy: f32) -> Self {
        Self {
            position: [x, y],
            velocity: [vx, vy],
        }
    }
}

impl Default for Particle {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

/// Extended particle for SPH simulation (32 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SphParticle {
    pub position: [f32; 2],   // 8 bytes
    pub velocity: [f32; 2],   // 8 bytes
    pub density: f32,         // 4 bytes
    pub pressure: f32,        // 4 bytes
    pub force: [f32; 2],      // 8 bytes
}

impl SphParticle {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            position: [x, y],
            velocity: [0.0, 0.0],
            density: 0.0,
            pressure: 0.0,
            force: [0.0, 0.0],
        }
    }
}

impl Default for SphParticle {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

/// Extended particle for 3D SPH simulation (64 bytes)
/// Uses double density relaxation (density + near_density)
/// Layout matches WGSL vec3 alignment (16-byte aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SphParticle3D {
    pub position: [f32; 3],     // 12 bytes @ 0
    pub _pad0: f32,             // 4 bytes @ 12 (vec3 padding)
    pub velocity: [f32; 3],     // 12 bytes @ 16
    pub _pad1: f32,             // 4 bytes @ 28 (vec3 padding)
    pub force: [f32; 3],        // 12 bytes @ 32
    pub density: f32,           // 4 bytes @ 44
    pub near_density: f32,      // 4 bytes @ 48
    pub _padding: [f32; 3],     // 12 bytes @ 52 (pad to 64)
}

impl SphParticle3D {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: [x, y, z],
            _pad0: 0.0,
            velocity: [0.0, 0.0, 0.0],
            _pad1: 0.0,
            force: [0.0, 0.0, 0.0],
            density: 0.0,
            near_density: 0.0,
            _padding: [0.0, 0.0, 0.0],
        }
    }
}

impl Default for SphParticle3D {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}
