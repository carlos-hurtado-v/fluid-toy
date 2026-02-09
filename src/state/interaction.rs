//! Mouse force, spray particles, and force mode configuration

/// Mouse force interaction mode (repr matches GPU constants)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForceMode {
    Push = 0,
    Pull = 1,
    Vortex = 2,
    Explode = 3,
    Drain = 4,
}

/// Mouse force interaction configuration
#[derive(Debug, Clone)]
pub struct MouseForceConfig {
    pub mode: ForceMode,
    pub radius: f32,
    pub strength: f32,
}

impl Default for MouseForceConfig {
    fn default() -> Self {
        Self {
            mode: ForceMode::Push,
            radius: 0.5,
            strength: 30.0,
        }
    }
}

/// Spray particle configuration
#[derive(Debug, Clone)]
pub struct SprayConfig {
    pub enabled: bool,
    pub emission_threshold: f32,
    pub spray_count: u32,
    pub lifetime: f32,
    pub lifetime_variation: f32,
    pub drag: f32,
    pub speed_multiplier: f32,
    pub velocity_jitter: f32,
    pub particle_size: f32,
    pub max_particles: u32,
}

impl Default for SprayConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            emission_threshold: 10.0,
            spray_count: 5,
            lifetime: 0.4,
            lifetime_variation: 0.4,
            drag: 3.0,
            speed_multiplier: 0.65,
            velocity_jitter: 1.5,
            particle_size: 0.004,
            max_particles: 100_000,
        }
    }
}

impl SprayConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }
}

// --- GPU structs ---

/// GPU-compatible mouse force parameters (matches WGSL struct layout, 48 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMouseForce {
    pub position: [f32; 3],   // 12 bytes: 3D position of force
    pub radius: f32,          //  4 bytes → 16
    pub strength: f32,        //  4 bytes
    pub is_active: u32,       //  4 bytes
    pub mode: u32,            //  4 bytes: ForceMode enum value
    pub _pad: f32,            //  4 bytes → 32
    pub direction: [f32; 3],  // 12 bytes: camera ray dir (vortex axis)
    pub _pad2: f32,           //  4 bytes → 48
}

impl Default for GpuMouseForce {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            radius: 0.5,
            strength: 30.0,
            is_active: 0,
            mode: 0,
            _pad: 0.0,
            direction: [0.0, 0.0, -1.0],
            _pad2: 0.0,
        }
    }
}

/// GPU spray simulation parameters (48 bytes, uniform buffer)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSprayParams {
    pub emission_threshold: f32,  // 0
    pub spray_count: u32,         // 4
    pub lifetime: f32,            // 8
    pub lifetime_variation: f32,  // 12
    pub drag: f32,                // 16
    pub speed_multiplier: f32,    // 20
    pub velocity_jitter: f32,     // 24
    pub dt: f32,                  // 28
    pub max_particles: u32,       // 32
    pub num_sph_particles: u32,   // 36
    pub frame_count: u32,         // 40
    pub gravity_y: f32,           // 44
}

/// GPU spray particle (32 bytes, scalar f32 to avoid WGSL vec3 alignment issues)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSprayParticle {
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub lifetime: f32,
    pub vel_x: f32,
    pub vel_y: f32,
    pub vel_z: f32,
    pub max_lifetime: f32,
}

/// GPU spray render parameters (16 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSprayRenderParams {
    pub particle_size: f32,
    pub max_particles: u32,
    pub _pad: [f32; 2],
}
