//! Mouse force, spray particles, and force mode configuration

/// Mouse force interaction mode (repr matches GPU constants)
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ForceMode {
    Push = 0,
    Pull = 1,
    Vortex = 2,
    Explode = 3,
    Drain = 4,
}

/// Mouse force interaction configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MouseForceConfig {
    pub mode: ForceMode,
    pub radius: f32,
    pub strength: f32,
}

impl Default for MouseForceConfig {
    fn default() -> Self {
        Self {
            mode: ForceMode::Vortex,
            radius: 0.35,
            strength: 50.0,
        }
    }
}

/// Whitewater (spray / foam / bubbles) configuration — Ihmsen et al. 2012 diffuse particles
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SprayConfig {
    pub enabled: bool,
    /// Minimum fluid particle speed (m/s) for any emission (kinetic energy gate)
    pub min_speed: f32,
    /// Diffuse particles per second per fluid particle at full potential
    pub emission_rate: f32,
    /// Foam lifetime ceiling (s). Only foam decays — spray persists until it
    /// lands and bubbles until they surface (SPlisHSPlasH lifetime rule); the
    /// per-particle value is energy-scaled at birth between
    /// `lifetime*(1-variation)` and `lifetime*(1+variation)`.
    pub lifetime: f32,
    pub lifetime_variation: f32,
    pub drag: f32,
    pub speed_multiplier: f32,
    pub velocity_jitter: f32,
    pub particle_size: f32,
    pub max_particles: u32,
    /// Trapped-air potential weight (converging relative velocities → air entrainment)
    pub k_trapped_air: f32,
    /// Wave-crest potential weight (surface curvature on outward-moving particles)
    pub k_wave_crest: f32,
    /// Bubble upward acceleration as a multiple of |gravity|
    pub bubble_buoyancy: f32,
    /// Per-step blend of bubble velocity toward local fluid velocity (0–1)
    pub bubble_drag: f32,
    pub bubbles_visible: bool,
    /// Master scale on the surface foam coverage response (1 = calibrated)
    pub foam_coverage: f32,
    /// Master scale on entrained-air milkiness in the water (1 = calibrated)
    pub aeration_strength: f32,
}

impl Default for SprayConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_speed: 0.8,
            emission_rate: 30.0,
            lifetime: 2.8,
            lifetime_variation: 0.5,
            drag: 1.5,
            speed_multiplier: 0.85,
            velocity_jitter: 0.9,
            particle_size: 0.0015,
            max_particles: 100_000,
            k_trapped_air: 1.25,
            k_wave_crest: 1.4,
            bubble_buoyancy: 2.5,
            bubble_drag: 0.6,
            bubbles_visible: true,
            foam_coverage: 0.8,
            aeration_strength: 0.95,
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

/// GPU whitewater simulation parameters (80 bytes, uniform buffer)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSprayParams {
    pub min_speed: f32,           // 0
    pub emission_rate: f32,       // 4
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
    pub k_trapped_air: f32,       // 48
    pub k_wave_crest: f32,        // 52
    /// Auto-calibrated clamp ceilings for the emission potentials (EMA of
    /// per-frame maxima, SPlisHSPlasH-style); the shader uses [0.1*max, max]
    pub ta_limit: f32,            // 56
    pub bubble_buoyancy: f32,     // 60
    pub bubble_drag: f32,         // 64
    pub wc_limit: f32,            // 68
    pub _pad: [f32; 2],           // 72
}

/// GPU diffuse particle (48 bytes, scalar f32 to avoid WGSL vec3 alignment issues).
/// `kind` is reclassified every step from the fluid neighbor count:
/// 0 = spray (airborne), 1 = foam (at surface), 2 = bubble (submerged).
/// `age` is wall-clock seconds since spawn (lifetime decays at kind-dependent
/// rates, so it can't serve as an age measure).
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
    pub kind: u32,
    pub age: f32,
    pub _pad: [f32; 2],
}

/// GPU spray render parameters (16 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSprayRenderParams {
    pub particle_size: f32,
    pub max_particles: u32,
    pub bubbles_visible: u32,
    /// 1 = foam draws via the screen-space density field (MC mode), so the
    /// sprite pass skips foam particles; 0 = legacy foam sprites (other modes)
    pub foam_as_field: u32,
}
