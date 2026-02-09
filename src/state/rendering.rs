//! Rendering, environment, lighting, and quality configuration

/// Fluid render mode selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidRenderMode {
    /// Simple particle spheres (fast, debug-friendly)
    Particles,
    /// Marching cubes mesh generation (true surface)
    MarchingCubes,
}

impl Default for FluidRenderMode {
    fn default() -> Self {
        Self::MarchingCubes
    }
}

/// Which HDR environment map to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrEnvironment {
    Farmland,
    PureSky,
}

/// Whether background shows solid color or HDR environment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundMode {
    SolidColor,
    Environment,
}

/// Environment/background configuration (unified for all modes)
#[derive(Debug, Clone)]
pub struct EnvironmentConfig {
    /// Whether to show solid color or HDR environment as background
    pub background_mode: BackgroundMode,
    /// Solid background color (RGB, 0-1)
    pub background_color: [f32; 3],
    /// Which HDR environment to load
    pub hdr_selection: HdrEnvironment,
    /// Environment intensity/exposure multiplier
    pub environment_intensity: f32,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            background_mode: BackgroundMode::Environment,
            background_color: [0.02, 0.02, 0.05],
            hdr_selection: HdrEnvironment::Farmland,
            environment_intensity: 1.0,
        }
    }
}

impl EnvironmentConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }

    pub fn to_gpu_params(&self) -> GpuEnvironmentParams {
        GpuEnvironmentParams {
            use_env_background: match self.background_mode {
                BackgroundMode::Environment => 1,
                BackgroundMode::SolidColor => 0,
            },
            background_r: self.background_color[0],
            background_g: self.background_color[1],
            background_b: self.background_color[2],
            env_intensity: self.environment_intensity,
            _pad: [0.0; 3],
        }
    }
}

/// Lighting configuration
#[derive(Debug, Clone)]
pub struct LightingConfig {
    /// Enable directional light (sun)
    pub sun_enabled: bool,
    /// Sun direction (normalized, points toward the sun)
    pub sun_direction: [f32; 3],
    /// Sun color (RGB, can be > 1 for HDR)
    pub sun_color: [f32; 3],
    /// Sun intensity multiplier
    pub sun_intensity: f32,
}

impl Default for LightingConfig {
    fn default() -> Self {
        // Default sun position: upper-right-front, warm sunlight color
        Self {
            sun_enabled: true,
            sun_direction: [0.4, 0.8, 0.3],  // ~55° elevation, visible specular highlights
            sun_color: [0.98, 0.82, 0.6],    // Warm white sunlight
            sun_intensity: 2.0,
        }
    }
}

impl LightingConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }

    /// Get normalized sun direction
    pub fn sun_direction_normalized(&self) -> [f32; 3] {
        let [x, y, z] = self.sun_direction;
        let len = (x * x + y * y + z * z).sqrt();
        if len > 0.0001 {
            [x / len, y / len, z / len]
        } else {
            [0.0, 1.0, 0.0]
        }
    }

    pub fn to_gpu_params(&self) -> GpuLightParams {
        GpuLightParams {
            sun_direction: self.sun_direction_normalized(),
            sun_enabled: if self.sun_enabled { 1 } else { 0 },
            sun_color: self.sun_color,
            sun_intensity: self.sun_intensity,
            _pad_unused: 0.0,
            _pad0: [0.0; 3],
            _padding: [0.0; 3],
            _pad1: 0.0,
        }
    }
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
    /// Rendering mode (particles or marching cubes)
    pub render_mode: FluidRenderMode,
    /// Marching cubes surface threshold (normalized, auto-scales with kernel_radius)
    /// Higher = tighter surface around dense fluid, lower = captures smaller droplets
    pub mc_threshold: f32,
    /// Refraction strength - how much the background distorts through water
    pub refraction_strength: f32,
    /// Deep water color - what you see looking into deep water
    pub deep_water_color: [f32; 3],
    /// Surface smoothing - blur radius for MC density field (0 = off, higher = smoother)
    pub mc_blur_radius: u32,
    /// Water surface roughness for PBR specular (0.01 = mirror, 0.5 = rough)
    pub water_roughness: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            particle_radius: 0.02,
            particle_color: [0.2, 0.4, 0.9],
            color_by_velocity: true,
            render_mode: FluidRenderMode::MarchingCubes,
            mc_threshold: 3.8,
            refraction_strength: 0.085,
            deep_water_color: [0.01, 0.04, 0.1],
            mc_blur_radius: 3,
            water_roughness: 0.15,
        }
    }
}

impl RenderConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }

    /// Compute the actual iso_value for marching cubes from the normalized threshold.
    /// The threshold is multiplied by the Poly6 kernel peak at r=0 for the MC kernel radius
    /// (kernel_radius * 2.5), making it stable across kernel_radius changes.
    pub fn compute_iso_value(&self, kernel_radius: f32) -> f32 {
        let h_mc = kernel_radius * 2.5;
        let pi = std::f32::consts::PI;
        // Poly6 peak at r=0: 315 / (64 * pi * h^3)
        let poly6_peak = 315.0 / (64.0 * pi * h_mc * h_mc * h_mc);
        self.mc_threshold * poly6_peak
    }

    /// Calculate the visual margin for boundary compensation
    /// Screen-space rendering expands particles significantly (4.5x), particles mode does not
    pub fn visual_margin(&self) -> f32 {
        match self.render_mode {
            FluidRenderMode::MarchingCubes => self.particle_radius,
            FluidRenderMode::Particles => self.particle_radius,
        }
    }

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
        }
    }
}

/// MSAA sample count options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsaaSamples {
    Off = 1,
    X2 = 2,
    X4 = 4,
    X8 = 8,
}

impl MsaaSamples {
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn label(self) -> &'static str {
        match self {
            MsaaSamples::Off => "Off",
            MsaaSamples::X2 => "2x",
            MsaaSamples::X4 => "4x",
            MsaaSamples::X8 => "8x",
        }
    }
}

/// Quality settings for rendering
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// MSAA sample count
    pub msaa: MsaaSamples,
    /// FXAA (Fast Approximate Anti-Aliasing) - post-process AA
    pub fxaa_enabled: bool,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            msaa: MsaaSamples::X4,  // 4x MSAA by default
            fxaa_enabled: true,     // FXAA enabled by default
        }
    }
}

// --- GPU structs ---

/// GPU-compatible render parameters (matches WGSL struct layout)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRenderParams {
    pub particle_radius: f32,
    pub color_by_velocity: u32,
    pub _padding1: [u32; 2],
    pub particle_color: [f32; 4],
}

/// GPU-compatible lighting parameters
/// Note: WGSL vec3<f32> has 16-byte alignment, so we need explicit padding
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLightParams {
    pub sun_direction: [f32; 3],   // 12 bytes, offset 0
    pub sun_enabled: u32,           // 4 bytes, offset 12
    pub sun_color: [f32; 3],        // 12 bytes, offset 16
    pub sun_intensity: f32,         // 4 bytes, offset 28
    pub _pad_unused: f32,           // 4 bytes, offset 32 (was specular_power)
    pub _pad0: [f32; 3],            // 12 bytes, offset 36 (aligns _padding to offset 48)
    pub _padding: [f32; 3],         // 12 bytes, offset 48 (matches WGSL vec3 alignment)
    pub _pad1: f32,                 // 4 bytes, offset 60 (struct padding to reach 64)
}
// Total: 64 bytes (matches WGSL struct alignment)

/// GPU spherical harmonics coefficients (144 bytes, uniform buffer)
/// 9 coefficients × vec4<f32> (RGB + pad per coefficient)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuShCoefficients {
    pub coeffs: [[f32; 4]; 9],
}

impl Default for GpuShCoefficients {
    fn default() -> Self {
        Self { coeffs: [[0.0; 4]; 9] }
    }
}

/// GPU environment parameters (32 bytes, uniform buffer)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuEnvironmentParams {
    pub use_env_background: u32,  // 0 = solid color, 1 = environment map
    pub background_r: f32,
    pub background_g: f32,
    pub background_b: f32,
    pub env_intensity: f32,
    pub _pad: [f32; 3],
}
