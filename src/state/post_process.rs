//! Post-processing configuration

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AoDebugMode {
    Off,
    RawAo,
    AppliedFactor,
}

impl AoDebugMode {
    pub fn as_u32(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::RawAo => 1,
            Self::AppliedFactor => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::RawAo => "Raw AO",
            Self::AppliedFactor => "Applied AO",
        }
    }
}

/// Post-processing settings
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PostProcessConfig {
    /// Master enable for all post-processing
    pub enabled: bool,

    // === Exposure / Tone Mapping ===
    pub exposure: f32,
    /// ACES filmic tonemapping for cinematic look
    pub tonemapping_enabled: bool,

    // === Color Grading ===
    pub saturation: f32,
    pub contrast: f32,
    pub brightness: f32,
    /// Color temperature shift (-1 = cool/blue, +1 = warm/orange)
    pub temperature: f32,

    // === Vignette ===
    pub vignette_enabled: bool,
    pub vignette_intensity: f32,
    pub vignette_smoothness: f32,

    // === Bloom ===
    pub bloom_enabled: bool,
    pub bloom_intensity: f32,
    pub bloom_threshold: f32,

    // === Chromatic Aberration ===
    pub chromatic_aberration_enabled: bool,
    pub chromatic_aberration_intensity: f32,

    // === Anamorphic Streaks ===
    pub streaks_enabled: bool,
    pub streaks_intensity: f32,
    pub streaks_threshold: f32,
    /// Streak tint color [R, G, B]
    pub streaks_tint: [f32; 3],

    // === Ambient Occlusion (GTAO) ===
    pub ao_enabled: bool,
    pub ao_intensity: f32,
    pub ao_radius: f32,
    pub ao_debug_mode: AoDebugMode,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            enabled: true,

            // Exposure / Tonemapping
            exposure: 1.15,
            tonemapping_enabled: true,

            // Color grading
            saturation: 1.15,
            contrast: 1.05,
            brightness: 0.0,
            temperature: 0.0,

            // Vignette
            vignette_enabled: true,
            vignette_intensity: 0.30,
            vignette_smoothness: 0.35,

            // Bloom
            bloom_enabled: true,
            bloom_intensity: 0.40,
            bloom_threshold: 0.85,

            // Chromatic aberration
            chromatic_aberration_enabled: true,
            chromatic_aberration_intensity: 0.0060,

            // Anamorphic streaks (cyan tint by default for sci-fi look)
            streaks_enabled: true,
            streaks_intensity: 0.15,
            streaks_threshold: 0.75,
            streaks_tint: [0.1, 0.17, 0.25], // Slight cyan/blue tint

            // Ambient Occlusion
            ao_enabled: true,
            ao_intensity: 0.8,
            ao_radius: 0.2,
            ao_debug_mode: AoDebugMode::Off,
        }
    }
}

impl PostProcessConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }
}

/// GPU-compatible post-process parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPostProcessParams {
    // Exposure
    pub exposure: f32,

    // Color grading
    pub saturation: f32,
    pub contrast: f32,
    pub brightness: f32,
    pub temperature: f32,

    // Vignette
    pub vignette_enabled: u32,
    pub vignette_intensity: f32,
    pub vignette_smoothness: f32,

    // Chromatic aberration
    pub chromatic_aberration_enabled: u32,
    pub chromatic_aberration_intensity: f32,

    // Bloom (applied in separate pass, but threshold checked here)
    pub bloom_enabled: u32,
    pub bloom_intensity: f32,
    pub bloom_threshold: f32,

    // Tonemapping
    pub tonemapping_enabled: u32,

    // Anamorphic streaks
    pub streaks_enabled: u32,
    pub streaks_intensity: f32,
    pub streaks_threshold: f32,
    pub streaks_tint_r: f32,
    pub streaks_tint_g: f32,
    pub streaks_tint_b: f32,

    // Ambient Occlusion
    pub ao_enabled: u32,
    pub ao_debug_mode: u32,
    pub ao_intensity: f32,
    pub _padding: f32,
}

impl PostProcessConfig {
    pub fn to_gpu_params(&self) -> GpuPostProcessParams {
        GpuPostProcessParams {
            exposure: self.exposure,
            saturation: self.saturation,
            contrast: self.contrast,
            brightness: self.brightness,
            temperature: self.temperature,
            vignette_enabled: if self.vignette_enabled { 1 } else { 0 },
            vignette_intensity: self.vignette_intensity,
            vignette_smoothness: self.vignette_smoothness,
            chromatic_aberration_enabled: if self.chromatic_aberration_enabled { 1 } else { 0 },
            chromatic_aberration_intensity: self.chromatic_aberration_intensity,
            bloom_enabled: if self.bloom_enabled { 1 } else { 0 },
            bloom_intensity: self.bloom_intensity,
            bloom_threshold: self.bloom_threshold,
            tonemapping_enabled: if self.tonemapping_enabled { 1 } else { 0 },
            streaks_enabled: if self.streaks_enabled { 1 } else { 0 },
            streaks_intensity: self.streaks_intensity,
            streaks_threshold: self.streaks_threshold,
            streaks_tint_r: self.streaks_tint[0],
            streaks_tint_g: self.streaks_tint[1],
            streaks_tint_b: self.streaks_tint[2],
            ao_enabled: if self.ao_enabled { 1 } else { 0 },
            ao_debug_mode: self.ao_debug_mode.as_u32(),
            ao_intensity: self.ao_intensity,
            _padding: 0.0,
        }
    }
}
