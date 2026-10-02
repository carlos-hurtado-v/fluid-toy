//! Rendering, environment, lighting, and quality configuration

/// Fluid render mode selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum FluidRenderMode {
    /// Simple particle spheres (fast, debug-friendly)
    Particles,
    /// Marching cubes mesh generation (true surface)
    #[default]
    MarchingCubes,
    /// Screen-space fluid rendering (depth smoothing + narrow-range filter)
    ScreenSpace,
}


/// MC refraction debug views: the water pixels show what the refraction code
/// did instead of the shaded result. Values are written as plain numbers (post
/// processing and FXAA are bypassed while one is on) so captures can be read
/// back exactly: `scripts/debug_decode.py` holds the encoding and the id
/// tables, and turns a capture into a false-color map + per-region histogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum McDebugView {
    #[default]
    Off,
    /// R = refraction path id / 16, G = final lookup kind / 8, B = (mirror bounces + 1) / 8
    Paths,
    /// R, G = screen uv of the final background-texture lookup, B = lookup kind / 8
    Lookup,
    /// How far the lookup jumps between neighbouring pixels: log2(1 + texels) / 8
    /// in all channels. Banding, staircases and aliasing show up as speckle
    Jump,
    /// R = cos of the final exit angle (0 = grazing / near critical), G = water
    /// path to the first exit / 4 m, B = exit interface id / 8 (wall plane,
    /// back face, back face near a wall)
    Exit,
    /// R = interface of the last mirror reflection / 8, G = exit interface / 8
    /// (same ids as Exit), B = (mirror bounces + 1) / 8
    Mirror,
}

impl McDebugView {
    pub const ALL: [McDebugView; 6] = [
        McDebugView::Off,
        McDebugView::Paths,
        McDebugView::Lookup,
        McDebugView::Jump,
        McDebugView::Exit,
        McDebugView::Mirror,
    ];

    pub fn as_u32(self) -> u32 {
        match self {
            McDebugView::Off => 0,
            McDebugView::Paths => 1,
            McDebugView::Lookup => 2,
            McDebugView::Jump => 3,
            McDebugView::Exit => 4,
            McDebugView::Mirror => 5,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            McDebugView::Off => "Off",
            McDebugView::Paths => "Paths",
            McDebugView::Lookup => "Lookup",
            McDebugView::Jump => "Jump",
            McDebugView::Exit => "Exit",
            McDebugView::Mirror => "Mirror",
        }
    }
}

/// Marching cubes grid resolution presets
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum McGridResolution {
    /// 80³ voxels — fast, lower surface detail
    Low,
    /// 128³ voxels — balanced
    Medium,
    /// 200³ voxels — high detail, smooth surfaces
    #[default]
    High,
}

impl McGridResolution {
    pub fn grid_size(self) -> u32 {
        match self {
            Self::Low => 80,
            Self::Medium => 128,
            Self::High => 200,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "Low (80³)",
            Self::Medium => "Medium (128³)",
            Self::High => "High (200³)",
        }
    }

    pub const ALL: [McGridResolution; 3] = [Self::Low, Self::Medium, Self::High];
}


/// Which HDR environment map to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum HdrEnvironment {
    Farmland,
    PureSky,
}

/// Whether background shows solid color or HDR environment
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BackgroundMode {
    SolidColor,
    Environment,
}

/// Environment/background configuration (unified for all modes)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EnvironmentConfig {
    /// Whether to show solid color or HDR environment as background
    pub background_mode: BackgroundMode,
    /// Solid background color (RGB, 0-1)
    pub background_color: [f32; 3],
    /// Which HDR environment to load
    pub hdr_selection: HdrEnvironment,
    /// Environment intensity/exposure multiplier
    pub environment_intensity: f32,
    /// Project the HDR's lower hemisphere onto a ground plane under the
    /// container (HDRI backdrop): the photographed ground gains parallax and
    /// depth, so the scene stands on it and refraction/SSR see it as geometry
    pub ground_projection: bool,
    /// Height above the ground the HDR was captured from; sets the scale of
    /// the projected ground (larger = coarser grass)
    pub ground_capture_height: f32,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            background_mode: BackgroundMode::SolidColor,
            // Neutral gray (sRGB 102/109/106) — reads foam/water contrast
            // better than black or a sky backdrop
            background_color: [0.132, 0.153, 0.143],
            hdr_selection: HdrEnvironment::PureSky,
            environment_intensity: 1.0,
            ground_projection: true,
            ground_capture_height: 1.7,
        }
    }
}

impl EnvironmentConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }

    /// Backdrop params; the ground plane and its occluder come from the
    /// scene (`ground`), the rest from this config
    pub fn to_gpu_params(&self, ground: &GroundStaging) -> GpuEnvironmentParams {
        GpuEnvironmentParams {
            use_env_background: match self.background_mode {
                BackgroundMode::Environment => 1,
                BackgroundMode::SolidColor => 0,
            },
            background_r: self.background_color[0],
            background_g: self.background_color[1],
            background_b: self.background_color[2],
            env_intensity: self.environment_intensity,
            ground_y: ground.ground_y,
            ground_capture_height: self.ground_capture_height,
            ground_enabled: self.ground_projection as u32,
            occluder_half_x: ground.occluder_half_extents[0],
            occluder_half_z: ground.occluder_half_extents[1],
            occluder_height: ground.occluder_height,
            ground_sky_share: ground.sky_share,
        }
    }
}

/// The HDR map's own sun, split out of the ambient SH and lit analytically
#[derive(Debug, Clone, Copy)]
pub struct EnvironmentSun {
    /// Toward the sun (normalized)
    pub direction: [f32; 3],
    /// Irradiance on a surface facing the sun, already scaled by the
    /// environment intensity
    pub irradiance: [f32; 3],
}

/// Scene-derived inputs to the ground-projected backdrop
#[derive(Debug, Clone, Copy)]
pub struct GroundStaging {
    /// World height of the ground plane
    pub ground_y: f32,
    /// Footprint half extents (X, Z; centered at the origin) of an opaque
    /// box standing on the ground (the pool), for its contact occlusion
    pub occluder_half_extents: [f32; 2],
    /// Its height above the ground; 0 = no occluder
    pub occluder_height: f32,
    /// Fraction of the ground's light that comes from the sky rather than
    /// the sun: the part the occluder's contact occlusion can remove
    pub sky_share: f32,
}

/// Lighting configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LightingConfig {
    /// Enable directional light (sun)
    pub sun_enabled: bool,
    /// Sun direction (normalized, points toward the sun)
    pub sun_direction: [f32; 3],
    /// Sun color (RGB, can be > 1 for HDR)
    pub sun_color: [f32; 3],
    /// Sun intensity multiplier
    pub sun_intensity: f32,
    /// In Environment mode, light with the HDR map's own sun (when it has a
    /// distinct one): its direction and measured irradiance drive the analytic
    /// sun, and the ambient SH is computed with the sun disk removed so it is
    /// not counted twice. Shadows, caustics and glints then match the sky.
    /// The manual direction/color/intensity apply otherwise.
    pub sun_from_environment: bool,
    /// Runtime: the HDR sun while it applies (set by the app each frame;
    /// never serialized)
    #[serde(skip)]
    pub environment_sun: Option<EnvironmentSun>,
}

impl Default for LightingConfig {
    fn default() -> Self {
        // Default sun position: upper-right-front, warm sunlight color
        Self {
            sun_enabled: true,
            sun_direction: [0.6, 0.5, 0.3],  // lower sun, longer specular streaks
            sun_color: [0.98, 0.82, 0.6],    // Warm white sunlight
            sun_intensity: 2.0,
            sun_from_environment: true,
            environment_sun: None,
        }
    }
}

impl LightingConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }

    /// The HDR sun when it is in effect
    pub fn environment_sun_active(&self) -> Option<EnvironmentSun> {
        self.environment_sun.filter(|_| self.sun_from_environment)
    }

    /// Effective sun light (color x intensity, or the HDR sun's irradiance),
    /// zero when the sun is off
    pub fn sun_rgb(&self) -> [f32; 3] {
        if !self.sun_enabled {
            return [0.0; 3];
        }
        match self.environment_sun_active() {
            Some(sun) => sun.irradiance,
            None => self.sun_color.map(|c| c * self.sun_intensity),
        }
    }

    /// Get normalized effective sun direction (HDR sun when active, else manual)
    pub fn sun_direction_normalized(&self) -> [f32; 3] {
        if let Some(sun) = self.environment_sun_active() {
            return sun.direction;
        }
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
            // Premultiplied: every shader uses color * intensity
            sun_color: self.sun_rgb(),
            sun_intensity: 1.0,
            ambient_intensity: 1.0,
            _pad0: [0.0; 3],
            _padding: [0.0; 3],
            _pad1: 0.0,
        }
    }

    /// Like `to_gpu_params` but with the SH ambient scale set (spray renderer)
    pub fn to_gpu_params_with_ambient(&self, ambient_intensity: f32) -> GpuLightParams {
        GpuLightParams {
            ambient_intensity,
            ..self.to_gpu_params()
        }
    }
}

/// Caustics configuration (light-space forward splatting onto the pool floor).
/// Active only in Marching Cubes mode with the OpaquePool container and sun enabled.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CausticsConfig {
    /// Enable caustics + water shadows on the pool floor
    pub enabled: bool,
    /// Caustic brightness multiplier (1 = energy-conserving)
    pub intensity: f32,
    /// How strongly water removes direct sun from the floor (1 = physical)
    pub shadow_strength: f32,
    /// Contrast exponent on the caustic map (1 = physical; >1 sharpens
    /// filaments and deepens dark lanes without raising mean brightness)
    pub focus: f32,
    /// Chromatic dispersion: per-channel IOR spread (0 = white caustics)
    pub dispersion: f32,
    /// Photon splat footprint as a multiple of photon spacing on the floor
    pub splat_size: f32,
    /// Gaussian blur sigma applied to the caustic map (in map texels)
    pub blur_sigma: f32,
    /// Temporal EMA history weight (0 = no smoothing, higher = more stable)
    pub temporal_smoothing: f32,
    /// Procedural micro-ripple gain for the caustics light pass. Supplies the
    /// sub-mesh surface detail that forms fine filaments; decoupled from the
    /// surface ripple_strength (caustic focusing needs more than specular).
    pub ripple_strength: f32,
    /// Light-space raster resolution (config-only, applied at startup)
    pub light_resolution: u32,
}

impl Default for CausticsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            intensity: 1.5,
            shadow_strength: 1.0,
            focus: 2.3,
            dispersion: 0.15,
            splat_size: 1.15,
            blur_sigma: 1.25,
            temporal_smoothing: 0.5,
            ripple_strength: 0.145,
            light_resolution: 768,
        }
    }
}

impl CausticsConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }
}

/// Rendering configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
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
    /// Marching-cubes density kernel scale relative to SPH kernel radius.
    /// Lower values preserve smaller features and reduce "chunky" meshing.
    pub mc_density_radius_scale: f32,
    /// Anisotropic MC kernels (Yu & Turk): per-particle ellipsoid splats fitted
    /// to the local particle distribution. Flattens calm surfaces and thins
    /// splash sheets; isolated droplets stay spherical.
    pub mc_anisotropy: bool,
    /// Anisotropy strength: 0 = isotropic spheres, 1 = full Yu & Turk
    /// (eigenvalue stretch + center smoothing).
    pub mc_anisotropy_strength: f32,
    /// Refraction strength - how much the background distorts through water
    /// (SS renderer, and the MC renderer's legacy path)
    pub refraction_strength: f32,
    /// MC: physically based refraction (Snell at the front surface, again at
    /// the back face; rays land on the floor/walls via the depth buffer).
    /// Off = legacy screen-space UV offset scaled by `refraction_strength`.
    pub mc_physical_refraction: bool,
    /// Physical water medium (MC + SS): pure-water absorption plus single
    /// scattering of sun and sky light along the true in-water path. The body
    /// color emerges from it; Clarity sets turbidity and the water color its
    /// spectral shape. Off = legacy hand-tuned model (uses deep_water_color).
    pub physical_water_medium: bool,
    /// MC refraction debug view (see McDebugView); bypasses post-processing
    pub mc_debug_view: McDebugView,
    /// Deep water color - what you see looking into deep water (legacy medium)
    pub deep_water_color: [f32; 3],
    /// Surface smoothing - blur radius for MC density field in voxels (0 = off).
    /// Low-pass on the density texture: smooths the bulk surface but erodes thin
    /// features (sheets/droplets) whose field width is comparable to the window.
    /// Default 1: keeps splash droplets and sheets. Still water's particle-scale
    /// lumps ("orbeez") are handled by `mc_calm_smoothing` instead; raising this
    /// to 3 calms them only partly and erases most droplets (A/B'd 2026-10-01).
    pub mc_blur_radius: u32,
    /// Calm-surface smoothing strength (0 = off, 1 = full): bulk-gated wide
    /// low-pass of the MC density field. Flattens the particle-scale lumps on
    /// still water where it is thick, leaving splash sheets and droplets to
    /// `mc_blur_radius` alone.
    pub mc_calm_smoothing: f32,
    /// Water surface roughness for PBR specular (0.01 = mirror, 0.5 = rough)
    pub water_roughness: f32,
    /// Micro-ripple normal perturbation strength (0 = glass-smooth, 1 = choppy)
    pub ripple_strength: f32,
    /// Water clarity (0 = murky, 1 = crystal clear) — controls absorption and depth tinting
    pub water_clarity: f32,
    /// Screen-space reflections enabled
    pub ssr_enabled: bool,
    /// Marching cubes grid resolution
    pub mc_grid_resolution: McGridResolution,
    /// Screen-space billboard radius scale (multiplied by kernel_radius).
    /// Particles sit at 0.6×kernel_radius spacing: 0.4 ≈ Splash's 0.67× spacing
    /// (max splash detail, granular silhouettes), 0.6 ≈ 1× spacing (smoother,
    /// fuller body — empirically reads more liquid at coarse particle counts).
    pub ss_radius_scale: f32,
    /// Narrow-range filter size constant (Splash "blurFilterSize").
    /// Filter pixel radius ≈ this × projected particle diameter × 0.05.
    pub ss_filter_size: u32,
    /// Screen-space 1D filter iterations (each = H + V pass). Plus 2D refinement.
    pub ss_filter_iterations: u32,
    /// Screen-space debug view (0=off, 1=depth, 2=filtered depth, 3=normals, 4=thickness)
    pub ss_debug_view: u32,
    /// Narrow-range filter depth window, in particle radii. The paper's base
    /// window is ~1-3 r (dynamic expansion handles slopes); Splash shipped 10,
    /// which depth-merges any surfaces within 10 radii and mushes churn.
    pub ss_nr_range: f32,
    /// Far-outlier/background clamp offset (mu), in particle radii. Equal to
    /// ss_nr_range keeps the filter response continuous at the window edge.
    pub ss_nr_offset: f32,
    /// Temporal EMA on the filtered SS depth (reprojected, validity-gated)
    pub ss_temporal: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            particle_radius: 0.02,
            particle_color: [0.08, 0.22, 0.34],
            color_by_velocity: true,
            render_mode: FluidRenderMode::MarchingCubes,
            mc_threshold: 0.95,
            mc_density_radius_scale: 1.0,
            mc_anisotropy: true,
            mc_anisotropy_strength: 1.0,
            refraction_strength: 0.045,
            mc_physical_refraction: true,
            physical_water_medium: true,
            mc_debug_view: McDebugView::Off,
            deep_water_color: [0.005, 0.03, 0.08],
            mc_blur_radius: 1,
            mc_calm_smoothing: 1.0,
            water_roughness: 0.1,
            ripple_strength: 0.011,
            water_clarity: 0.65,
            ssr_enabled: true,
            mc_grid_resolution: McGridResolution::default(),
            ss_radius_scale: 0.6,
            ss_filter_size: 12,
            ss_filter_iterations: 3,
            ss_debug_view: 0,
            ss_nr_range: 3.0,
            ss_nr_offset: 3.0,
            ss_temporal: true,
        }
    }
}

impl RenderConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }

    /// Compute the actual iso_value for marching cubes from the normalized threshold.
    /// The threshold is multiplied by the Poly6 kernel peak at r=0 for the MC kernel radius,
    /// making it stable across kernel_radius changes and density radius scale changes.
    pub fn compute_iso_value(&self, kernel_radius: f32) -> f32 {
        let h_mc = kernel_radius * self.mc_density_radius_scale;
        let pi = std::f32::consts::PI;
        // Poly6 peak at r=0: 315 / (64 * pi * h^3)
        let poly6_peak = 315.0 / (64.0 * pi * h_mc * h_mc * h_mc);
        self.mc_threshold * poly6_peak
    }

    /// Calculate the visual margin for boundary compensation
    /// Screen-space rendering expands particles significantly (4.5x), particles mode does not
    pub fn visual_margin(&self) -> f32 {
        // No margin — hard backstop clamps at the visual wall.
        // MC clips the mesh to the container. Particle billboards may overflow slightly.
        0.0
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct QualityConfig {
    /// MSAA sample count
    pub msaa: MsaaSamples,
    /// FXAA (Fast Approximate Anti-Aliasing) - post-process AA
    pub fxaa_enabled: bool,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            msaa: MsaaSamples::X4, // 4x MSAA by default
            fxaa_enabled: true,    // calms fine-grained foam shimmer in motion
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
    /// SH ambient scale (environment intensity). Read by spray_render.wgsl;
    /// other consumers have their own env params and declare this as padding.
    pub ambient_intensity: f32,     // 4 bytes, offset 32 (was specular_power)
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
    /// Ground-projected backdrop (see `GroundStaging`)
    pub ground_y: f32,
    pub ground_capture_height: f32,
    pub ground_enabled: u32,
    pub occluder_half_x: f32,
    pub occluder_half_z: f32,
    pub occluder_height: f32,
    pub ground_sky_share: f32,
}

impl Default for GpuEnvironmentParams {
    fn default() -> Self {
        Self {
            use_env_background: 1,
            background_r: 0.0,
            background_g: 0.0,
            background_b: 0.0,
            env_intensity: 1.0,
            ground_y: -1.0,
            ground_capture_height: 1.7,
            ground_enabled: 0,
            occluder_half_x: 0.0,
            occluder_half_z: 0.0,
            occluder_height: 0.0,
            ground_sky_share: 1.0,
        }
    }
}

/// GPU SSR parameters (16 bytes, uniform buffer)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSsrParams {
    pub max_distance: f32,
    pub thickness: f32,
    pub enabled: u32,
    pub _pad: u32,
}

impl Default for GpuSsrParams {
    fn default() -> Self {
        Self {
            max_distance: 10.0,
            thickness: 0.15,
            enabled: 1,
            _pad: 0,
        }
    }
}
