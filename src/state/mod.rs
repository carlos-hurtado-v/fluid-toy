//! Central state management - single source of truth for the application

pub mod post_process;
pub mod simulation;
pub mod rendering;
pub mod rigid_body;
pub mod interaction;

pub use post_process::{AoDebugMode, PostProcessConfig};
pub use simulation::*;
pub use rendering::*;
pub use rigid_body::*;
pub use interaction::*;

/// Complete application state - GUI binds to this.
/// Serializes to the JSON config format (`--config` / Export Config);
/// runtime values are skipped. Missing sections/fields fall back to defaults.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
#[serde(default)]
pub struct AppState {
    pub simulation: SimulationConfig,
    pub container: ContainerConfig,
    pub sph: SphConfig,
    pub rendering: RenderConfig,
    pub environment: EnvironmentConfig,
    pub lighting: LightingConfig,
    pub quality: QualityConfig,
    pub post_process: PostProcessConfig,
    pub camera: CameraConfig,
    pub rigid_body: RigidBodyConfig,
    pub spray: SprayConfig,
    pub mouse_force: MouseForceConfig,
    #[serde(skip)]
    pub runtime: RuntimeState,
}

/// Camera configuration for 3D viewing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
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

/// Runtime state - changes during execution
#[derive(Debug, Clone)]
pub struct RuntimeState {
    /// Number of particles
    pub particle_count: u32,
    /// Frames per second
    pub fps: f32,
    /// Simulation time elapsed
    pub time_elapsed: f32,
    /// Frame counter for spray RNG seed
    pub frame_count: u32,
    /// Path of the last Export Config write (GUI feedback)
    pub last_export: Option<String>,
    /// Live auto-calibrated whitewater potential ceilings (GUI readout)
    pub spray_ta_limit: f32,
    pub spray_wc_limit: f32,
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            particle_count: 0,
            fps: 0.0,
            time_elapsed: 0.0,
            frame_count: 0,
            last_export: None,
            spray_ta_limit: 0.0,
            spray_wc_limit: 0.0,
        }
    }
}
