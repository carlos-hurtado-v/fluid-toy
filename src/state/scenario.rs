//! Scenario configuration — initial fluid layout, timed events, and
//! measurement probes. This is what turns the sandbox into an experiment rig:
//! a config file can place water columns (dam break), script parameter changes
//! at simulation times (gate release, tilt excitation), and log surface
//! heights / fluid extents to the stats CSV.
//!
//! Everything here is pure data loaded from the JSON config; the runtime
//! (event cursor, probe readbacks) lives in `App`.

/// Maximum number of height probes (GPU buffer capacity)
pub const MAX_PROBES: usize = 8;

/// A box of fluid spawned on reset, axis-aligned in world space.
/// Particles fill the box on the standard lattice (spacing = 0.6 × kernel
/// radius — solver tuning depends on it), so the box dimensions are quantized
/// to the lattice. Blocks are clamped into the container interior.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FluidBlockConfig {
    /// Box center in world coordinates
    pub center: [f32; 3],
    /// Full extents along X/Y/Z
    pub size: [f32; 3],
    /// Initial velocity of every particle in the block
    pub velocity: [f32; 3],
}

impl Default for FluidBlockConfig {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            size: [0.5, 0.5, 0.5],
            velocity: [0.0; 3],
        }
    }
}

/// A timed configuration change: at simulation time `time` (seconds), apply
/// `set` — a `path=value` string with exactly the CLI `--set` syntax, e.g.
/// `"rigid_bodies.0.enabled=false"` or `"container.tilt_x_target=0.3"`.
/// Fired once per reset; Reset Sim replays the schedule from t=0.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ScenarioEvent {
    /// Simulation time in seconds (frame × substep_dt × substeps)
    pub time: f32,
    /// `path=value` override, identical syntax to `--set`
    pub set: String,
}

impl Default for ScenarioEvent {
    fn default() -> Self {
        Self {
            time: 0.0,
            set: String::new(),
        }
    }
}

/// A surface-height probe: reports the highest fluid particle inside a
/// vertical cylinder of `radius` around the (x, z) column, in world Y.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ProbeConfig {
    pub x: f32,
    pub z: f32,
    /// Sampling cylinder radius (world units)
    pub radius: f32,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            x: 0.0,
            z: 0.0,
            radius: 0.06,
        }
    }
}

/// Scenario definition: initial fluid blocks, timed events, height probes.
/// Empty (the default) means legacy behavior — the centered initial cube,
/// no events, no probe columns in the stats CSV.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ScenarioConfig {
    /// Fluid boxes spawned on reset; empty = legacy centered cube
    /// (`simulation.initial_cube_size`)
    pub fluid_blocks: Vec<FluidBlockConfig>,
    /// Timed `--set`-style overrides, fired once each when sim time passes them
    pub events: Vec<ScenarioEvent>,
    /// Surface-height probe columns (max 8), logged to stats and shown in GUI
    pub probes: Vec<ProbeConfig>,
}

impl ScenarioConfig {
    /// True when the scenario changes nothing (pure legacy behavior)
    pub fn is_empty(&self) -> bool {
        self.fluid_blocks.is_empty() && self.events.is_empty() && self.probes.is_empty()
    }
}
