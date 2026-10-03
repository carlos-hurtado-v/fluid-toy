//! Command-line launch options — config files, parameter overrides, and
//! automation (frame captures, stats logging, auto-exit) for headless-ish
//! tuning runs.
//!
//! The JSON config mirrors `AppState` (minus runtime fields): what "Export
//! Config" writes from the GUI is exactly what `--config` loads. `--set`
//! paths follow the JSON structure, e.g. `--set rendering.mc_threshold=0.6`
//! or `--set lighting.sun_direction=[0.4,0.8,0.3]`.

use std::path::PathBuf;

use crate::state::AppState;

const USAGE: &str = "\
fluid-toy [options]

Options:
  --config <file.json>     Load app state from a JSON config (partial configs
                           fall back to defaults per missing field)
  --set <path=value>       Override a single config value (repeatable), e.g.
                           --set sph.kernel_radius=0.05
                           --set rendering.render_mode=MarchingCubes
                           --set lighting.sun_direction=[0.4,0.8,0.3]
                           --set rigid_bodies.0.enabled=false (array index)
  --save-config <file>     Write the effective config (defaults + --config +
                           --set) to a file and exit
  --capture <f1,f2,...>    Save a PNG of the rendered scene (no GUI) after the
                           given simulation frames; exits after the last one
                           unless --stay is given
  --out <dir>              Output directory for captures (default: captures)
  --stats <file.csv>       Append per-frame stats (particle/spray/foam/bubble
                           counts, MC vertices) to a CSV
  --exit-after <frame>     Exit after the given simulation frame
  --stay                   Keep running after captures/--exit-after milestones
  --size <WxH>             Window size in logical pixels (default: 1000x700)
  --snapshot <f1,f2,...>   Write a snapshot (PNG + config JSON + .state) into
                           --out after the given frames, like F12 does
  --load-state <file>      Start from a saved simulation state (.state, from
                           F12 or --snapshot); pair it with the snapshot's
                           JSON via --config. Frame numbers for --capture,
                           --snapshot and --exit-after then count from the
                           loaded moment
  --hold                   With --load-state: keep the simulation frozen on
                           the loaded state (frame-exact render A/B); frames
                           count rendered frames, so --capture 1 is the state
                           as saved and later frames let temporal history
                           (AO, caustics, SS) settle
  --probe <x,y>            Pixel probe (repeatable): record every step of the
                           water shader's refraction at this pixel and write
                           frame_NNNNN_probe.json next to each --capture PNG
                           (decode: scripts/probe_decode.py). Coordinates are
                           capture-PNG pixels (physical; x right, y down)
  --probe-line <x0,y0,x1,y1>
                           Probe every pixel on this line (e.g. across a
                           stripe); at most 64 probed pixels in total
  --dump-field             With --capture: also write the marching-cubes density
                           field of each captured frame (frame_NNNNN_field.bin
                           + .json; scripts/field_profile.py measures the
                           surface in it)
  --profile <file.json>    Time every frame phase on the CPU and the GPU
                           (timestamp queries) and print / write the averages
                           at exit: where a slow configuration spends its time
  --help                   Show this help
";

/// Comma-separated frame numbers (`--capture 60,180`)
fn frame_list(text: &str, flag: &str) -> Result<Vec<u64>, String> {
    text.split(',')
        .map(|part| {
            part.trim()
                .parse::<u64>()
                .map_err(|_| format!("{flag}: '{part}' is not a frame number"))
        })
        .collect()
}

/// Comma-separated pixel coordinates (`--probe 640,512`)
fn number_list(text: &str, flag: &str) -> Result<Vec<u32>, String> {
    text.split(',')
        .map(|part| {
            part.trim()
                .parse::<u32>()
                .map_err(|_| format!("{flag}: '{part}' is not a pixel coordinate"))
        })
        .collect()
}

/// Every pixel on the segment a -> b (Bresenham), endpoints included
fn line_pixels(a: [u32; 2], b: [u32; 2]) -> Vec<[u32; 2]> {
    let (mut x, mut y) = (a[0] as i64, a[1] as i64);
    let (x1, y1) = (b[0] as i64, b[1] as i64);
    let (dx, dy) = ((x1 - x).abs(), -(y1 - y).abs());
    let (sx, sy) = (if x < x1 { 1 } else { -1 }, if y < y1 { 1 } else { -1 });
    let mut err = dx + dy;
    let mut out = Vec::new();
    loop {
        out.push([x as u32, y as u32]);
        if x == x1 && y == y1 {
            return out;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Parsed command-line options.
pub struct LaunchOptions {
    pub config_path: Option<PathBuf>,
    pub sets: Vec<(String, String)>,
    pub save_config: Option<PathBuf>,
    pub capture_frames: Vec<u64>,
    pub out_dir: PathBuf,
    pub stats_path: Option<PathBuf>,
    pub exit_after: Option<u64>,
    pub stay: bool,
    pub window_size: Option<(u32, u32)>,
    pub snapshot_frames: Vec<u64>,
    pub load_state: Option<PathBuf>,
    pub hold: bool,
    pub probe_pixels: Vec<[u32; 2]>,
    pub dump_field: bool,
    pub profile_path: Option<PathBuf>,
}

impl Default for LaunchOptions {
    fn default() -> Self {
        Self {
            config_path: None,
            sets: Vec::new(),
            save_config: None,
            capture_frames: Vec::new(),
            out_dir: PathBuf::from("captures"),
            stats_path: None,
            exit_after: None,
            stay: false,
            window_size: None,
            snapshot_frames: Vec::new(),
            load_state: None,
            hold: false,
            probe_pixels: Vec::new(),
            dump_field: false,
            profile_path: None,
        }
    }
}

impl LaunchOptions {
    /// Whether this run is automated (captures, stats, or a scripted exit).
    pub fn is_automated(&self) -> bool {
        !self.capture_frames.is_empty()
            || !self.snapshot_frames.is_empty()
            || self.stats_path.is_some()
            || self.exit_after.is_some()
    }

    /// Parse from process args. Prints usage and exits on --help or error.
    pub fn parse_or_exit() -> Self {
        match Self::parse(std::env::args().skip(1)) {
            Ok(opts) => opts,
            Err(msg) => {
                if msg == "help" {
                    print!("{USAGE}");
                    std::process::exit(0);
                }
                eprintln!("error: {msg}\n");
                eprint!("{USAGE}");
                std::process::exit(2);
            }
        }
    }

    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut opts = Self::default();
        let mut args = args.peekable();

        fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
            args.next().ok_or_else(|| format!("{flag} requires a value"))
        }

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => return Err("help".into()),
                "--config" => opts.config_path = Some(PathBuf::from(value(&mut args, "--config")?)),
                "--set" => {
                    let kv = value(&mut args, "--set")?;
                    let (path, val) = kv
                        .split_once('=')
                        .ok_or_else(|| format!("--set expects path=value, got '{kv}'"))?;
                    opts.sets.push((path.to_string(), val.to_string()));
                }
                "--save-config" => {
                    opts.save_config = Some(PathBuf::from(value(&mut args, "--save-config")?))
                }
                "--capture" => {
                    opts.capture_frames.extend(frame_list(&value(&mut args, "--capture")?, "--capture")?)
                }
                "--snapshot" => {
                    opts.snapshot_frames.extend(frame_list(&value(&mut args, "--snapshot")?, "--snapshot")?)
                }
                "--load-state" => {
                    opts.load_state = Some(PathBuf::from(value(&mut args, "--load-state")?))
                }
                "--hold" => opts.hold = true,
                "--dump-field" => opts.dump_field = true,
                "--profile" => opts.profile_path = Some(PathBuf::from(value(&mut args, "--profile")?)),
                "--probe" => {
                    let v = value(&mut args, "--probe")?;
                    let n = number_list(&v, "--probe")?;
                    if n.len() != 2 {
                        return Err(format!("--probe expects x,y, got '{v}'"));
                    }
                    opts.probe_pixels.push([n[0], n[1]]);
                }
                "--probe-line" => {
                    let v = value(&mut args, "--probe-line")?;
                    let n = number_list(&v, "--probe-line")?;
                    if n.len() != 4 {
                        return Err(format!("--probe-line expects x0,y0,x1,y1, got '{v}'"));
                    }
                    opts.probe_pixels.extend(line_pixels([n[0], n[1]], [n[2], n[3]]));
                }
                "--out" => opts.out_dir = PathBuf::from(value(&mut args, "--out")?),
                "--stats" => opts.stats_path = Some(PathBuf::from(value(&mut args, "--stats")?)),
                "--exit-after" => {
                    let v = value(&mut args, "--exit-after")?;
                    opts.exit_after =
                        Some(v.parse::<u64>().map_err(|_| {
                            format!("--exit-after: '{v}' is not a frame number")
                        })?);
                }
                "--stay" => opts.stay = true,
                "--size" => {
                    let v = value(&mut args, "--size")?;
                    let (w, h) = v
                        .split_once(['x', 'X'])
                        .ok_or_else(|| format!("--size expects WxH, got '{v}'"))?;
                    let w = w.parse::<u32>().map_err(|_| format!("--size: bad width '{w}'"))?;
                    let h = h.parse::<u32>().map_err(|_| format!("--size: bad height '{h}'"))?;
                    opts.window_size = Some((w.max(1), h.max(1)));
                }
                other => return Err(format!("unknown option '{other}'")),
            }
        }

        opts.capture_frames.sort_unstable();
        opts.capture_frames.dedup();
        opts.snapshot_frames.sort_unstable();
        opts.snapshot_frames.dedup();
        if opts.hold && opts.load_state.is_none() {
            return Err("--hold needs --load-state (it freezes the loaded state)".into());
        }
        let mut seen = std::collections::HashSet::new();
        opts.probe_pixels.retain(|p| seen.insert(*p));
        if opts.probe_pixels.len() > crate::render::marching_cubes::PROBE_MAX_PIXELS {
            return Err(format!(
                "{} probed pixels, at most {} (shorten the --probe-line)",
                opts.probe_pixels.len(),
                crate::render::marching_cubes::PROBE_MAX_PIXELS
            ));
        }
        if !opts.probe_pixels.is_empty() && opts.capture_frames.is_empty() {
            return Err("--probe writes its records on --capture frames: add --capture".into());
        }
        Ok(opts)
    }

    /// Build the effective AppState: defaults → config file → --set overrides.
    pub fn build_app_state(&self) -> Result<AppState, String> {
        let mut state = AppState::default();

        if let Some(path) = &self.config_path {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            let mut tree: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;
            migrate_legacy_config(&mut tree);
            state = serde_json::from_value(tree)
                .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;
        }

        if !self.sets.is_empty() {
            let mut tree = serde_json::to_value(&state)
                .map_err(|e| format!("internal: state to json failed: {e}"))?;
            for (path, raw) in &self.sets {
                apply_set(&mut tree, path, raw)?;
            }
            state = serde_json::from_value(tree)
                .map_err(|e| format!("--set produced an invalid config: {e}"))?;
        }

        validate_scenario(&state)?;
        Ok(state)
    }
}

/// Fail fast on scenario mistakes: bad event paths surface at launch (with
/// the same helpful key listing as `--set` typos), not minutes into a run.
fn validate_scenario(state: &AppState) -> Result<(), String> {
    if state.scenario.probes.len() > crate::state::MAX_PROBES {
        return Err(format!(
            "scenario.probes: {} probes configured, max is {}",
            state.scenario.probes.len(),
            crate::state::MAX_PROBES
        ));
    }
    if state.scenario.events.is_empty() {
        return Ok(());
    }
    let tree = serde_json::to_value(state)
        .map_err(|e| format!("internal: state to json failed: {e}"))?;
    for (i, event) in state.scenario.events.iter().enumerate() {
        let (path, raw) = event.set.split_once('=').ok_or_else(|| {
            format!(
                "scenario.events.{i}: set expects path=value, got '{}'",
                event.set
            )
        })?;
        // Dry-run against a scratch copy; events applying to the evolved
        // live state later can only differ in array lengths, not keys.
        let mut scratch = tree.clone();
        apply_set(&mut scratch, path, raw)
            .map_err(|e| format!("scenario.events.{i}: {e}"))?;
        serde_json::from_value::<AppState>(scratch)
            .map_err(|e| format!("scenario.events.{i}: '{}' produces an invalid config: {e}", event.set))?;
    }
    Ok(())
}

/// Migrate legacy config keys so old export files keep loading:
/// `rigid_body` (single object with `held: bool`) became `rigid_bodies`
/// (array with `motion: Static|Kinematic|Dynamic`).
fn migrate_legacy_config(tree: &mut serde_json::Value) {
    let Some(obj) = tree.as_object_mut() else {
        return;
    };
    if obj.contains_key("rigid_bodies") {
        obj.remove("rigid_body");
        return;
    }
    if let Some(mut rb) = obj.remove("rigid_body") {
        if let Some(rbo) = rb.as_object_mut() {
            let held = rbo
                .remove("held")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            rbo.entry("motion").or_insert(serde_json::Value::String(
                if held { "Static" } else { "Dynamic" }.to_string(),
            ));
        }
        obj.insert("rigid_bodies".to_string(), serde_json::Value::Array(vec![rb]));
    }
}

/// Set a dotted-path key in a JSON tree. The key must already exist (catches
/// typos); numeric path parts index into arrays (e.g.
/// `rigid_bodies.0.enabled=false`). The value is parsed as JSON, falling back
/// to a bare string so enum variants can be written without quotes.
/// Shared by CLI `--set` overrides and runtime scenario events.
pub(crate) fn apply_set(root: &mut serde_json::Value, path: &str, raw: &str) -> Result<(), String> {
    let parts: Vec<&str> = path.split('.').collect();
    let (last, walk) = parts
        .split_last()
        .ok_or_else(|| "--set: empty path".to_string())?;

    let mut current = root;
    for part in walk {
        current = step_into(current, part, path)?;
    }

    let value = serde_json::from_str(raw)
        .unwrap_or_else(|_| serde_json::Value::String(raw.to_string()));

    if current.is_array() {
        let slot = step_into(current, last, path)?;
        *slot = value;
    } else {
        check_key(current, last, path)?;
        current
            .as_object_mut()
            .unwrap()
            .insert(last.to_string(), value);
    }
    Ok(())
}

/// Walk one path segment: object key, or array index for numeric parts.
fn step_into<'a>(
    node: &'a mut serde_json::Value,
    key: &str,
    full_path: &str,
) -> Result<&'a mut serde_json::Value, String> {
    if node.is_array() {
        let len = node.as_array().map(|a| a.len()).unwrap_or(0);
        let idx: usize = key.parse().map_err(|_| {
            format!("--set {full_path}: '{key}' is not an array index (array has {len} entries)")
        })?;
        node.get_mut(idx).ok_or_else(|| {
            format!("--set {full_path}: index {idx} out of range (array has {len} entries)")
        })
    } else {
        check_key(node, key, full_path)?;
        Ok(node.get_mut(key).unwrap())
    }
}

/// Verify `key` exists in the object at `node`, with a helpful error listing
/// the keys that do exist.
fn check_key(node: &serde_json::Value, key: &str, full_path: &str) -> Result<(), String> {
    let obj = node.as_object().ok_or_else(|| {
        format!("--set {full_path}: '{key}' cannot be set (parent is not an object)")
    })?;
    if !obj.contains_key(key) {
        let known: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
        return Err(format!(
            "--set {full_path}: unknown key '{key}' (known: {})",
            known.join(", ")
        ));
    }
    Ok(())
}

/// Serialize the app state to pretty JSON (the Export Config format).
pub fn config_to_json(state: &AppState) -> String {
    serde_json::to_string_pretty(state).expect("AppState serialization cannot fail")
}

/// Find the next free `configs/export_NNN.json` path and write the state there.
pub fn export_config(state: &AppState) -> std::io::Result<PathBuf> {
    let dir = PathBuf::from("configs");
    std::fs::create_dir_all(&dir)?;
    let mut n = 1u32;
    let path = loop {
        let candidate = dir.join(format!("export_{n:03}.json"));
        if !candidate.exists() {
            break candidate;
        }
        n += 1;
    };
    std::fs::write(&path, config_to_json(state))?;
    Ok(path)
}
