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
  --help                   Show this help
";

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
        }
    }
}

impl LaunchOptions {
    /// Whether this run is automated (captures, stats, or a scripted exit).
    pub fn is_automated(&self) -> bool {
        !self.capture_frames.is_empty() || self.stats_path.is_some() || self.exit_after.is_some()
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
                    for part in value(&mut args, "--capture")?.split(',') {
                        let frame = part
                            .trim()
                            .parse::<u64>()
                            .map_err(|_| format!("--capture: '{part}' is not a frame number"))?;
                        opts.capture_frames.push(frame);
                    }
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
        Ok(opts)
    }

    /// Build the effective AppState: defaults → config file → --set overrides.
    pub fn build_app_state(&self) -> Result<AppState, String> {
        let mut state = AppState::default();

        if let Some(path) = &self.config_path {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            state = serde_json::from_str(&text)
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

        Ok(state)
    }
}

/// Set a dotted-path key in a JSON tree. The key must already exist (catches
/// typos); the value is parsed as JSON, falling back to a bare string so enum
/// variants can be written without quotes.
fn apply_set(root: &mut serde_json::Value, path: &str, raw: &str) -> Result<(), String> {
    let parts: Vec<&str> = path.split('.').collect();
    let (last, walk) = parts
        .split_last()
        .ok_or_else(|| "--set: empty path".to_string())?;

    let mut current = root;
    for part in walk {
        check_key(current, part, path)?;
        current = current.get_mut(*part).unwrap();
    }
    check_key(current, last, path)?;

    let value = serde_json::from_str(raw)
        .unwrap_or_else(|_| serde_json::Value::String(raw.to_string()));
    current
        .as_object_mut()
        .unwrap()
        .insert(last.to_string(), value);
    Ok(())
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
