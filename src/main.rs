//! Fluid Toy - A framework for GPU-accelerated particle simulations

mod app;
mod gpu;
mod gui;
mod launch;
mod render;
mod simulation;
mod state;

use app::App;
use launch::LaunchOptions;
use winit::event_loop::EventLoop;

fn main() {
    env_logger::init();

    let options = LaunchOptions::parse_or_exit();
    let state = match options.build_app_state() {
        Ok(state) => state,
        Err(msg) => {
            eprintln!("error: {msg}");
            std::process::exit(2);
        }
    };

    // --save-config: dump the effective config and exit (no GPU needed)
    if let Some(path) = &options.save_config {
        if let Err(e) = std::fs::write(path, launch::config_to_json(&state)) {
            eprintln!("error: failed to write {}: {e}", path.display());
            std::process::exit(1);
        }
        println!("Wrote config to {}", path.display());
        return;
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new(state, options);

    event_loop.run_app(&mut app).expect("Event loop error");
}
