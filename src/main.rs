//! Fluid Toy - A framework for GPU-accelerated particle simulations

mod app;
mod gpu;
mod gui;
mod render;
mod simulation;
mod state;

use app::App;
use winit::event_loop::EventLoop;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();

    event_loop.run_app(&mut app).expect("Event loop error");
}
