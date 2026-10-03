//! Simulation tab: transport and solver settings, SPH parameters, mouse
//! force, live measurements

use super::{section, GuiAction};
use crate::state::{AppState, ForceMode};

pub(super) fn show(ui: &mut egui::Ui, state: &mut AppState, action: &mut GuiAction) {
    section(ui, "Simulation", |ui| simulation(ui, state, action));
    section(ui, "SPH Physics", |ui| sph_physics(ui, state));
    section(ui, "Mouse Force", |ui| mouse_force(ui, state));
    section(ui, "Measurements", |ui| measurements(ui, state));
}

/// Play / pause, reset, solver settings and the particle budget
fn simulation(ui: &mut egui::Ui, state: &mut AppState, action: &mut GuiAction) {
    ui.horizontal(|ui| {
        if ui.button(if state.simulation.paused { "▶ Play" } else { "⏸ Pause" }).clicked() {
            state.simulation.paused = !state.simulation.paused;
        }
        if ui.button("↺ Reset Sim").clicked() {
            *action = GuiAction::ResetSimulation;
        }
    });

    ui.add_space(8.0);

    ui.add(
        egui::Slider::new(&mut state.simulation.gravity, 0.0..=30.0)
            .text("Gravity")
    );
    ui.add(
        egui::Slider::new(&mut state.simulation.damping, 0.0..=1.0)
            .text("Bounce")
    );
    ui.add(
        egui::Slider::new(&mut state.simulation.simulation_speed, 0.25..=2.0)
            .text("Sim Speed")
    );

    let mut substeps = state.simulation.substeps as i32;
    ui.add(
        egui::Slider::new(&mut substeps, 1..=8)
            .text("Substeps (quality)")
    );
    state.simulation.substeps = substeps as u32;

    let mut pcisph_iters = state.simulation.pcisph_iterations as i32;
    ui.add(
        egui::Slider::new(&mut pcisph_iters, 2..=8)
            .text("Pressure Iters (PCISPH)")
    );
    state.simulation.pcisph_iterations = pcisph_iters as u32;

    ui.add_space(8.0);
    ui.separator();
    ui.label("Particle Settings (requires reset):");

    ui.add(
        egui::Slider::new(&mut state.simulation.initial_cube_size, 5..=64)
            .text("Initial Cube Size")
    );
    ui.label(format!("  = {} particles",
        state.simulation.initial_cube_size.pow(3)));

    ui.add(
        egui::Slider::new(&mut state.simulation.max_particles, 1000..=500_000)
            .text("Max Particles")
            .logarithmic(true)
    );
}

fn sph_physics(ui: &mut egui::Ui, state: &mut AppState) {
    ui.add(
        egui::Slider::new(&mut state.sph.kernel_radius, 0.02..=0.15)
            .text("Kernel Radius")
    );
    ui.label(format!("Rest Density: {:.0}", state.sph.rest_density()));
    ui.add(
        egui::Slider::new(&mut state.sph.near_stiffness, 0.05..=2.0)
            .text("Near Stiffness")
    );
    ui.add(
        egui::Slider::new(&mut state.sph.viscosity, 0.01..=5.0)
            .text("Viscosity")
    );
    ui.add(
        egui::Slider::new(&mut state.sph.mass, 0.1..=5.0)
            .text("Particle Mass")
    );
    ui.add(
        egui::Slider::new(&mut state.sph.surface_tension, 0.0..=0.02)
            .text("Surface Tension")
    );
    ui.add(
        egui::Slider::new(&mut state.sph.wall_stiffness, 50.0..=500.0)
            .text("Wall Stiffness")
    );
    ui.add(
        egui::Slider::new(&mut state.sph.xsph_epsilon, 0.0..=0.5)
            .text("XSPH Smoothing")
    );
    ui.add(
        egui::Slider::new(&mut state.sph.boundary_density, 0.0..=1.25)
            .text("Boundary Density")
    );
}

fn mouse_force(ui: &mut egui::Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        ui.label("Mode:");
        ui.selectable_value(&mut state.mouse_force.mode, ForceMode::Push, "Push");
        ui.selectable_value(&mut state.mouse_force.mode, ForceMode::Pull, "Pull");
        ui.selectable_value(&mut state.mouse_force.mode, ForceMode::Vortex, "Vortex");
        ui.selectable_value(&mut state.mouse_force.mode, ForceMode::Explode, "Explode");
        ui.selectable_value(&mut state.mouse_force.mode, ForceMode::Drain, "Drain");
    });
    ui.add_space(4.0);
    ui.add(
        egui::Slider::new(&mut state.mouse_force.radius, 0.1..=2.0)
            .text("Radius")
    );
    ui.add(
        egui::Slider::new(&mut state.mouse_force.strength, 1.0..=100.0)
            .text("Strength")
    );
}

/// Live fluid measurements (probe pass) + scenario status
fn measurements(ui: &mut egui::Ui, state: &AppState) {
    if !state.scenario.is_empty() {
        ui.label(format!(
            "Scenario: {} fluid blocks, {}/{} events fired",
            state.scenario.fluid_blocks.len(),
            state.runtime.scenario_events_fired,
            state.scenario.events.len(),
        ));
        if !state.scenario.fluid_blocks.is_empty() {
            ui.label("(Initial Cube Size is ignored; Reset replays the scenario)");
        }
        ui.separator();
    }
    match &state.runtime.measurements {
        Some(m) => {
            let f = |v: Option<f32>| {
                v.map_or("-".to_string(), |v| format!("{v:.3}"))
            };
            ui.label(format!("Fluid X: [{}, {}]", f(m.min_x), f(m.max_x)));
            ui.label(format!("Fluid Z: [{}, {}]", f(m.min_z), f(m.max_z)));
            ui.label(format!("Surface max Y: {}", f(m.max_y)));
            for (i, (h, p)) in m
                .probe_heights
                .iter()
                .zip(&state.scenario.probes)
                .enumerate()
            {
                ui.label(format!(
                    "Probe {i} at ({:.2}, {:.2}): {}",
                    p.x,
                    p.z,
                    f(*h)
                ));
            }
        }
        None => {
            ui.label("No measurements yet (runs while simulating)");
        }
    }
}
