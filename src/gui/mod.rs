//! GUI module - egui integration for parameter control

use crate::state::{AppState, FluidRenderMode, RenderConfig, SimulationConfig};

/// Renders the control panel and returns any triggered action
pub fn render_control_panel(ctx: &egui::Context, state: &mut AppState) -> GuiAction {
    let mut action = GuiAction::None;

    egui::Window::new("Controls")
        .default_pos([10.0, 10.0])
        .default_width(250.0)
        .resizable(true)
        .collapsible(true)
        .show(ctx, |ui| {
            // Simulation controls
            ui.collapsing("Simulation", |ui| {
                ui.horizontal(|ui| {
                    if ui.button(if state.simulation.paused { "▶ Play" } else { "⏸ Pause" }).clicked() {
                        state.simulation.paused = !state.simulation.paused;
                    }
                    if ui.button("↺ Reset Sim").clicked() {
                        action = GuiAction::ResetSimulation;
                    }
                });

                ui.add_space(8.0);

                ui.add(
                    egui::Slider::new(&mut state.simulation.gravity, 0.0..=30.0)
                        .text("Gravity")
                );

                ui.add_space(4.0);
                ui.label("Container Tilt:");
                ui.add(
                    egui::Slider::new(&mut state.simulation.tilt_x, -std::f32::consts::PI..=std::f32::consts::PI)
                        .text("Tilt X (↕)")
                        .suffix(" rad")
                );
                ui.add(
                    egui::Slider::new(&mut state.simulation.tilt_z, -std::f32::consts::PI..=std::f32::consts::PI)
                        .text("Tilt Z (↔)")
                        .suffix(" rad")
                );
                if ui.button("Reset Tilt").clicked() {
                    state.simulation.tilt_x = 0.0;
                    state.simulation.tilt_z = 0.0;
                }
                if ui.button("Flip Upside Down").clicked() {
                    state.simulation.tilt_x = std::f32::consts::PI;
                    state.simulation.tilt_z = 0.0;
                }

                // Show container orientation
                let tilt_deg_x = state.simulation.tilt_x.to_degrees();
                let tilt_deg_z = state.simulation.tilt_z.to_degrees();
                ui.label(format!("Container tilt: {:.0}° x {:.0}°", tilt_deg_x, tilt_deg_z));

                ui.add_space(4.0);
                ui.add(
                    egui::Slider::new(&mut state.simulation.damping, 0.0..=1.0)
                        .text("Bounce")
                );
                ui.add(
                    egui::Slider::new(&mut state.simulation.delta_time, 0.001..=0.05)
                        .text("Time Step")
                );

                ui.add_space(8.0);
                ui.separator();
                ui.label("Particle Settings (requires reset):");

                ui.add(
                    egui::Slider::new(&mut state.simulation.initial_cube_size, 5..=30)
                        .text("Initial Cube Size")
                );
                ui.label(format!("  = {} particles",
                    state.simulation.initial_cube_size.pow(3)));

                ui.add(
                    egui::Slider::new(&mut state.simulation.max_particles, 1000..=100_000)
                        .text("Max Particles")
                        .logarithmic(true)
                );

                ui.add_space(4.0);
                ui.label("Bounds:");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut state.simulation.bounds.0).speed(0.01).range(0.1..=1.0));
                    ui.label("x");
                    ui.add(egui::DragValue::new(&mut state.simulation.bounds.1).speed(0.01).range(0.1..=1.0));
                });
            });

            ui.add_space(8.0);

            // SPH Physics controls
            ui.collapsing("SPH Physics", |ui| {
                ui.add(
                    egui::Slider::new(&mut state.sph.kernel_radius, 0.02..=0.15)
                        .text("Kernel Radius")
                );
                ui.add(
                    egui::Slider::new(&mut state.sph.rest_density, 1000.0..=30000.0)
                        .text("Rest Density")
                        .logarithmic(true)
                );
                ui.add(
                    egui::Slider::new(&mut state.sph.stiffness, 1.0..=200.0)
                        .text("Stiffness")
                );
                ui.add(
                    egui::Slider::new(&mut state.sph.viscosity, 1.0..=500.0)
                        .text("Viscosity")
                        .logarithmic(true)
                );
                ui.add(
                    egui::Slider::new(&mut state.sph.mass, 0.1..=5.0)
                        .text("Particle Mass")
                );
                ui.add(
                    egui::Slider::new(&mut state.sph.wall_stiffness, 1000.0..=20000.0)
                        .text("Wall Stiffness")
                );
            });

            ui.add_space(8.0);

            // Rendering controls
            ui.collapsing("Rendering", |ui| {
                ui.label("Render Mode:");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut state.rendering.render_mode, FluidRenderMode::ScreenSpace, "Screen-Space");
                    ui.selectable_value(&mut state.rendering.render_mode, FluidRenderMode::MarchingCubes, "Marching Cubes");
                    ui.selectable_value(&mut state.rendering.render_mode, FluidRenderMode::Particles, "Particles");
                });
                ui.add_space(4.0);

                ui.add(
                    egui::Slider::new(&mut state.rendering.particle_radius, 0.005..=0.05)
                        .text("Particle Size")
                );

                if state.rendering.render_mode == FluidRenderMode::Particles {
                    ui.checkbox(&mut state.rendering.color_by_velocity, "Color by velocity");
                }

                ui.add_space(4.0);
                ui.label("Particle Color:");
                egui::color_picker::color_edit_button_rgb(ui, &mut state.rendering.particle_color);

                ui.add_space(4.0);
                ui.label("Background:");
                egui::color_picker::color_edit_button_rgb(ui, &mut state.rendering.background_color);

                ui.add_space(8.0);
                ui.add(
                    egui::Slider::new(&mut state.rendering.env_rotation, 0.0..=std::f32::consts::TAU)
                        .text("Environment Rotation")
                );
            });

            ui.add_space(16.0);
            ui.separator();

            if ui.button("Reset to Defaults").clicked() {
                action = GuiAction::ResetDefaults;
            }

            ui.add_space(8.0);
            ui.label(format!("Particles: {}", state.runtime.particle_count));
            ui.label(format!("FPS: {:.0}", state.runtime.fps));
        });

    action
}

/// Actions that the GUI can trigger
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuiAction {
    None,
    ResetSimulation,
    ResetDefaults,
}

/// Default configs for reset functionality
impl SimulationConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }
}

impl RenderConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }
}
