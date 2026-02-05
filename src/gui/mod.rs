//! GUI module - egui integration for parameter control

use crate::state::{AppState, ContainerConfig, FluidRenderMode, SimulationConfig};

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
            });

            ui.add_space(8.0);

            // Container controls
            ui.collapsing("Container", |ui| {
                ui.label("Dimensions:");
                ui.add(
                    egui::Slider::new(&mut state.container.width, 0.5..=3.0)
                        .text("Width (X)")
                );
                ui.add(
                    egui::Slider::new(&mut state.container.depth, 0.5..=3.0)
                        .text("Depth (Z)")
                );
                ui.add(
                    egui::Slider::new(&mut state.container.height, 0.5..=3.0)
                        .text("Height (Y)")
                );

                ui.add_space(4.0);
                ui.add(
                    egui::Slider::new(&mut state.container.floor_y, -1.5..=0.5)
                        .text("Floor Position")
                );
                ui.label(format!("  Ceiling at: {:.2}", state.container.ceiling_y()));

                ui.add_space(8.0);
                ui.separator();
                ui.label("Tilt:");
                ui.add(
                    egui::Slider::new(&mut state.container.tilt_x, -std::f32::consts::PI..=std::f32::consts::PI)
                        .text("Tilt X (↕)")
                        .suffix(" rad")
                );
                ui.add(
                    egui::Slider::new(&mut state.container.tilt_z, -std::f32::consts::PI..=std::f32::consts::PI)
                        .text("Tilt Z (↔)")
                        .suffix(" rad")
                );

                ui.horizontal(|ui| {
                    if ui.button("Reset Tilt").clicked() {
                        state.container.tilt_x = 0.0;
                        state.container.tilt_z = 0.0;
                    }
                    if ui.button("Flip Upside Down").clicked() {
                        state.container.tilt_x = std::f32::consts::PI;
                        state.container.tilt_z = 0.0;
                    }
                });

                // Show container orientation
                let tilt_deg_x = state.container.tilt_x.to_degrees();
                let tilt_deg_z = state.container.tilt_z.to_degrees();
                ui.label(format!("Tilt: {:.0}° x {:.0}°", tilt_deg_x, tilt_deg_z));
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

                if state.rendering.render_mode == FluidRenderMode::ScreenSpace
                    || state.rendering.render_mode == FluidRenderMode::MarchingCubes
                {
                    ui.add_space(8.0);
                    ui.separator();
                    ui.label("Surface Detail:");
                    ui.add(
                        egui::Slider::new(&mut state.rendering.ripple_scale, 1.0..=50.0)
                            .text("Ripple Scale")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.rendering.ripple_strength, 0.0..=1.0)
                            .text("Ripple Strength")
                    );
                }

                ui.add_space(4.0);
                ui.label("Particle Color:");
                egui::color_picker::color_edit_button_rgb(ui, &mut state.rendering.particle_color);

                ui.add_space(4.0);
                ui.label("Background:");
                egui::color_picker::color_edit_button_rgb(ui, &mut state.rendering.background_color);
            });

            ui.add_space(8.0);

            // Post-processing controls
            ui.collapsing("Post Processing", |ui| {
                ui.checkbox(&mut state.post_process.enabled, "Enable Post Processing");

                if state.post_process.enabled {
                    ui.add_space(8.0);
                    ui.separator();

                    // Exposure & Tonemapping
                    ui.label("Exposure & Tonemapping:");
                    ui.add(
                        egui::Slider::new(&mut state.post_process.exposure, 0.1..=3.0)
                            .text("Exposure")
                    );
                    ui.checkbox(&mut state.post_process.tonemapping_enabled, "ACES Tonemapping");

                    ui.add_space(8.0);
                    ui.separator();

                    // Color Grading
                    ui.label("Color Grading:");
                    ui.add(
                        egui::Slider::new(&mut state.post_process.saturation, 0.0..=2.0)
                            .text("Saturation")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.post_process.contrast, 0.5..=2.0)
                            .text("Contrast")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.post_process.brightness, -0.5..=0.5)
                            .text("Brightness")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.post_process.temperature, -1.0..=1.0)
                            .text("Temperature")
                    );

                    ui.add_space(8.0);
                    ui.separator();

                    // Bloom
                    ui.checkbox(&mut state.post_process.bloom_enabled, "Bloom");
                    if state.post_process.bloom_enabled {
                        ui.add(
                            egui::Slider::new(&mut state.post_process.bloom_intensity, 0.0..=2.0)
                                .text("Intensity")
                        );
                        ui.add(
                            egui::Slider::new(&mut state.post_process.bloom_threshold, 0.0..=2.0)
                                .text("Threshold")
                        );
                    }

                    ui.add_space(8.0);
                    ui.separator();

                    // Vignette
                    ui.checkbox(&mut state.post_process.vignette_enabled, "Vignette");
                    if state.post_process.vignette_enabled {
                        ui.add(
                            egui::Slider::new(&mut state.post_process.vignette_intensity, 0.0..=1.0)
                                .text("Intensity")
                        );
                        ui.add(
                            egui::Slider::new(&mut state.post_process.vignette_smoothness, 0.0..=1.0)
                                .text("Smoothness")
                        );
                    }

                    ui.add_space(8.0);
                    ui.separator();

                    // Chromatic Aberration
                    ui.checkbox(&mut state.post_process.chromatic_aberration_enabled, "Chromatic Aberration");
                    if state.post_process.chromatic_aberration_enabled {
                        ui.add(
                            egui::Slider::new(&mut state.post_process.chromatic_aberration_intensity, 0.0..=0.05)
                                .text("Intensity")
                        );
                    }

                    ui.add_space(8.0);
                    ui.separator();

                    // Anamorphic Streaks
                    ui.checkbox(&mut state.post_process.streaks_enabled, "Anamorphic Streaks");
                    if state.post_process.streaks_enabled {
                        ui.add(
                            egui::Slider::new(&mut state.post_process.streaks_intensity, 0.0..=2.0)
                                .text("Intensity")
                        );
                        ui.add(
                            egui::Slider::new(&mut state.post_process.streaks_threshold, 0.0..=1.5)
                                .text("Threshold")
                        );
                        ui.label("Streak Tint:");
                        egui::color_picker::color_edit_button_rgb(ui, &mut state.post_process.streaks_tint);
                    }

                    ui.add_space(8.0);
                    if ui.button("Reset Post Processing").clicked() {
                        state.post_process.reset_defaults();
                    }
                }
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

impl ContainerConfig {
    pub fn gui_reset_defaults(&mut self) {
        *self = Self::default();
    }
}

