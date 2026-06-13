//! GUI module - egui integration for parameter control

use crate::state::{AoDebugMode, AppState, BackgroundMode, ContainerStyle, FluidRenderMode, ForceMode, HdrEnvironment, McGridResolution, RigidBodyShape, SimulationConfig};

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
            });

            ui.add_space(8.0);

            // Container controls
            ui.collapsing("Container", |ui| {
                ui.label("Style:");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut state.container.style, ContainerStyle::Wireframe, "Wireframe");
                    ui.selectable_value(&mut state.container.style, ContainerStyle::OpaquePool, "Pool");
                });

                if state.container.style == ContainerStyle::OpaquePool {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label("Tile Color:");
                        egui::color_picker::color_edit_button_rgb(ui, &mut state.container.tile_color);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Grout Color:");
                        egui::color_picker::color_edit_button_rgb(ui, &mut state.container.grout_color);
                    });
                    ui.add(
                        egui::Slider::new(&mut state.container.tile_scale, 5.0..=50.0)
                            .text("Tile Scale")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.container.grout_width, 0.01..=0.10)
                            .text("Grout Width")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.container.specular_strength, 0.0..=1.0)
                            .text("Specular")
                    );
                }

                ui.add_space(4.0);
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
                    egui::Slider::new(&mut state.container.tilt_x_target, -std::f32::consts::PI..=std::f32::consts::PI)
                        .text("Tilt X (↕)")
                        .suffix(" rad")
                );
                ui.add(
                    egui::Slider::new(&mut state.container.tilt_z_target, -std::f32::consts::PI..=std::f32::consts::PI)
                        .text("Tilt Z (↔)")
                        .suffix(" rad")
                );

                ui.horizontal(|ui| {
                    if ui.button("Reset Tilt").clicked() {
                        state.container.tilt_x_target = 0.0;
                        state.container.tilt_z_target = 0.0;
                    }
                    if ui.button("Flip Upside Down").clicked() {
                        state.container.tilt_x_target = std::f32::consts::PI;
                        state.container.tilt_z_target = 0.0;
                    }
                });

                let tilt_deg_x = state.container.tilt_x_target.to_degrees();
                let tilt_deg_z = state.container.tilt_z_target.to_degrees();
                ui.label(format!("Tilt: {:.0}° x {:.0}°", tilt_deg_x, tilt_deg_z));
            });

            ui.add_space(8.0);

            // Rigid Body controls
            ui.collapsing("Rigid Body", |ui| {
                ui.checkbox(&mut state.rigid_body.enabled, "Enable");

                if state.rigid_body.enabled {
                    ui.add_space(4.0);
                    ui.label("Shape:");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut state.rigid_body.shape, RigidBodyShape::Cube, "Cube");
                        ui.selectable_value(&mut state.rigid_body.shape, RigidBodyShape::Sphere, "Sphere");
                        ui.selectable_value(&mut state.rigid_body.shape, RigidBodyShape::Cylinder, "Cylinder");
                        ui.selectable_value(&mut state.rigid_body.shape, RigidBodyShape::Torus, "Torus");
                        ui.selectable_value(&mut state.rigid_body.shape, RigidBodyShape::Custom, "Duck");
                    });

                    ui.add_space(4.0);
                    ui.checkbox(&mut state.rigid_body.held, "Held (manual position)");

                    ui.horizontal(|ui| {
                        if ui.button("Drop").clicked() {
                            state.rigid_body.held = false;
                            state.rigid_body.velocity = [0.0; 3];
                        }
                        if ui.button("Reset").clicked() {
                            state.rigid_body.held = true;
                            state.rigid_body.position = [0.0, 0.2, 0.0];
                            state.rigid_body.velocity = [0.0; 3];
                            state.rigid_body.orientation = [0.0, 0.0, 0.0, 1.0];
                            state.rigid_body.angular_velocity = [0.0; 3];
                        }
                        if ui.button("Reset Rotation").clicked() {
                            state.rigid_body.orientation = [0.0, 0.0, 0.0, 1.0];
                            state.rigid_body.angular_velocity = [0.0; 3];
                        }
                    });

                    ui.add_space(4.0);
                    ui.add(
                        egui::Slider::new(&mut state.rigid_body.half_extent, 0.05..=0.5)
                            .text("Size")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.rigid_body.density, 10.0..=10000.0)
                            .text("Density")
                            .logarithmic(true)
                    );
                    ui.label(format!("  Fluid density: {:.0}", state.sph.rest_density()));

                    if state.rigid_body.held {
                        ui.add_space(4.0);
                        ui.label("Position:");
                        ui.add(
                            egui::Slider::new(&mut state.rigid_body.position[0], -1.0..=1.0)
                                .text("X")
                        );
                        ui.add(
                            egui::Slider::new(&mut state.rigid_body.position[1], -1.0..=1.0)
                                .text("Y")
                        );
                        ui.add(
                            egui::Slider::new(&mut state.rigid_body.position[2], -1.0..=1.0)
                                .text("Z")
                        );
                    }

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label("Color:");
                        egui::color_picker::color_edit_button_rgb(ui, &mut state.rigid_body.color);
                    });
                }
            });

            ui.add_space(8.0);

            // SPH Physics controls
            ui.collapsing("SPH Physics", |ui| {
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
            });

            ui.add_space(8.0);

            // Mouse Force controls
            ui.collapsing("Mouse Force", |ui| {
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
            });

            ui.add_space(8.0);

            // Whitewater (spray / foam / bubbles) controls
            ui.collapsing("Whitewater (Spray & Foam)", |ui| {
                ui.checkbox(&mut state.spray.enabled, "Enable");

                if state.spray.enabled {
                    ui.add_space(4.0);
                    ui.add(
                        egui::Slider::new(&mut state.spray.k_trapped_air, 0.0..=3.0)
                            .text("Trapped Air")
                    ).on_hover_text("Emission from converging flow (impacts, churning)");
                    ui.add(
                        egui::Slider::new(&mut state.spray.k_wave_crest, 0.0..=3.0)
                            .text("Wave Crest")
                    ).on_hover_text("Emission from fast-moving convex surface curvature");
                    ui.add(
                        egui::Slider::new(&mut state.spray.emission_rate, 1.0..=60.0)
                            .text("Emission Rate")
                            .suffix("/s")
                    ).on_hover_text("Diffuse particles per second per fluid particle at full potential");
                    ui.add(
                        egui::Slider::new(&mut state.spray.min_speed, 0.2..=3.0)
                            .text("Min Speed")
                            .suffix(" m/s")
                    ).on_hover_text("Fluid slower than this never emits");
                    ui.label(format!(
                        "Auto limits - TA {:.2}, WC {:.2}",
                        state.runtime.spray_ta_limit, state.runtime.spray_wc_limit
                    )).on_hover_text(
                        "Self-calibrated potential ceilings (EMA of per-frame maxima); \
                         emission saturates at these values",
                    );
                    ui.add_space(4.0);
                    ui.add(
                        egui::Slider::new(&mut state.spray.lifetime, 0.5..=8.0)
                            .text("Foam Lifetime")
                            .suffix("s")
                    ).on_hover_text(
                        "Only foam ages out (energy-scaled per particle at birth); \
                         spray persists until it lands, bubbles until they surface",
                    );
                    ui.add(
                        egui::Slider::new(&mut state.spray.lifetime_variation, 0.0..=1.0)
                            .text("Lifetime Variation")
                    ).on_hover_text("Energy-scaled spread of per-particle foam lifetimes (staggers fade-out)");
                    ui.add(
                        egui::Slider::new(&mut state.spray.drag, 0.0..=5.0)
                            .text("Air Drag")
                    ).on_hover_text("Deceleration of airborne spray (foam and bubbles follow the fluid instead)");
                    ui.add(
                        egui::Slider::new(&mut state.spray.bubble_buoyancy, 0.0..=8.0)
                            .text("Bubble Buoyancy")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.spray.bubble_drag, 0.0..=1.0)
                            .text("Bubble Drag")
                    ).on_hover_text("How strongly bubbles follow the surrounding fluid");
                    ui.add(
                        egui::Slider::new(&mut state.spray.speed_multiplier, 0.2..=2.0)
                            .text("Velocity Inherit")
                    ).on_hover_text("Fraction of the emitter's velocity newborns launch with");
                    ui.add(
                        egui::Slider::new(&mut state.spray.velocity_jitter, 0.0..=3.0)
                            .text("Velocity Jitter")
                    ).on_hover_text("Random velocity added at spawn (m/s), spreads the launch cone");
                    ui.add_space(4.0);
                    ui.add(
                        egui::Slider::new(&mut state.spray.particle_size, 0.001..=0.05)
                            .text("Particle Size")
                    );
                    ui.add_space(4.0);
                    ui.add(
                        egui::Slider::new(&mut state.spray.foam_coverage, 0.0..=3.0)
                            .text("Foam Coverage")
                    ).on_hover_text("Surface foam response: lower = sparser lace, higher = denser carpet (1 = calibrated)");
                    ui.add(
                        egui::Slider::new(&mut state.spray.aeration_strength, 0.0..=3.0)
                            .text("Aeration")
                    ).on_hover_text("Entrained-air milkiness inside the water (vortex cores, plunge plumes); 1 = calibrated");
                    ui.checkbox(&mut state.spray.bubbles_visible, "Show Bubbles");
                }
            });

            ui.add_space(8.0);

            // Rendering controls
            ui.collapsing("Rendering", |ui| {
                ui.label("Render Mode:");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut state.rendering.render_mode, FluidRenderMode::MarchingCubes, "Marching Cubes");
                    ui.selectable_value(&mut state.rendering.render_mode, FluidRenderMode::ScreenSpace, "Screen Space");
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

                if state.rendering.render_mode == FluidRenderMode::MarchingCubes {
                    ui.add_space(8.0);
                    ui.separator();
                    let prev_resolution = state.rendering.mc_grid_resolution;
                    egui::ComboBox::from_label("Grid Resolution")
                        .selected_text(state.rendering.mc_grid_resolution.label())
                        .show_ui(ui, |ui| {
                            for res in McGridResolution::ALL {
                                ui.selectable_value(&mut state.rendering.mc_grid_resolution, res, res.label());
                            }
                        });
                    if state.rendering.mc_grid_resolution != prev_resolution {
                        action = GuiAction::RebuildMcGrid;
                    }
                    let mut blur_val = state.rendering.mc_blur_radius as i32;
                    ui.add(
                        egui::Slider::new(&mut blur_val, 0..=5)
                            .text("Surface Smoothing")
                    ).on_hover_text(
                        "Post-blur of the density field (radius in voxels).\n\
                         Rounds the bulk surface but erases thin sheets and droplets\n\
                         — every step roughly halves the smallest surviving feature.\n\
                         With Anisotropic Kernels on, 0-1 is recommended.",
                    );
                    state.rendering.mc_blur_radius = blur_val as u32;
                    ui.add(
                        egui::Slider::new(&mut state.rendering.mc_density_radius_scale, 1.0..=3.0)
                            .text("Density Radius Scale")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.rendering.mc_threshold, 0.1..=1.5)
                            .text("Surface Threshold")
                    );
                    ui.checkbox(&mut state.rendering.mc_anisotropy, "Anisotropic Kernels (Yu & Turk)")
                        .on_hover_text(
                            "Fit per-particle ellipsoids to the local particle distribution:\n\
                             flattens calm surfaces, thins splash sheets, keeps droplets round.\n\
                             Density Radius Scale 1.0 is recommended (and cheapest) when enabled.",
                        );
                    if state.rendering.mc_anisotropy {
                        ui.add(
                            egui::Slider::new(&mut state.rendering.mc_anisotropy_strength, 0.0..=1.0)
                                .text("Anisotropy Strength")
                        );
                    }
                    ui.add(
                        egui::Slider::new(&mut state.rendering.water_roughness, 0.01..=0.5)
                            .text("Roughness")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.rendering.ripple_strength, 0.0..=0.06)
                            .text("Ripple Strength")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.rendering.water_clarity, 0.0..=1.0)
                            .text("Clarity")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.rendering.refraction_strength, 0.0..=0.10)
                            .text("Refraction")
                    );
                    ui.checkbox(&mut state.rendering.ssr_enabled, "Screen-Space Reflections");
                    ui.label("Deep Water Color:");
                    egui::color_picker::color_edit_button_rgb(ui, &mut state.rendering.deep_water_color);
                }

                if state.rendering.render_mode == FluidRenderMode::ScreenSpace {
                    ui.add_space(8.0);
                    ui.separator();
                    ui.add(
                        egui::Slider::new(&mut state.rendering.ss_radius_scale, 0.25..=1.0)
                            .text("Radius Scale")
                    );
                    let mut filter_size = state.rendering.ss_filter_size as i32;
                    ui.add(
                        egui::Slider::new(&mut filter_size, 4..=30)
                            .text("Filter Size")
                    );
                    state.rendering.ss_filter_size = filter_size as u32;
                    let mut filter_iters = state.rendering.ss_filter_iterations as i32;
                    ui.add(
                        egui::Slider::new(&mut filter_iters, 0..=5)
                            .text("Filter Passes")
                    );
                    state.rendering.ss_filter_iterations = filter_iters as u32;
                    ui.add(
                        egui::Slider::new(&mut state.rendering.water_roughness, 0.01..=0.5)
                            .text("Roughness")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.rendering.water_clarity, 0.0..=1.0)
                            .text("Clarity")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.rendering.refraction_strength, 0.0..=0.10)
                            .text("Refraction")
                    );
                    ui.label("Deep Water Color:");
                    egui::color_picker::color_edit_button_rgb(ui, &mut state.rendering.deep_water_color);

                    ui.add_space(4.0);
                    ui.separator();
                    ui.label("Debug View:");
                    egui::ComboBox::from_id_salt("ss_debug")
                        .selected_text(match state.rendering.ss_debug_view {
                            1 => "Depth",
                            2 => "Normals",
                            3 => "Thickness",
                            4 => "Coverage",
                            _ => "Off",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut state.rendering.ss_debug_view, 0, "Off");
                            ui.selectable_value(&mut state.rendering.ss_debug_view, 1, "Depth");
                            ui.selectable_value(&mut state.rendering.ss_debug_view, 2, "Normals");
                            ui.selectable_value(&mut state.rendering.ss_debug_view, 3, "Thickness");
                            ui.selectable_value(&mut state.rendering.ss_debug_view, 4, "Coverage");
                        });
                }

                ui.add_space(4.0);
                ui.label("Particle Color:");
                egui::color_picker::color_edit_button_rgb(ui, &mut state.rendering.particle_color);
            });

            ui.add_space(8.0);

            // Environment controls
            ui.collapsing("Environment", |ui| {
                ui.label("Background Mode:");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut state.environment.background_mode, BackgroundMode::Environment, "HDR Environment");
                    ui.selectable_value(&mut state.environment.background_mode, BackgroundMode::SolidColor, "Solid Color");
                });

                if state.environment.background_mode == BackgroundMode::SolidColor {
                    ui.add_space(4.0);
                    ui.label("Background Color:");
                    egui::color_picker::color_edit_button_rgb(ui, &mut state.environment.background_color);
                }

                ui.add_space(8.0);
                ui.separator();
                ui.label("HDR Environment Map:");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut state.environment.hdr_selection, HdrEnvironment::Farmland, "Farmland");
                    ui.selectable_value(&mut state.environment.hdr_selection, HdrEnvironment::PureSky, "Pure Sky");
                });

                ui.add_space(4.0);
                ui.add(
                    egui::Slider::new(&mut state.environment.environment_intensity, 0.1..=3.0)
                        .text("Intensity")
                );
            });

            ui.add_space(8.0);

            // Lighting controls
            ui.collapsing("Lighting", |ui| {
                ui.checkbox(&mut state.lighting.sun_enabled, "Enable Sun Light");

                if state.lighting.sun_enabled {
                    ui.add_space(8.0);

                    ui.label("Sun Direction:");
                    ui.add(
                        egui::Slider::new(&mut state.lighting.sun_direction[0], -1.0..=1.0)
                            .text("X")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.lighting.sun_direction[1], 0.0..=1.0)
                            .text("Y (up)")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.lighting.sun_direction[2], -1.0..=1.0)
                            .text("Z")
                    );

                    ui.add_space(8.0);
                    ui.label("Sun Color:");
                    egui::color_picker::color_edit_button_rgb(ui, &mut state.lighting.sun_color);

                    ui.add(
                        egui::Slider::new(&mut state.lighting.sun_intensity, 0.0..=5.0)
                            .text("Intensity")
                    );

                }

                ui.add_space(8.0);
                ui.separator();
                ui.checkbox(&mut state.caustics.enabled, "Caustics (Pool Floor)");
                if state.caustics.enabled {
                    let active = state.lighting.sun_enabled
                        && state.container.style == ContainerStyle::OpaquePool
                        && state.rendering.render_mode == FluidRenderMode::MarchingCubes;
                    if !active {
                        ui.label("(needs Sun + Pool container + Marching Cubes)");
                    }
                    ui.add(
                        egui::Slider::new(&mut state.caustics.intensity, 0.0..=3.0)
                            .text("Intensity")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.caustics.shadow_strength, 0.0..=1.0)
                            .text("Water Shadow")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.caustics.focus, 1.0..=4.0)
                            .text("Focus")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.caustics.dispersion, 0.0..=1.0)
                            .text("Dispersion")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.caustics.ripple_strength, 0.0..=0.25)
                            .text("Ripple Detail")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.caustics.splat_size, 0.35..=2.5)
                            .text("Splat Size")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.caustics.blur_sigma, 0.3..=3.0)
                            .text("Blur")
                    );
                    ui.add(
                        egui::Slider::new(&mut state.caustics.temporal_smoothing, 0.0..=0.95)
                            .text("Temporal Smoothing")
                    );
                }
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
                    ui.separator();

                    // Ambient Occlusion
                    ui.checkbox(&mut state.post_process.ao_enabled, "Ambient Occlusion (GTAO)");
                    if state.post_process.ao_enabled {
                        ui.add(
                            egui::Slider::new(&mut state.post_process.ao_intensity, 0.0..=3.0)
                                .text("Intensity")
                        );
                        ui.add(
                            egui::Slider::new(&mut state.post_process.ao_radius, 0.05..=0.5)
                                .text("Radius")
                        );
                        egui::ComboBox::from_label("AO Debug")
                            .selected_text(state.post_process.ao_debug_mode.label())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut state.post_process.ao_debug_mode, AoDebugMode::Off, AoDebugMode::Off.label());
                                ui.selectable_value(&mut state.post_process.ao_debug_mode, AoDebugMode::RawAo, AoDebugMode::RawAo.label());
                                ui.selectable_value(&mut state.post_process.ao_debug_mode, AoDebugMode::AppliedFactor, AoDebugMode::AppliedFactor.label());
                            });
                    }

                    ui.add_space(8.0);
                    if ui.button("Reset Post Processing").clicked() {
                        state.post_process.reset_defaults();
                    }
                }
            });

            ui.add_space(8.0);

            // Quality settings
            ui.collapsing("Quality", |ui| {
                ui.label("Anti-Aliasing (MSAA):");
                ui.horizontal(|ui| {
                    use crate::state::MsaaSamples;
                    for option in [MsaaSamples::Off, MsaaSamples::X2, MsaaSamples::X4, MsaaSamples::X8] {
                        if ui.selectable_label(state.quality.msaa == option, option.label()).clicked() {
                            state.quality.msaa = option;
                        }
                    }
                });
                ui.label("(Requires restart to take effect)");

                ui.add_space(8.0);
                ui.checkbox(&mut state.quality.fxaa_enabled, "FXAA (Post-Process AA)");
            });

            ui.add_space(16.0);
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Reset to Defaults").clicked() {
                    action = GuiAction::ResetDefaults;
                }
                if ui.button("Export Config")
                    .on_hover_text("Save all current settings + camera to configs/export_NNN.json\n(reload with: fluid-toy --config <file>)")
                    .clicked()
                {
                    action = GuiAction::ExportConfig;
                }
            });
            if let Some(path) = &state.runtime.last_export {
                ui.label(format!("Saved: {path}"));
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
    RebuildMcGrid,
    ExportConfig,
}

/// Default configs for reset functionality
impl SimulationConfig {
    pub fn reset_defaults(&mut self) {
        *self = Self::default();
    }
}


