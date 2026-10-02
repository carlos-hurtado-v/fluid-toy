//! GUI module - egui integration for parameter control

use crate::state::{AoDebugMode, AppState, BackgroundMode, ContainerStyle, FluidRenderMode, ForceMode, HdrEnvironment, McGridResolution, RigidBodyConfig, RigidBodyMotion, RigidBodyShape, SimulationConfig, MAX_RIGID_BODIES};

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
            ui.collapsing("Rigid Bodies", |ui| {
                let mut remove_idx: Option<usize> = None;

                for (i, body) in state.rigid_bodies.iter_mut().enumerate() {
                    let title = format!("Body {} — {:?} {:?}", i + 1, body.motion, body.shape);
                    egui::CollapsingHeader::new(title)
                        .id_salt(i)
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut body.enabled, "Enable");
                                if ui.button("Remove").clicked() {
                                    remove_idx = Some(i);
                                }
                            });

                            ui.add_space(4.0);
                            ui.label("Shape:");
                            ui.horizontal_wrapped(|ui| {
                                ui.selectable_value(&mut body.shape, RigidBodyShape::Cube, "Cube");
                                ui.selectable_value(&mut body.shape, RigidBodyShape::Sphere, "Sphere");
                                ui.selectable_value(&mut body.shape, RigidBodyShape::Cylinder, "Cylinder");
                                ui.selectable_value(&mut body.shape, RigidBodyShape::Torus, "Torus");
                                ui.selectable_value(&mut body.shape, RigidBodyShape::Propeller, "Propeller");
                                ui.selectable_value(&mut body.shape, RigidBodyShape::Custom, "Duck");
                            });

                            if body.shape == RigidBodyShape::Propeller {
                                ui.add(egui::Slider::new(&mut body.prop_blades, 2..=6).text("Blades"));
                                ui.add(
                                    egui::Slider::new(&mut body.prop_pitch_deg, 0.0..=60.0)
                                        .text("Blade Pitch (deg)"),
                                );
                            }

                            ui.add_space(4.0);
                            ui.label("Motion:");
                            ui.horizontal(|ui| {
                                ui.selectable_value(&mut body.motion, RigidBodyMotion::Static, "Static");
                                ui.selectable_value(&mut body.motion, RigidBodyMotion::Kinematic, "Kinematic");
                                ui.selectable_value(&mut body.motion, RigidBodyMotion::Dynamic, "Dynamic");
                            });

                            match body.motion {
                                RigidBodyMotion::Static => {}
                                RigidBodyMotion::Kinematic => {
                                    ui.add(
                                        egui::Slider::new(&mut body.spin_rpm, -300.0..=300.0)
                                            .text("Spin (RPM)"),
                                    );
                                }
                                RigidBodyMotion::Dynamic => {
                                    ui.horizontal(|ui| {
                                        if ui.button("Stop Motion").clicked() {
                                            body.velocity = [0.0; 3];
                                            body.angular_velocity = [0.0; 3];
                                        }
                                        if ui.button("Reset Rotation").clicked() {
                                            body.orientation = [0.0, 0.0, 0.0, 1.0];
                                            body.angular_velocity = [0.0; 3];
                                        }
                                    });
                                }
                            }

                            ui.add_space(4.0);
                            ui.add(egui::Slider::new(&mut body.half_extent, 0.05..=0.5).text("Size"));
                            if body.motion == RigidBodyMotion::Dynamic {
                                ui.add(
                                    egui::Slider::new(&mut body.relative_density, 0.05..=3.0)
                                        .text("Density (x fluid)")
                                        .logarithmic(true),
                                );
                                ui.label("  1.0 = neutral buoyancy, above 1 sinks");
                            }

                            ui.add_space(4.0);
                            ui.label("Position:");
                            ui.add(egui::Slider::new(&mut body.position[0], -1.5..=1.5).text("X"));
                            ui.add(egui::Slider::new(&mut body.position[1], -1.5..=1.5).text("Y"));
                            ui.add(egui::Slider::new(&mut body.position[2], -1.5..=1.5).text("Z"));

                            if body.motion != RigidBodyMotion::Dynamic {
                                ui.add_space(4.0);
                                ui.label("Orientation (deg):");
                                ui.add(egui::Slider::new(&mut body.euler_deg[0], -180.0..=180.0).text("Rot X"));
                                ui.add(egui::Slider::new(&mut body.euler_deg[1], -180.0..=180.0).text("Rot Y"));
                                ui.add(egui::Slider::new(&mut body.euler_deg[2], -180.0..=180.0).text("Rot Z"));
                            }

                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.label("Color:");
                                egui::color_picker::color_edit_button_rgb(ui, &mut body.color);
                            });
                        });
                }

                if let Some(i) = remove_idx {
                    state.rigid_bodies.remove(i);
                }

                ui.add_space(4.0);
                if state.rigid_bodies.len() < MAX_RIGID_BODIES {
                    if ui.button("+ Add Body").clicked() {
                        state.rigid_bodies.push(RigidBodyConfig::default());
                    }
                } else {
                    ui.label(format!("Max {} bodies", MAX_RIGID_BODIES));
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
                ui.add(
                    egui::Slider::new(&mut state.sph.boundary_density, 0.0..=1.25)
                        .text("Boundary Density")
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
                        "Lifetime of foam PARTICLES (energy-scaled per particle at birth); \
                         spray persists until it lands, bubbles until they surface. With the \
                         Surface Foam Map on, top-surface foam lives in the map instead \
                         (Foam Persistence) and this only covers foam on walls and overhangs",
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
                        egui::Slider::new(&mut state.spray.particle_size, 0.001..=0.01)
                            .logarithmic(true)
                            .text("Particle Size")
                    ).on_hover_text(
                        "Sprite size of spray streaks and bubbles, grain of the foam field; \
                         total foam amount is size-invariant (Coverage/Aeration set that)",
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
                    ui.checkbox(&mut state.spray.foam_map, "Surface Foam Map")
                        .on_hover_text(
                            "Marching Cubes: foam settling on the top surface moves into an \
                             advected 2D layer that the surface flow stretches into patches \
                             and strands. Off = every foam particle rendered individually",
                        );
                    ui.add_enabled(
                        state.spray.foam_map,
                        egui::Slider::new(&mut state.spray.foam_persistence, 0.5..=60.0)
                            .logarithmic(true)
                            .text("Foam Persistence")
                            .suffix(" s"),
                    ).on_hover_text("Half-life of surface foam: clean water ~1-3 s, pool or sea water 10 s and up");
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
                         Default 1; Calm Surface Smoothing handles still water.",
                    );
                    state.rendering.mc_blur_radius = blur_val as u32;
                    ui.add(
                        egui::Slider::new(&mut state.rendering.mc_calm_smoothing, 0.0..=1.0)
                            .text("Calm Surface Smoothing")
                    ).on_hover_text(
                        "Wide smoothing applied only where the water is thick (bulk\n\
                         surfaces): flattens the particle-scale lumps on still water\n\
                         while splash sheets and droplets keep Surface Smoothing alone.\n\
                         0 = off.",
                    );
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
                    water_medium_controls(ui, state);
                    ui.checkbox(&mut state.rendering.mc_physical_refraction, "Physical Refraction")
                        .on_hover_text(
                            "Snell refraction at the surface and again on the way out of the \
                             body: the pool floor shows its true apparent depth and drops and \
                             crests act as lenses. Off = legacy screen-space offset",
                        );
                    ui.add_enabled(
                        !state.rendering.mc_physical_refraction,
                        egui::Slider::new(&mut state.rendering.refraction_strength, 0.0..=0.10)
                            .text("Refraction (legacy)")
                    );
                    ui.checkbox(&mut state.rendering.ssr_enabled, "Screen-Space Reflections");
                    egui::ComboBox::from_label("Refraction Debug")
                        .selected_text(state.rendering.mc_debug_view.label())
                        .show_ui(ui, |ui| {
                            for view in crate::state::McDebugView::ALL {
                                ui.selectable_value(&mut state.rendering.mc_debug_view, view, view.label());
                            }
                        })
                        .response
                        .on_hover_text(
                            "Water pixels show what physical refraction did, as data \
                             (post-processing off while active). Paths: route taken (R), \
                             final lookup (G), mirror bounces (B). Lookup: lookup uv. \
                             Jump: lookup jump between pixels (bright = banding/aliasing). \
                             Exit: exit angle cosine, water path, exit interface. \
                             Mirror: interface of the last mirror reflection, exit \
                             interface, bounces. \
                             Decode captures with scripts/debug_decode.py",
                        );
                    egui::ComboBox::from_label("Behind-Surface Rays")
                        .selected_text(state.rendering.mc_silhouette_exit.label())
                        .show_ui(ui, |ui| {
                            for mode in crate::state::McSilhouetteExit::ALL {
                                ui.selectable_value(&mut state.rendering.mc_silhouette_exit, mode, mode.label());
                            }
                        })
                        .response
                        .on_hover_text(
                            "Glass tank: a ray inside the water that passes behind a nearer \
                             layer of water surface (as the camera sees it) looks like it left \
                             the water. Exit: old behaviour (stripes in mirrors). Continue: \
                             assume it is still in the water (removes those stripes, but can \
                             draw stair steps along the outline in mirrors)",
                        );
                    ui.checkbox(&mut state.rendering.mc_front_face_exit, "Front-Face Exits")
                        .on_hover_text(
                            "Glass tank: rays inside the water also detect leaving through \
                             a surface the camera sees from its side (the free surface seen \
                             from above). Off: they are noticed only at the water's outline \
                             on screen, with the outline's normal (stripes in mirrors when \
                             looking across the surface)",
                        );
                    ui.checkbox(&mut state.rendering.mc_volume_trace, "World-Space Water Test")
                        .on_hover_text(
                            "Glass tank: rays inside the water test the density field the mesh \
                             is built from to find where they leave it, instead of the depth \
                             buffers (which cannot see behind bodies, behind a nearer fold of \
                             the surface, or off screen: stripes and stair steps in mirrors). \
                             With it on, Behind-Surface Rays and Front-Face Exits have no effect",
                        );
                    ui.checkbox(&mut state.rendering.mc_filtered_lookup, "Filtered Lookups")
                        .on_hover_text(
                            "What a refracted or mirrored ray finally shows is read over                              its footprint on screen (mip chain + anisotropic filtering)                              instead of one sample. Off: lookups that shrink the image skip                              texels (streaks and sparkle in grazing mirrors, shimmer in motion)",
                        );
                    deep_water_color_control(ui, state);
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
                        egui::Slider::new(&mut state.rendering.ss_nr_range, 1.0..=15.0)
                            .text("NR Depth Range")
                            .suffix(" r")
                    ).on_hover_text(
                        "Narrow-range filter window in particle radii. Small (~3, per the \
                         paper) keeps nearby surfaces separate in churn; large (10, Splash) \
                         depth-merges them into blobs",
                    );
                    ui.add(
                        egui::Slider::new(&mut state.rendering.ss_nr_offset, 0.5..=10.0)
                            .text("NR Clamp Offset")
                            .suffix(" r")
                    ).on_hover_text(
                        "Where far-side and background samples clamp, in particle radii; \
                         equal to the range keeps the filter response continuous",
                    );
                    ui.checkbox(&mut state.rendering.ss_temporal, "Temporal Smoothing")
                        .on_hover_text(
                            "Reprojected EMA on the filtered depth; damps churn shimmer \
                             and specular fireflies",
                        );
                    ui.checkbox(&mut state.rendering.mc_anisotropy, "Anisotropic Splats")
                        .on_hover_text(
                            "Stretch splats into Yu & Turk ellipsoids (same records and \
                             toggle as the MC mode); flattens calm surfaces and thin sheets",
                        );
                    ui.add(
                        egui::Slider::new(&mut state.rendering.water_roughness, 0.01..=0.5)
                            .text("Roughness")
                    );
                    water_medium_controls(ui, state);
                    ui.add(
                        egui::Slider::new(&mut state.rendering.refraction_strength, 0.0..=0.10)
                            .text("Refraction")
                    );
                    deep_water_color_control(ui, state);

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
                ui.checkbox(&mut state.environment.ground_projection, "Ground Projection")
                    .on_hover_text(
                        "Project the HDR's ground onto a plane under the container, so \
                         the scene stands on it (with parallax) instead of floating over \
                         a ground at infinity",
                    );
                ui.add_enabled(
                    state.environment.ground_projection,
                    egui::Slider::new(&mut state.environment.ground_capture_height, 0.5..=4.0)
                        .text("Ground Scale")
                        .suffix(" m"),
                ).on_hover_text("Height the HDR was shot from: larger = coarser ground texture");
            });

            ui.add_space(8.0);

            // Lighting controls
            ui.collapsing("Lighting", |ui| {
                ui.checkbox(&mut state.lighting.sun_enabled, "Enable Sun Light");

                if state.lighting.sun_enabled {
                    ui.add_space(8.0);

                    ui.checkbox(&mut state.lighting.sun_from_environment, "Sun Follows HDR")
                        .on_hover_text(
                            "In Environment mode, aim the sun at the HDR map's own sun so \
                             shadows, caustics and glints match the sky (maps without a \
                             distinct sun keep the manual direction)",
                        );
                    let following_hdr = state.lighting.environment_sun_active().is_some();
                    ui.label(if following_hdr { "Sun Direction (following HDR):" } else { "Sun Direction:" });
                    ui.add_enabled_ui(!following_hdr, |ui| {
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
                    });

                    ui.add_space(8.0);
                    ui.add_enabled_ui(!following_hdr, |ui| {
                        ui.label("Sun Color:");
                        egui::color_picker::color_edit_button_rgb(ui, &mut state.lighting.sun_color);

                        ui.add(
                            egui::Slider::new(&mut state.lighting.sun_intensity, 0.0..=5.0)
                                .text("Intensity")
                        );
                    });
                    if let Some(sun) = state.lighting.environment_sun_active() {
                        ui.label(format!(
                            "HDR sun irradiance: {:.2} / {:.2} / {:.2}",
                            sun.irradiance[0], sun.irradiance[1], sun.irradiance[2],
                        ));
                    }

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
                            egui::Slider::new(&mut state.post_process.bloom_threshold, 0.0..=16.0)
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
                            egui::Slider::new(&mut state.post_process.streaks_threshold, 0.0..=32.0)
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

            // Live fluid measurements (probe pass) + scenario status
            ui.collapsing("Measurements", |ui| {
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
            });

            ui.add_space(8.0);
            ui.label(format!("Particles: {}", state.runtime.particle_count));
            ui.label(format!("FPS: {:.0}", state.runtime.fps));
        });

    action
}

/// Water medium controls shared by the MC and SS sections (one shared state)
fn water_medium_controls(ui: &mut egui::Ui, state: &mut AppState) {
    ui.checkbox(&mut state.rendering.physical_water_medium, "Physical Water Medium")
        .on_hover_text(
            "Pure-water absorption plus single scattering of sun and sky light \
             along the true in-water path: the body color comes from the light, \
             not a hand-set deep color. Off = legacy hand-tuned model",
        );
    let clarity_hint = if state.rendering.physical_water_medium {
        "Turbidity (scattering): 1 = very clear pool, 0 = murky. Particle color sets its tint"
    } else {
        "Optical density of the legacy model: 1 = crystal clear, 0 = murky"
    };
    ui.add(
        egui::Slider::new(&mut state.rendering.water_clarity, 0.0..=1.0)
            .text("Clarity")
    ).on_hover_text(clarity_hint);
}

/// Legacy deep color: has no effect under the physical water medium
fn deep_water_color_control(ui: &mut egui::Ui, state: &mut AppState) {
    ui.add_enabled_ui(!state.rendering.physical_water_medium, |ui| {
        ui.label("Deep Water Color (legacy medium):");
        egui::color_picker::color_edit_button_rgb(ui, &mut state.rendering.deep_water_color);
    });
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


