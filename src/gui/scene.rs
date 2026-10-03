//! Scene tab: the container, the backdrop behind it, and the rigid bodies

use super::section;
use crate::state::{AppState, BackgroundMode, ContainerStyle, HdrEnvironment, RigidBodyConfig, RigidBodyMotion, RigidBodyShape, MAX_RIGID_BODIES};

pub(super) fn show(ui: &mut egui::Ui, state: &mut AppState) {
    section(ui, "Container", |ui| container(ui, state));
    section(ui, "Environment", |ui| environment(ui, state));
    // Last: the one group that keeps growing (up to MAX_RIGID_BODIES bodies)
    section(ui, "Rigid Bodies", |ui| rigid_bodies(ui, state));
}

fn container(ui: &mut egui::Ui, state: &mut AppState) {
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
}

fn environment(ui: &mut egui::Ui, state: &mut AppState) {
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
}

fn rigid_bodies(ui: &mut egui::Ui, state: &mut AppState) {
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
}
