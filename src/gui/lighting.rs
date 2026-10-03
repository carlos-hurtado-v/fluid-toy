//! Lighting tab: the sun and pool caustics (the environment map is on the
//! Scene tab)

use super::section;
use crate::state::{AppState, ContainerStyle, FluidRenderMode};

pub(super) fn show(ui: &mut egui::Ui, state: &mut AppState) {
    section(ui, "Sun", |ui| sun(ui, state));
    section(ui, "Caustics", |ui| caustics(ui, state));
}

fn sun(ui: &mut egui::Ui, state: &mut AppState) {
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
}

fn caustics(ui: &mut egui::Ui, state: &mut AppState) {
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
}
