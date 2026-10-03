//! Post tab: the post-processing chain and anti-aliasing

use super::section;
use crate::state::{AoDebugMode, AppState};

pub(super) fn show(ui: &mut egui::Ui, state: &mut AppState) {
    section(ui, "Post Processing", |ui| post_processing(ui, state));
    section(ui, "Quality", |ui| quality(ui, state));
}

fn post_processing(ui: &mut egui::Ui, state: &mut AppState) {
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
}

fn quality(ui: &mut egui::Ui, state: &mut AppState) {
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
}
