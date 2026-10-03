//! Whitewater tab: spray, foam and bubbles

use crate::state::AppState;

pub(super) fn show(ui: &mut egui::Ui, state: &mut AppState) {
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
}
