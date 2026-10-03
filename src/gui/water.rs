//! Water tab: render mode and how the fluid is drawn in each mode

use super::{section, GuiAction};
use crate::state::{AppState, FluidRenderMode, McGridResolution};

pub(super) fn show(ui: &mut egui::Ui, state: &mut AppState, action: &mut GuiAction) {
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
    ui.horizontal(|ui| {
        ui.label("Particle Color:");
        egui::color_picker::color_edit_button_rgb(ui, &mut state.rendering.particle_color);
    });
    ui.add_space(4.0);

    match state.rendering.render_mode {
        FluidRenderMode::MarchingCubes => {
            section(ui, "Surface", |ui| mc_surface(ui, state, action));
            section(ui, "Appearance", |ui| mc_appearance(ui, state));
            section(ui, "Refraction", |ui| mc_refraction(ui, state));
        }
        FluidRenderMode::ScreenSpace => {
            section(ui, "Surface", |ui| ss_surface(ui, state));
            section(ui, "Appearance", |ui| ss_appearance(ui, state));
            ss_debug_view(ui, state);
        }
        FluidRenderMode::Particles => {}
    }
}

/// Marching Cubes: the density field and the mesh extracted from it
fn mc_surface(ui: &mut egui::Ui, state: &mut AppState, action: &mut GuiAction) {
    let prev_resolution = state.rendering.mc_grid_resolution;
    egui::ComboBox::from_label("Grid Resolution")
        .selected_text(state.rendering.mc_grid_resolution.label())
        .show_ui(ui, |ui| {
            for res in McGridResolution::ALL {
                ui.selectable_value(&mut state.rendering.mc_grid_resolution, res, res.label());
            }
        });
    if state.rendering.mc_grid_resolution != prev_resolution {
        *action = GuiAction::RebuildMcGrid;
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
    ui.add_enabled(
        state.rendering.mc_calm_smoothing > 0.0,
        egui::Slider::new(&mut state.rendering.mc_normal_denoise, 0.0..=4.0)
            .text("Calm Normal Denoise (deg)")
    ).on_hover_text(
        "Removes the faint particle-scale ripple left in calm water's\n\
         normals (marbled grazing reflections): normals within about\n\
         this angle of a wider average take its direction, larger\n\
         differences are real shape and stay. Never moves a normal by\n\
         more than half this angle; geometry untouched. Needs Calm\n\
         Surface Smoothing. 0 = off.",
    );
    ui.checkbox(&mut state.rendering.mc_wet_bodies, "Water Meets Bodies")
        .on_hover_text(
            "The water surface runs into rigid bodies at the level it has\n\
             next to them (cube, sphere, cylinder, torus, propeller).\n\
             Off: the surface stops short of a body and sinks toward it,\n\
             a moat a few centimetres deep at every waterline.",
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
}

/// Marching Cubes: surface finish, water medium, reflections
fn mc_appearance(ui: &mut egui::Ui, state: &mut AppState) {
    ui.add(
        egui::Slider::new(&mut state.rendering.water_roughness, 0.01..=0.5)
            .text("Roughness")
    );
    ui.add(
        egui::Slider::new(&mut state.rendering.ripple_strength, 0.0..=0.06)
            .text("Ripple Strength")
    );
    water_medium_controls(ui, state);
    ui.checkbox(&mut state.rendering.ssr_enabled, "Screen-Space Reflections");
    deep_water_color_control(ui, state);
}

/// Marching Cubes: refraction model, its in-water test and debug views
fn mc_refraction(ui: &mut egui::Ui, state: &mut AppState) {
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
    ui.checkbox(&mut state.rendering.mc_volume_trace, "World-Space Water Test")
        .on_hover_text(
            "Glass tank: rays inside the water test the density field the mesh \
             is built from to find where they leave it, instead of the depth \
             buffers (which cannot see behind bodies, behind a nearer fold of \
             the surface, or off screen: stripes and stair steps in mirrors). \
             Off: the older screen-space test, tuned by the two settings below",
        );
    // Only the screen-space test reads these two
    ui.add_enabled_ui(!state.rendering.mc_volume_trace, |ui| {
        egui::ComboBox::from_label("Behind-Surface Rays")
            .selected_text(state.rendering.mc_silhouette_exit.label())
            .show_ui(ui, |ui| {
                for mode in crate::state::McSilhouetteExit::ALL {
                    ui.selectable_value(&mut state.rendering.mc_silhouette_exit, mode, mode.label());
                }
            })
            .response
            .on_hover_text(
                "Screen-space test only (World-Space Water Test off). A ray inside \
                 the water that passes behind a nearer layer of water surface (as \
                 the camera sees it) looks like it left the water. Exit: old \
                 behaviour (stripes in mirrors). Continue: assume it is still in \
                 the water (removes those stripes, but can draw stair steps along \
                 the outline in mirrors)",
            )
            .on_disabled_hover_text("No effect while World-Space Water Test is on");
        ui.checkbox(&mut state.rendering.mc_front_face_exit, "Front-Face Exits")
            .on_hover_text(
                "Screen-space test only (World-Space Water Test off). Rays inside \
                 the water also detect leaving through a surface the camera sees \
                 from its side (the free surface seen from above). Off: they are \
                 noticed only at the water's outline on screen, with the outline's \
                 normal (stripes in mirrors when looking across the surface)",
            )
            .on_disabled_hover_text("No effect while World-Space Water Test is on");
    });
    ui.checkbox(&mut state.rendering.mc_filtered_lookup, "Filtered Lookups")
        .on_hover_text(
            "What a refracted or mirrored ray finally shows is read over its \
             footprint on screen (mip chain + anisotropic filtering) instead of \
             one sample. A small effect: slightly less sparkle where a \
             refraction shrinks the image",
        );
}

/// Screen Space: splats and the depth filter
fn ss_surface(ui: &mut egui::Ui, state: &mut AppState) {
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
}

/// Screen Space: surface finish, water medium, refraction
fn ss_appearance(ui: &mut egui::Ui, state: &mut AppState) {
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
}

fn ss_debug_view(ui: &mut egui::Ui, state: &mut AppState) {
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
