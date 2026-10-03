//! GUI module - egui integration for parameter control
//!
//! One floating "Controls" window: an action row and a tab row that stay put,
//! and below them the selected tab's sections in a scroll area, so the window
//! never grows past the app window. One file per tab. FPS and particle count
//! sit in a click-through overlay in the top-right corner, visible whatever
//! the window shows.

mod lighting;
mod post;
mod scene;
mod simulation;
mod water;
mod whitewater;

use crate::state::{AppState, SimulationConfig};

/// Room left under the scroll area: the Controls window's own bottom margin
/// plus a gap to the bottom of the app window
const BOTTOM_MARGIN: f32 = 16.0;
/// Starting width of the window's content: fits the widest tab (Water), so
/// the window does not widen the first time that tab is opened
const DEFAULT_WIDTH: f32 = 340.0;
/// The scroll area never gets shorter than this; a window dragged too low to
/// fit it is moved back up by egui
const MIN_SCROLL_HEIGHT: f32 = 160.0;

/// Tabs of the control panel, in display order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
enum Tab {
    #[default]
    Simulation,
    Scene,
    Water,
    Whitewater,
    Lighting,
    Post,
}

impl Tab {
    const ALL: [Self; 6] = [
        Self::Simulation,
        Self::Scene,
        Self::Water,
        Self::Whitewater,
        Self::Lighting,
        Self::Post,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Simulation => "Simulation",
            Self::Scene => "Scene",
            Self::Water => "Water",
            Self::Whitewater => "Whitewater",
            Self::Lighting => "Lighting",
            Self::Post => "Post",
        }
    }
}

/// Renders the control panel and the status overlay, returns any triggered action
pub fn render_control_panel(ctx: &egui::Context, state: &mut AppState) -> GuiAction {
    let mut action = GuiAction::None;

    status_overlay(ctx, state);

    // Which tab is open is GUI-only state: it lives in egui's memory, not in
    // AppState, so it never reaches exported configs or `--set` paths
    let tab_id = egui::Id::new("controls_tab");
    let mut tab = ctx.data(|d| d.get_temp::<Tab>(tab_id)).unwrap_or_default();

    let screen = ctx.content_rect();
    egui::Window::new("Controls")
        .default_pos([10.0, 10.0])
        .default_width(DEFAULT_WIDTH)
        // Width is the user's to drag; height follows the content. On the
        // axis that is not resizable, min/max height only set the layout
        // budget egui hands the content, not the size shown: asking for the
        // whole screen (egui caps it to what fits) leaves it to the scroll
        // area below to decide how tall the window gets
        .resizable([true, false])
        .min_height(screen.height())
        .collapsible(true)
        .show(ctx, |ui| {
            // A thin scroll bar that is always drawn while a tab overflows
            // (the default one only appears on hover)
            ui.spacing_mut().scroll = egui::style::ScrollStyle::thin();

            action_row(ui, state, &mut action);
            ui.separator();
            tab_grid(ui, &mut tab);
            ui.separator();

            // As tall as the tab's content, up to the bottom of the app
            // window; each tab keeps its own scroll position
            let room = screen.bottom() - ui.cursor().top() - BOTTOM_MARGIN;
            egui::ScrollArea::vertical()
                .id_salt(tab)
                .max_height(room.max(MIN_SCROLL_HEIGHT))
                .auto_shrink([false, true])
                .show(ui, |ui| match tab {
                    Tab::Simulation => simulation::show(ui, state, &mut action),
                    Tab::Scene => scene::show(ui, state),
                    Tab::Water => water::show(ui, state, &mut action),
                    Tab::Whitewater => whitewater::show(ui, state),
                    Tab::Lighting => lighting::show(ui, state),
                    Tab::Post => post::show(ui, state),
                });
        });

    ctx.data_mut(|d| d.insert_temp(tab_id, tab));

    action
}

/// The tab strip: a fixed 3 x 2 grid of equal cells, so no tab moves when the
/// window changes width
fn tab_grid(ui: &mut egui::Ui, tab: &mut Tab) {
    ui.columns(3, |columns| {
        for (i, t) in Tab::ALL.into_iter().enumerate() {
            columns[i % 3].vertical_centered_justified(|ui| {
                let button = egui::Button::selectable(*tab == t, t.label()).frame_when_inactive(true);
                if ui.add(button).clicked() {
                    *tab = t;
                }
            });
        }
    });
}

/// Whole-app actions, pinned above the tabs
fn action_row(ui: &mut egui::Ui, state: &AppState, action: &mut GuiAction) {
    ui.horizontal(|ui| {
        if ui.button("Reset to Defaults").clicked() {
            *action = GuiAction::ResetDefaults;
        }
        if ui.button("Export Config")
            .on_hover_text("Save all current settings + camera to configs/export_NNN.json\n(reload with: fluid-toy --config <file>)")
            .clicked()
        {
            *action = GuiAction::ExportConfig;
        }
    });
    if let Some(path) = &state.runtime.last_export {
        ui.label(format!("Saved: {path}"));
    }
}

/// FPS and particle count in the top-right corner, whatever the Controls
/// window shows. Nothing in it takes input, so clicks and drags over it still
/// reach the camera and the mouse force
fn status_overlay(ctx: &egui::Context, state: &AppState) {
    egui::Area::new(egui::Id::new("status_overlay"))
        .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                let line = |ui: &mut egui::Ui, text: String| {
                    ui.add(egui::Label::new(text).selectable(false));
                };
                line(ui, format!("FPS: {:.0}", state.runtime.fps));
                line(ui, format!("Particles: {}", state.runtime.particle_count));
            });
        });
}

/// One collapsible group of a tab. Open by default: the tab already narrows
/// what is on screen, and a tab too long for the window scrolls
fn section(ui: &mut egui::Ui, title: &str, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(title)
        .default_open(true)
        .show(ui, add_contents);
    ui.add_space(4.0);
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
