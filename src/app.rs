//! Application state and event handling
//!
//! `App` owns the GPU context, the simulation, every renderer, the camera and
//! egui, and implements winit's `ApplicationHandler`. Its methods are split by
//! concern across the submodules:
//!
//! - `init`: building the simulation and the renderers (startup, resets)
//! - `sync`: pushing config/GUI state to the GPU each frame (params, SH,
//!   environment map), scenario events, particle spawning
//! - `frame`: one frame, as a sequence of phases
//! - `scene`: the fluid render modes (Particles, ScreenSpace, MarchingCubes)
//! - `automation`: captures, snapshots, state files, stats, auto-exit
//!
//! New per-frame work goes into a phase method in the matching submodule (or
//! a new submodule); `update_and_render` in frame.rs stays a list of phase
//! calls, and this file stays the struct plus window events.

mod automation;
mod frame;
mod init;
mod scene;
mod sync;

use std::collections::VecDeque;
use std::io::Write as _;
use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use crate::gpu::GpuContext;
use crate::launch::LaunchOptions;
use crate::render::env_background::EnvBackground;
use crate::render::environment::{ShCoefficients, SunEstimate};
use crate::render::mesh_loader::SdfData;
use crate::render::{
    Camera, CausticsRenderer, ContainerRenderer, GtaoRenderer, MarchingCubesRenderer, ParticleRenderer3D,
    PostProcessRenderer, RigidBodyRenderer, ScreenSpaceFluidRenderer, SprayRenderer, WireframeRenderer,
};
use crate::simulation::{ProbeSystem, SphSimulation3DGrid, SpraySystem};
use crate::state::{AppState, BackgroundMode, ContainerStyle, FluidRenderMode, HdrEnvironment};

pub struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    renderer: Option<ParticleRenderer3D>,
    mc_renderer: Option<MarchingCubesRenderer>,
    ss_renderer: Option<ScreenSpaceFluidRenderer>,
    /// Which water renderer's front depth the whitewater splat gate is bound
    /// to; None forces a rebind (set after resize / renderer recreation)
    spray_depth_bound: Option<FluidRenderMode>,
    wireframe_renderer: Option<WireframeRenderer>,
    container_renderer: Option<ContainerRenderer>,
    caustics_renderer: Option<CausticsRenderer>,
    rigid_body_renderer: Option<RigidBodyRenderer>,
    rigid_body_depth_view: Option<wgpu::TextureView>,  // Fallback depth for modes without shared depth
    spray_system: Option<SpraySystem>,
    spray_renderer: Option<SprayRenderer>,
    spray_prev_enabled: bool,
    post_process_renderer: Option<PostProcessRenderer>,
    gtao_renderer: Option<GtaoRenderer>,
    prev_camera_params: Option<crate::render::GpuCameraParams>,
    // Environment map (used by MC renderer + env background)
    #[allow(dead_code)]
    env_texture: Option<wgpu::Texture>,
    env_view: Option<wgpu::TextureView>,
    env_sampler: Option<wgpu::Sampler>,
    current_hdr: HdrEnvironment,
    sh_coefficients: Option<ShCoefficients>,
    /// Surface foam map (advected 2D foam layer; MC water shader binds it)
    foam_map: Option<crate::simulation::FoamMap>,
    /// Sun found in the loaded HDR map (None when the map is too diffuse)
    hdr_sun: Option<SunEstimate>,
    // Last (mode, solid color bits, hdr) the SH uniform buffers were pushed
    // for. SolidColor mode uses a uniform SH of the background color so the
    // diffuse ambient matches the visible surround instead of the HDR sky.
    last_sh_key: Option<(BackgroundMode, [u32; 3], HdrEnvironment, bool)>,
    // Environment background pass (for Particles mode)
    env_background: Option<EnvBackground>,
    sph_simulation: Option<SphSimulation3DGrid>,
    probe_system: Option<ProbeSystem>,
    sdf_data: Option<SdfData>,
    camera: Camera,
    state: AppState,
    /// Per-event fired flags for scenario events (reset replays the schedule)
    scenario_fired: Vec<bool>,
    // Frame timing
    last_frame_time: Instant,
    // Mouse state for camera control (left button)
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    // Mouse state for force interaction (right button)
    right_mouse_pressed: bool,
    explode_fired: bool, // One-shot tracking for Explode mode
    current_mouse_pos: (f64, f64),
    // Spawn state (middle button - continuous while held)
    middle_mouse_pressed: bool,
    // egui
    egui_ctx: egui::Context,
    egui_winit: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
    // Automation (--capture / --stats / --exit-after)
    launch: LaunchOptions,
    /// Simulation frames completed (deterministic clock for captures/stats)
    sim_frame_index: u64,
    /// Simulated seconds elapsed (sim_frame_index × frame_dt, dt-change aware)
    sim_time: f64,
    pending_captures: VecDeque<u64>,
    /// --snapshot frames (PNG + config + .state into --out)
    pending_snapshots: VecDeque<u64>,
    had_captures: bool,
    /// Clock for --capture / --snapshot / --exit-after: simulated frames, or
    /// rendered frames while --hold freezes a loaded state. Equals
    /// sim_frame_index unless a state was loaded (then it counts from there).
    milestone_frame: u64,
    /// F12: save the next frame (GUI-free) together with its exact config
    snapshot_requested: bool,
    stats_file: Option<std::io::BufWriter<std::fs::File>>,
    should_exit: bool,
}

impl App {
    pub fn new(mut state: AppState, launch: LaunchOptions) -> Self {
        if launch.hold {
            // Frozen on the loaded state: render-only frames
            state.simulation.paused = true;
        } else if launch.is_automated() && state.simulation.paused {
            println!("note: automation flags require a running simulation; ignoring paused=true");
            state.simulation.paused = false;
        }

        let stats_file = launch.stats_path.as_ref().map(|path| {
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                let _ = std::fs::create_dir_all(parent);
            }
            let file = std::fs::File::create(path).unwrap_or_else(|e| {
                eprintln!("error: cannot create stats file {}: {e}", path.display());
                std::process::exit(1);
            });
            let mut writer = std::io::BufWriter::new(file);
            let mut header = String::from(
                "frame,sim_time,particles,mc_vertices,spray_total,spray,foam,bubbles,fps,ta_limit,wc_limit,fluid_max_x,fluid_min_x,fluid_max_y",
            );
            for i in 0..state.scenario.probes.len().min(crate::state::MAX_PROBES) {
                header.push_str(&format!(",probe{i}_h"));
            }
            header.push('\n');
            let _ = writer.write_all(header.as_bytes());
            writer
        });

        if !launch.capture_frames.is_empty() || !launch.snapshot_frames.is_empty() {
            if let Err(e) = std::fs::create_dir_all(&launch.out_dir) {
                eprintln!(
                    "error: cannot create capture dir {}: {e}",
                    launch.out_dir.display()
                );
                std::process::exit(1);
            }
        }

        let pending_captures: VecDeque<u64> = launch.capture_frames.iter().copied().collect();
        let pending_snapshots: VecDeque<u64> = launch.snapshot_frames.iter().copied().collect();
        let had_captures = !pending_captures.is_empty() || !pending_snapshots.is_empty();
        let scenario_fired = vec![false; state.scenario.events.len()];

        Self {
            window: None,
            gpu: None,
            renderer: None,
            mc_renderer: None,
            ss_renderer: None,
            spray_depth_bound: None,
            wireframe_renderer: None,
            container_renderer: None,
            caustics_renderer: None,
            rigid_body_renderer: None,
            rigid_body_depth_view: None,
            spray_system: None,
            spray_renderer: None,
            spray_prev_enabled: true, // default: enabled
            post_process_renderer: None,
            gtao_renderer: None,
            prev_camera_params: None,
            env_texture: None,
            env_view: None,
            env_sampler: None,
            current_hdr: HdrEnvironment::Farmland,
            sh_coefficients: None,
            foam_map: None,
            hdr_sun: None,
            last_sh_key: None,
            env_background: None,
            sph_simulation: None,
            probe_system: None,
            sdf_data: None,
            camera: Camera::default(),
            state,
            scenario_fired,
            last_frame_time: Instant::now(),
            mouse_pressed: false,
            last_mouse_pos: None,
            right_mouse_pressed: false,
            explode_fired: false,
            current_mouse_pos: (0.0, 0.0),
            middle_mouse_pressed: false,
            egui_ctx: egui::Context::default(),
            egui_winit: None,
            egui_renderer: None,
            launch,
            sim_frame_index: 0,
            sim_time: 0.0,
            pending_captures,
            pending_snapshots,
            had_captures,
            milestone_frame: 0,
            snapshot_requested: false,
            stats_file,
            should_exit: false,
        }
    }

    fn simulation_substep_dt(&self) -> f32 {
        self.state.simulation.substep_dt()
    }

    /// Caustics run only for the MC water surface, in the opaque pool
    /// (the floor is the receiver), lit by the sun.
    fn caustics_active(&self) -> bool {
        self.state.caustics.enabled
            && self.state.rendering.render_mode == FluidRenderMode::MarchingCubes
            && self.state.container.style == ContainerStyle::OpaquePool
            && self.state.lighting.sun_enabled
    }

    fn cursor_ray(&self) -> ([f32; 3], [f32; 3]) {
        let gpu = self.gpu.as_ref().unwrap();
        self.camera.screen_to_ray(
            self.current_mouse_pos.0 as f32,
            self.current_mouse_pos.1 as f32,
            gpu.config.width as f32,
            gpu.config.height as f32,
        )
    }
    /// The window was resized: reconfigure the surface and every
    /// screen-sized target
    fn resize(&mut self, width: u32, height: u32) {
        if let Some(gpu) = &mut self.gpu {
            gpu.resize(width, height);
            self.camera.set_aspect(width as f32, height as f32);
            if let Some(renderer) = &mut self.renderer {
                renderer.resize(&gpu.device, width, height);
            }
            let env_view = self.env_view.as_ref().unwrap();
            let env_sampler = self.env_sampler.as_ref().unwrap();
            if let Some(mc_renderer) = &mut self.mc_renderer {
                mc_renderer.resize(&gpu.device, env_view, env_sampler, width, height);
            }
            // Resize recreated the water front depths — force the
            // per-frame sync to rebind the foam splat gate for the
            // active mode
            self.spray_depth_bound = None;
            if let (Some(ss_renderer), Some(mc_renderer)) =
                (&mut self.ss_renderer, &self.mc_renderer)
            {
                ss_renderer.resize(
                    &gpu.device, env_view, env_sampler,
                    mc_renderer.foam_density_view(),
                    width, height,
                );
            }
            self.rigid_body_depth_view = Some(create_depth_texture(
                &gpu.device, width, height,
            ));
            if let Some(pp_renderer) = &mut self.post_process_renderer {
                pp_renderer.resize(&gpu.device, width, height);
            }
            if let Some(gtao) = &mut self.gtao_renderer {
                gtao.resize(&gpu.device, width, height);
            }
        }
    }

    fn key_pressed(&mut self, key_code: KeyCode) {
        let tilt_speed = 0.05; // Radians per key event
        match key_code {
            KeyCode::Space => {
                self.state.simulation.paused = !self.state.simulation.paused;
            }
            // Arrow keys for tilting
            KeyCode::ArrowLeft => {
                self.state.container.tilt_z_target -= tilt_speed;
            }
            KeyCode::ArrowRight => {
                self.state.container.tilt_z_target += tilt_speed;
            }
            KeyCode::ArrowUp => {
                self.state.container.tilt_x_target -= tilt_speed;
            }
            KeyCode::ArrowDown => {
                self.state.container.tilt_x_target += tilt_speed;
            }
            // Home to reset tilt AND camera
            KeyCode::Home => {
                self.state.container.tilt_x_target = 0.0;
                self.state.container.tilt_z_target = 0.0;
                self.camera.reset();
            }
            // End to flip upside down
            KeyCode::End => {
                self.state.container.tilt_x_target = std::f32::consts::PI;
                self.state.container.tilt_z_target = 0.0;
            }
            // F12: snapshot (frame + exact config, same instant)
            KeyCode::F12 => {
                self.snapshot_requested = true;
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes().with_title("Fluid Toy");
        // --size is in physical pixels so captures have exact dimensions
        let window_attrs = match self.launch.window_size {
            Some((w, h)) => window_attrs.with_inner_size(winit::dpi::PhysicalSize::new(w, h)),
            None => window_attrs.with_inner_size(winit::dpi::LogicalSize::new(1000, 700)),
        };

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
        self.window = Some(window.clone());

        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Let egui handle events first
        if let Some(egui_winit) = &mut self.egui_winit {
            let response = egui_winit.on_window_event(self.window.as_ref().unwrap(), &event);
            if response.consumed {
                // Reset mouse state if egui consumed the event
                if matches!(event, WindowEvent::MouseInput { .. }) {
                    self.mouse_pressed = false;
                    self.last_mouse_pos = None;
                }
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => self.resize(new_size.width, new_size.height),
            WindowEvent::MouseInput { state, button, .. } => {
                match button {
                    MouseButton::Left => {
                        self.mouse_pressed = state == ElementState::Pressed;
                        if !self.mouse_pressed {
                            self.last_mouse_pos = None;
                        }
                    }
                    MouseButton::Right => {
                        self.right_mouse_pressed = state == ElementState::Pressed;
                        if !self.right_mouse_pressed {
                            self.explode_fired = false;
                        }
                    }
                    MouseButton::Middle => {
                        // Spawn particles while middle button held
                        self.middle_mouse_pressed = state == ElementState::Pressed;
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Always track mouse position for force interaction
                self.current_mouse_pos = (position.x, position.y);

                // Orbit camera control on left drag
                if self.mouse_pressed {
                    if let Some((last_x, last_y)) = self.last_mouse_pos {
                        let delta_x = (position.x - last_x) as f32;
                        let delta_y = (position.y - last_y) as f32;
                        self.camera.rotate(delta_x * 0.01, -delta_y * 0.01);
                    }
                    self.last_mouse_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 0.5,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                self.camera.zoom(scroll);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                // Handle keys on press only
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(key_code) = event.physical_key {
                        self.key_pressed(key_code);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.update_and_render();
                if self.should_exit {
                    event_loop.exit();
                    return;
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("RigidBody Fallback Depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
