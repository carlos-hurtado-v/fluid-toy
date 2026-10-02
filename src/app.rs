//! Application state and event handling

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

use wgpu::util::DeviceExt;

use crate::gpu::GpuContext;
use crate::launch::LaunchOptions;
use crate::gui::{self, GuiAction};
use crate::render::{Camera, CausticsRenderer, ContainerRenderer, GpuPoolStyle, GtaoRenderer, MarchingCubesRenderer, ParticleRenderer3D, PostProcessRenderer, RigidBodyDraw, RigidBodyRenderer, ScreenSpaceFluidRenderer, SprayRenderer, WireframeRenderer};
use crate::state::ContainerStyle;
use crate::simulation::{ProbeSystem, SphSimulation3DGrid, SpraySystem, create_particle_block};
use crate::render::environment::{load_embedded_environment_map, SunEstimate};
use crate::render::container_renderer::{POOL_WALL_HEIGHT_FRACTION, WALL_THICKNESS};
use crate::render::mesh_loader::{self, SdfData};
use crate::state::{AppState, BackgroundMode, EnvironmentSun, FluidRenderMode, GroundStaging, ForceMode, GpuMouseForce, GpuShCoefficients, GpuSprayParams, GpuSprayRenderParams, HdrEnvironment, RigidBodyMotion, integrate_rigid_body};
use crate::render::environment::ShCoefficients;

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
    // Environment background rendering (for Particles mode)
    env_bg_pipeline: Option<wgpu::RenderPipeline>,
    env_bg_bind_group: Option<wgpu::BindGroup>,
    env_bg_bind_group_layout: Option<wgpu::BindGroupLayout>,
    env_params_buffer: Option<wgpu::Buffer>,
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
            env_bg_pipeline: None,
            env_bg_bind_group: None,
            env_bg_bind_group_layout: None,
            env_params_buffer: None,
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

    fn create_initial_particles(&self) -> Vec<crate::simulation::SphParticle3D> {
        // Scenario fluid blocks take precedence over the legacy centered cube
        if !self.state.scenario.fluid_blocks.is_empty() {
            let particles = self.create_scenario_particles();
            if !particles.is_empty() {
                return particles;
            }
            eprintln!(
                "[scenario] fluid_blocks produced no particles; falling back to the initial cube"
            );
        }

        // Keep the original lattice spacing (solver tuning depends on this),
        // but place the block low enough to avoid immediate ceiling collisions.
        let spacing = self.state.sph.kernel_radius * 0.6;
        let cube_size = self.state.simulation.initial_cube_size;
        let mut particles = create_particle_block(spacing, cube_size);

        let source_min_y = 0.2;
        let block_height = (cube_size.saturating_sub(1) as f32) * spacing;
        let margin = self.state.rendering.visual_margin();
        let min_y = self.state.container.floor_y + margin + spacing * 0.25;
        let max_y = self.state.container.ceiling_y() - margin - spacing * 0.25;
        let target_min_y = (max_y - block_height).max(min_y);
        let y_shift = target_min_y - source_min_y;

        for p in &mut particles {
            p.position[1] += y_shift;
        }

        particles
    }

    /// Build particles from scenario fluid blocks: world-space boxes filled on
    /// the standard lattice (0.6 × kernel radius — solver tuning depends on
    /// it), clamped into the container interior, capped at max_particles.
    fn create_scenario_particles(&self) -> Vec<crate::simulation::SphParticle3D> {
        use crate::simulation::particle::rand_f32;

        let spacing = self.state.sph.kernel_radius * 0.6;
        let cap = self.state.simulation.max_particles as usize;
        let container = &self.state.container;
        let (forward, inverse) = container.rotation_matrices();
        let center_y = container.floor_y + container.height / 2.0;
        let margin = self.state.rendering.visual_margin() + spacing * 0.25;
        let lim = [
            (container.width / 2.0 - margin).max(0.0),
            (container.height / 2.0 - margin).max(0.0),
            (container.depth / 2.0 - margin).max(0.0),
        ];
        let rot = |m: &[[f32; 4]; 3], v: [f32; 3]| {
            [
                m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
                m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
                m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
            ]
        };

        let mut particles = Vec::new();
        let mut requested = 0usize;
        'blocks: for block in &self.state.scenario.fluid_blocks {
            // Points per axis: lattice-quantized, at least one
            let counts: [u32; 3] = std::array::from_fn(|a| {
                (block.size[a].max(0.0) / spacing + 1e-4) as u32 + 1
            });
            requested += counts.iter().map(|&c| c as usize).product::<usize>();

            for iy in 0..counts[1] {
                for iz in 0..counts[2] {
                    for ix in 0..counts[0] {
                        if particles.len() >= cap {
                            break 'blocks;
                        }
                        let jitter = 0.0005 * rand_f32();
                        let idx = [ix, iy, iz];
                        let mut world = [0.0f32; 3];
                        for a in 0..3 {
                            let span = (counts[a] - 1) as f32 * spacing;
                            world[a] = block.center[a] - span / 2.0
                                + idx[a] as f32 * spacing
                                + jitter;
                        }
                        // Clamp into the (possibly tilted) container interior
                        let rel = [world[0], world[1] - center_y, world[2]];
                        let mut local = rot(&inverse, rel);
                        for a in 0..3 {
                            local[a] = local[a].clamp(-lim[a], lim[a]);
                        }
                        let clamped = rot(&forward, local);
                        let mut p = crate::simulation::SphParticle3D::new(
                            clamped[0],
                            clamped[1] + center_y,
                            clamped[2],
                        );
                        p.velocity = block.velocity;
                        particles.push(p);
                    }
                }
            }
        }
        if requested > particles.len() {
            eprintln!(
                "[scenario] fluid_blocks request {requested} particles; capped at max_particles = {cap}"
            );
        }
        particles
    }

    /// Fire scenario events whose time has come, once each per reset.
    /// Runs on the deterministic sim clock, so automation runs replay exactly.
    fn pump_scenario_events(&mut self) {
        // Events list can change size if an event edits the scenario itself;
        // keep flags aligned (new entries start unfired)
        if self.scenario_fired.len() != self.state.scenario.events.len() {
            self.scenario_fired
                .resize(self.state.scenario.events.len(), false);
        }

        let mut due: Vec<usize> = (0..self.state.scenario.events.len())
            .filter(|&i| {
                !self.scenario_fired[i]
                    && f64::from(self.state.scenario.events[i].time) <= self.sim_time
            })
            .collect();
        if due.is_empty() {
            return;
        }
        // Same-frame events fire in time order
        due.sort_by(|&a, &b| {
            self.state.scenario.events[a]
                .time
                .partial_cmp(&self.state.scenario.events[b].time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for i in due {
            self.scenario_fired[i] = true;
            let event = self.state.scenario.events[i].clone();
            let Some((path, raw)) = event.set.split_once('=') else {
                eprintln!(
                    "[scenario] t={:.3}: bad event '{}' (expected path=value)",
                    event.time, event.set
                );
                continue;
            };
            match self.apply_state_set(path, raw) {
                Ok(()) => {
                    self.state.runtime.scenario_events_fired += 1;
                    println!(
                        "[scenario] t={:.3}s (frame {}): set {}",
                        event.time, self.sim_frame_index, event.set
                    );
                }
                Err(e) => eprintln!("[scenario] t={:.3}: {e}", event.time),
            }
        }
    }

    /// Apply a `--set`-style `path=value` to the live state (the scenario
    /// event mechanism). Round-trips through the JSON config form, so it
    /// accepts exactly what the CLI accepts; serde-skipped runtime fields are
    /// carried over from the live state afterwards.
    fn apply_state_set(&mut self, path: &str, raw: &str) -> Result<(), String> {
        let mut tree = serde_json::to_value(&self.state)
            .map_err(|e| format!("state to json failed: {e}"))?;
        crate::launch::apply_set(&mut tree, path, raw)?;
        let mut new_state: AppState = serde_json::from_value(tree)
            .map_err(|e| format!("'{path}={raw}' produces an invalid config: {e}"))?;

        new_state.runtime = self.state.runtime.clone();
        for (nb, ob) in new_state
            .rigid_bodies
            .iter_mut()
            .zip(&self.state.rigid_bodies)
        {
            nb.spin_angle = ob.spin_angle;
        }
        self.state = new_state;
        Ok(())
    }

    fn initialize(&mut self, window: Arc<Window>) {
        // Automation runs render uncapped (no vsync wait between frames)
        let gpu = pollster::block_on(GpuContext::new(window.clone(), self.launch.is_automated()));

        // Initialize camera from state
        self.camera.distance = self.state.camera.distance;
        self.camera.yaw = self.state.camera.yaw;
        self.camera.pitch = self.state.camera.pitch;
        self.camera.target = self.state.camera.target;
        self.camera.fov = self.state.camera.fov;
        self.camera.set_aspect(gpu.config.width as f32, gpu.config.height as f32);

        // Create renderer with state-driven params
        let camera_params = self.camera.to_gpu_params();
        let render_params = self.state.rendering.to_gpu_params();
        let renderer = ParticleRenderer3D::new(
            &gpu.device,
            crate::render::HDR_FORMAT,
            &camera_params,
            &render_params,
            gpu.config.width,
            gpu.config.height,
        );

        // Create initial 3D SPH particles (dam break style - half the box)
        // Spacing = 0.6 * h (slightly looser than reference to reduce initial pressure)
        let particles = self.create_initial_particles();
        self.state.runtime.particle_count = particles.len() as u32;

        // Load duck mesh SDF for custom rigid body collision
        let sdf_data = match mesh_loader::load_embedded_duck() {
            Ok(loaded_mesh) => {
                let sdf = loaded_mesh.sdf;
                if sdf.is_some() {
                    log::info!("Duck SDF loaded for rigid body collision");
                }
                sdf
            }
            Err(e) => {
                log::error!("Failed to load duck.glb for SDF: {}", e);
                None
            }
        };

        // Create 3D SPH simulation (O(n²) version - works correctly)
        let sph_params = self.state.sph.to_gpu_params_3d(
            self.state.runtime.particle_count,
            self.simulation_substep_dt(),
        );
        let container_geom = self.state.container.to_gpu_geometry(
                self.state.sph.wall_stiffness,
                self.state.simulation.damping,
                self.state.container.style == ContainerStyle::OpaquePool,
                0.0,
            );
        let sph_simulation = SphSimulation3DGrid::new(
            &gpu.device,
            &gpu.queue,
            &particles,
            sph_params,
            container_geom,
            self.state.simulation.max_particles,
            sdf_data.as_ref(),
        );

        let probe_system = ProbeSystem::new(
            &gpu.device,
            sph_simulation.particle_buffer(),
            &self.state.scenario.probes,
        );

        // Load environment map (shared by SS + MC renderers)
        let (env_texture, env_view, env_sampler, sh_coefficients, hdr_sun) = load_embedded_environment_map(
            &gpu.device,
            &gpu.queue,
            self.state.environment.hdr_selection,
        ).expect("Failed to load environment map");

        // Surface foam map (its textures are bound by the MC water shader)
        let foam_map = crate::simulation::FoamMap::new(&gpu.device);

        // Create marching cubes renderer (shares environment map)
        let mut mc_renderer = MarchingCubesRenderer::new(
            &gpu.device,
            crate::render::HDR_FORMAT,
            &env_view,
            &env_sampler,
            gpu.config.width,
            gpu.config.height,
            self.state.quality.msaa.as_u32(),
            self.state.rendering.mc_grid_resolution.grid_size(),
            &foam_map,
        );
        if !self.launch.probe_pixels.is_empty() {
            mc_renderer.enable_probe(&gpu.device, &env_view, &env_sampler, &self.launch.probe_pixels);
            println!(
                "probe: recording {} pixel(s) on capture frames",
                self.launch.probe_pixels.len()
            );
        }

        // Create wireframe renderer for container visualization
        let wireframe_renderer = WireframeRenderer::new(
            &gpu.device,
            gpu.config.format,
            &camera_params,
            &container_geom,
        );

        // Create rigid body renderer + fallback depth texture (before caustics:
        // the splat pass binds the body array for photon occlusion)
        let rigid_body_renderer = RigidBodyRenderer::new(
            &gpu.device,
            &gpu.queue,
            crate::render::HDR_FORMAT,
            &camera_params,
            self.state.quality.msaa.as_u32(),
        );
        let rigid_body_depth_view = create_depth_texture(
            &gpu.device, gpu.config.width, gpu.config.height,
        );

        // Create caustics renderer (raster source = MC mesh; output sampled by container)
        let caustics_renderer = CausticsRenderer::new(
            &gpu.device,
            mc_renderer.mesh_vertex_buffer(),
            rigid_body_renderer.bodies_buffer(),
            &container_geom,
            &self.state.caustics,
        );

        // Create opaque pool container renderer
        let pool_style = GpuPoolStyle::from_config(
            &self.state.container,
            &self.state.lighting,
            self.state.environment.environment_intensity,
            0.0,
            0.0,
            1.0,
        );
        let gpu_sh = GpuShCoefficients { coeffs: sh_coefficients.coeffs };
        let container_renderer = ContainerRenderer::new(
            &gpu.device,
            crate::render::HDR_FORMAT,
            &camera_params,
            &container_geom,
            &pool_style,
            &self.state.container,
            self.state.quality.msaa.as_u32(),
            self.state.sph.kernel_radius,
            &gpu_sh,
            caustics_renderer.display_view(),
            caustics_renderer.sampler(),
        );

        // Create whitewater system and renderer
        let spray_params = self.build_spray_params(0);
        let spray_system = SpraySystem::new(
            &gpu.device,
            sph_simulation.sorted_particle_buffer(),
            sph_simulation.sph_params_buffer(),
            sph_simulation.container_geom_buffer(),
            sph_simulation.cell_starts_buffer(),
            sph_simulation.cell_counts_buffer(),
            sph_simulation.grid_params_buffer(),
            self.state.spray.max_particles,
            &spray_params,
        );
        let spray_renderer = SprayRenderer::new(
            &gpu.device,
            crate::render::HDR_FORMAT,
            &camera_params,
            spray_system.spray_buffer(),
            &self.build_spray_render_params(),
            self.state.quality.msaa.as_u32(),
        );

        // Create post-process renderer
        let post_process_params = self.state.post_process.to_gpu_params();
        let post_process_renderer = PostProcessRenderer::new(
            &gpu.device,
            &gpu.queue,
            gpu.config.format,
            gpu.config.width,
            gpu.config.height,
            &post_process_params,
        );

        // Create GTAO renderer
        let gtao_renderer = GtaoRenderer::new(
            &gpu.device,
            gpu.config.width,
            gpu.config.height,
        );

        // Create environment background pipeline (for Particles mode HDR background)
        let env_bg_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Env Background Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mc_environment.wgsl").into()),
        });

        let env_params_gpu = self.state.environment.to_gpu_params(&self.ground_staging());
        let env_params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Env Params Buffer"),
            contents: bytemuck::bytes_of(&env_params_gpu),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let env_bg_bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Env BG BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let env_bg_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Env BG BG"),
            layout: &env_bg_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: renderer.camera_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&env_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&env_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: env_params_buffer.as_entire_binding(),
                },
            ],
        });

        let env_bg_pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Env BG Pipeline Layout"),
            bind_group_layouts: &[&env_bg_bind_group_layout],
            push_constant_ranges: &[],
        });

        let env_bg_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Env BG Pipeline"),
            layout: Some(&env_bg_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &env_bg_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &env_bg_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::render::HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Setup egui
        let egui_winit = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let egui_renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            gpu.config.format,
            egui_wgpu::RendererOptions::default(),
        );

        // Upload SH coefficients to MC renderer before moving locals
        mc_renderer.update_sh_coefficients(&gpu.queue, &gpu_sh);

        // Create screen-space fluid renderer
        let ss_renderer = ScreenSpaceFluidRenderer::new(
            &gpu.device,
            crate::render::HDR_FORMAT,
            &env_view,
            &env_sampler,
            &camera_params,
            &self.state.lighting.to_gpu_params(),
            &crate::render::marching_cubes::GpuWaterParams::default(),
            &gpu_sh,
            mc_renderer.foam_density_view(),
            gpu.config.width,
            gpu.config.height,
        );

        // Wire the MC front depth into the foam splat pass (depth-aware foam);
        // the per-frame sync rebinds to the SS depth when that mode is active
        {
            let mut spray_renderer = spray_renderer;
            spray_renderer.set_depth_view(
                &gpu.device,
                spray_system.spray_buffer(),
                mc_renderer.front_depth_view(),
            );
            self.spray_renderer = Some(spray_renderer);
            self.spray_depth_bound = Some(FluidRenderMode::MarchingCubes);
        }

        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.mc_renderer = Some(mc_renderer);
        self.ss_renderer = Some(ss_renderer);
        self.wireframe_renderer = Some(wireframe_renderer);
        self.container_renderer = Some(container_renderer);
        self.caustics_renderer = Some(caustics_renderer);
        self.rigid_body_renderer = Some(rigid_body_renderer);
        self.rigid_body_depth_view = Some(rigid_body_depth_view);
        self.spray_system = Some(spray_system);
        self.post_process_renderer = Some(post_process_renderer);
        self.gtao_renderer = Some(gtao_renderer);
        self.prev_camera_params = Some(camera_params);
        self.env_texture = Some(env_texture);
        self.env_view = Some(env_view);
        self.env_sampler = Some(env_sampler);
        self.sh_coefficients = Some(sh_coefficients);
        self.hdr_sun = hdr_sun;
        self.foam_map = Some(foam_map);
        self.current_hdr = self.state.environment.hdr_selection;
        self.env_bg_pipeline = Some(env_bg_pipeline);
        self.env_bg_bind_group = Some(env_bg_bind_group);
        self.env_bg_bind_group_layout = Some(env_bg_bind_group_layout);
        self.env_params_buffer = Some(env_params_buffer);
        self.sph_simulation = Some(sph_simulation);
        self.probe_system = Some(probe_system);
        self.sdf_data = sdf_data;
        self.egui_winit = Some(egui_winit);
        self.egui_renderer = Some(egui_renderer);

        if let Some(path) = self.launch.load_state.clone() {
            if let Err(e) = self.load_state(&path) {
                eprintln!("error: --load-state {}: {e}", path.display());
                std::process::exit(1);
            }
        }
    }

    fn reset_simulation(&mut self) {
        // Replay the scenario from t=0: rewind the deterministic sim clock
        // and re-arm every timed event
        self.scenario_fired = vec![false; self.state.scenario.events.len()];
        self.state.runtime.scenario_events_fired = 0;
        self.state.runtime.measurements = None;
        self.sim_frame_index = 0;
        self.sim_time = 0.0;
        self.milestone_frame = 0;

        // Reset dynamic rigid body velocity and rotation (keep position);
        // Static/Kinematic bodies re-derive their pose from euler/spin anyway
        for body in &mut self.state.rigid_bodies {
            body.velocity = [0.0; 3];
            body.angular_velocity = [0.0; 3];
            if body.motion == RigidBodyMotion::Dynamic {
                body.orientation = [0.0, 0.0, 0.0, 1.0];
            }
            body.spin_angle = 0.0;
        }

        if let Some(gpu) = &self.gpu {
            let particles = self.create_initial_particles();
            self.state.runtime.particle_count = particles.len() as u32;

            let sph_params = self.state.sph.to_gpu_params_3d(
                self.state.runtime.particle_count,
                self.simulation_substep_dt(),
            );
            let container_geom = self.state.container.to_gpu_geometry(
                self.state.sph.wall_stiffness,
                self.state.simulation.damping,
                self.state.container.style == ContainerStyle::OpaquePool,
                0.0,
            );
            self.sph_simulation = Some(SphSimulation3DGrid::new(
                &gpu.device,
                &gpu.queue,
                &particles,
                sph_params,
                container_geom,
                self.state.simulation.max_particles,
                self.sdf_data.as_ref(),
            ));

            let camera_params = self.camera.to_gpu_params();
            let render_params = self.state.rendering.to_gpu_params();
            self.renderer = Some(ParticleRenderer3D::new(
                &gpu.device,
                crate::render::HDR_FORMAT,
                &camera_params,
                &render_params,
                gpu.config.width,
                gpu.config.height,
            ));

            // Recreate whitewater system with new simulation's buffers
            if let Some(sph_sim) = &self.sph_simulation {
                let spray_params = self.build_spray_params(0);
                let spray_system = SpraySystem::new(
                    &gpu.device,
                    sph_sim.sorted_particle_buffer(),
                    sph_sim.sph_params_buffer(),
                    sph_sim.container_geom_buffer(),
                    sph_sim.cell_starts_buffer(),
                    sph_sim.cell_counts_buffer(),
                    sph_sim.grid_params_buffer(),
                    self.state.spray.max_particles,
                    &spray_params,
                );
                let camera_params = self.camera.to_gpu_params();
                let mut spray_renderer = SprayRenderer::new(
                    &gpu.device,
                    crate::render::HDR_FORMAT,
                    &camera_params,
                    spray_system.spray_buffer(),
                    &self.build_spray_render_params(),
                    self.state.quality.msaa.as_u32(),
                );
                if let Some(mc_renderer) = &self.mc_renderer {
                    spray_renderer.set_depth_view(
                        &gpu.device,
                        spray_system.spray_buffer(),
                        mc_renderer.front_depth_view(),
                    );
                    // New spray buffer: let the per-frame sync rebind for the
                    // active mode (SS uses its own front depth)
                    self.spray_depth_bound = Some(FluidRenderMode::MarchingCubes);
                }
                self.spray_renderer = Some(spray_renderer);
                self.spray_system = Some(spray_system);
            }

            // Fresh simulation: no foam on its surface yet
            if let Some(foam_map) = self.foam_map.as_mut() {
                foam_map.request_reset();
            }

            // Probes bind the new simulation's particle buffer
            if let Some(sph_sim) = &self.sph_simulation {
                self.probe_system = Some(ProbeSystem::new(
                    &gpu.device,
                    sph_sim.particle_buffer(),
                    &self.state.scenario.probes,
                ));
            }
            self.state.runtime.frame_count = 0;
        }
    }

    /// Caustics run only for the MC water surface, in the opaque pool
    /// (the floor is the receiver), lit by the sun.
    fn caustics_active(&self) -> bool {
        self.state.caustics.enabled
            && self.state.rendering.render_mode == FluidRenderMode::MarchingCubes
            && self.state.container.style == ContainerStyle::OpaquePool
            && self.state.lighting.sun_enabled
    }

    fn reset_defaults(&mut self) {
        self.state.simulation.reset_defaults();
        self.state.sph.reset_defaults();
        self.state.rendering.reset_defaults();
        self.state.camera.reset_defaults();
        self.state.lighting.reset_defaults();
        self.state.caustics.reset_defaults();
        self.state.container.reset_defaults();
        self.state.rigid_bodies.clear();
        self.state.spray.reset_defaults();
        self.state.environment.reset_defaults();
        // Reset camera to defaults
        self.camera.distance = self.state.camera.distance;
        self.camera.yaw = self.state.camera.yaw;
        self.camera.pitch = self.state.camera.pitch;
        self.camera.target = self.state.camera.target;
        self.camera.fov = self.state.camera.fov;
        self.reset_simulation();
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
            WindowEvent::Resized(new_size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(new_size.width, new_size.height);
                    self.camera.set_aspect(new_size.width as f32, new_size.height as f32);
                    if let Some(renderer) = &mut self.renderer {
                        renderer.resize(&gpu.device, new_size.width, new_size.height);
                    }
                    let env_view = self.env_view.as_ref().unwrap();
                    let env_sampler = self.env_sampler.as_ref().unwrap();
                    if let Some(mc_renderer) = &mut self.mc_renderer {
                        mc_renderer.resize(&gpu.device, env_view, env_sampler, new_size.width, new_size.height);
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
                            new_size.width, new_size.height,
                        );
                    }
                    self.rigid_body_depth_view = Some(create_depth_texture(
                        &gpu.device, new_size.width, new_size.height,
                    ));
                    if let Some(pp_renderer) = &mut self.post_process_renderer {
                        pp_renderer.resize(&gpu.device, new_size.width, new_size.height);
                    }
                    if let Some(gtao) = &mut self.gtao_renderer {
                        gtao.resize(&gpu.device, new_size.width, new_size.height);
                    }
                }
            }
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

impl App {
    fn cursor_ray(&self) -> ([f32; 3], [f32; 3]) {
        let gpu = self.gpu.as_ref().unwrap();
        self.camera.screen_to_ray(
            self.current_mouse_pos.0 as f32,
            self.current_mouse_pos.1 as f32,
            gpu.config.width as f32,
            gpu.config.height as f32,
        )
    }

    /// Push SH irradiance to all consumers when the ambient source changes.
    /// Environment mode uses the HDR map's coefficients; SolidColor mode uses
    /// a uniform SH of the background color, so diffuse ambient agrees with
    /// the visible surround instead of silently keeping the HDR sky's light.
    fn refresh_sh_coefficients(&mut self) {
        let env = &self.state.environment;
        // With the HDR sun split out (lit analytically), ambient is the sky only
        let sky_only = self.state.lighting.sun_from_environment && self.hdr_sun.is_some();
        let key = (
            env.background_mode,
            [
                env.background_color[0].to_bits(),
                env.background_color[1].to_bits(),
                env.background_color[2].to_bits(),
            ],
            self.current_hdr,
            sky_only,
        );
        if self.last_sh_key == Some(key) {
            return;
        }
        let coeffs = match env.background_mode {
            BackgroundMode::Environment => match (&self.sh_coefficients, &self.hdr_sun) {
                (_, Some(sun)) if sky_only => sun.sky_sh.coeffs,
                (Some(sh), _) => sh.coeffs,
                (None, _) => return,
            },
            BackgroundMode::SolidColor => ShCoefficients::uniform(env.background_color).coeffs,
        };
        let gpu_sh = GpuShCoefficients { coeffs };
        let gpu = self.gpu.as_ref().unwrap();
        if let Some(mc_renderer) = &self.mc_renderer {
            mc_renderer.update_sh_coefficients(&gpu.queue, &gpu_sh);
        }
        if let Some(ss_renderer) = &self.ss_renderer {
            ss_renderer.update_sh_coefficients(&gpu.queue, &gpu_sh);
        }
        if let Some(container_r) = &self.container_renderer {
            container_r.update_sh_coefficients(&gpu.queue, &gpu_sh);
        }
        if let Some(rb_renderer) = &self.rigid_body_renderer {
            rb_renderer.update_sh_coefficients(&gpu.queue, &gpu_sh);
        }
        if let Some(spray_renderer) = &self.spray_renderer {
            spray_renderer.update_sh_coefficients(&gpu.queue, &gpu_sh);
        }
        self.last_sh_key = Some(key);
    }

    /// Where the projected ground sits and what stands on it (the pool box),
    /// plus the sky's share of the ground's light for the box's contact occlusion
    fn ground_staging(&self) -> GroundStaging {
        let c = &self.state.container;
        let (aabb_min, _) = c.tilted_aabb();
        let pool = c.style == ContainerStyle::OpaquePool;
        let t = if pool { WALL_THICKNESS } else { 0.0 };
        GroundStaging {
            // A hair below the base: coplanar with the pool's outer bottom it
            // would z-fight. The tank's water rests on it (glass is thin).
            ground_y: aabb_min[1] - t - 0.003,
            occluder_half_extents: [c.half_width() + t, c.half_depth() + t],
            occluder_height: if pool { c.height * POOL_WALL_HEIGHT_FRACTION + t } else { 0.0 },
            sky_share: self.ground_sky_share(),
        }
    }

    /// Sky vs sun share of the light on open ground, from the HDR's sky-only
    /// SH and measured sun (all sky when the map has no distinct sun)
    fn ground_sky_share(&self) -> f32 {
        let Some(sun) = &self.hdr_sun else { return 1.0 };
        let c = &sun.sky_sh.coeffs;
        // SH irradiance on an upward-facing surface (n = +Y), per channel
        let e_up = |k: usize| 0.282095 * c[0][k] + 0.488603 * c[1][k] - 0.315392 * c[6][k] - 0.546274 * c[8][k];
        let luminance = |v: [f32; 3]| 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
        let sky = luminance([e_up(0), e_up(1), e_up(2)]).max(0.0);
        let sun_e = luminance(sun.irradiance) * sun.direction[1].max(0.0);
        if sky + sun_e > 0.0 { sky / (sky + sun_e) } else { 1.0 }
    }

    fn reload_environment_map(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let selection = self.state.environment.hdr_selection;

        let (env_texture, env_view, env_sampler, sh_coefficients, hdr_sun) = load_embedded_environment_map(
            &gpu.device,
            &gpu.queue,
            selection,
        ).expect("Failed to load environment map");
        self.hdr_sun = hdr_sun;

        // Rebuild MC renderer bind groups for the new env texture
        if let Some(mc_renderer) = &mut self.mc_renderer {
            mc_renderer.rebuild_env_bind_groups(&gpu.device, &env_view, &env_sampler);
        }

        // Rebuild SS renderer bind groups for the new env texture
        if let (Some(ss_renderer), Some(mc_renderer)) =
            (&mut self.ss_renderer, &self.mc_renderer)
        {
            ss_renderer.rebuild_env_bind_groups(
                &gpu.device, &env_view, &env_sampler,
                mc_renderer.foam_density_view(),
            );
        }

        self.sh_coefficients = Some(sh_coefficients);
        // SH consumers are refreshed next sync (background-mode aware)
        self.last_sh_key = None;

        // Rebuild env background bind group (for Particles mode)
        if let (Some(layout), Some(renderer), Some(buf)) = (
            &self.env_bg_bind_group_layout,
            &self.renderer,
            &self.env_params_buffer,
        ) {
            self.env_bg_bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Env BG BG"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: renderer.camera_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&env_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&env_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buf.as_entire_binding(),
                    },
                ],
            }));
        }

        self.env_texture = Some(env_texture);
        self.env_view = Some(env_view);
        self.env_sampler = Some(env_sampler);
        self.current_hdr = selection;

        log::info!("Switched environment map to {:?}", selection);
    }

    fn sync_gpu_state(&mut self) {
        self.refresh_sh_coefficients();
        // The analytic sun follows the visible HDR sun (when it has one)
        let env_intensity = self.state.environment.environment_intensity;
        self.state.lighting.environment_sun =
            if self.state.environment.background_mode == BackgroundMode::Environment {
                self.hdr_sun.map(|sun| EnvironmentSun {
                    direction: sun.direction,
                    irradiance: sun.irradiance.map(|e| e * env_intensity),
                })
            } else {
                None
            };

        // Compute mouse force before borrowing sph_sim mutably
        let mouse_force = if self.right_mouse_pressed {
            let (ray_origin, ray_dir) = self.cursor_ray();
            let hit = self.camera.ray_plane_intersection(ray_origin, ray_dir, -0.6)
                .or_else(|| self.camera.ray_plane_intersection(ray_origin, ray_dir, 0.0))
                .unwrap_or([0.0, 0.0, 0.0]);

            let cfg = &self.state.mouse_force;
            let mode = cfg.mode;

            // Explode mode: one-shot — only active on first frame of click
            let is_active = if mode == ForceMode::Explode {
                if self.explode_fired {
                    0
                } else {
                    self.explode_fired = true;
                    1
                }
            } else {
                1
            };

            GpuMouseForce {
                position: hit,
                radius: cfg.radius,
                strength: cfg.strength,
                is_active,
                mode: mode as u32,
                _pad: 0.0,
                direction: ray_dir,
                _pad2: 0.0,
            }
        } else {
            GpuMouseForce::default()
        };

        let gpu = self.gpu.as_ref().unwrap();
        let substep_dt = self.simulation_substep_dt();
        let caustics_on = self.caustics_active();
        if let (Some(sph_sim), Some(renderer)) = (&mut self.sph_simulation, &self.renderer) {
            let sph_params = self.state.sph.to_gpu_params_3d(
                self.state.runtime.particle_count,
                substep_dt,
            );
            sph_sim.update_sph_params(&gpu.queue, &sph_params);
            sph_sim.set_pcisph_iterations(self.state.simulation.pcisph_iterations);

            let is_pool = self.state.container.style == ContainerStyle::OpaquePool;
            let container_geom = self.state.container.to_gpu_geometry(
                self.state.sph.wall_stiffness,
                self.state.simulation.damping,
                is_pool,
                0.0, // clip_margin updated by MC renderer when needed
            );
            sph_sim.update_container_geometry(&gpu.queue, &container_geom);

            // Update wireframe container visualization
            if let Some(wireframe) = &self.wireframe_renderer {
                wireframe.update_container_geometry(&gpu.queue, &container_geom);
            }

            // Update opaque pool container renderer
            if let Some(container_r) = &mut self.container_renderer {
                let (caustic_strength, shadow_strength) = if caustics_on {
                    (self.state.caustics.intensity, self.state.caustics.shadow_strength)
                } else {
                    (0.0, 0.0)
                };
                let pool_style = GpuPoolStyle::from_config(
                    &self.state.container,
                    &self.state.lighting,
                    self.state.environment.environment_intensity,
                    caustic_strength,
                    shadow_strength,
                    self.state.caustics.focus.max(0.1),
                );
                container_r.update_container_geometry(&gpu.queue, &container_geom);
                container_r.update_pool_style(&gpu.queue, &pool_style);
                container_r.update_camera(&gpu.queue, &self.camera.to_gpu_params());
                container_r.maybe_rebuild_mesh(&gpu.device, &self.state.container, self.state.sph.kernel_radius);
            }
            // Drop caustic temporal history while inactive so re-enabling
            // starts from the current frame instead of stale data
            if !caustics_on {
                if let Some(caustics) = &mut self.caustics_renderer {
                    caustics.invalidate_history();
                }
            }

            // Update gravity (based on tilt)
            let gravity = self.state.simulation.to_gpu_gravity();
            sph_sim.update_gravity(&gpu.queue, &gravity);

            sph_sim.update_mouse_force(&gpu.queue, &mouse_force);

            // Update rigid bodies (re-derive Static/Kinematic poses from the
            // GUI euler/spin state so slider edits apply immediately)
            for body in &mut self.state.rigid_bodies {
                body.refresh_pose();
            }
            let gpu_bodies: Vec<_> = self
                .state
                .rigid_bodies
                .iter()
                .map(|b| b.to_gpu_rigid_body(self.state.sph.wall_stiffness))
                .collect();
            sph_sim.update_rigid_bodies(&gpu.queue, &gpu_bodies);

            // Update camera
            let camera_params = self.camera.to_gpu_params();
            renderer.update_camera(&gpu.queue, &camera_params);
            if let Some(wireframe) = &self.wireframe_renderer {
                wireframe.update_camera(&gpu.queue, &camera_params);
            }
            if let Some(rb_renderer) = &mut self.rigid_body_renderer {
                rb_renderer.update_camera(&gpu.queue, &camera_params);
                let lighting = &self.state.lighting;
                rb_renderer.update_light(&gpu.queue, &crate::render::rigid_body_renderer::GpuRbLightParams {
                    sun_dir: lighting.sun_direction_normalized(),
                    ibl_strength: self.state.environment.environment_intensity,
                    sun_rgb: lighting.sun_rgb(),
                    _pad0: 0.0,
                });
                rb_renderer.update_container_geometry(&gpu.queue, &container_geom);
                let light_dir = lighting.sun_direction_normalized();
                let mut renders = Vec::new();
                let mut draws = Vec::new();
                for body in self.state.rigid_bodies.iter().filter(|b| b.enabled) {
                    renders.push(body.to_gpu_render(light_dir));
                    draws.push(RigidBodyDraw {
                        shape: body.shape,
                        vertex_count: body.render_vertex_count(),
                    });
                }
                rb_renderer.update_bodies(&gpu.queue, &renders, &draws);
                if let Some(mc_renderer) = &self.mc_renderer {
                    mc_renderer.update_bodies(&gpu.queue, &renders);
                }
            }
            if let Some(spray_renderer) = &self.spray_renderer {
                spray_renderer.update_camera(&gpu.queue, &camera_params);
                spray_renderer.update_params(&gpu.queue, &self.build_spray_render_params());
                spray_renderer.update_light(
                    &gpu.queue,
                    &self.state.lighting.to_gpu_params_with_ambient(
                        self.state.environment.environment_intensity,
                    ),
                );
                spray_renderer.update_container_geometry(&gpu.queue, &container_geom);
            }

            let render_params = self.state.rendering.to_gpu_params();
            renderer.update_params(&gpu.queue, &render_params);
            renderer.update_light_params(&gpu.queue, &self.state.lighting.to_gpu_params());
        }
    }

    fn build_spray_params(&self, frame_count: u32) -> GpuSprayParams {
        let spray = &self.state.spray;
        // Auto-calibrated potential ceilings (EMA over per-frame maxima)
        let (ta_limit, wc_limit) = self.spray_system.as_ref().map_or(
            (
                crate::simulation::spray::AUTO_TA_INIT,
                crate::simulation::spray::AUTO_WC_INIT,
            ),
            |s| s.auto_limits(),
        );
        GpuSprayParams {
            min_speed: spray.min_speed,
            emission_rate: spray.emission_rate,
            lifetime: spray.lifetime,
            lifetime_variation: spray.lifetime_variation,
            drag: spray.drag,
            speed_multiplier: spray.speed_multiplier,
            velocity_jitter: spray.velocity_jitter,
            dt: self.state.simulation.substep_dt() * self.state.simulation.substeps as f32,
            max_particles: spray.max_particles,
            num_sph_particles: self.state.runtime.particle_count,
            frame_count,
            gravity_y: -self.state.simulation.gravity,
            k_trapped_air: spray.k_trapped_air,
            k_wave_crest: spray.k_wave_crest,
            ta_limit,
            bubble_buoyancy: spray.bubble_buoyancy,
            bubble_drag: spray.bubble_drag,
            wc_limit,
            _pad: [0.0; 2],
        }
    }

    fn build_spray_render_params(&self) -> GpuSprayRenderParams {
        // In MC and SS modes foam renders as a screen-space density field;
        // the sprite pass then draws only spray streaks and bubbles
        let foam_as_field = matches!(
            self.state.rendering.render_mode,
            FluidRenderMode::MarchingCubes | FluidRenderMode::ScreenSpace
        );
        GpuSprayRenderParams {
            particle_size: self.state.spray.particle_size,
            max_particles: self.state.spray.max_particles,
            bubbles_visible: self.state.spray.bubbles_visible as u32,
            foam_as_field: foam_as_field as u32,
        }
    }

    fn spawn_particles(&mut self) {
        if !self.middle_mouse_pressed {
            return;
        }
        let (ray_origin, ray_dir) = self.cursor_ray();
        let spawn_pos = self.camera.ray_plane_intersection(ray_origin, ray_dir, -0.5)
            .or_else(|| self.camera.ray_plane_intersection(ray_origin, ray_dir, 0.0))
            .unwrap_or([0.0, 0.0, 0.0]);

        let gpu = self.gpu.as_ref().unwrap();
        let substep_dt = self.simulation_substep_dt();
        if let Some(sph_sim) = &mut self.sph_simulation {

            let spawned = sph_sim.spawn_particles(&gpu.queue, spawn_pos, 10, 0.08);
            self.state.runtime.particle_count = sph_sim.num_particles();

            if spawned > 0 {
                let sph_params = self.state.sph.to_gpu_params_3d(
                    self.state.runtime.particle_count,
                    substep_dt,
                );
                sph_sim.update_sph_params(&gpu.queue, &sph_params);
            }
        }
    }

    fn update_and_render(&mut self) {
        // Early return if not initialized
        if self.gpu.is_none() || self.sph_simulation.is_none() || self.renderer.is_none() {
            return;
        }

        // Calculate FPS
        let now = Instant::now();
        let delta = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        // Smooth FPS with exponential moving average
        if delta > 0.0 {
            let instant_fps = 1.0 / delta;
            self.state.runtime.fps = self.state.runtime.fps * 0.9 + instant_fps * 0.1;
        }

        // Visual time (ripples etc.): wall clock normally; fixed sim step in
        // automation runs so captures are reproducible across machines
        let frame_dt = self.state.simulation.substep_dt() * self.state.simulation.substeps as f32;
        // --hold freezes everything that moves, ripples included (until the
        // GUI unpauses)
        let held = self.launch.hold && self.state.simulation.paused;
        if self.launch.is_automated() {
            if !self.state.simulation.paused {
                self.state.runtime.time_elapsed += frame_dt;
            }
        } else if !held {
            self.state.runtime.time_elapsed += delta;
        }

        let window = self.window.as_ref().unwrap();
        let egui_winit = self.egui_winit.as_mut().unwrap();

        // Run egui
        let raw_input = egui_winit.take_egui_input(window);
        let mut gui_action = GuiAction::None;
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            gui_action = gui::render_control_panel(ctx, &mut self.state);
        });

        egui_winit.handle_platform_output(window, full_output.platform_output);

        let tris = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        // Now get the GPU resources
        let gpu = self.gpu.as_ref().unwrap();
        let egui_renderer = self.egui_renderer.as_mut().unwrap();

        // Update egui textures
        for (id, image_delta) in &full_output.textures_delta.set {
            egui_renderer.update_texture(&gpu.device, &gpu.queue, *id, image_delta);
        }

        // Fire due scenario events before this frame's state reaches the GPU
        self.pump_scenario_events();

        // Sync state to GPU
        self.sync_gpu_state();

        // Detect HDR environment switch
        if self.state.environment.hdr_selection != self.current_hdr {
            self.reload_environment_map();
        }

        // Handle particle spawning (middle mouse held = continuous stream)
        self.spawn_particles();

        // Get current frame texture
        let gpu = self.gpu.as_ref().unwrap();
        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Main Encoder"),
            });

        // Smoothly interpolate container tilt toward target each frame (total frame time)
        if !held {
            self.state.container.update_tilt(frame_dt);
        }

        // Run SPH simulation if not paused (multiple sub-steps for stability)
        // Note: Grid simulation manages its own command encoding/submission
        let num_substeps = self.state.simulation.substeps;
        let substep_dt = self.simulation_substep_dt();
        if !self.state.simulation.paused {
            if let Some(sph_sim) = &mut self.sph_simulation {
                // Only Dynamic bodies consume the reaction accumulators;
                // Static/Kinematic skip the clear AND the blocking readback
                // (saves a hard GPU sync per frame).
                let any_dynamic = self
                    .state
                    .rigid_bodies
                    .iter()
                    .any(|b| b.enabled && b.motion == RigidBodyMotion::Dynamic);
                let any_spinning = self
                    .state
                    .rigid_bodies
                    .iter()
                    .any(|b| b.enabled && b.motion == RigidBodyMotion::Kinematic && b.spin_rpm != 0.0);
                // Clear accumulators once, then accumulate over all sub-steps
                if any_dynamic {
                    sph_sim.clear_rigid_body_accum(&gpu.queue);
                }
                for _ in 0..num_substeps {
                    // Advance kinematic spin per substep so fast bodies sweep
                    // smoothly instead of jumping once per frame (each step()
                    // submits its own command buffer, so the write lands
                    // between substeps)
                    if any_spinning {
                        for body in &mut self.state.rigid_bodies {
                            body.advance_kinematic(substep_dt);
                        }
                        let gpu_bodies: Vec<_> = self
                            .state
                            .rigid_bodies
                            .iter()
                            .map(|b| b.to_gpu_rigid_body(self.state.sph.wall_stiffness))
                            .collect();
                        sph_sim.update_rigid_bodies(&gpu.queue, &gpu_bodies);
                    }
                    sph_sim.step(&gpu.device, &gpu.queue);
                }
                if any_dynamic {
                    sph_sim.read_rigid_body_accum(&gpu.device);
                }
            }

            // Run spray system after SPH completes
            if self.state.spray.enabled && self.spray_system.is_some() {
                let needs_reset = !self.spray_prev_enabled;
                self.state.runtime.frame_count = self.state.runtime.frame_count.wrapping_add(1);
                let spray_params = self.build_spray_params(self.state.runtime.frame_count);

                if let Some(spray_sys) = self.spray_system.as_mut() {
                    // Reset spray on re-enable to clear stale frozen particles
                    if needs_reset {
                        spray_sys.reset(&gpu.queue);
                    }
                    spray_sys.update_params(&gpu.queue, &spray_params);
                    spray_sys.step(&gpu.device, &gpu.queue, self.state.runtime.particle_count);

                    // Publish the live auto-limits for the GUI readout
                    let (ta, wc) = spray_sys.auto_limits();
                    self.state.runtime.spray_ta_limit = ta;
                    self.state.runtime.spray_wc_limit = wc;
                }
            }
        }
        self.spray_prev_enabled = self.state.spray.enabled;

        // Advance the deterministic frame clock (drives captures and stats)
        let stepped = !self.state.simulation.paused;
        if stepped {
            self.sim_frame_index += 1;
            self.sim_time += frame_dt as f64;
        }
        if stepped || held {
            self.milestone_frame += 1;
        }

        // Measure fluid extents + probe heights on the post-integrate state
        if stepped {
            if let Some(probe) = self.probe_system.as_mut() {
                probe.collect(&gpu.device);
                probe.encode(&gpu.queue, &mut encoder, self.state.runtime.particle_count);
            }
        }

        // Surface foam map: settled foam particles deposit + retire, the
        // layer advects with the smoothed surface flow (MC water renders it)
        if let Some(foam_map) = self.foam_map.as_mut() {
            let spray = &self.state.spray;
            let active = spray.enabled
                && spray.foam_map
                && self.state.rendering.render_mode == FluidRenderMode::MarchingCubes
                && self.spray_system.is_some();
            let max_spray = self.spray_system.as_ref().map_or(0, |s| s.capacity());
            foam_map.update(
                &gpu.queue,
                active,
                stepped,
                self.state.container.width,
                self.state.container.depth,
                self.state.sph.kernel_radius,
                self.state.simulation.substep_dt() * self.state.simulation.substeps as f32,
                spray.foam_persistence,
                self.state.runtime.particle_count,
                max_spray,
            );
            if active && stepped {
                if let (Some(sph_sim), Some(spray_sys)) = (&self.sph_simulation, &self.spray_system) {
                    foam_map.encode(
                        &gpu.device,
                        &mut encoder,
                        sph_sim.particle_buffer(),
                        spray_sys.spray_buffer(),
                        sph_sim.container_geom_buffer(),
                    );
                }
            }
        }

        // Prepare a swapchain readback if a capture is due this frame
        let mut frame_capture = false;
        while self
            .pending_captures
            .front()
            .is_some_and(|&f| self.milestone_frame >= f)
        {
            self.pending_captures.pop_front();
            frame_capture = true;
        }
        let mut scheduled_snapshot = false;
        while self
            .pending_snapshots
            .front()
            .is_some_and(|&f| self.milestone_frame >= f)
        {
            self.pending_snapshots.pop_front();
            scheduled_snapshot = true;
        }
        let snapshot = std::mem::take(&mut self.snapshot_requested);
        let capture_due = frame_capture || scheduled_snapshot || snapshot;
        // Pixel probe: fresh records for this frame's water pass
        if let Some(mc) = &self.mc_renderer {
            mc.reset_probe(&gpu.queue);
        }
        let capture = if capture_due {
            let unpadded_bytes_per_row = gpu.config.width * 4;
            let padded_bytes_per_row = unpadded_bytes_per_row
                .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Capture Readback"),
                size: padded_bytes_per_row as u64 * gpu.config.height as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            Some((buffer, padded_bytes_per_row))
        } else {
            None
        };

        // Integrate dynamic rigid bodies on CPU (Static/Kinematic bodies are
        // pose-driven and intentionally skip the container clamp so obstacles
        // can be embedded in walls/floor)
        if !self.state.simulation.paused {
            if let Some(sph_sim) = &self.sph_simulation {
                let accums = *sph_sim.rigid_body_accums();
                let gravity = self.state.simulation.gravity_vector();
                let fluid_density = self.state.sph.rest_density();
                let kernel_radius = self.state.sph.kernel_radius;
                let particles_per_volume = fluid_density / self.state.sph.mass.max(1e-6);
                for (i, body) in self
                    .state
                    .rigid_bodies
                    .iter_mut()
                    .take(accums.len())
                    .enumerate()
                {
                    if body.enabled && body.motion == RigidBodyMotion::Dynamic {
                        integrate_rigid_body(
                            body,
                            &self.state.container,
                            substep_dt,
                            num_substeps,
                            gravity,
                            fluid_density,
                            kernel_radius,
                            particles_per_volume,
                            &accums[i],
                        );
                    }
                }
            }
        }

        // Scene renderers always draw into the HDR scene buffer (HDR_FORMAT);
        // post-processing (or its passthrough when disabled) writes the screen
        // A refraction debug view writes data, not color: no exposure, tonemap,
        // bloom or FXAA may touch it
        let debug_view_on = self.state.rendering.render_mode == FluidRenderMode::MarchingCubes
            && self.state.rendering.mc_debug_view != crate::state::McDebugView::Off;
        let post_process_enabled = self.state.post_process.enabled && !debug_view_on;
        let render_target = self
            .post_process_renderer
            .as_ref()
            .expect("post-process renderer owns the HDR scene buffer")
            .scene_view();

        // Render fluid or particles based on render mode,
        // then render rigid body with depth testing against the fluid
        let mut fluid_depth_view: Option<&wgpu::TextureView> = None;
        if let Some(sph_sim) = &self.sph_simulation {
            match self.state.rendering.render_mode {
                FluidRenderMode::Particles => {
                    // Particle rendering (individual spheres)
                    let use_env = self.state.environment.background_mode == BackgroundMode::Environment;

                    // Render environment background if HDR mode
                    if use_env {
                        if let (Some(pipeline), Some(bind_group), Some(buf)) = (
                            &self.env_bg_pipeline,
                            &self.env_bg_bind_group,
                            &self.env_params_buffer,
                        ) {
                            // Update env params
                            let env_params = self.state.environment.to_gpu_params(&self.ground_staging());
                            gpu.queue.write_buffer(buf, 0, bytemuck::bytes_of(&env_params));

                            // Update camera in particle renderer (needed for inv matrices in env shader)
                            if let Some(renderer) = &self.renderer {
                                let camera_params = self.camera.to_gpu_params();
                                renderer.update_camera(&gpu.queue, &camera_params);
                            }

                            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Env Background Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: render_target,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            pass.set_pipeline(pipeline);
                            pass.set_bind_group(0, bind_group, &[]);
                            pass.draw(0..3, 0..1);
                        }
                    }

                    if let Some(renderer) = &self.renderer {
                        renderer.render(
                            &mut encoder,
                            render_target,
                            sph_sim.particle_buffer(),
                            sph_sim.num_particles(),
                            &self.state.environment.background_color,
                            !use_env, // clear_background: only clear if solid color mode
                        );
                        // Share particle depth buffer for rigid body occlusion
                        fluid_depth_view = Some(renderer.depth_view());
                    }
                }
                FluidRenderMode::ScreenSpace => {
                    // Whitewater splat depth gate follows the active water
                    // renderer's front depth (1 frame stale, same as MC)
                    if self.spray_depth_bound != Some(FluidRenderMode::ScreenSpace) {
                        if let (Some(spray_renderer), Some(ss_renderer), Some(spray_system)) = (
                            &mut self.spray_renderer,
                            &self.ss_renderer,
                            &self.spray_system,
                        ) {
                            spray_renderer.set_depth_view(
                                &gpu.device,
                                spray_system.spray_buffer(),
                                ss_renderer.front_depth_view(),
                            );
                            self.spray_depth_bound = Some(FluidRenderMode::ScreenSpace);
                        }
                    }
                    // Run the shared Yu & Turk anisotropy fit so the SS splats
                    // can stretch into ellipsoids (records live in the MC
                    // renderer; sorted-particle indexed)
                    let aniso_on = self.state.rendering.mc_anisotropy && sph_sim.num_particles() > 0;
                    let h_mc = self.state.sph.kernel_radius
                        * self.state.rendering.mc_density_radius_scale;
                    if aniso_on {
                        if let Some(mc_renderer) = &mut self.mc_renderer {
                            mc_renderer.update_aniso_params(
                                &gpu.queue,
                                true,
                                self.state.rendering.mc_anisotropy_strength,
                                self.state.sph.kernel_radius,
                                h_mc,
                            );
                            mc_renderer.run_anisotropy(
                                &mut encoder,
                                &gpu.device,
                                sph_sim.sorted_particle_buffer(),
                                sph_sim.cell_starts_buffer(),
                                sph_sim.cell_counts_buffer(),
                                sph_sim.grid_params_buffer(),
                                sph_sim.num_particles(),
                            );
                        }
                    }

                    // Screen-space fluid rendering with narrow-range depth filter
                    if let (Some(ss_renderer), Some(mc_renderer)) =
                        (&self.ss_renderer, &self.mc_renderer)
                    {
                        let camera_params = self.camera.to_gpu_params();
                        ss_renderer.update_camera(&gpu.queue, &camera_params);
                        ss_renderer.update_light_params(&gpu.queue, &self.state.lighting.to_gpu_params());
                        let use_env = self.state.environment.background_mode == BackgroundMode::Environment;
                        let water_params = crate::render::marching_cubes::GpuWaterParams {
                            water_color: self.state.rendering.particle_color,
                            roughness: self.state.rendering.water_roughness,
                            ior: 1.333,
                            env_intensity: self.state.environment.environment_intensity,
                            use_env_background: if use_env { 1 } else { 0 },
                            background_r: self.state.environment.background_color[0],
                            background_g: self.state.environment.background_color[1],
                            background_b: self.state.environment.background_color[2],
                            time: self.state.runtime.time_elapsed,
                            refraction_strength: self.state.rendering.refraction_strength,
                            deep_color_r: self.state.rendering.deep_water_color[0],
                            deep_color_g: self.state.rendering.deep_water_color[1],
                            deep_color_b: self.state.rendering.deep_water_color[2],
                            ripple_strength: self.state.rendering.ripple_strength,
                            clarity: self.state.rendering.water_clarity,
                            _pad1: self.state.rendering.ss_debug_view as f32,
                            foam_coverage: self.state.spray.foam_coverage,
                            aeration_strength: self.state.spray.aeration_strength,
                            physical_medium: if self.state.rendering.physical_water_medium { 1.0 } else { 0.0 },
                            body_count: 0,
                            debug_view: 0,
                            _pad_m: 0.0,
                        };
                        ss_renderer.update_water_params(&gpu.queue, &water_params);
                        let env_params = self.state.environment.to_gpu_params(&self.ground_staging());
                        ss_renderer.update_env_params(&gpu.queue, &env_params);

                        // Scene objects go into the SS background pass (refraction +
                        // depth-aware composite), mirroring the MC renderer.
                        let rb_for_ss = if self.state.rigid_bodies.iter().any(|b| b.enabled) {
                            self.rigid_body_renderer.as_ref()
                        } else {
                            None
                        };
                        let spray_for_ss = if self.state.spray.enabled {
                            self.spray_renderer.as_ref()
                        } else {
                            None
                        };
                        let container_for_ss = if self.state.container.style == ContainerStyle::OpaquePool {
                            self.container_renderer.as_ref()
                        } else {
                            None
                        };

                        // Splat whitewater into the shared half-res field
                        // (owned by the MC renderer, composited by ss_composite;
                        // cleared even when spray is off so no stale foam lingers)
                        if let (Some(spray_renderer), Some(mc_renderer)) =
                            (&self.spray_renderer, &self.mc_renderer)
                        {
                            spray_renderer.render_foam_density(
                                &mut encoder,
                                mc_renderer.foam_density_view(),
                                self.state.spray.enabled,
                            );
                        }

                        let ss_radius = self.state.sph.kernel_radius * self.state.rendering.ss_radius_scale;
                        let particle_spacing = self.state.sph.kernel_radius * 0.6;
                        // Ellipsoid records are normalized to h_mc; the SS
                        // surface sits at ss_radius
                        let aniso_surface_scale = ss_radius / h_mc.max(1e-6);
                        ss_renderer.render(
                            &gpu.device,
                            &gpu.queue,
                            &mut encoder,
                            render_target,
                            sph_sim.sorted_particle_buffer(),
                            sph_sim.num_particles(),
                            &camera_params,
                            ss_radius,
                            particle_spacing,
                            mc_renderer.aniso_buffer(),
                            aniso_on,
                            aniso_surface_scale,
                            self.state.rendering.ss_filter_size,
                            self.state.rendering.ss_filter_iterations,
                            self.state.rendering.ss_nr_range,
                            self.state.rendering.ss_nr_offset,
                            self.state.rendering.ss_temporal,
                            self.camera.fov,
                            rb_for_ss,
                            spray_for_ss,
                            container_for_ss,
                        );
                        fluid_depth_view = Some(ss_renderer.depth_view());
                    }
                }
                FluidRenderMode::MarchingCubes => {
                    // Rebind the whitewater splat depth gate to the MC front
                    // depth after a mode switch or resize
                    if self.spray_depth_bound != Some(FluidRenderMode::MarchingCubes) {
                        if let (Some(spray_renderer), Some(mc_renderer), Some(spray_system)) = (
                            &mut self.spray_renderer,
                            &self.mc_renderer,
                            &self.spray_system,
                        ) {
                            spray_renderer.set_depth_view(
                                &gpu.device,
                                spray_system.spray_buffer(),
                                mc_renderer.front_depth_view(),
                            );
                            self.spray_depth_bound = Some(FluidRenderMode::MarchingCubes);
                        }
                    }
                    // Marching cubes surface mesh rendering
                    let caustics_on = self.caustics_active();
                    let env_params = self.state.environment.to_gpu_params(&self.ground_staging());
                    if let Some(mc_renderer) = &mut self.mc_renderer {
                        let camera_params = self.camera.to_gpu_params();
                        mc_renderer.update_camera(&gpu.queue, &camera_params);
                        mc_renderer.update_light_params(&gpu.queue, &self.state.lighting.to_gpu_params());
                        let use_env = self.state.environment.background_mode == BackgroundMode::Environment;
                        mc_renderer.update_water_params(
                            &gpu.queue,
                            &self.state.rendering.particle_color,
                            self.state.rendering.water_roughness,
                            self.state.environment.environment_intensity,
                            use_env,
                            &self.state.environment.background_color,
                            self.state.runtime.time_elapsed,
                            self.state.rendering.refraction_strength,
                            &self.state.rendering.deep_water_color,
                            self.state.rendering.ripple_strength,
                            self.state.rendering.water_clarity,
                            self.state.rendering.mc_physical_refraction,
                            self.state.rendering.physical_water_medium,
                            self.state.spray.foam_coverage,
                            self.state.spray.aeration_strength,
                            // Enabled bodies occupy the front of the body array
                            self.state
                                .rigid_bodies
                                .iter()
                                .filter(|b| b.enabled)
                                .count()
                                .min(crate::state::MAX_RIGID_BODIES) as u32,
                            self.state.rendering.mc_debug_view.as_u32(),
                        );
                        mc_renderer.update_env_params(&gpu.queue, &env_params);
                        mc_renderer.set_ssr_enabled(&gpu.queue, self.state.rendering.ssr_enabled);

                        // Update MC grid bounds to cover the full container + margin
                        // (anisotropic ellipsoids can reach up to ANISO_MAX_STRETCH × the kernel radius)
                        let (aabb_min, aabb_max) = self.state.container.tilted_aabb();
                        let aniso_stretch = if self.state.rendering.mc_anisotropy {
                            crate::render::marching_cubes::ANISO_MAX_STRETCH
                        } else {
                            1.0
                        };
                        let mc_margin = self.state.sph.kernel_radius * self.state.rendering.mc_density_radius_scale * aniso_stretch + 0.05;
                        mc_renderer.set_bounds(
                            [aabb_min[0] - mc_margin, aabb_min[1] - mc_margin, aabb_min[2] - mc_margin],
                            [aabb_max[0] + mc_margin, aabb_max[1] + mc_margin, aabb_max[2] + mc_margin],
                        );

                        // Update container geometry for MC clipping (with clip_margin from MC cell_size)
                        let mc_geom = {
                            let c = &self.state.container;
                            let is_pool = c.style == ContainerStyle::OpaquePool;
                            let grid_extent = (aabb_max[0] - aabb_min[0] + 2.0 * mc_margin)
                                .max(aabb_max[1] - aabb_min[1] + 2.0 * mc_margin)
                                .max(aabb_max[2] - aabb_min[2] + 2.0 * mc_margin);
                            let mc_cell_size = grid_extent / mc_renderer.grid_size() as f32;
                            let geom = c.to_gpu_geometry(
                                self.state.sph.wall_stiffness,
                                self.state.simulation.damping,
                                is_pool,
                                mc_cell_size * 1.5,
                            );
                            mc_renderer.update_container_geometry(&gpu.queue, &geom);
                            geom
                        };

                        let iso_value = self.state.rendering.compute_iso_value(self.state.sph.kernel_radius);
                        let blur_radius = self.state.rendering.mc_blur_radius;
                        mc_renderer.update_params(
                            &gpu.queue,
                            self.state.sph.kernel_radius * self.state.rendering.mc_density_radius_scale,
                            iso_value,
                            sph_sim.num_particles(),
                            blur_radius,
                        );
                        mc_renderer.update_calm_smoothing(
                            &gpu.queue,
                            self.state.rendering.mc_calm_smoothing,
                            self.state.sph.kernel_radius,
                        );
                        mc_renderer.update_aniso_params(
                            &gpu.queue,
                            self.state.rendering.mc_anisotropy,
                            self.state.rendering.mc_anisotropy_strength,
                            self.state.sph.kernel_radius,
                            self.state.sph.kernel_radius * self.state.rendering.mc_density_radius_scale,
                        );
                        mc_renderer.generate(
                            &mut encoder,
                            &gpu.device,
                            sph_sim.sorted_particle_buffer(),
                            sph_sim.cell_starts_buffer(),
                            sph_sim.cell_counts_buffer(),
                            sph_sim.grid_params_buffer(),
                            blur_radius,
                            sph_sim.num_particles(),
                            self.state.rendering.mc_anisotropy,
                            self.state.rendering.mc_calm_smoothing,
                        );
                        // Splat foam into the density field the water shader
                        // composites (cleared even when spray is off so no
                        // stale foam lingers on the surface)
                        if let Some(spray_renderer) = &self.spray_renderer {
                            spray_renderer.render_foam_density(
                                &mut encoder,
                                mc_renderer.foam_density_view(),
                                self.state.spray.enabled,
                            );
                        }
                        // Caustics: raster the fresh MC mesh from the sun and
                        // splat refracted photons onto the pool floor map. Runs
                        // before mc_renderer.render() so the container picks up
                        // this frame's map in both background and main passes.
                        if caustics_on {
                            if let Some(caustics) = &mut self.caustics_renderer {
                                // Enabled bodies occupy the front of the render
                                // body array (update_bodies uploads them in order)
                                let body_count = self
                                    .state
                                    .rigid_bodies
                                    .iter()
                                    .filter(|b| b.enabled)
                                    .count()
                                    .min(crate::state::MAX_RIGID_BODIES) as u32;
                                caustics.update(
                                    &gpu.queue,
                                    &mc_geom,
                                    self.state.lighting.sun_direction_normalized(),
                                    &self.state.caustics,
                                    self.state.rendering.water_clarity,
                                    self.state.runtime.time_elapsed,
                                    body_count,
                                );
                                let container_mesh = self
                                    .container_renderer
                                    .as_ref()
                                    .map(|c| c.mesh_buffers());
                                caustics.run(
                                    &mut encoder,
                                    mc_renderer.mesh_indirect_buffer(),
                                    container_mesh,
                                );
                            }
                        }
                        // Pass rigid body renderer into MC pass for proper MSAA depth testing
                        let rb_for_mc = if self.state.rigid_bodies.iter().any(|b| b.enabled) {
                            self.rigid_body_renderer.as_ref()
                        } else {
                            None
                        };
                        let spray_for_mc = if self.state.spray.enabled {
                            self.spray_renderer.as_ref()
                        } else {
                            None
                        };
                        let container_for_mc = if self.state.container.style == ContainerStyle::OpaquePool {
                            self.container_renderer.as_ref()
                        } else {
                            None
                        };
                        mc_renderer.render(
                            &mut encoder,
                            render_target,
                            &self.state.environment.background_color,
                            rb_for_mc,
                            spray_for_mc,
                            container_for_mc,
                        );
                    }
                }
            }
        }

        // Render spray particles for Particles mode
        // (MC and SS modes draw spray inside their own passes)
        if self.state.spray.enabled && self.state.rendering.render_mode == FluidRenderMode::Particles {
            if let Some(spray_renderer) = &self.spray_renderer {
                let depth_view = fluid_depth_view
                    .unwrap_or_else(|| self.rigid_body_depth_view.as_ref().unwrap());
                let use_fluid_depth = fluid_depth_view.is_some();

                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Spray Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: render_target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: if use_fluid_depth {
                                wgpu::LoadOp::Load
                            } else {
                                wgpu::LoadOp::Clear(1.0)
                            },
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                spray_renderer.render(&mut render_pass);
            }
        }

        // Render rigid bodies with depth testing against fluid surface
        // (MC and SS modes handle this inside their own passes above)
        if self.state.rigid_bodies.iter().any(|b| b.enabled)
            && self.state.rendering.render_mode == FluidRenderMode::Particles
        {
            if let Some(rb_renderer) = &self.rigid_body_renderer {
                let depth_view = fluid_depth_view
                    .unwrap_or_else(|| self.rigid_body_depth_view.as_ref().unwrap());
                let use_fluid_depth = fluid_depth_view.is_some();

                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Rigid Body Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: render_target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            // Load fluid depth if available, otherwise clear (everything passes)
                            load: if use_fluid_depth {
                                wgpu::LoadOp::Load
                            } else {
                                wgpu::LoadOp::Clear(1.0)
                            },
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rb_renderer.render(&mut render_pass);
            }
        }

        // Run GTAO if enabled and post-processing is on
        if post_process_enabled && self.state.post_process.ao_enabled {
            if let Some(gtao) = &mut self.gtao_renderer {
                // Get the depth view from the appropriate renderer
                let depth_view = match self.state.rendering.render_mode {
                    FluidRenderMode::MarchingCubes => {
                        self.mc_renderer.as_ref().map(|mc| mc.front_depth_view())
                    }
                    FluidRenderMode::ScreenSpace => {
                        self.ss_renderer.as_ref().map(|ss| ss.front_depth_view())
                    }
                    FluidRenderMode::Particles => {
                        self.renderer.as_ref().map(|r| r.depth_view())
                    }
                };

                if let Some(depth_view) = depth_view {
                    let camera_params = self.camera.to_gpu_params();

                    // Compute previous VP matrix
                    let prev_cam = self.prev_camera_params.unwrap_or(camera_params);
                    let prev_vp = crate::render::gtao::GpuPrevViewProjection {
                        matrix: mat4_mul(prev_cam.projection, prev_cam.view),
                    };

                    // Rebuild bind groups with current depth view
                    gtao.rebuild_bind_groups(&gpu.device, depth_view);

                    gtao.render(
                        &mut encoder,
                        &gpu.queue,
                        &camera_params,
                        self.state.post_process.ao_radius,
                        &prev_vp,
                        gpu.config.width,
                        gpu.config.height,
                    );

                    // Update post-process AO bind group
                    if let Some(pp) = &mut self.post_process_renderer {
                        pp.update_ao_bind_group(&gpu.device, gtao.ao_view());
                    }

                    // Save current camera for next frame's reprojection
                    self.prev_camera_params = Some(camera_params);
                }
            }
        }

        // Post-process the HDR scene onto the screen; disabled = plain
        // passthrough (no exposure, tonemap or effects; clipped at 1)
        if let Some(pp) = &self.post_process_renderer {
            if post_process_enabled {
                let pp_params = self.state.post_process.to_gpu_params();
                pp.update_params(&gpu.queue, &pp_params);
                pp.render(&mut encoder, &view, self.state.post_process.bloom_enabled, self.state.post_process.streaks_enabled, self.state.quality.fxaa_enabled);
            } else {
                pp.update_params(&gpu.queue, &crate::state::PostProcessConfig::passthrough_gpu_params());
                pp.render(&mut encoder, &view, false, false, self.state.quality.fxaa_enabled && !debug_view_on);
            }
        }

        // Render wireframe container visualization (on top of fluid, below UI)
        // Skip when using opaque pool style
        if let Some(wireframe) = &self.wireframe_renderer {
        if self.state.container.style == ContainerStyle::Wireframe {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Wireframe Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            wireframe.render(&mut render_pass);
        }
        }

        // Capture the scene as rendered so far (everything except the GUI)
        if let Some((buffer, padded_bytes_per_row)) = &capture {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &output.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(*padded_bytes_per_row),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: gpu.config.width,
                    height: gpu.config.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Render egui
        let egui_renderer = self.egui_renderer.as_mut().unwrap();
        let window = self.window.as_ref().unwrap();
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gpu.config.width, gpu.config.height],
            pixels_per_point: window.scale_factor() as f32,
        };

        egui_renderer.update_buffers(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &tris,
            &screen_descriptor,
        );

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // forget_lifetime is needed because egui_wgpu::Renderer::render requires 'static
            let mut render_pass = render_pass.forget_lifetime();
            egui_renderer.render(&mut render_pass, &tris, &screen_descriptor);
        }

        // Cleanup egui textures
        for id in &full_output.textures_delta.free {
            egui_renderer.free_texture(id);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Read back marching cubes vertex count for next frame. Stats runs
        // block for an exact per-frame value (deterministic CSV rows);
        // interactive frames poll so the CPU never stalls on the GPU.
        if self.state.rendering.render_mode == FluidRenderMode::MarchingCubes {
            if let Some(mc_renderer) = &mut self.mc_renderer {
                if self.stats_file.is_some() {
                    mc_renderer.read_vertex_count(&gpu.device);
                } else {
                    mc_renderer.poll_vertex_count(&gpu.device);
                }
            }
        }

        // Probe readback mirrors the same policy: blocking for exact stats
        // rows, async map for interactive frames
        if let Some(probe) = self.probe_system.as_mut() {
            if self.stats_file.is_some() {
                if stepped {
                    self.state.runtime.measurements = probe.read_blocking(&gpu.device);
                }
            } else {
                probe.arm_map();
                if let Some(m) = probe.latest() {
                    self.state.runtime.measurements = Some(m.clone());
                }
            }
        }

        // Finish any pending capture (map readback, write PNG)
        if let Some((buffer, padded_bytes_per_row)) = capture {
            if frame_capture {
                let path = self
                    .launch
                    .out_dir
                    .join(format!("frame_{:05}.png", self.milestone_frame));
                self.save_capture(&buffer, padded_bytes_per_row, &path);
                let probe_path = self
                    .launch
                    .out_dir
                    .join(format!("frame_{:05}_probe.json", self.milestone_frame));
                self.save_probe(&probe_path);
            }
            if snapshot {
                self.save_snapshot(&buffer, padded_bytes_per_row, false);
            }
            if scheduled_snapshot {
                self.save_snapshot(&buffer, padded_bytes_per_row, true);
            }
        }

        // Append a stats row for every simulated frame
        if stepped && self.stats_file.is_some() {
            let spray_counts = if self.state.spray.enabled {
                let device = &self.gpu.as_ref().unwrap().device;
                self.spray_system
                    .as_ref()
                    .map_or([0; 4], |s| s.read_stats(device))
            } else {
                [0; 4]
            };
            let mc_vertices = if self.state.rendering.render_mode == FluidRenderMode::MarchingCubes
            {
                self.mc_renderer.as_ref().map_or(0, |mc| mc.vertex_count())
            } else {
                0
            };
            let mut row = format!(
                "{},{:.4},{},{},{},{},{},{},{:.1},{:.4},{:.4}",
                self.sim_frame_index,
                self.sim_time,
                self.state.runtime.particle_count,
                mc_vertices,
                spray_counts[0],
                spray_counts[1],
                spray_counts[2],
                spray_counts[3],
                self.state.runtime.fps,
                self.state.runtime.spray_ta_limit,
                self.state.runtime.spray_wc_limit,
            );
            // Measurement columns (empty cell = no fluid in range)
            let fmt = |v: Option<f32>| v.map_or(String::new(), |v| format!("{v:.5}"));
            let m = self.state.runtime.measurements.as_ref();
            row.push_str(&format!(
                ",{},{},{}",
                fmt(m.and_then(|m| m.max_x)),
                fmt(m.and_then(|m| m.min_x)),
                fmt(m.and_then(|m| m.max_y)),
            ));
            for k in 0..self.state.scenario.probes.len().min(crate::state::MAX_PROBES) {
                let h = m.and_then(|m| m.probe_heights.get(k).copied().flatten());
                row.push_str(&format!(",{}", fmt(h)));
            }
            row.push('\n');
            if let Some(file) = self.stats_file.as_mut() {
                let _ = file.write_all(row.as_bytes());
                let _ = file.flush();
            }
        }

        // Automation exit: leave once every requested milestone is reached
        if self.launch.is_automated() && !self.launch.stay && !self.should_exit {
            let captures_done =
                self.pending_captures.is_empty() && self.pending_snapshots.is_empty();
            let exit_frame_reached = self
                .launch
                .exit_after
                .is_none_or(|n| self.milestone_frame >= n);
            let has_milestone = self.had_captures || self.launch.exit_after.is_some();
            if has_milestone && captures_done && exit_frame_reached {
                println!(
                    "Automation milestones reached at frame {} — exiting",
                    self.milestone_frame
                );
                self.should_exit = true;
            }
        }

        // Handle GUI actions after rendering
        match gui_action {
            GuiAction::ResetSimulation => self.reset_simulation(),
            GuiAction::ResetDefaults => self.reset_defaults(),
            GuiAction::RebuildMcGrid => {
                if let Some(mc_renderer) = self.mc_renderer.as_mut() {
                    let gpu = self.gpu.as_ref().unwrap();
                    mc_renderer.rebuild_grid(
                        &gpu.device,
                        self.state.rendering.mc_grid_resolution.grid_size(),
                    );
                }
            }
            GuiAction::ExportConfig => self.export_config(),
            GuiAction::None => {}
        }
    }

    /// Map a completed swapchain readback and write it out as a PNG named
    /// after the current simulation frame.
    fn save_capture(&self, buffer: &wgpu::Buffer, padded_bytes_per_row: u32, path: &std::path::Path) -> bool {
        let gpu = self.gpu.as_ref().unwrap();
        let (width, height) = (gpu.config.width, gpu.config.height);

        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        {
            let data = slice.get_mapped_range();
            for row in 0..height {
                let start = (row * padded_bytes_per_row) as usize;
                pixels.extend_from_slice(&data[start..start + (width * 4) as usize]);
            }
        }
        buffer.unmap();

        // Swapchain is BGRA on most Windows backends; PNG expects RGBA.
        // Alpha is meaningless post-composite, so force opaque.
        let swap_bgra = matches!(
            gpu.config.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        for px in pixels.chunks_exact_mut(4) {
            if swap_bgra {
                px.swap(0, 2);
            }
            px[3] = 255;
        }

        match image::RgbaImage::from_raw(width, height, pixels) {
            Some(img) => match img.save(path) {
                Ok(()) => {
                    println!("Captured {}", path.display());
                    true
                }
                Err(e) => {
                    eprintln!("error: failed to save capture {}: {e}", path.display());
                    false
                }
            },
            None => {
                eprintln!("error: capture buffer size mismatch");
                false
            }
        }
    }

    /// Snapshot: this frame as a GUI-free PNG, the exact config (live camera
    /// included) as JSON and the simulation state as `.state`, all from the
    /// same instant. F12 writes captures/snapshots/snap_NNN_WxH.*; scheduled
    /// (`--snapshot`) ones go to --out as snap_fNNNNN_WxH.* (milestone frame).
    /// Repro: `--config <json> --load-state <state> --size WxH [--hold]`.
    fn save_snapshot(&mut self, buffer: &wgpu::Buffer, padded_bytes_per_row: u32, scheduled: bool) {
        let (width, height) = {
            let gpu = self.gpu.as_ref().unwrap();
            (gpu.config.width, gpu.config.height)
        };
        let (dir, stem) = if scheduled {
            let stem = format!("snap_f{:05}_{width}x{height}", self.milestone_frame);
            (self.launch.out_dir.clone(), stem)
        } else {
            let dir = std::path::PathBuf::from("captures").join("snapshots");
            if let Err(e) = std::fs::create_dir_all(&dir) {
                eprintln!("error: cannot create {}: {e}", dir.display());
                return;
            }
            let mut n = 1u32;
            let stem = loop {
                let stem = format!("snap_{n:03}_{width}x{height}");
                if !dir.join(format!("{stem}.png")).exists() {
                    break stem;
                }
                n += 1;
            };
            (dir, stem)
        };
        let png = dir.join(format!("{stem}.png"));
        let json = dir.join(format!("{stem}.json"));
        let sim_state = dir.join(format!("{stem}.state"));
        if !self.save_capture(buffer, padded_bytes_per_row, &png) {
            return;
        }
        let mut state = self.state.clone();
        state.camera.distance = self.camera.distance;
        state.camera.yaw = self.camera.yaw;
        state.camera.pitch = self.camera.pitch;
        state.camera.target = self.camera.target;
        state.camera.fov = self.camera.fov;
        let message = match std::fs::write(&json, crate::launch::config_to_json(&state))
            .map_err(|e| format!("snapshot config failed: {e}"))
            .and_then(|()| self.save_state(&sim_state))
        {
            Ok(()) => format!(
                "snapshot {} (repro: --config {} --load-state {} --size {width}x{height}, \
                 add --hold to freeze it)",
                png.display(),
                json.display(),
                sim_state.display(),
            ),
            Err(e) => format!("snapshot failed: {e}"),
        };
        println!("{message}");
        self.state.runtime.last_export = Some(message);
    }

    /// Pixel probe: this frame's refraction records as JSON (no-op unless
    /// --probe); decoded by scripts/probe_decode.py
    fn save_probe(&self, path: &std::path::Path) {
        let (Some(gpu), Some(mc)) = (self.gpu.as_ref(), self.mc_renderer.as_ref()) else {
            return;
        };
        if !mc.probe_enabled() {
            return;
        }
        let dump = mc.read_probe(&gpu.device, &gpu.queue);
        let probed: std::collections::HashSet<[u32; 2]> =
            dump.fragments.iter().map(|f| f.pixel).collect();
        let overflowed = dump.fragments.iter().filter(|f| f.overflow).count();
        let mut value = serde_json::to_value(&dump).unwrap_or_default();
        if let Some(obj) = value.as_object_mut() {
            obj.insert("frame".into(), self.milestone_frame.into());
            obj.insert("size".into(), serde_json::json!([gpu.config.width, gpu.config.height]));
            obj.insert(
                "render_mode".into(),
                format!("{:?}", self.state.rendering.render_mode).into(),
            );
        }
        let text = serde_json::to_string(&value).unwrap_or_default();
        match std::fs::write(path, text) {
            Ok(()) => {
                println!(
                    "probe: {} fragment(s) on {}/{} pixel(s) -> {}",
                    dump.fragments.len(),
                    probed.len(),
                    dump.pixels.len(),
                    path.display()
                );
                if self.state.rendering.render_mode != FluidRenderMode::MarchingCubes {
                    println!("probe: warning: only the MarchingCubes water shader is instrumented");
                }
                if overflowed > 0 || dump.slots_dropped > 0 {
                    println!(
                        "probe: warning: {overflowed} fragment(s) ran out of event slots, {} fragment(s) dropped",
                        dump.slots_dropped
                    );
                }
            }
            Err(e) => eprintln!("error: cannot write {}: {e}", path.display()),
        }
    }

    /// Write the simulation state (GPU buffers + CPU clocks) to a `.state`
    /// file; see simulation/snapshot.rs for what is in it and why.
    fn save_state(&self, path: &std::path::Path) -> Result<(), String> {
        use crate::simulation::snapshot::{read_gpu, SimState, StateHeader, STATE_VERSION};
        let gpu = self.gpu.as_ref().ok_or("no GPU")?;
        let sph_sim = self.sph_simulation.as_ref().ok_or("no simulation")?;
        let spray = self.spray_system.as_ref().ok_or("no spray system")?;
        let foam_map = self.foam_map.as_ref().ok_or("no foam map")?;

        let mut sources = sph_sim.snapshot_sources();
        sources.extend(spray.snapshot_sources());
        sources.extend(foam_map.snapshot_sources());
        let blobs = read_gpu(&gpu.device, &gpu.queue, sources);

        let (grid_cell_size, grid_total_cells) = sph_sim.grid_layout();
        let (ta, wc) = spray.auto_limits();
        let header = StateHeader {
            version: STATE_VERSION,
            particle_count: sph_sim.num_particles(),
            grid_cell_size,
            grid_total_cells,
            spray_capacity: spray.capacity(),
            sim_frame_index: self.sim_frame_index,
            sim_time: self.sim_time,
            time_elapsed: self.state.runtime.time_elapsed,
            spray_frame_count: self.state.runtime.frame_count,
            spray_auto_limits: [ta, wc],
            foam_map: foam_map.snapshot_scalars(),
            scenario_fired: self.scenario_fired.clone(),
            scenario_events_fired: self.state.runtime.scenario_events_fired,
            spin_angles: self.state.rigid_bodies.iter().map(|b| b.spin_angle).collect(),
            blobs: Vec::new(),
        };
        SimState::new(header, blobs).write(path)
    }

    /// Restore a `.state` file over the freshly initialized simulation. The
    /// config (camera, tunables, rigid body poses) comes from the snapshot's
    /// JSON via --config; this restores what the config cannot express.
    fn load_state(&mut self, path: &std::path::Path) -> Result<(), String> {
        let state = crate::simulation::snapshot::SimState::read(path)?;
        let h = &state.header;
        let substep_dt = self.simulation_substep_dt();
        let gpu = self.gpu.as_ref().ok_or("no GPU")?;
        let sph_sim = self.sph_simulation.as_mut().ok_or("no simulation")?;
        let spray = self.spray_system.as_mut().ok_or("no spray system")?;
        let foam_map = self.foam_map.as_mut().ok_or("no foam map")?;

        // Layout checks first: a mismatched grid or ring buffer would load as
        // garbage, not fail
        let (cell_size, total_cells) = sph_sim.grid_layout();
        if h.grid_cell_size != cell_size || h.grid_total_cells != total_cells {
            return Err(format!(
                "saved on a kernel_radius {} grid ({} cells), this config builds {} ({} cells); \
                 load it with the snapshot's --config",
                h.grid_cell_size, h.grid_total_cells, cell_size, total_cells
            ));
        }
        if h.spray_capacity != spray.capacity() {
            return Err(format!(
                "saved with spray.max_particles {}, this config has {}",
                h.spray_capacity,
                spray.capacity()
            ));
        }
        sph_sim.restore_snapshot(&gpu.queue, &state)?;
        spray.restore_snapshot(&gpu.queue, &state)?;
        foam_map.restore_snapshot(&gpu.queue, &state)?;

        self.state.runtime.particle_count = h.particle_count;
        let sph_params = self.state.sph.to_gpu_params_3d(h.particle_count, substep_dt);
        sph_sim.update_sph_params(&gpu.queue, &sph_params);

        self.sim_frame_index = h.sim_frame_index;
        self.sim_time = h.sim_time;
        self.state.runtime.time_elapsed = h.time_elapsed;
        self.state.runtime.frame_count = h.spray_frame_count;
        // Event flags realign to the config's list in pump_scenario_events
        self.scenario_fired = h.scenario_fired.clone();
        self.state.runtime.scenario_events_fired = h.scenario_events_fired;
        if h.spin_angles.len() != self.state.rigid_bodies.len() {
            eprintln!(
                "warning: state has {} rigid bodies, config has {}; spin angles matched by index",
                h.spin_angles.len(),
                self.state.rigid_bodies.len()
            );
        }
        for (body, &angle) in self.state.rigid_bodies.iter_mut().zip(&h.spin_angles) {
            body.spin_angle = angle;
        }
        // The loaded state is already whitewater-live: no re-enable reset
        self.spray_prev_enabled = self.state.spray.enabled;

        println!(
            "loaded state {} (sim frame {}, t={:.3}s, {} particles){}",
            path.display(),
            h.sim_frame_index,
            h.sim_time,
            h.particle_count,
            if self.launch.hold { ", held" } else { "" },
        );
        Ok(())
    }

    /// Write the current state (including the live camera pose) to
    /// configs/export_NNN.json.
    fn export_config(&mut self) {
        // Sync the live camera back into the serializable config
        self.state.camera.distance = self.camera.distance;
        self.state.camera.yaw = self.camera.yaw;
        self.state.camera.pitch = self.camera.pitch;
        self.state.camera.target = self.camera.target;
        self.state.camera.fov = self.camera.fov;

        match crate::launch::export_config(&self.state) {
            Ok(path) => {
                let display = path.display().to_string();
                println!("Exported config to {display}");
                self.state.runtime.last_export = Some(display);
            }
            Err(e) => {
                eprintln!("error: config export failed: {e}");
                self.state.runtime.last_export = Some(format!("export failed: {e}"));
            }
        }
    }
}

/// Multiply two 4x4 column-major matrices
fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            result[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    result
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

