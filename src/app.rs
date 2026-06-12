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
use crate::render::{Camera, ContainerRenderer, GpuPoolStyle, GtaoRenderer, MarchingCubesRenderer, ParticleRenderer3D, PostProcessRenderer, RigidBodyRenderer, ScreenSpaceFluidRenderer, SprayRenderer, WireframeRenderer};
use crate::state::ContainerStyle;
use crate::simulation::{SphSimulation3DGrid, SpraySystem, create_particle_block};
use crate::render::environment::load_embedded_environment_map;
use crate::render::mesh_loader::{self, SdfData};
use crate::state::{AppState, BackgroundMode, FluidRenderMode, ForceMode, GpuMouseForce, GpuShCoefficients, GpuSprayParams, GpuSprayRenderParams, HdrEnvironment, integrate_rigid_body, clamp_rigid_body_to_container};
use crate::render::environment::ShCoefficients;

pub struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    renderer: Option<ParticleRenderer3D>,
    mc_renderer: Option<MarchingCubesRenderer>,
    ss_renderer: Option<ScreenSpaceFluidRenderer>,
    wireframe_renderer: Option<WireframeRenderer>,
    container_renderer: Option<ContainerRenderer>,
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
    // Environment background rendering (for Particles mode)
    env_bg_pipeline: Option<wgpu::RenderPipeline>,
    env_bg_bind_group: Option<wgpu::BindGroup>,
    env_bg_bind_group_layout: Option<wgpu::BindGroupLayout>,
    env_params_buffer: Option<wgpu::Buffer>,
    sph_simulation: Option<SphSimulation3DGrid>,
    sdf_data: Option<SdfData>,
    camera: Camera,
    state: AppState,
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
    had_captures: bool,
    stats_file: Option<std::io::BufWriter<std::fs::File>>,
    should_exit: bool,
}

impl App {
    pub fn new(mut state: AppState, launch: LaunchOptions) -> Self {
        if launch.is_automated() && state.simulation.paused {
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
            let _ = writer
                .write_all(b"frame,sim_time,particles,mc_vertices,spray_total,spray,foam,bubbles,fps,ta_limit,wc_limit\n");
            writer
        });

        if !launch.capture_frames.is_empty() {
            if let Err(e) = std::fs::create_dir_all(&launch.out_dir) {
                eprintln!(
                    "error: cannot create capture dir {}: {e}",
                    launch.out_dir.display()
                );
                std::process::exit(1);
            }
        }

        let pending_captures: VecDeque<u64> = launch.capture_frames.iter().copied().collect();
        let had_captures = !pending_captures.is_empty();

        Self {
            window: None,
            gpu: None,
            renderer: None,
            mc_renderer: None,
            ss_renderer: None,
            wireframe_renderer: None,
            container_renderer: None,
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
            env_bg_pipeline: None,
            env_bg_bind_group: None,
            env_bg_bind_group_layout: None,
            env_params_buffer: None,
            sph_simulation: None,
            sdf_data: None,
            camera: Camera::default(),
            state,
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
            had_captures,
            stats_file,
            should_exit: false,
        }
    }

    fn simulation_substep_dt(&self) -> f32 {
        self.state.simulation.substep_dt()
    }

    fn create_initial_particles(&self) -> Vec<crate::simulation::SphParticle3D> {
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
            gpu.config.format,
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

        // Load environment map (shared by SS + MC renderers)
        let (env_texture, env_view, env_sampler, sh_coefficients) = load_embedded_environment_map(
            &gpu.device,
            &gpu.queue,
            self.state.environment.hdr_selection,
        ).expect("Failed to load environment map");

        // Create marching cubes renderer (shares environment map)
        let mc_renderer = MarchingCubesRenderer::new(
            &gpu.device,
            gpu.config.format,
            &env_view,
            &env_sampler,
            gpu.config.width,
            gpu.config.height,
            self.state.quality.msaa.as_u32(),
            self.state.rendering.mc_grid_resolution.grid_size(),
        );

        // Create wireframe renderer for container visualization
        let wireframe_renderer = WireframeRenderer::new(
            &gpu.device,
            gpu.config.format,
            &camera_params,
            &container_geom,
        );

        // Create opaque pool container renderer
        let pool_style = GpuPoolStyle::from_config(
            &self.state.container,
            self.state.lighting.sun_direction_normalized(),
        );
        let gpu_sh = GpuShCoefficients { coeffs: sh_coefficients.coeffs };
        let container_renderer = ContainerRenderer::new(
            &gpu.device,
            gpu.config.format,
            &camera_params,
            &container_geom,
            &pool_style,
            &self.state.container,
            self.state.quality.msaa.as_u32(),
            self.state.sph.kernel_radius,
            &gpu_sh,
        );

        // Create rigid body renderer + fallback depth texture
        let rb_render_params = self.state.rigid_body.to_gpu_render(
            self.state.lighting.sun_direction_normalized(),
        );
        let rigid_body_renderer = RigidBodyRenderer::new(
            &gpu.device,
            &gpu.queue,
            gpu.config.format,
            &camera_params,
            &rb_render_params,
            self.state.quality.msaa.as_u32(),
        );
        let rigid_body_depth_view = create_depth_texture(
            &gpu.device, gpu.config.width, gpu.config.height,
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
            gpu.config.format,
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

        let env_params_gpu = self.state.environment.to_gpu_params();
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
                    format: gpu.config.format,
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
            gpu.config.format,
            &env_view,
            &env_sampler,
            &camera_params,
            &self.state.lighting.to_gpu_params(),
            &crate::render::marching_cubes::GpuWaterParams::default(),
            &gpu_sh,
            gpu.config.width,
            gpu.config.height,
        );

        // Wire the MC front depth into the foam splat pass (depth-aware foam)
        {
            let mut spray_renderer = spray_renderer;
            spray_renderer.set_depth_view(
                &gpu.device,
                spray_system.spray_buffer(),
                mc_renderer.front_depth_view(),
            );
            self.spray_renderer = Some(spray_renderer);
        }

        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.mc_renderer = Some(mc_renderer);
        self.ss_renderer = Some(ss_renderer);
        self.wireframe_renderer = Some(wireframe_renderer);
        self.container_renderer = Some(container_renderer);
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
        self.current_hdr = self.state.environment.hdr_selection;
        self.env_bg_pipeline = Some(env_bg_pipeline);
        self.env_bg_bind_group = Some(env_bg_bind_group);
        self.env_bg_bind_group_layout = Some(env_bg_bind_group_layout);
        self.env_params_buffer = Some(env_params_buffer);
        self.sph_simulation = Some(sph_simulation);
        self.sdf_data = sdf_data;
        self.egui_winit = Some(egui_winit);
        self.egui_renderer = Some(egui_renderer);
    }

    fn reset_simulation(&mut self) {
        // Reset rigid body velocity and rotation (keep position)
        self.state.rigid_body.velocity = [0.0; 3];
        self.state.rigid_body.angular_velocity = [0.0; 3];
        self.state.rigid_body.orientation = [0.0, 0.0, 0.0, 1.0];

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
                gpu.config.format,
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
                    gpu.config.format,
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
                }
                self.spray_renderer = Some(spray_renderer);
                self.spray_system = Some(spray_system);
            }
            self.state.runtime.frame_count = 0;
        }
    }

    fn reset_defaults(&mut self) {
        self.state.simulation.reset_defaults();
        self.state.sph.reset_defaults();
        self.state.rendering.reset_defaults();
        self.state.camera.reset_defaults();
        self.state.lighting.reset_defaults();
        self.state.container.reset_defaults();
        self.state.rigid_body.reset_defaults();
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
            let response = egui_winit.on_window_event(&self.window.as_ref().unwrap(), &event);
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
                    // MC resize recreated the front depth — rebind it in the
                    // foam splat pass
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
                    }
                    if let Some(ss_renderer) = &mut self.ss_renderer {
                        ss_renderer.resize(&gpu.device, env_view, env_sampler, new_size.width, new_size.height);
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

    fn reload_environment_map(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let selection = self.state.environment.hdr_selection;

        let (env_texture, env_view, env_sampler, sh_coefficients) = load_embedded_environment_map(
            &gpu.device,
            &gpu.queue,
            selection,
        ).expect("Failed to load environment map");

        // Rebuild MC renderer bind groups and update SH coefficients
        if let Some(mc_renderer) = &mut self.mc_renderer {
            mc_renderer.rebuild_env_bind_groups(&gpu.device, &env_view, &env_sampler);
            let gpu_sh = GpuShCoefficients { coeffs: sh_coefficients.coeffs };
            mc_renderer.update_sh_coefficients(&gpu.queue, &gpu_sh);
        }

        // Rebuild SS renderer bind groups and update SH coefficients
        if let Some(ss_renderer) = &mut self.ss_renderer {
            ss_renderer.rebuild_env_bind_groups(&gpu.device, &env_view, &env_sampler);
            let gpu_sh = GpuShCoefficients { coeffs: sh_coefficients.coeffs };
            ss_renderer.update_sh_coefficients(&gpu.queue, &gpu_sh);
        }

        // Update container renderer SH coefficients
        if let Some(container_r) = &self.container_renderer {
            let gpu_sh = GpuShCoefficients { coeffs: sh_coefficients.coeffs };
            container_r.update_sh_coefficients(&gpu.queue, &gpu_sh);
        }

        self.sh_coefficients = Some(sh_coefficients);

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
                let pool_style = GpuPoolStyle::from_config(
                    &self.state.container,
                    self.state.lighting.sun_direction_normalized(),
                );
                container_r.update_container_geometry(&gpu.queue, &container_geom);
                container_r.update_pool_style(&gpu.queue, &pool_style);
                container_r.update_camera(&gpu.queue, &self.camera.to_gpu_params());
                container_r.maybe_rebuild_mesh(&gpu.device, &self.state.container, self.state.sph.kernel_radius);
            }

            // Update gravity (based on tilt)
            let gravity = self.state.simulation.to_gpu_gravity();
            sph_sim.update_gravity(&gpu.queue, &gravity);

            sph_sim.update_mouse_force(&gpu.queue, &mouse_force);

            // Update rigid body
            let rb_params = self.state.rigid_body.to_gpu_rigid_body(self.state.sph.wall_stiffness);
            sph_sim.update_rigid_body(&gpu.queue, &rb_params);

            // Update camera
            let camera_params = self.camera.to_gpu_params();
            renderer.update_camera(&gpu.queue, &camera_params);
            if let Some(wireframe) = &self.wireframe_renderer {
                wireframe.update_camera(&gpu.queue, &camera_params);
            }
            if let Some(rb_renderer) = &mut self.rigid_body_renderer {
                rb_renderer.update_camera(&gpu.queue, &camera_params);
                let rb_render = self.state.rigid_body.to_gpu_render(
                    self.state.lighting.sun_direction_normalized(),
                );
                rb_renderer.update_params(&gpu.queue, &rb_render);
                rb_renderer.set_shape(self.state.rigid_body.shape);
                rb_renderer.set_vertex_count(self.state.rigid_body.shape.vertex_count());
            }
            if let Some(spray_renderer) = &self.spray_renderer {
                spray_renderer.update_camera(&gpu.queue, &camera_params);
                spray_renderer.update_params(&gpu.queue, &self.build_spray_render_params());
                spray_renderer.update_light(&gpu.queue, &self.state.lighting.to_gpu_params());
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
        // In MC mode foam renders as a screen-space density field; the sprite
        // pass then draws only spray streaks and bubbles
        let foam_as_field = self.state.rendering.render_mode == FluidRenderMode::MarchingCubes;
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
        if self.launch.is_automated() {
            if !self.state.simulation.paused {
                self.state.runtime.time_elapsed += frame_dt;
            }
        } else {
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
        self.state.container.update_tilt(frame_dt);

        // Run SPH simulation if not paused (multiple sub-steps for stability)
        // Note: Grid simulation manages its own command encoding/submission
        let num_substeps = self.state.simulation.substeps;
        let substep_dt = self.simulation_substep_dt();
        if !self.state.simulation.paused {
            if let Some(sph_sim) = &mut self.sph_simulation {
                // Clear accumulator once, then accumulate over all sub-steps
                sph_sim.clear_rigid_body_accum(&gpu.queue);
                for _ in 0..num_substeps {
                    sph_sim.step(&gpu.device, &gpu.queue);
                }
                // Read back total accumulated rigid body forces
                sph_sim.read_rigid_body_accum(&gpu.device);
            }

            // Run spray system after SPH completes
            if self.state.spray.enabled && self.spray_system.is_some() {
                let needs_reset = !self.spray_prev_enabled;
                self.state.runtime.frame_count = self.state.runtime.frame_count.wrapping_add(1);
                let spray_params = self.build_spray_params(self.state.runtime.frame_count);

                let spray_sys = self.spray_system.as_mut().unwrap();
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
        self.spray_prev_enabled = self.state.spray.enabled;

        // Advance the deterministic frame clock (drives captures and stats)
        let stepped = !self.state.simulation.paused;
        if stepped {
            self.sim_frame_index += 1;
            self.sim_time += frame_dt as f64;
        }

        // Prepare a swapchain readback if a capture is due this frame
        let mut capture_due = false;
        while self
            .pending_captures
            .front()
            .is_some_and(|&f| self.sim_frame_index >= f)
        {
            self.pending_captures.pop_front();
            capture_due = true;
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

        // Integrate rigid body on CPU
        if self.state.rigid_body.enabled && !self.state.rigid_body.held && !self.state.simulation.paused {
            if let Some(sph_sim) = &self.sph_simulation {
                let accum = sph_sim.rigid_body_accum();
                integrate_rigid_body(
                    &mut self.state.rigid_body,
                    &self.state.container,
                    substep_dt,
                    num_substeps,
                    self.state.simulation.gravity_vector(),
                    accum,
                );
            }
        }

        // Clamp held rigid body to container (physics mode clamps in integrate_rigid_body)
        if self.state.rigid_body.enabled && self.state.rigid_body.held {
            clamp_rigid_body_to_container(
                &mut self.state.rigid_body,
                &self.state.container,
                false,
            );
        }

        // Determine render target (post-process intermediate or direct to screen)
        let post_process_enabled = self.state.post_process.enabled;
        let render_target = if post_process_enabled {
            if let Some(pp) = &self.post_process_renderer {
                pp.scene_view()
            } else {
                &view
            }
        } else {
            &view
        };

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
                            let env_params = self.state.environment.to_gpu_params();
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
                    // Screen-space fluid rendering with narrow-range depth filter
                    if let Some(ss_renderer) = &self.ss_renderer {
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
                            foam_coverage: 1.0,
                            aeration_strength: 1.0,
                        };
                        ss_renderer.update_water_params(&gpu.queue, &water_params);
                        let env_params = self.state.environment.to_gpu_params();
                        ss_renderer.update_env_params(&gpu.queue, &env_params);

                        // Scene objects go into the SS background pass (refraction +
                        // depth-aware composite), mirroring the MC renderer.
                        let rb_for_ss = if self.state.rigid_body.enabled {
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

                        let ss_radius = self.state.sph.kernel_radius * self.state.rendering.ss_radius_scale;
                        let particle_spacing = self.state.sph.kernel_radius * 0.6;
                        ss_renderer.render(
                            &gpu.device,
                            &gpu.queue,
                            &mut encoder,
                            render_target,
                            sph_sim.particle_buffer(),
                            sph_sim.num_particles(),
                            &camera_params,
                            ss_radius,
                            particle_spacing,
                            self.state.rendering.ss_filter_size,
                            self.state.rendering.ss_filter_iterations,
                            self.camera.fov,
                            rb_for_ss,
                            spray_for_ss,
                            container_for_ss,
                        );
                        fluid_depth_view = Some(ss_renderer.depth_view());
                    }
                }
                FluidRenderMode::MarchingCubes => {
                    // Marching cubes surface mesh rendering
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
                            self.state.spray.foam_coverage,
                            self.state.spray.aeration_strength,
                        );
                        let env_params = self.state.environment.to_gpu_params();
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
                        {
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
                        }

                        let iso_value = self.state.rendering.compute_iso_value(self.state.sph.kernel_radius);
                        let blur_radius = self.state.rendering.mc_blur_radius;
                        mc_renderer.update_params(
                            &gpu.queue,
                            self.state.sph.kernel_radius * self.state.rendering.mc_density_radius_scale,
                            iso_value,
                            sph_sim.num_particles(),
                            blur_radius,
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
                        // Pass rigid body renderer into MC pass for proper MSAA depth testing
                        let rb_for_mc = if self.state.rigid_body.enabled {
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

        // Render rigid body cube with depth testing against fluid surface
        // (MC and SS modes handle this inside their own passes above)
        if self.state.rigid_body.enabled && self.state.rendering.render_mode == FluidRenderMode::Particles {
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

        // Apply post-processing if enabled
        if post_process_enabled {
            if let Some(pp) = &self.post_process_renderer {
                let pp_params = self.state.post_process.to_gpu_params();
                pp.update_params(&gpu.queue, &pp_params);
                pp.render(&mut encoder, &view, self.state.post_process.bloom_enabled, self.state.post_process.streaks_enabled, self.state.quality.fxaa_enabled);
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

        // Read back marching cubes vertex count for next frame
        if self.state.rendering.render_mode == FluidRenderMode::MarchingCubes {
            if let Some(mc_renderer) = &mut self.mc_renderer {
                mc_renderer.read_vertex_count(&gpu.device);
            }
        }

        // Finish any pending capture (map readback, write PNG)
        if let Some((buffer, padded_bytes_per_row)) = capture {
            self.save_capture(&buffer, padded_bytes_per_row);
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
            let row = format!(
                "{},{:.4},{},{},{},{},{},{},{:.1},{:.4},{:.4}\n",
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
            let file = self.stats_file.as_mut().unwrap();
            let _ = file.write_all(row.as_bytes());
            let _ = file.flush();
        }

        // Automation exit: leave once every requested milestone is reached
        if self.launch.is_automated() && !self.launch.stay && !self.should_exit {
            let captures_done = self.pending_captures.is_empty();
            let exit_frame_reached = self
                .launch
                .exit_after
                .is_none_or(|n| self.sim_frame_index >= n);
            let has_milestone = self.had_captures || self.launch.exit_after.is_some();
            if has_milestone && captures_done && exit_frame_reached {
                println!(
                    "Automation milestones reached at frame {} — exiting",
                    self.sim_frame_index
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
    fn save_capture(&self, buffer: &wgpu::Buffer, padded_bytes_per_row: u32) {
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

        let path = self
            .launch
            .out_dir
            .join(format!("frame_{:05}.png", self.sim_frame_index));
        match image::RgbaImage::from_raw(width, height, pixels) {
            Some(img) => match img.save(&path) {
                Ok(()) => println!("Captured {}", path.display()),
                Err(e) => eprintln!("error: failed to save capture {}: {e}", path.display()),
            },
            None => eprintln!("error: capture buffer size mismatch"),
        }
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

