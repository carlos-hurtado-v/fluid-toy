//! Application state and event handling

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
use crate::gui::{self, GuiAction};
use crate::render::{Camera, FluidRenderer, GpuContainerParams, ParticleRenderer3D, PostProcessRenderer, ScreenSpaceFluidRenderer, WireframeRenderer};
use crate::simulation::{SphParticle3D, SphSimulation3DGrid};
use crate::state::{AppState, FluidRenderMode, GpuMouseForce};

pub struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    renderer: Option<ParticleRenderer3D>,
    fluid_renderer: Option<FluidRenderer>,
    ss_renderer: Option<ScreenSpaceFluidRenderer>,
    wireframe_renderer: Option<WireframeRenderer>,
    post_process_renderer: Option<PostProcessRenderer>,
    sph_simulation: Option<SphSimulation3DGrid>,
    camera: Camera,
    state: AppState,
    // Frame timing
    last_frame_time: Instant,
    // Mouse state for camera control (left button)
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    // Mouse state for force interaction (right button)
    right_mouse_pressed: bool,
    current_mouse_pos: (f64, f64),
    // Spawn state (middle button - continuous while held)
    middle_mouse_pressed: bool,
    // egui
    egui_ctx: egui::Context,
    egui_winit: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            renderer: None,
            fluid_renderer: None,
            ss_renderer: None,
            wireframe_renderer: None,
            post_process_renderer: None,
            sph_simulation: None,
            camera: Camera::default(),
            state: AppState::default(),
            last_frame_time: Instant::now(),
            mouse_pressed: false,
            last_mouse_pos: None,
            right_mouse_pressed: false,
            current_mouse_pos: (0.0, 0.0),
            middle_mouse_pressed: false,
            egui_ctx: egui::Context::default(),
            egui_winit: None,
            egui_renderer: None,
        }
    }

    fn initialize(&mut self, window: Arc<Window>) {
        let gpu = pollster::block_on(GpuContext::new(window.clone()));

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
        let spacing = self.state.sph.kernel_radius * 0.6;
        let particles = create_sph_particle_block(spacing, &self.state);
        self.state.runtime.particle_count = particles.len() as u32;

        // Create 3D SPH simulation (O(n²) version - works correctly)
        let sph_params = self.state.sph.to_gpu_params_3d(
            self.state.runtime.particle_count,
            self.state.simulation.delta_time,
        );
        let bounds_params = self.state.container.to_gpu_bounds_3d(
                self.state.sph.wall_stiffness,
                self.state.rendering.visual_margin(),
            );
        let sph_simulation = SphSimulation3DGrid::new(
            &gpu.device,
            &gpu.queue,
            &particles,
            sph_params,
            bounds_params,
            self.state.simulation.max_particles,
        );

        // Create fluid renderer (screen-space, kept for comparison)
        let fluid_renderer = FluidRenderer::new(
            &gpu.device,
            gpu.config.format,
            &camera_params,
            gpu.config.width,
            gpu.config.height,
        );

        // Create screen-space fluid renderer (photorealistic)
        let ss_renderer = ScreenSpaceFluidRenderer::new(
            &gpu.device,
            &gpu.queue,
            gpu.config.format,
            &camera_params,
            gpu.config.width,
            gpu.config.height,
        );

        // Create wireframe renderer for container visualization
        let container_params = GpuContainerParams::from_config(&self.state.container);
        let wireframe_renderer = WireframeRenderer::new(
            &gpu.device,
            gpu.config.format,
            &camera_params,
            &container_params,
        );

        // Create post-process renderer
        let post_process_params = self.state.post_process.to_gpu_params();
        let post_process_renderer = PostProcessRenderer::new(
            &gpu.device,
            gpu.config.format,
            gpu.config.width,
            gpu.config.height,
            &post_process_params,
        );

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

        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.fluid_renderer = Some(fluid_renderer);
        self.ss_renderer = Some(ss_renderer);
        self.wireframe_renderer = Some(wireframe_renderer);
        self.post_process_renderer = Some(post_process_renderer);
        self.sph_simulation = Some(sph_simulation);
        self.egui_winit = Some(egui_winit);
        self.egui_renderer = Some(egui_renderer);
    }

    fn reset_simulation(&mut self) {
        if let Some(gpu) = &self.gpu {
            let spacing = self.state.sph.kernel_radius * 0.6;
            let particles = create_sph_particle_block(spacing, &self.state);
            self.state.runtime.particle_count = particles.len() as u32;

            let sph_params = self.state.sph.to_gpu_params_3d(
                self.state.runtime.particle_count,
                self.state.simulation.delta_time,
            );
            let bounds_params = self.state.container.to_gpu_bounds_3d(
                self.state.sph.wall_stiffness,
                self.state.rendering.visual_margin(),
            );
            self.sph_simulation = Some(SphSimulation3DGrid::new(
                &gpu.device,
                &gpu.queue,
                &particles,
                sph_params,
                bounds_params,
                self.state.simulation.max_particles,
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
        }
    }

    fn reset_defaults(&mut self) {
        self.state.simulation.reset_defaults();
        self.state.sph.reset_defaults();
        self.state.rendering.reset_defaults();
        self.state.camera.reset_defaults();
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

        let window_attrs = Window::default_attributes()
            .with_title("Fluid Toy")
            .with_inner_size(winit::dpi::LogicalSize::new(1000, 700));

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
                    if let Some(ss_renderer) = &mut self.ss_renderer {
                        ss_renderer.resize(&gpu.device, new_size.width, new_size.height);
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
                                self.state.container.tilt_z -= tilt_speed;
                            }
                            KeyCode::ArrowRight => {
                                self.state.container.tilt_z += tilt_speed;
                            }
                            KeyCode::ArrowUp => {
                                self.state.container.tilt_x -= tilt_speed;
                            }
                            KeyCode::ArrowDown => {
                                self.state.container.tilt_x += tilt_speed;
                            }
                            // Home to reset tilt AND camera
                            KeyCode::Home => {
                                self.state.container.tilt_x = 0.0;
                                self.state.container.tilt_z = 0.0;
                                self.camera.reset();
                            }
                            // End to flip upside down
                            KeyCode::End => {
                                self.state.container.tilt_x = std::f32::consts::PI;
                                self.state.container.tilt_z = 0.0;
                            }
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.update_and_render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl App {
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
        if let (Some(sph_sim), Some(renderer)) = (&self.sph_simulation, &self.renderer) {
            let sph_params = self.state.sph.to_gpu_params_3d(
                self.state.runtime.particle_count,
                self.state.simulation.delta_time,
            );
            sph_sim.update_sph_params(&gpu.queue, &sph_params);

            let bounds_params = self.state.container.to_gpu_bounds_3d(
                self.state.sph.wall_stiffness,
                self.state.rendering.visual_margin(),
            );
            sph_sim.update_bounds_params(&gpu.queue, &bounds_params);

            // Update wireframe container visualization
            if let Some(wireframe) = &self.wireframe_renderer {
                let container_params = GpuContainerParams::from_config(&self.state.container);
                wireframe.update_container(&gpu.queue, &container_params);
            }

            // Update gravity (based on tilt)
            let gravity = self.state.simulation.to_gpu_gravity();
            sph_sim.update_gravity(&gpu.queue, &gravity);

            // Update mouse force
            let mouse_force = if self.right_mouse_pressed {
                // Cast ray from camera through mouse position
                let screen_width = gpu.config.width as f32;
                let screen_height = gpu.config.height as f32;
                let (ray_origin, ray_dir) = self.camera.screen_to_ray(
                    self.current_mouse_pos.0 as f32,
                    self.current_mouse_pos.1 as f32,
                    screen_width,
                    screen_height,
                );

                // Intersect with horizontal plane at y = -0.6 (where fluid settles)
                // If that fails, try y = 0 (center), then use origin as fallback
                let hit = self.camera.ray_plane_intersection(ray_origin, ray_dir, -0.6)
                    .or_else(|| self.camera.ray_plane_intersection(ray_origin, ray_dir, 0.0))
                    .unwrap_or([0.0, 0.0, 0.0]);

                GpuMouseForce {
                    position: hit,
                    radius: 0.5,
                    strength: 30.0,
                    is_active: 1,
                    _padding: [0.0; 2],
                }
            } else {
                GpuMouseForce::default()
            };
            sph_sim.update_mouse_force(&gpu.queue, &mouse_force);

            // Update camera
            let camera_params = self.camera.to_gpu_params();
            renderer.update_camera(&gpu.queue, &camera_params);
            if let Some(wireframe) = &self.wireframe_renderer {
                wireframe.update_camera(&gpu.queue, &camera_params);
            }

            let render_params = self.state.rendering.to_gpu_params();
            renderer.update_params(&gpu.queue, &render_params);
        }

        // Handle particle spawning (middle mouse held = continuous stream)
        if self.middle_mouse_pressed {
            if let Some(sph_sim) = &mut self.sph_simulation {
                let screen_width = gpu.config.width as f32;
                let screen_height = gpu.config.height as f32;
                let (ray_origin, ray_dir) = self.camera.screen_to_ray(
                    self.current_mouse_pos.0 as f32,
                    self.current_mouse_pos.1 as f32,
                    screen_width,
                    screen_height,
                );

                // Find spawn position via ray-plane intersection
                let spawn_pos = self.camera.ray_plane_intersection(ray_origin, ray_dir, -0.5)
                    .or_else(|| self.camera.ray_plane_intersection(ray_origin, ray_dir, 0.0))
                    .unwrap_or([0.0, 0.0, 0.0]);

                // Spawn a small batch each frame for continuous stream effect
                let spawned = sph_sim.spawn_particles(&gpu.queue, spawn_pos, 10, 0.08);
                self.state.runtime.particle_count = sph_sim.num_particles();

                if spawned > 0 {
                    // Update SPH params with new particle count
                    let sph_params = self.state.sph.to_gpu_params_3d(
                        self.state.runtime.particle_count,
                        self.state.simulation.delta_time,
                    );
                    sph_sim.update_sph_params(&gpu.queue, &sph_params);
                }
            }
        }

        // Get current frame texture
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

        // Run SPH simulation if not paused (multiple sub-steps for stability)
        // Note: Grid simulation manages its own command encoding/submission
        if !self.state.simulation.paused {
            if let Some(sph_sim) = &self.sph_simulation {
                // Run 2 sub-steps per frame (matches reference)
                for _ in 0..2 {
                    sph_sim.step(&gpu.device, &gpu.queue);
                }
            }
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

        // Render fluid or particles based on render mode
        if let Some(sph_sim) = &self.sph_simulation {
            match self.state.rendering.render_mode {
                FluidRenderMode::ScreenSpace => {
                    // Screen-space fluid rendering (photorealistic)
                    if let Some(ss_renderer) = &self.ss_renderer {
                        let camera_params = self.camera.to_gpu_params();
                        ss_renderer.update_camera(&gpu.queue, &camera_params);
                        // Identity scene rotation (camera orbits, scene stays fixed)
                        let identity = [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ];
                        ss_renderer.update_params(
                            &gpu.queue,
                            self.state.rendering.particle_radius,
                            gpu.config.width,
                            gpu.config.height,
                            &camera_params,
                            &identity,
                        );
                        ss_renderer.render(
                            &mut encoder,
                            render_target,
                            sph_sim.particle_buffer(),
                            sph_sim.num_particles(),
                            &self.state.rendering.background_color,
                        );
                    }
                }
                FluidRenderMode::Particles => {
                    // Particle rendering (individual spheres)
                    if let Some(renderer) = &self.renderer {
                        renderer.render(
                            &mut encoder,
                            render_target,
                            sph_sim.particle_buffer(),
                            sph_sim.num_particles(),
                            &self.state.rendering.background_color,
                        );
                    }
                }
            }
        }

        // Apply post-processing if enabled
        if post_process_enabled {
            if let Some(pp) = &self.post_process_renderer {
                let pp_params = self.state.post_process.to_gpu_params();
                pp.update_params(&gpu.queue, &pp_params);
                pp.render(&mut encoder, &view, self.state.post_process.bloom_enabled, self.state.post_process.streaks_enabled);
            }
        }

        // Render wireframe container visualization (on top of fluid, below UI)
        if let Some(wireframe) = &self.wireframe_renderer {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Wireframe Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep fluid rendering
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

        // Render egui
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

        // Handle GUI actions after rendering
        match gui_action {
            GuiAction::ResetSimulation => self.reset_simulation(),
            GuiAction::ResetDefaults => self.reset_defaults(),
            GuiAction::None => {}
        }
    }
}

/// Create a cube of particles for testing (controlled count)
fn create_sph_particle_block(spacing: f32, state: &AppState) -> Vec<SphParticle3D> {
    let mut particles = Vec::new();

    // Create a cube based on initial_cube_size setting (N×N×N particles)
    let count = state.simulation.initial_cube_size;
    let size = (count as f32 - 1.0) * spacing;
    let half = size / 2.0;

    for y in 0..count {
        for z in 0..count {
            for x in 0..count {
                let px = -half + (x as f32) * spacing;
                let py = 0.2 + (y as f32) * spacing;  // Start above center
                let pz = -half + (z as f32) * spacing;
                // Small jitter to prevent perfectly aligned particles
                let jitter = 0.0005 * rand_f32();
                particles.push(SphParticle3D::new(px + jitter, py + jitter, pz + jitter));
            }
        }
    }

    particles
}

/// Simple pseudo-random float (not cryptographic, just for jitter)
fn rand_f32() -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    static mut SEED: u64 = 0;
    unsafe {
        SEED = SEED.wrapping_add(1);
        let mut hasher = DefaultHasher::new();
        (SEED, SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()).hash(&mut hasher);
        (hasher.finish() % 1000) as f32 / 1000.0
    }
}
