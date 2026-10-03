//! One frame: `update_and_render` runs the phases below in order. Each
//! phase is a method; a new one is added here (or in scene.rs /
//! automation.rs) and called from the list, not written inline.

use std::time::Instant;

use super::App;
use crate::gui::{self, GuiAction};
use crate::state::{integrate_rigid_body, ContainerStyle, FluidRenderMode, RigidBodyMotion};

/// What the GUI pass produced this frame
struct GuiFrame {
    /// Tessellated egui output, drawn on top of the finished frame
    tris: Vec<egui::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
    /// What the control panel asked for (handled after the frame is presented)
    action: GuiAction,
}

impl App {
    pub(super) fn update_and_render(&mut self) {
        // Early return if not initialized
        if self.gpu.is_none() || self.sph_simulation.is_none() || self.renderer.is_none() {
            return;
        }

        let (frame_dt, held) = self.tick_clock();
        let gui = self.run_gui();

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

        // Simulate, then encode what measures or follows the new state
        let stepped = self.step_simulation(frame_dt, held);
        self.encode_measurements(&mut encoder, stepped);
        let capture = self.begin_capture();
        self.integrate_rigid_bodies();

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
            .scene_view()
            .clone();

        // Render the fluid (and the scene objects that go with it) by render
        // mode, then everything that works on the finished scene
        self.render_fluid(&mut encoder, &render_target);
        self.encode_gtao(&mut encoder, post_process_enabled);
        self.encode_post_process(&mut encoder, &view, post_process_enabled, debug_view_on);
        self.encode_wireframe(&mut encoder, &view);
        self.encode_capture_copy(&mut encoder, &output.texture, &capture);
        self.encode_gui(&mut encoder, &view, &gui);

        self.gpu.as_ref().unwrap().queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // After the submit: readbacks, automation outputs, GUI requests
        self.read_back_measurements(stepped);
        self.finish_capture(capture);
        self.write_stats_row(stepped);
        self.check_automation_exit();
        self.handle_gui_action(gui.action);
    }

    /// FPS estimate and the visual clock. Returns (frame_dt, held): the
    /// simulated time one frame advances, and whether `--hold` is freezing
    /// everything that moves.
    fn tick_clock(&mut self) -> (f32, bool) {
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
        (frame_dt, held)
    }

    /// Run the control panel and upload the texture changes egui made
    fn run_gui(&mut self) -> GuiFrame {
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

        GuiFrame {
            tris,
            textures_delta: full_output.textures_delta,
            action: gui_action,
        }
    }

    /// Run the SPH substeps and the whitewater system (unless paused) and
    /// advance the deterministic clocks. Returns whether the simulation
    /// stepped this frame.
    fn step_simulation(&mut self, frame_dt: f32, held: bool) -> bool {
        let gpu = self.gpu.as_ref().unwrap();

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
        stepped
    }

    /// Passes over the post-integrate state: the wave gauges / extents and
    /// the surface foam map
    fn encode_measurements(&mut self, encoder: &mut wgpu::CommandEncoder, stepped: bool) {
        let gpu = self.gpu.as_ref().unwrap();

        // Measure fluid extents + probe heights on the post-integrate state
        if stepped {
            if let Some(probe) = self.probe_system.as_mut() {
                probe.collect(&gpu.device);
                probe.encode(&gpu.queue, encoder, self.state.runtime.particle_count);
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
                        encoder,
                        sph_sim.particle_buffer(),
                        spray_sys.spray_buffer(),
                        sph_sim.container_geom_buffer(),
                    );
                }
            }
        }
    }

    /// Integrate dynamic rigid bodies on CPU (Static/Kinematic bodies are
    /// pose-driven and intentionally skip the container clamp so obstacles
    /// can be embedded in walls/floor)
    fn integrate_rigid_bodies(&mut self) {
        let num_substeps = self.state.simulation.substeps;
        let substep_dt = self.simulation_substep_dt();
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
    }

    /// GTAO on the active water renderer's front depth (when ambient
    /// occlusion and post-processing are on)
    fn encode_gtao(&mut self, encoder: &mut wgpu::CommandEncoder, post_process_enabled: bool) {
        let gpu = self.gpu.as_ref().unwrap();

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
                        encoder,
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
    }

    fn encode_post_process(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        post_process_enabled: bool,
        debug_view_on: bool,
    ) {
        let gpu = self.gpu.as_ref().unwrap();

        // Post-process the HDR scene onto the screen; disabled = plain
        // passthrough (no exposure, tonemap or effects; clipped at 1)
        if let Some(pp) = &self.post_process_renderer {
            if post_process_enabled {
                let pp_params = self.state.post_process.to_gpu_params();
                pp.update_params(&gpu.queue, &pp_params);
                pp.render(encoder, view, self.state.post_process.bloom_enabled, self.state.post_process.streaks_enabled, self.state.quality.fxaa_enabled);
            } else {
                pp.update_params(&gpu.queue, &crate::state::PostProcessConfig::passthrough_gpu_params());
                pp.render(encoder, view, false, false, self.state.quality.fxaa_enabled && !debug_view_on);
            }
        }
    }

    fn encode_wireframe(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        // Render wireframe container visualization (on top of fluid, below UI)
        // Skip when using opaque pool style
        if let Some(wireframe) = &self.wireframe_renderer {
            if self.state.container.style == ContainerStyle::Wireframe {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Wireframe Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
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
    }

    /// Draw the control panel on top of the finished frame
    fn encode_gui(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, gui: &GuiFrame) {
        let gpu = self.gpu.as_ref().unwrap();

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
            encoder,
            &gui.tris,
            &screen_descriptor,
        );

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            egui_renderer.render(&mut render_pass, &gui.tris, &screen_descriptor);
        }

        // Cleanup egui textures
        for id in &gui.textures_delta.free {
            egui_renderer.free_texture(id);
        }
    }

    /// Readbacks that feed the GUI and the stats rows (after the submit)
    fn read_back_measurements(&mut self, stepped: bool) {
        let gpu = self.gpu.as_ref().unwrap();

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
    }

    fn handle_gui_action(&mut self, gui_action: GuiAction) {
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
