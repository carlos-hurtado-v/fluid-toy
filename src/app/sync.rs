//! Pushing config/GUI state to the GPU each frame: simulation and renderer
//! parameters, SH ambient, the environment map, plus what changes the state
//! on the way (scenario events, particle spawning).

use super::App;
use crate::render::container_renderer::{POOL_WALL_HEIGHT_FRACTION, WALL_THICKNESS};
use crate::render::environment::{load_embedded_environment_map, ShCoefficients};
use crate::render::{GpuPoolStyle, RigidBodyDraw};
use crate::state::{
    AppState, BackgroundMode, ContainerStyle, EnvironmentSun, FluidRenderMode, ForceMode, GpuMouseForce,
    GpuShCoefficients, GpuSprayParams, GpuSprayRenderParams, GroundStaging,
};

impl App {
    /// Fire scenario events whose time has come, once each per reset.
    /// Runs on the deterministic sim clock, so automation runs replay exactly.
    pub(super) fn pump_scenario_events(&mut self) {
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
    pub(super) fn apply_state_set(&mut self, path: &str, raw: &str) -> Result<(), String> {
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

    /// Push SH irradiance to all consumers when the ambient source changes.
    /// Environment mode uses the HDR map's coefficients; SolidColor mode uses
    /// a uniform SH of the background color, so diffuse ambient agrees with
    /// the visible surround instead of silently keeping the HDR sky's light.
    pub(super) fn refresh_sh_coefficients(&mut self) {
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
    pub(super) fn ground_staging(&self) -> GroundStaging {
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
    pub(super) fn ground_sky_share(&self) -> f32 {
        let Some(sun) = &self.hdr_sun else { return 1.0 };
        let c = &sun.sky_sh.coeffs;
        // SH irradiance on an upward-facing surface (n = +Y), per channel
        let e_up = |k: usize| 0.282095 * c[0][k] + 0.488603 * c[1][k] - 0.315392 * c[6][k] - 0.546274 * c[8][k];
        let luminance = |v: [f32; 3]| 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
        let sky = luminance([e_up(0), e_up(1), e_up(2)]).max(0.0);
        let sun_e = luminance(sun.irradiance) * sun.direction[1].max(0.0);
        if sky + sun_e > 0.0 { sky / (sky + sun_e) } else { 1.0 }
    }

    pub(super) fn reload_environment_map(&mut self) {
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
        if let (Some(env_background), Some(renderer)) = (&mut self.env_background, &self.renderer) {
            env_background.rebuild_bind_group(&gpu.device, renderer.camera_buffer(), &env_view, &env_sampler);
        }

        self.env_texture = Some(env_texture);
        self.env_view = Some(env_view);
        self.env_sampler = Some(env_sampler);
        self.current_hdr = selection;

        log::info!("Switched environment map to {:?}", selection);
    }

    pub(super) fn sync_gpu_state(&mut self) {
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
        let mouse_force = self.mouse_force();

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

    /// The right-mouse force for this frame (inactive unless the button is
    /// held; Explode fires once per click)
    fn mouse_force(&mut self) -> GpuMouseForce {
        if self.right_mouse_pressed {
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
        }
    }

    pub(super) fn build_spray_params(&self, frame_count: u32) -> GpuSprayParams {
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

    pub(super) fn build_spray_render_params(&self) -> GpuSprayRenderParams {
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

    pub(super) fn spawn_particles(&mut self) {
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
}
