//! The fluid render modes: what `update_and_render` draws into the HDR
//! scene buffer. Each mode also draws the scene objects that share its
//! passes (container, rigid bodies, whitewater).

use super::App;
use crate::render::marching_cubes::GpuWaterParams;
use crate::state::{BackgroundMode, ContainerStyle, FluidRenderMode, GpuContainerGeometry};

impl App {
    /// Render fluid or particles based on render mode
    pub(super) fn render_fluid(&mut self, encoder: &mut wgpu::CommandEncoder, render_target: &wgpu::TextureView) {
        match self.state.rendering.render_mode {
            FluidRenderMode::Particles => self.render_particles(encoder, render_target),
            FluidRenderMode::ScreenSpace => self.render_screen_space(encoder, render_target),
            FluidRenderMode::MarchingCubes => self.render_marching_cubes(encoder, render_target),
        }
    }

    /// Debug billboards: backdrop, particles, then whitewater and rigid
    /// bodies depth-tested against them
    fn render_particles(&self, encoder: &mut wgpu::CommandEncoder, render_target: &wgpu::TextureView) {
        let gpu = self.gpu.as_ref().unwrap();
        let Some(sph_sim) = &self.sph_simulation else {
            return;
        };
        let mut fluid_depth_view: Option<&wgpu::TextureView> = None;

        // Particle rendering (individual spheres)
        let use_env = self.state.environment.background_mode == BackgroundMode::Environment;

        // Render environment background if HDR mode
        if use_env {
            if let Some(env_background) = &self.env_background {
                let env_params = self.state.environment.to_gpu_params(&self.ground_staging());
                env_background.update_params(&gpu.queue, &env_params);

                // Update camera in particle renderer (needed for inv matrices in env shader)
                if let Some(renderer) = &self.renderer {
                    let camera_params = self.camera.to_gpu_params();
                    renderer.update_camera(&gpu.queue, &camera_params);
                }

                env_background.encode(encoder, render_target);
            }
        }

        if let Some(renderer) = &self.renderer {
            renderer.render(
                encoder,
                render_target,
                sph_sim.particle_buffer(),
                sph_sim.num_particles(),
                &self.state.environment.background_color,
                !use_env, // clear_background: only clear if solid color mode
            );
            // Share particle depth buffer for rigid body occlusion
            fluid_depth_view = Some(renderer.depth_view());
        }

        // Whitewater and rigid bodies on top, depth-tested against the
        // particles (MC and SS modes draw both inside their own passes).
        // Without a particle depth buffer the fallback depth is cleared, so
        // everything passes.
        let (depth_view, depth_load) = match fluid_depth_view {
            Some(view) => (view, wgpu::LoadOp::Load),
            None => (self.rigid_body_depth_view.as_ref().unwrap(), wgpu::LoadOp::Clear(1.0)),
        };
        if self.state.spray.enabled {
            if let Some(spray_renderer) = &self.spray_renderer {
                let mut render_pass =
                    begin_overlay_pass(encoder, "Spray Render Pass", render_target, depth_view, depth_load);
                spray_renderer.render(&mut render_pass);
            }
        }
        if self.state.rigid_bodies.iter().any(|b| b.enabled) {
            if let Some(rb_renderer) = &self.rigid_body_renderer {
                let mut render_pass =
                    begin_overlay_pass(encoder, "Rigid Body Render Pass", render_target, depth_view, depth_load);
                rb_renderer.render(&mut render_pass);
            }
        }
    }

    /// Screen-space fluid: splats, filtered depth, composite (scene objects
    /// go into its background pass)
    fn render_screen_space(&mut self, encoder: &mut wgpu::CommandEncoder, render_target: &wgpu::TextureView) {
        // Whitewater splat depth gate follows the active water
        // renderer's front depth (1 frame stale, same as MC)
        self.bind_spray_depth_gate(FluidRenderMode::ScreenSpace);

        let gpu = self.gpu.as_ref().unwrap();
        let Some(sph_sim) = &self.sph_simulation else {
            return;
        };

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
                    encoder,
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
            ss_renderer.update_water_params(&gpu.queue, &self.ss_water_params());
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
                    encoder,
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
                encoder,
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
        }
    }

    /// Marching cubes: field + mesh generation, whitewater field, caustics,
    /// then the water pass (scene objects go into its passes)
    fn render_marching_cubes(&mut self, encoder: &mut wgpu::CommandEncoder, render_target: &wgpu::TextureView) {
        // Rebind the whitewater splat depth gate to the MC front
        // depth after a mode switch or resize
        self.bind_spray_depth_gate(FluidRenderMode::MarchingCubes);

        // Marching cubes surface mesh rendering
        let Some(mc_geom) = self.update_marching_cubes() else {
            return;
        };

        let gpu = self.gpu.as_ref().unwrap();
        let Some(sph_sim) = &self.sph_simulation else {
            return;
        };
        let caustics_on = self.caustics_active();
        if let Some(mc_renderer) = &mut self.mc_renderer {
            mc_renderer.generate(
                encoder,
                &gpu.device,
                sph_sim.sorted_particle_buffer(),
                sph_sim.cell_starts_buffer(),
                sph_sim.cell_counts_buffer(),
                sph_sim.grid_params_buffer(),
                self.state.rendering.mc_blur_radius,
                sph_sim.num_particles(),
                self.state.rendering.mc_anisotropy,
                self.state.rendering.mc_calm_smoothing,
            );
            // Splat foam into the density field the water shader
            // composites (cleared even when spray is off so no
            // stale foam lingers on the surface)
            if let Some(spray_renderer) = &self.spray_renderer {
                spray_renderer.render_foam_density(
                    encoder,
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
                        encoder,
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
                encoder,
                render_target,
                &self.state.environment.background_color,
                rb_for_mc,
                spray_for_mc,
                container_for_mc,
            );
        }
    }

    /// Upload this frame's parameters to the MC renderer. Returns the
    /// container geometry it was given (clip margin from the MC cell size),
    /// which the caustics pass uses too.
    fn update_marching_cubes(&mut self) -> Option<GpuContainerGeometry> {
        let gpu = self.gpu.as_ref().unwrap();
        let num_particles = self.sph_simulation.as_ref()?.num_particles();
        let env_params = self.state.environment.to_gpu_params(&self.ground_staging());
        let mc_renderer = self.mc_renderer.as_mut()?;

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
            self.state.rendering.mc_silhouette_exit.as_u32(),
            self.state.rendering.mc_front_face_exit,
            &env_params,
            self.state.rendering.mc_filtered_lookup,
            self.state.rendering.mc_volume_trace,
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

        // Update container geometry for the MC renderer (clip_margin from MC cell_size).
        // clip_enabled (pool only) gates fragment discards and the volume trace's wall
        // inset; the density field is bounded at the walls in both styles.
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
            num_particles,
            blur_radius,
        );
        mc_renderer.update_calm_smoothing(
            &gpu.queue,
            self.state.rendering.mc_calm_smoothing,
            self.state.sph.kernel_radius,
            iso_value,
        );
        mc_renderer.update_voxel_normals(
            &gpu.queue,
            self.state.rendering.mc_normal_denoise,
            self.state.rendering.mc_calm_smoothing,
        );
        mc_renderer.update_wall_bound(
            &gpu.queue,
            self.state.sph.kernel_radius,
            self.state.container.style == ContainerStyle::OpaquePool,
        );
        mc_renderer.update_aniso_params(
            &gpu.queue,
            self.state.rendering.mc_anisotropy,
            self.state.rendering.mc_anisotropy_strength,
            self.state.sph.kernel_radius,
            self.state.sph.kernel_radius * self.state.rendering.mc_density_radius_scale,
        );
        Some(mc_geom)
    }

    /// Point the whitewater splat depth gate at the front depth of the water
    /// renderer `mode` uses (after a mode switch, a resize or a spray rebuild)
    fn bind_spray_depth_gate(&mut self, mode: FluidRenderMode) {
        if self.spray_depth_bound == Some(mode) {
            return;
        }
        let gpu = self.gpu.as_ref().unwrap();
        let front_depth = match mode {
            FluidRenderMode::ScreenSpace => self.ss_renderer.as_ref().map(|r| r.front_depth_view()),
            FluidRenderMode::MarchingCubes => self.mc_renderer.as_ref().map(|r| r.front_depth_view()),
            FluidRenderMode::Particles => None,
        };
        if let (Some(spray_renderer), Some(front_depth), Some(spray_system)) =
            (&mut self.spray_renderer, front_depth, &self.spray_system)
        {
            spray_renderer.set_depth_view(&gpu.device, spray_system.spray_buffer(), front_depth);
            self.spray_depth_bound = Some(mode);
        }
    }

    /// Water shading parameters for the screen-space composite (it shares the
    /// MC water shader's struct; the refraction fields it has no use for stay 0)
    fn ss_water_params(&self) -> GpuWaterParams {
        let use_env = self.state.environment.background_mode == BackgroundMode::Environment;
        GpuWaterParams {
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
            silhouette_exit: 0,
            front_exit: 0,
            ground_enabled: 0,
            ground_y: 0.0,
            ground_capture_height: 0.0,
            filtered_lookup: 0,
            volume_trace: 0,
            _pad_g: [0; 2],
        }
    }
}

/// A pass that draws on top of the scene buffer, depth-tested against
/// `depth_view`
fn begin_overlay_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    label: &str,
    render_target: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    depth_load: wgpu::LoadOp<f32>,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
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
                load: depth_load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    })
}
