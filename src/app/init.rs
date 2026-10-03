//! Building the simulation and the renderers: startup and resets.

use std::sync::Arc;

use winit::window::Window;

use super::{create_depth_texture, App};
use crate::gpu::GpuContext;
use crate::render::env_background::EnvBackground;
use crate::render::environment::load_embedded_environment_map;
use crate::render::mesh_loader::{self, SdfData};
use crate::render::{
    GpuCameraParams, CausticsRenderer, ContainerRenderer, GpuPoolStyle, GtaoRenderer, MarchingCubesRenderer, ParticleRenderer3D,
    PostProcessRenderer, RigidBodyRenderer, ScreenSpaceFluidRenderer, SprayRenderer, WireframeRenderer,
};
use crate::simulation::{create_particle_block, ProbeSystem, SphParticle3D, SphSimulation3DGrid, SpraySystem};
use crate::state::{ContainerStyle, FluidRenderMode, GpuContainerGeometry, GpuShCoefficients, RigidBodyMotion};

/// The duck mesh's SDF, for custom rigid body collision
fn load_duck_sdf() -> Option<SdfData> {
    match mesh_loader::load_embedded_duck() {
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
    }
}

impl App {
    pub(super) fn create_initial_particles(&self) -> Vec<crate::simulation::SphParticle3D> {
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
    pub(super) fn create_scenario_particles(&self) -> Vec<crate::simulation::SphParticle3D> {
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

    pub(super) fn initialize(&mut self, window: Arc<Window>) {
        // Automation runs render uncapped (no vsync wait between frames)
        let gpu = pollster::block_on(GpuContext::new(
            window.clone(),
            self.launch.is_automated(),
            self.launch.profile_path.is_some(),
        ));
        if let Some(path) = &self.launch.profile_path {
            crate::gpu::profile::init(&gpu.device, &gpu.queue, path.clone());
        }

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
        self.sdf_data = load_duck_sdf();

        // Create 3D SPH simulation and its wave gauges
        let container_geom = self.sim_container_geometry();
        let sph_simulation = self.create_simulation(&gpu, &particles);
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

        // Create the scene objects around the water (wireframe box, rigid
        // bodies, caustics, opaque pool) and the fallback depth texture
        let gpu_sh = GpuShCoefficients { coeffs: sh_coefficients.coeffs };
        let (wireframe_renderer, rigid_body_renderer, caustics_renderer, container_renderer) =
            self.create_scene_renderers(&gpu, &camera_params, &container_geom, &mc_renderer, &gpu_sh);
        let rigid_body_depth_view = create_depth_texture(
            &gpu.device, gpu.config.width, gpu.config.height,
        );

        // Create whitewater system and renderer, with the MC front depth wired
        // into the foam splat pass (depth-aware foam); the per-frame sync
        // rebinds to the SS depth when that mode is active
        let (spray_system, spray_renderer) =
            self.create_whitewater(&gpu, &sph_simulation, Some(mc_renderer.front_depth_view()));

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

        // Create environment background pass (for Particles mode HDR background)
        let env_background = EnvBackground::new(
            &gpu.device,
            renderer.camera_buffer(),
            &env_view,
            &env_sampler,
            &self.state.environment.to_gpu_params(&self.ground_staging()),
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

        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.mc_renderer = Some(mc_renderer);
        self.ss_renderer = Some(ss_renderer);
        self.spray_renderer = Some(spray_renderer);
        self.spray_depth_bound = Some(FluidRenderMode::MarchingCubes);
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
        self.env_background = Some(env_background);
        self.sph_simulation = Some(sph_simulation);
        self.probe_system = Some(probe_system);
        self.egui_winit = Some(egui_winit);
        self.egui_renderer = Some(egui_renderer);

        if let Some(path) = self.launch.load_state.clone() {
            if let Err(e) = self.load_state(&path) {
                eprintln!("error: --load-state {}: {e}", path.display());
                std::process::exit(1);
            }
        }
    }

    /// The scene objects around the water: the wireframe box, rigid bodies,
    /// caustics (raster source = the MC mesh, output sampled by the pool)
    /// and the opaque pool container
    fn create_scene_renderers(
        &self,
        gpu: &GpuContext,
        camera_params: &GpuCameraParams,
        container_geom: &GpuContainerGeometry,
        mc_renderer: &MarchingCubesRenderer,
        gpu_sh: &GpuShCoefficients,
    ) -> (WireframeRenderer, RigidBodyRenderer, CausticsRenderer, ContainerRenderer) {
        // Create wireframe renderer for container visualization
        let wireframe_renderer = WireframeRenderer::new(
            &gpu.device,
            gpu.config.format,
            camera_params,
            container_geom,
        );

        // Create rigid body renderer (before caustics:
        // the splat pass binds the body array for photon occlusion)
        let rigid_body_renderer = RigidBodyRenderer::new(
            &gpu.device,
            &gpu.queue,
            crate::render::HDR_FORMAT,
            camera_params,
            self.state.quality.msaa.as_u32(),
        );

        // Create caustics renderer (raster source = MC mesh; output sampled by container)
        let caustics_renderer = CausticsRenderer::new(
            &gpu.device,
            mc_renderer.mesh_vertex_buffer(),
            rigid_body_renderer.bodies_buffer(),
            container_geom,
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
        let container_renderer = ContainerRenderer::new(
            &gpu.device,
            crate::render::HDR_FORMAT,
            camera_params,
            container_geom,
            &pool_style,
            &self.state.container,
            self.state.quality.msaa.as_u32(),
            self.state.sph.kernel_radius,
            gpu_sh,
            caustics_renderer.display_view(),
            caustics_renderer.sampler(),
        );

        (wireframe_renderer, rigid_body_renderer, caustics_renderer, container_renderer)
    }

    /// Container geometry as the simulation sees it (no MC clip margin)
    fn sim_container_geometry(&self) -> GpuContainerGeometry {
        self.state.container.to_gpu_geometry(
            self.state.sph.wall_stiffness,
            self.state.simulation.damping,
            self.state.container.style == ContainerStyle::OpaquePool,
            0.0,
        )
    }

    /// A fresh SPH simulation over `particles` with the current solver and
    /// container parameters (`runtime.particle_count` must already match)
    fn create_simulation(&self, gpu: &GpuContext, particles: &[SphParticle3D]) -> SphSimulation3DGrid {
        let sph_params = self.state.sph.to_gpu_params_3d(
            self.state.runtime.particle_count,
            self.simulation_substep_dt(),
        );
        SphSimulation3DGrid::new(
            &gpu.device,
            &gpu.queue,
            particles,
            sph_params,
            self.sim_container_geometry(),
            self.state.simulation.max_particles,
            self.sdf_data.as_ref(),
        )
    }

    /// The whitewater system over a simulation's buffers, and its renderer.
    /// `front_depth`: the water front depth the foam splat pass is gated by
    /// (the MC renderer's; the per-frame sync rebinds it per render mode).
    fn create_whitewater(
        &self,
        gpu: &GpuContext,
        sph_sim: &SphSimulation3DGrid,
        front_depth: Option<&wgpu::TextureView>,
    ) -> (SpraySystem, SprayRenderer) {
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
        if let Some(front_depth) = front_depth {
            spray_renderer.set_depth_view(&gpu.device, spray_system.spray_buffer(), front_depth);
        }
        (spray_system, spray_renderer)
    }

    pub(super) fn reset_simulation(&mut self) {
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
            self.sph_simulation = Some(self.create_simulation(gpu, &particles));

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
                let front_depth = self.mc_renderer.as_ref().map(|mc| mc.front_depth_view());
                let (spray_system, spray_renderer) = self.create_whitewater(gpu, sph_sim, front_depth);
                if front_depth.is_some() {
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

    pub(super) fn reset_defaults(&mut self) {
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
