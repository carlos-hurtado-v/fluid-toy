//! Marching Cubes fluid surface renderer
//!
//! Generates a triangle mesh from the particle density field with marching
//! cubes and shades it as water. This file only wires the passes together
//! (construction, per-frame order, resize / grid rebuild / HDR switch); each
//! pass lives in its own module:
//!
//! - field: mc_anisotropy -> mc_field (density, blur) -> calm_smoothing ->
//!   wall_bound -> voxel_normals -> mc_mesh
//! - screen: mc_faces (front/back G-buffers) -> mc_backdrop + mc_background
//!   (scene behind the water, mips) -> mc_ssr -> mc_water (mc_probe for
//!   `--probe`)
//!
//! A new pass gets its own `mc_<pass>.rs` with `new` / `update` / `encode`
//! (and `resize` / `rebuild_bind_groups` if it owns sized resources), on the
//! wall_bound.rs pattern; this file only constructs it and calls it.

use std::cell::Cell;

use super::calm_smoothing::CalmSmoothing;
use super::mc_anisotropy::AnisotropyPass;
use super::mc_backdrop::Backdrop;
use super::mc_background::Background;
use super::mc_faces::FaceBuffers;
use super::mc_field::{BlurPass, DensityInputs, DensityPass, FieldTextures, GpuGridParams, SimBuffers};
use super::mc_mesh::{MeshPass, MAX_VERTICES};
use super::mc_probe::PixelProbe;
use super::mc_ssr::Ssr;
use super::mc_water::{FoamMapViews, SharedBuffers, VolumeBindings, WaterBindings, WaterPass};
use super::voxel_normals::VoxelNormals;
use super::wall_bound::WallBound;
use super::ContainerRenderer;
use super::RigidBodyRenderer;
use super::SprayRenderer;
use crate::render::GpuCameraParams;
use crate::state::{GpuContainerGeometry, GpuEnvironmentParams, GpuLightParams, GpuShCoefficients};

pub use super::mc_anisotropy::ANISO_MAX_STRETCH;
pub use super::mc_probe::{ProbeDump, PROBE_MAX_PIXELS};
pub use super::mc_water::GpuWaterParams;

/// Foam density field (half-res: cheaper splatting + free smoothing when the
/// water shader samples it with a linear filter)
fn create_foam_density_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let foam_density_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Foam Density Texture"),
        size: wgpu::Extent3d {
            width: (width / 2).max(1),
            height: (height / 2).max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: super::spray_renderer::FOAM_DENSITY_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let foam_density_view =
        foam_density_texture.create_view(&wgpu::TextureViewDescriptor::default());
    (foam_density_texture, foam_density_view)
}

pub struct MarchingCubesRenderer {
    // --- The density field and the passes that fill it, in frame order ---
    field: FieldTextures,
    // Anisotropic kernel pass (Yu & Turk): covariance + eigensolve per particle
    aniso: AnisotropyPass,
    density: DensityPass,
    blur: BlurPass,
    // Bulk-gated smoothing of calm water (half-res helper fields + combine)
    calm: CalmSmoothing,
    // Ends the field on the container walls (last field pass)
    wall_bound: WallBound,
    // Normal at every voxel of the final field, for generate and the water shader
    voxel_normals: VoxelNormals,
    mesh: MeshPass,
    /// Which field texture this frame's generate() left the result in
    result_in_b: bool,
    /// The sim buffers the density / anisotropy passes cached their bind
    /// groups over; a sim rebuild swaps them out
    cached_sim_buffers: Option<[wgpu::Buffer; 4]>,

    // --- Screen-space inputs of the water pass ---
    faces: FaceBuffers,
    background: Background,
    backdrop: Backdrop,
    ssr: Ssr,
    /// rendering.mc_filtered_lookup (set with the water params): whether the
    /// background's mip chain is needed this frame
    filtered_lookup: Cell<bool>,
    // Half-res foam density field, splatted by SprayRenderer and composited
    // by the water shader (foam reads as connected patches, not sprites)
    _foam_density_texture: wgpu::Texture,
    foam_density_view: wgpu::TextureView,
    foam_map: FoamMapViews,

    // --- The water pass ---
    buffers: SharedBuffers,
    water: WaterPass,
    // Pixel probe (--probe): the records its shader variant writes
    probe: PixelProbe,

    // Grid bounds
    grid_min: [f32; 3],
    grid_max: [f32; 3],
    // MC grid resolution (cells per dimension)
    grid_size: u32,

    // Screen dimensions
    width: u32,
    height: u32,
}

impl MarchingCubesRenderer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        env_texture_view: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
        width: u32,
        height: u32,
        sample_count: u32,
        grid_size: u32,
        foam_map: &crate::simulation::FoamMap,
    ) -> Self {
        // Clamp sample count to valid values (1, 2, 4, 8)
        // Note: Not all GPUs support 8x MSAA - wgpu will validate this
        let sample_count = match sample_count {
            1 => 1,
            2 => 2,
            4 => 4,
            8 => 8,
            _ => 4, // Default to 4x if invalid
        };

        // Grid bounds (matching simulation domain)
        let grid_min = [-1.0f32, -1.0, -1.0];
        let grid_max = [1.0f32, 1.0, 1.0];
        let extent_x = grid_max[0] - grid_min[0];
        let extent_y = grid_max[1] - grid_min[1];
        let extent_z = grid_max[2] - grid_min[2];
        let cell_size = extent_x.max(extent_y).max(extent_z) / grid_size as f32;

        let buffers = SharedBuffers::new(
            device,
            &GpuGridParams {
                grid_min,
                grid_size,
                grid_max,
                cell_size,
                kernel_radius: 0.1,
                iso_value: 500.0, // Will be tuned based on rest_density
                num_particles: 0,
                max_vertices: MAX_VERTICES,
            },
        );

        // The density field and the passes that fill it
        let field = FieldTextures::new(device, grid_size);
        let calm = CalmSmoothing::new(device, grid_size, &field.view_a, &field.view_b);
        let wall_bound = WallBound::new(
            device,
            grid_size,
            &field.view_a,
            &field.view_b,
            &buffers.grid_params,
            &buffers.container_geom,
        );
        let voxel_normals = VoxelNormals::new(
            device,
            grid_size,
            &field.view_a,
            &field.view_b,
            &buffers.grid_params,
            calm.gate_view(),
        );
        let aniso = AnisotropyPass::new(device);
        let density = DensityPass::new(device);
        let blur = BlurPass::new(device, grid_size, &field);
        let mesh = MeshPass::new(device, &field, &buffers.grid_params, voxel_normals.view());

        // Screen-space inputs of the water pass
        let faces = FaceBuffers::new(
            device,
            width,
            height,
            &buffers.camera,
            mesh.vertex_buffer(),
            &buffers.container_geom,
        );
        let background = Background::new(device, surface_format, width, height);
        let backdrop = Backdrop::new(device, surface_format, sample_count, &buffers.camera, env_texture_view, env_sampler);
        let ssr = Ssr::new(device, width, height, &buffers.camera, &faces, &background);
        let (foam_density_texture, foam_density_view) = create_foam_density_texture(device, width, height);
        let foam_map = FoamMapViews {
            map_view: foam_map.foam_view().clone(),
            surface_view: foam_map.surface_view().clone(),
            params: foam_map.params_buffer().clone(),
            coords_view: foam_map.coords_view().clone(),
        };

        // The water pass
        let probe = PixelProbe::new(device);
        let water = WaterPass::new(
            device,
            surface_format,
            width,
            height,
            sample_count,
            &WaterBindings {
                buffers: &buffers,
                vertices: mesh.vertex_buffer(),
                env_view: env_texture_view,
                env_sampler,
                faces: &faces,
                background: &background,
                ssr: &ssr,
                foam_density_view: &foam_density_view,
                foam_map: &foam_map,
                probe: &probe,
            },
            &VolumeBindings {
                field: &field,
                grid_params: &buffers.grid_params,
                tri_table: mesh.tri_table_buffer(),
                voxel_normals: voxel_normals.view(),
            },
        );

        Self {
            field,
            aniso,
            density,
            blur,
            calm,
            wall_bound,
            voxel_normals,
            mesh,
            result_in_b: false,
            cached_sim_buffers: None,
            faces,
            background,
            backdrop,
            ssr,
            filtered_lookup: Cell::new(true),
            _foam_density_texture: foam_density_texture,
            foam_density_view,
            foam_map,
            buffers,
            water,
            probe,
            grid_min,
            grid_max,
            grid_size,
            width,
            height,
        }
    }

    /// Get the front-face depth view (for GTAO input)
    pub fn front_depth_view(&self) -> &wgpu::TextureView {
        self.faces.front_depth_view()
    }

    /// Target for the foam density splat pass (rendered by SprayRenderer)
    pub fn foam_density_view(&self) -> &wgpu::TextureView {
        &self.foam_density_view
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, params: &GpuCameraParams) {
        queue.write_buffer(&self.buffers.camera, 0, bytemuck::bytes_of(params));
    }

    pub fn update_light_params(&self, queue: &wgpu::Queue, params: &GpuLightParams) {
        queue.write_buffer(&self.buffers.light_params, 0, bytemuck::bytes_of(params));
    }

    pub fn update_sh_coefficients(&self, queue: &wgpu::Queue, coeffs: &GpuShCoefficients) {
        queue.write_buffer(&self.buffers.sh_coefficients, 0, bytemuck::bytes_of(coeffs));
    }

    /// Voxel edge length: the grid's largest extent over its resolution
    fn cell_size(&self) -> f32 {
        let extent_x = self.grid_max[0] - self.grid_min[0];
        let extent_y = self.grid_max[1] - self.grid_min[1];
        let extent_z = self.grid_max[2] - self.grid_min[2];
        extent_x.max(extent_y).max(extent_z) / self.grid_size as f32
    }

    pub fn update_params(&self, queue: &wgpu::Queue, kernel_radius: f32, iso_value: f32, num_particles: u32, blur_radius: u32) {
        let cell_size = self.cell_size();
        let params = GpuGridParams {
            grid_min: self.grid_min,
            grid_size: self.grid_size,
            grid_max: self.grid_max,
            cell_size,
            kernel_radius,
            iso_value,
            num_particles,
            max_vertices: MAX_VERTICES,
        };
        queue.write_buffer(&self.buffers.grid_params, 0, bytemuck::bytes_of(&params));

        // Update blur radius for all 3 direction buffers
        self.blur.update(queue, blur_radius, self.grid_size);
    }

    /// Where the field ends at the container walls. `kernel_radius` is the sim h.
    pub fn update_wall_bound(&self, queue: &wgpu::Queue, kernel_radius: f32, is_pool: bool) {
        self.wall_bound.update(queue, kernel_radius, self.cell_size(), is_pool);
    }

    /// Calm-surface smoothing strength (0 = off). `kernel_radius` is the sim h,
    /// `iso_value` the same one `update_params` gets.
    pub fn update_calm_smoothing(&self, queue: &wgpu::Queue, strength: f32, kernel_radius: f32, iso_value: f32) {
        self.calm.update(queue, strength, kernel_radius, self.cell_size(), iso_value);
    }

    /// Normal denoising on calm water (degrees, 0 = off; see
    /// `mc_voxel_normals.wgsl`). It leans on the calm gate field, so it
    /// follows the calm-smoothing strength down to off.
    pub fn update_voxel_normals(&self, queue: &wgpu::Queue, denoise_deg: f32, calm_smoothing: f32) {
        self.voxel_normals.update(queue, denoise_deg * calm_smoothing.clamp(0.0, 1.0));
    }

    /// Update anisotropic kernel parameters (Yu & Turk).
    /// `kernel_radius` is the sim h; `h_mc` the MC density kernel radius.
    pub fn update_aniso_params(&self, queue: &wgpu::Queue, enabled: bool, strength: f32, kernel_radius: f32, h_mc: f32) {
        self.aniso.update_params(queue, enabled, strength, kernel_radius, h_mc);
    }

    /// Update environment parameters (background mode, color, intensity)
    pub fn update_env_params(&self, queue: &wgpu::Queue, params: &GpuEnvironmentParams) {
        self.backdrop.update_params(queue, params);
    }

    /// Update grid bounds to match container dimensions
    pub fn set_bounds(&mut self, min: [f32; 3], max: [f32; 3]) {
        self.grid_min = min;
        self.grid_max = max;
    }

    /// Current grid resolution (cells per dimension)
    pub fn grid_size(&self) -> u32 {
        self.grid_size
    }

    /// Rebuild 3D density textures and dependent bind groups for a new grid resolution.
    /// Call when the user changes the MC grid resolution preset.
    pub fn rebuild_grid(&mut self, device: &wgpu::Device, new_grid_size: u32) {
        self.grid_size = new_grid_size;

        // Recreate density textures at new resolution
        let field = FieldTextures::new(device, new_grid_size);

        // Field passes that own per-resolution textures and bind groups
        self.calm = CalmSmoothing::new(device, new_grid_size, &field.view_a, &field.view_b);
        self.wall_bound = WallBound::new(
            device,
            new_grid_size,
            &field.view_a,
            &field.view_b,
            &self.buffers.grid_params,
            &self.buffers.container_geom,
        );
        self.voxel_normals = VoxelNormals::new(
            device,
            new_grid_size,
            &field.view_a,
            &field.view_b,
            &self.buffers.grid_params,
            self.calm.gate_view(),
        );
        self.water.rebuild_volume_bind_groups(
            device,
            &VolumeBindings {
                field: &field,
                grid_params: &self.buffers.grid_params,
                tri_table: self.mesh.tri_table_buffer(),
                voxel_normals: self.voxel_normals.view(),
            },
        );

        // Bind groups that reference the density texture views
        self.mesh.rebuild_bind_groups(device, &field, &self.buffers.grid_params, self.voxel_normals.view());
        self.blur.rebuild_bind_groups(device, &field);

        // Store the new textures (old ones are dropped automatically)
        self.field = field;

        // The cached density bind group references the old density view
        self.density.invalidate();
        self.aniso.invalidate();
    }

    /// Update container geometry (shared struct: geometry, rotation, physics, clip)
    pub fn update_container_geometry(&self, queue: &wgpu::Queue, geom: &GpuContainerGeometry) {
        queue.write_buffer(&self.buffers.container_geom, 0, bytemuck::bytes_of(geom));
    }

    /// MC mesh vertex storage buffer (allocated once; never recreated).
    /// Used by the caustics light-space raster pass.
    pub fn mesh_vertex_buffer(&self) -> &wgpu::Buffer {
        self.mesh.vertex_buffer()
    }

    /// GPU-driven indirect draw args for the MC mesh (vertex count filled by generate)
    pub fn mesh_indirect_buffer(&self) -> &wgpu::Buffer {
        self.mesh.indirect_buffer()
    }

    /// Update water shading parameters
    pub fn update_water_params(
        &self,
        queue: &wgpu::Queue,
        water_color: &[f32; 3],
        roughness: f32,
        env_intensity: f32,
        use_env_background: bool,
        background_color: &[f32; 3],
        time: f32,
        refraction_strength: f32,
        deep_color: &[f32; 3],
        ripple_strength: f32,
        clarity: f32,
        physical_refraction: bool,
        physical_medium: bool,
        foam_coverage: f32,
        aeration_strength: f32,
        body_count: u32,
        debug_view: u32,
        silhouette_exit: u32,
        front_exit: bool,
        env_params: &GpuEnvironmentParams,
        filtered_lookup: bool,
        volume_trace: bool,
    ) {
        self.filtered_lookup.set(filtered_lookup);
        let params = GpuWaterParams {
            water_color: *water_color,
            roughness,
            ior: 1.333,
            env_intensity,
            use_env_background: if use_env_background { 1 } else { 0 },
            background_r: background_color[0],
            background_g: background_color[1],
            background_b: background_color[2],
            time,
            refraction_strength,
            deep_color_r: deep_color[0],
            deep_color_g: deep_color[1],
            deep_color_b: deep_color[2],
            ripple_strength,
            clarity,
            _pad1: if physical_refraction { 1.0 } else { 0.0 },
            foam_coverage,
            aeration_strength,
            physical_medium: if physical_medium { 1.0 } else { 0.0 },
            body_count,
            debug_view,
            silhouette_exit,
            front_exit: front_exit as u32,
            ground_enabled: env_params.ground_enabled,
            ground_y: env_params.ground_y,
            ground_capture_height: env_params.ground_capture_height,
            filtered_lookup: filtered_lookup as u32,
            volume_trace: volume_trace as u32,
            _pad_g: [0; 2],
        };
        queue.write_buffer(&self.buffers.water_params, 0, bytemuck::bytes_of(&params));
    }

    /// Upload this frame's enabled rigid bodies (render layout, in order; the
    /// count rides in GpuWaterParams::body_count)
    pub fn update_bodies(&self, queue: &wgpu::Queue, bodies: &[crate::state::GpuRigidBodyRender]) {
        let count = bodies.len().min(crate::state::MAX_RIGID_BODIES);
        if count > 0 {
            queue.write_buffer(&self.buffers.bodies, 0, bytemuck::cast_slice(&bodies[..count]));
        }
    }

    /// Switch the water pass to the pixel-probe shader variant, recording at
    /// these pixels (PNG coordinates, at most PROBE_MAX_PIXELS)
    pub fn enable_probe(
        &mut self,
        device: &wgpu::Device,
        env_view: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
        pixels: &[[u32; 2]],
    ) {
        self.probe.enable(device, pixels);
        self.water.enable_probe(device);
        self.rebind_water(device, env_view, env_sampler);
    }

    pub fn probe_enabled(&self) -> bool {
        self.probe.enabled()
    }

    /// Empty the probe records before this frame's water pass (queue writes
    /// land ahead of the next submit)
    pub fn reset_probe(&self, queue: &wgpu::Queue) {
        self.probe.reset(queue);
    }

    /// Read the last frame's probe records (blocking)
    pub fn read_probe(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> ProbeDump {
        self.probe.read(device, queue)
    }

    /// Field dump (--dump-field): the density field the mesh was extracted
    /// from, as last generated — grid parameters + grid_size^3 f32 (x fastest).
    /// Voxel i sits at grid_min + i * cell_size (mc_generate's convention).
    pub fn read_field(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> (GpuGridParams, Vec<u8>) {
        self.field.read(device, queue, self.result_in_b, self.grid_size, &self.buffers.grid_params)
    }

    /// Set whether SSR is enabled and update GPU params
    pub fn set_ssr_enabled(&self, queue: &wgpu::Queue, enabled: bool) {
        self.ssr.set_enabled(queue, enabled);
    }

    /// Per-particle ellipsoid records from the last anisotropy pass, indexed
    /// by sorted particle index (valid after run_anisotropy this frame).
    pub fn aniso_buffer(&self) -> &wgpu::Buffer {
        self.aniso.records()
    }

    /// Invalidate cached bind groups if the sim handed us different buffer
    /// objects (sim rebuild on respawn/container change swaps them out).
    fn sync_sim_buffer_cache(&mut self, sim: &SimBuffers) {
        let unchanged = self.cached_sim_buffers.as_ref().is_some_and(|cached| sim.same_as(cached));
        if !unchanged {
            self.cached_sim_buffers = Some(sim.handles());
            self.density.invalidate();
            self.aniso.invalidate();
        }
    }

    /// Per-particle anisotropic kernel fit (covariance + eigensolve), Yu &
    /// Turk. Standalone entry so the screen-space renderer can run it without
    /// the rest of the MC pipeline; generate() calls it too. The separate
    /// compute pass gives an implicit barrier before any consumer.
    #[allow(clippy::too_many_arguments)]
    pub fn run_anisotropy(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        sorted_particle_buffer: &wgpu::Buffer,
        cell_starts_buffer: &wgpu::Buffer,
        cell_counts_buffer: &wgpu::Buffer,
        sph_grid_params_buffer: &wgpu::Buffer,
        num_particles: u32,
    ) {
        if num_particles == 0 {
            return;
        }
        let sim = SimBuffers {
            sorted_particles: sorted_particle_buffer,
            cell_starts: cell_starts_buffer,
            cell_counts: cell_counts_buffer,
            grid_params: sph_grid_params_buffer,
        };
        self.sync_sim_buffer_cache(&sim);
        // Growing the record buffer leaves the density bind group stale too
        if self.aniso.ensure_capacity(device, num_particles) {
            self.density.invalidate();
        }
        self.aniso.encode(encoder, device, &sim, &self.buffers.container_geom, num_particles);
    }

    /// Generate mesh from particles using SPH grid-accelerated density computation
    pub fn generate(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        sorted_particle_buffer: &wgpu::Buffer,
        cell_starts_buffer: &wgpu::Buffer,
        cell_counts_buffer: &wgpu::Buffer,
        sph_grid_params_buffer: &wgpu::Buffer,
        blur_radius: u32,
        num_particles: u32,
        aniso_enabled: bool,
        calm_smoothing: f32,
    ) {
        // Reset counter
        self.mesh.reset_counter(encoder);

        // Pass 0: Per-particle anisotropic kernel fit (covariance + eigensolve).
        // Runs before the density pass binds its inputs — it may grow the
        // record buffer, which that bind group references. Also syncs the sim
        // buffer cache; repeat the sync here for the aniso-disabled path.
        let sim = SimBuffers {
            sorted_particles: sorted_particle_buffer,
            cell_starts: cell_starts_buffer,
            cell_counts: cell_counts_buffer,
            grid_params: sph_grid_params_buffer,
        };
        if aniso_enabled && num_particles > 0 {
            self.run_anisotropy(
                encoder,
                device,
                sorted_particle_buffer,
                cell_starts_buffer,
                cell_counts_buffer,
                sph_grid_params_buffer,
                num_particles,
            );
        } else {
            self.sync_sim_buffer_cache(&sim);
        }

        // Pass 1: Generate density field (splat particles into texture A)
        self.density.encode(
            encoder,
            device,
            &sim,
            &DensityInputs {
                grid_params: &self.buffers.grid_params,
                field: &self.field.view_a,
                container_geom: &self.buffers.container_geom,
                aniso_records: self.aniso.records(),
                aniso_params: self.aniso.params_buffer(),
            },
            self.grid_size,
        );

        // Calm-surface smoothing helper fields, from the raw field in A
        // (before the base blur ping-pongs through it)
        let calm = calm_smoothing > 0.0;
        if calm {
            self.calm.encode_helpers(encoder);
        }

        // Pass 1.5: Blur density field (3 separable passes: X, Y, Z).
        // After blur the result is in texture B (odd number of passes:
        // a->b, b->a, a->b); without it, it stays in texture A
        let result_in_b = blur_radius > 0;
        if result_in_b {
            self.blur.encode(encoder, self.grid_size);
        }
        // Pass 1.6: blend calm bulk water toward the wide half-res field
        let result_in_b = if calm {
            self.calm.encode_combine(encoder, result_in_b)
        } else {
            result_in_b
        };

        // Pass 1.7: end the field on the container walls
        let result_in_b = self.wall_bound.encode(encoder, result_in_b);

        self.result_in_b = result_in_b;

        // Pass 1.8: the final field's normals, for generate and the water shader
        self.voxel_normals.encode(encoder, result_in_b);

        // Pass 2: Generate triangles (read from whichever texture has the
        // result), then hand the count to the indirect draw and the readback
        self.mesh.encode(encoder, result_in_b, self.grid_size);
    }

    /// Last read-back mesh vertex count (see `read_vertex_count`)
    pub fn vertex_count(&self) -> u32 {
        self.mesh.vertex_count()
    }

    /// Read back vertex count, blocking until the GPU finishes (call after
    /// submit). Exact for the frame just submitted — automation/stats runs
    /// use this so CSV rows stay deterministic.
    pub fn read_vertex_count(&mut self, device: &wgpu::Device) {
        self.mesh.read_vertex_count(device);
    }

    /// Non-blocking variant for interactive frames: harvests a previously
    /// issued readback if the GPU has finished it, then arms the next one.
    /// The count lags a frame or two, which only affects the GUI stat —
    /// rendering uses the GPU-side indirect buffer.
    pub fn poll_vertex_count(&mut self, device: &wgpu::Device) {
        self.mesh.poll_vertex_count(device);
    }

    /// Render the generated mesh with environment background.
    /// If `rigid_body` is provided, it will be rendered inside the same MSAA pass
    /// for proper depth testing against the fluid surface.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        _background_color: &[f32; 3],
        rigid_body: Option<&RigidBodyRenderer>,
        spray: Option<&SprayRenderer>,
        container: Option<&ContainerRenderer>,
    ) {
        let indirect_buffer = self.mesh.indirect_buffer();

        // Pass 0a: water front faces to depth + normal G-buffer (for GTAO + SSR)
        // Pass 0b: rigid body + container into front depth (depth-only)
        self.faces.encode_front(encoder, indirect_buffer, rigid_body, container);

        // Pass 1: back faces to back depth (for thickness calculation) and
        // their normals (refraction exit interface)
        self.faces.encode_back(encoder, indirect_buffer);

        // Pass 2: Render environment to background texture (for screen-space refraction)
        // Uses single-sampled depth and pipeline since background_texture is single-sampled
        {
            let mut env_pass = self.background.begin_pass(encoder);
            self.backdrop.draw_1x(&mut env_pass);

            // Render rigid body into background texture so the water shader's
            // screen-space refraction shows it through the water surface
            if let Some(rb) = rigid_body {
                rb.render(&mut env_pass);
            }

            // Render container into background (visible through water refraction + SSR)
            if let Some(ct) = container {
                ct.render(&mut env_pass);
            }

            // Render spray into background (visible through water refraction)
            if let Some(sp) = spray {
                sp.render(&mut env_pass);
            }
        }

        // Pass 2b: the background's mip chain, for filtered refraction lookups
        if self.filtered_lookup.get() {
            self.background.encode_mips(encoder);
        }

        // SSR compute pass: ray-march against background depth for screen-space reflections
        self.ssr.encode(encoder, self.width, self.height);

        // Pass 3: Render water mesh with screen-space refraction from background
        // Uses MSAA if enabled (renders to the MSAA target, resolves to color_view)
        {
            let mut pass = self.water.begin_pass(encoder, color_view);

            // Draw environment background first (at far plane, will show through where no water)
            self.backdrop.draw(&mut pass);

            // Draw rigid body into the carved-out gap before water mesh
            if let Some(rb) = rigid_body {
                rb.render_msaa(&mut pass);
            }

            // Draw container walls into the scene before water mesh
            if let Some(ct) = container {
                ct.render_msaa(&mut pass);
            }

            // Draw water mesh (samples background_texture for refraction)
            self.water.draw(&mut pass, indirect_buffer, self.result_in_b);

            // Draw whitewater after the water mesh: camera-biased foam wins the
            // depth test at the surface, while submerged bubbles fail it and
            // remain visible only through refraction (background pass copy)
            if let Some(sp) = spray {
                sp.render_msaa(&mut pass);
            }
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, env_view: &wgpu::TextureView, env_sampler: &wgpu::Sampler, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;

        // Every screen-sized target; SSR last (it binds the others' views)
        self.water.resize(device, width, height);
        self.faces.resize(device, width, height);
        self.background.resize(device, width, height);
        let (foam_density_texture, foam_density_view) = create_foam_density_texture(device, width, height);
        self._foam_density_texture = foam_density_texture;
        self.foam_density_view = foam_density_view;
        self.ssr.resize(device, width, height, &self.buffers.camera, &self.faces, &self.background);

        // Recreate render bind group with new textures
        self.rebind_water(device, env_view, env_sampler);
    }

    /// Rebuild the water pass's group 0 over the current size-dependent views
    /// (resize), environment texture (HDR switch) and probe buffer
    fn rebind_water(&mut self, device: &wgpu::Device, env_view: &wgpu::TextureView, env_sampler: &wgpu::Sampler) {
        let bind_group = self.water.create_bind_group(
            device,
            &WaterBindings {
                buffers: &self.buffers,
                vertices: self.mesh.vertex_buffer(),
                env_view,
                env_sampler,
                faces: &self.faces,
                background: &self.background,
                ssr: &self.ssr,
                foam_density_view: &self.foam_density_view,
                foam_map: &self.foam_map,
                probe: &self.probe,
            },
        );
        self.water.set_bind_group(bind_group);
    }

    /// Rebuild bind groups that reference environment texture (for HDR switching)
    pub fn rebuild_env_bind_groups(&mut self, device: &wgpu::Device, env_view: &wgpu::TextureView, env_sampler: &wgpu::Sampler) {
        self.backdrop.rebuild_bind_group(device, &self.buffers.camera, env_view, env_sampler);
        self.rebind_water(device, env_view, env_sampler);
    }
}
