//! Surface foam map: Eulerian 2D foam layer over the container (see
//! `foam_map.wgsl` for the model). Settled diffuse foam particles deposit into
//! it and retire; it is advected by a surface velocity field smoothed above
//! the SPH particle scale, so foam forms coherent patches, filaments and lines
//! instead of per-particle snow. The MC water shader composites it on the top
//! surface; particle foam keeps rendering everywhere else.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::simulation::snapshot::{write_texture, FoamMapScalars, GpuSource, SimState};

/// Fine foam grid (texels per side). Square over the container's larger
/// horizontal extent: ~4.3 mm texels for the default 2.22 m tank.
const FINE_DIM: u32 = 512;
/// Coarse surface grid (column tops + surface velocity), ~1.7 mm x 4
const COARSE_DIM: u32 = 128;
/// Surface layer depth below a column's top particle, in kernel radii
const SURFACE_BAND_H: f32 = 1.5;
/// Surface velocity smoothing sigma, in kernel radii. Particle-scale velocity
/// carries the SPH lattice; foam advected by it collapses onto the particles.
const VELOCITY_SMOOTHING_H: f32 = 1.0;
/// Gaussian spread of one particle's deposit, in kernel radii (~half the
/// particle spacing, so a patch of particles deposits a connected patch)
const DEPOSIT_SIGMA_H: f32 = 0.35;
/// Foam one settling particle deposits (map units x m^2): a patch of ~30k
/// particles per m^2 reaches the densest foam the compositor renders
const DEPOSIT_AMOUNT: f32 = 6.7e-5;
/// Must match NEWBORN_SPRAY_TIME in spray_simulate.wgsl
const GRACE_AGE: f32 = 0.15;
/// Bursting: foam density lost per second on top of the half-life decay, so
/// thin foam (lace, strings) dies within a couple of seconds while thick
/// patches ride the half-life
const BURST_RATE: f32 = 0.12;
/// Flow-map cycle (s): the lace pattern is carried by the flow for up to this
/// long before its phase restarts (hidden by the two-phase crossfade)
const FLOW_PERIOD: f32 = 1.0;

const FLAG_ACTIVE: u32 = 1;
const FLAG_RESET: u32 = 2;
const FLAG_RESTART_A: u32 = 4;
const FLAG_RESTART_B: u32 = 8;

/// Same layout as `FoamMapParams` in foam_map.wgsl / mc_render.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GpuFoamMapParams {
    pub origin_x: f32,
    pub origin_z: f32,
    pub fine_cell: f32,
    pub coarse_cell: f32,
    pub fine_dim: u32,
    pub coarse_dim: u32,
    pub num_particles: u32,
    pub max_spray: u32,
    pub dt: f32,
    pub decay: f32,
    pub surface_band: f32,
    pub deposit_amount: f32,
    pub deposit_sigma: f32,
    pub grace_age: f32,
    pub blur_sigma: f32,
    pub flags: u32,
    pub flow_phase: f32,
    pub burst: f32,
    pub _pad: [f32; 2],
}

fn buf(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
}

fn view(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(view) }
}

/// Bind group for one pass (auto layout: only the bindings its entry point uses)
fn bind_group(device: &wgpu::Device, pipeline: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry]) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Foam Map BG"),
        layout: &pipeline.get_bind_group_layout(0),
        entries,
    })
}

/// Buffers the particle-reading passes bind (change when the sim or the
/// spray system is recreated)
struct Sources {
    particles: wgpu::Buffer,
    spray: wgpu::Buffer,
    container: wgpu::Buffer,
    column_tops: wgpu::BindGroup,
    splat_velocity: wgpu::BindGroup,
    deposit: wgpu::BindGroup,
}

pub struct FoamMap {
    params: GpuFoamMapParams,
    params_buffer: wgpu::Buffer,
    col_top: wgpu::Buffer,
    vel_accum: wgpu::Buffer,
    deposit_accum: wgpu::Buffer,
    _surface: [wgpu::Texture; 2],
    surface_views: [wgpu::TextureView; 2],
    _foam: [wgpu::Texture; 4],
    foam_views: [wgpu::TextureView; 4],
    /// Flow-map coordinates: [0] current (rendered), [1] advection target
    coords: [wgpu::Texture; 2],
    coords_views: [wgpu::TextureView; 2],

    clear_pipeline: wgpu::ComputePipeline,
    column_tops_pipeline: wgpu::ComputePipeline,
    splat_velocity_pipeline: wgpu::ComputePipeline,
    resolve_pipeline: wgpu::ComputePipeline,
    blur_h_pipeline: wgpu::ComputePipeline,
    blur_v_pipeline: wgpu::ComputePipeline,
    deposit_pipeline: wgpu::ComputePipeline,
    inject_pipeline: wgpu::ComputePipeline,
    forward_pipeline: wgpu::ComputePipeline,
    backward_pipeline: wgpu::ComputePipeline,
    correct_pipeline: wgpu::ComputePipeline,
    coords_pipeline: wgpu::ComputePipeline,

    clear_bg: wgpu::BindGroup,
    resolve_bg: wgpu::BindGroup,
    blur_h_bg: wgpu::BindGroup,
    blur_v_bg: wgpu::BindGroup,
    inject_bg: wgpu::BindGroup,
    forward_bg: wgpu::BindGroup,
    backward_bg: wgpu::BindGroup,
    correct_bg: wgpu::BindGroup,
    coords_bg: wgpu::BindGroup,
    sources: Option<Sources>,
    /// Simulated time the map has advanced (drives the flow-map phases)
    flow_time: f32,

    /// Clear the map on the next active frame (sim reset / re-enable)
    reset_pending: bool,
    was_active: bool,
}

impl FoamMap {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Foam Map Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../shaders/container_common.wgsl"),
                    include_str!("../shaders/foam_map.wgsl")
                )
                .into(),
            ),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("Foam Map {}", entry)),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let clear_pipeline = pipeline("clear_coarse");
        let column_tops_pipeline = pipeline("column_tops");
        let splat_velocity_pipeline = pipeline("splat_velocity");
        let resolve_pipeline = pipeline("resolve");
        let blur_h_pipeline = pipeline("blur_h");
        let blur_v_pipeline = pipeline("blur_v");
        let deposit_pipeline = pipeline("deposit");
        let inject_pipeline = pipeline("inject");
        let forward_pipeline = pipeline("advect_forward");
        let backward_pipeline = pipeline("advect_backward");
        let correct_pipeline = pipeline("correct");
        let coords_pipeline = pipeline("advect_coords");

        let params = GpuFoamMapParams {
            fine_dim: FINE_DIM,
            coarse_dim: COARSE_DIM,
            fine_cell: 1.0,
            coarse_cell: 1.0,
            decay: 1.0,
            blur_sigma: 1.0,
            deposit_sigma: 0.01,
            ..Default::default()
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Foam Map Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let storage = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let coarse_cells = (COARSE_DIM * COARSE_DIM) as u64;
        let col_top = storage("Foam Map Column Tops", coarse_cells * 4);
        let vel_accum = storage("Foam Map Velocity Accum", coarse_cells * 12);
        let deposit_accum = storage("Foam Map Deposit Accum", (FINE_DIM * FINE_DIM) as u64 * 4);

        let texture = |label: &str, dim: u32, format: wgpu::TextureFormat| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: dim, height: dim, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                // COPY_*: state files save/restore the persistent layers
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let surface: [wgpu::Texture; 2] = std::array::from_fn(|_| {
            texture("Foam Map Surface", COARSE_DIM, wgpu::TextureFormat::Rgba32Float)
        });
        let surface_views: [wgpu::TextureView; 2] =
            std::array::from_fn(|i| surface[i].create_view(&wgpu::TextureViewDescriptor::default()));
        let foam: [wgpu::Texture; 4] =
            std::array::from_fn(|_| texture("Foam Map", FINE_DIM, wgpu::TextureFormat::R32Float));
        let foam_views: [wgpu::TextureView; 4] =
            std::array::from_fn(|i| foam[i].create_view(&wgpu::TextureViewDescriptor::default()));
        let coords: [wgpu::Texture; 2] = std::array::from_fn(|_| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Foam Map Flow Coords"),
                size: wgpu::Extent3d { width: FINE_DIM, height: FINE_DIM, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        });
        let coords_views: [wgpu::TextureView; 2] =
            std::array::from_fn(|i| coords[i].create_view(&wgpu::TextureViewDescriptor::default()));

        let pb = &params_buffer;
        let (s_final, s_tmp) = (&surface_views[0], &surface_views[1]);
        // Foam textures: A persistent, B injected, C forward, D backward
        let [fa, fb, fc, fd] = &foam_views;

        let clear_bg = bind_group(device, &clear_pipeline, &[
            buf(0, pb), buf(3, &col_top), buf(4, &vel_accum),
        ]);
        let resolve_bg = bind_group(device, &resolve_pipeline, &[
            buf(0, pb), buf(3, &col_top), buf(4, &vel_accum), view(5, s_final),
        ]);
        let blur_h_bg = bind_group(device, &blur_h_pipeline, &[buf(0, pb), view(5, s_tmp), view(6, s_final)]);
        let blur_v_bg = bind_group(device, &blur_v_pipeline, &[buf(0, pb), view(5, s_final), view(6, s_tmp)]);
        let inject_bg = bind_group(device, &inject_pipeline, &[
            buf(0, pb), buf(8, &deposit_accum), view(9, fa), view(10, fb),
        ]);
        let forward_bg = bind_group(device, &forward_pipeline, &[
            buf(0, pb), view(6, s_final), view(9, fb), view(10, fc),
        ]);
        let backward_bg = bind_group(device, &backward_pipeline, &[
            buf(0, pb), view(6, s_final), view(9, fc), view(10, fd),
        ]);
        let correct_bg = bind_group(device, &correct_pipeline, &[
            buf(0, pb), view(6, s_final), view(9, fc), view(10, fa), view(11, fb), view(12, fd),
        ]);
        let coords_bg = bind_group(device, &coords_pipeline, &[
            buf(0, pb), view(6, s_final), view(13, &coords_views[0]), view(14, &coords_views[1]),
        ]);

        Self {
            params,
            params_buffer,
            col_top,
            vel_accum,
            deposit_accum,
            _surface: surface,
            surface_views,
            _foam: foam,
            foam_views,
            coords,
            coords_views,
            clear_pipeline,
            column_tops_pipeline,
            splat_velocity_pipeline,
            resolve_pipeline,
            blur_h_pipeline,
            blur_v_pipeline,
            deposit_pipeline,
            inject_pipeline,
            forward_pipeline,
            backward_pipeline,
            correct_pipeline,
            coords_pipeline,
            clear_bg,
            resolve_bg,
            blur_h_bg,
            blur_v_bg,
            inject_bg,
            forward_bg,
            backward_bg,
            correct_bg,
            coords_bg,
            sources: None,
            flow_time: 0.0,
            reset_pending: true,
            was_active: false,
        }
    }

    /// Persistent foam map (container-local XZ, R32Float) for the water shader
    pub fn foam_view(&self) -> &wgpu::TextureView {
        &self.foam_views[0]
    }

    /// Coarse surface grid (rgba32float; .a = column top height) for the
    /// water shader's top-surface test
    pub fn surface_view(&self) -> &wgpu::TextureView {
        &self.surface_views[0]
    }

    /// Flow-map coordinates (rgba32float: phase A xy, phase B xy) for the
    /// water shader's advected lace pattern
    pub fn coords_view(&self) -> &wgpu::TextureView {
        &self.coords_views[0]
    }

    pub fn params_buffer(&self) -> &wgpu::Buffer {
        &self.params_buffer
    }

    /// Empty the map on the next active frame (simulation reset)
    pub fn request_reset(&mut self) {
        self.reset_pending = true;
    }

    /// State-file sources: the layers that persist across frames and that
    /// the water shader reads (foam density, smoothed surface, flow coords);
    /// the other textures and accumulators are rebuilt every stepped frame
    pub fn snapshot_sources(&self) -> Vec<(&'static str, GpuSource<'_>)> {
        vec![
            ("foam.density", GpuSource::Texture { texture: &self._foam[0], bytes_per_texel: 4 }),
            ("foam.surface", GpuSource::Texture { texture: &self._surface[0], bytes_per_texel: 16 }),
            ("foam.coords", GpuSource::Texture { texture: &self.coords[0], bytes_per_texel: 16 }),
        ]
    }

    pub fn snapshot_scalars(&self) -> FoamMapScalars {
        FoamMapScalars {
            flow_time: self.flow_time,
            was_active: self.was_active,
            reset_pending: self.reset_pending,
        }
    }

    pub fn restore_snapshot(&mut self, queue: &wgpu::Queue, state: &SimState) -> Result<(), String> {
        let layers = [
            ("foam.density", &self._foam[0], 4),
            ("foam.surface", &self._surface[0], 16),
            ("foam.coords", &self.coords[0], 16),
        ];
        for (name, texture, bpt) in layers {
            write_texture(queue, texture, bpt, state.blob(name)?, name)?;
        }
        let scalars = state.header.foam_map;
        self.flow_time = scalars.flow_time;
        self.was_active = scalars.was_active;
        self.reset_pending = scalars.reset_pending;
        Ok(())
    }

    /// Write this frame's params. `active` = the map owns top-surface foam
    /// (deposit + render); inactive leaves particle foam exactly as before.
    /// `stepped` = the simulation advances this frame (encode will run).
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        active: bool,
        stepped: bool,
        container_width: f32,
        container_depth: f32,
        kernel_radius: f32,
        dt: f32,
        half_life: f32,
        num_particles: u32,
        max_spray: u32,
    ) {
        if active && !self.was_active {
            self.reset_pending = true;
        }
        self.was_active = active;
        let half = 0.5 * container_width.max(container_depth);
        let fine_cell = 2.0 * half / FINE_DIM as f32;
        let coarse_cell = 2.0 * half / COARSE_DIM as f32;
        let mut flags = 0;
        if active {
            flags |= FLAG_ACTIVE;
            if self.reset_pending {
                flags |= FLAG_RESET;
            }
            if stepped {
                // Phase B runs half a cycle behind A; each restarts when its
                // cycle wraps this step
                let before = self.flow_time / FLOW_PERIOD;
                self.flow_time += dt;
                let after = self.flow_time / FLOW_PERIOD;
                if before.floor() != after.floor() {
                    flags |= FLAG_RESTART_A;
                }
                if (before + 0.5).floor() != (after + 0.5).floor() {
                    flags |= FLAG_RESTART_B;
                }
            }
        }
        self.params = GpuFoamMapParams {
            origin_x: -half,
            origin_z: -half,
            fine_cell,
            coarse_cell,
            fine_dim: FINE_DIM,
            coarse_dim: COARSE_DIM,
            num_particles,
            max_spray,
            dt,
            decay: (-std::f32::consts::LN_2 * dt / half_life.max(0.05)).exp(),
            surface_band: SURFACE_BAND_H * kernel_radius,
            deposit_amount: DEPOSIT_AMOUNT,
            deposit_sigma: DEPOSIT_SIGMA_H * kernel_radius,
            grace_age: GRACE_AGE,
            blur_sigma: (VELOCITY_SMOOTHING_H * kernel_radius / coarse_cell).max(0.5),
            flags,
            flow_phase: (self.flow_time / FLOW_PERIOD).fract(),
            burst: BURST_RATE * dt,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
    }

    fn ensure_sources(
        &mut self,
        device: &wgpu::Device,
        particles: &wgpu::Buffer,
        spray: &wgpu::Buffer,
        container: &wgpu::Buffer,
    ) {
        if let Some(s) = &self.sources {
            if &s.particles == particles && &s.spray == spray && &s.container == container {
                return;
            }
        }
        let pb = &self.params_buffer;
        let column_tops = bind_group(device, &self.column_tops_pipeline, &[
            buf(0, pb), buf(1, container), buf(2, particles), buf(3, &self.col_top),
        ]);
        let splat_velocity = bind_group(device, &self.splat_velocity_pipeline, &[
            buf(0, pb), buf(1, container), buf(2, particles), buf(3, &self.col_top), buf(4, &self.vel_accum),
        ]);
        let deposit = bind_group(device, &self.deposit_pipeline, &[
            buf(0, pb), buf(1, container), view(6, &self.surface_views[0]), buf(7, spray), buf(8, &self.deposit_accum),
        ]);
        self.sources = Some(Sources {
            particles: particles.clone(),
            spray: spray.clone(),
            container: container.clone(),
            column_tops,
            splat_velocity,
            deposit,
        });
    }

    /// Record one frame of the map (call after the spray step, only while
    /// active and the simulation advanced). `particles` = SPH particle buffer,
    /// `container` = the simulation's container geometry uniform.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        particles: &wgpu::Buffer,
        spray: &wgpu::Buffer,
        container: &wgpu::Buffer,
    ) {
        self.ensure_sources(device, particles, spray, container);
        let sources = self.sources.as_ref().unwrap();
        let coarse = COARSE_DIM.div_ceil(8);
        let fine = FINE_DIM.div_ceil(8);
        let particle_groups = self.params.num_particles.div_ceil(256).max(1);
        let spray_groups = self.params.max_spray.div_ceil(256).max(1);
        let steps: [(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32)); 12] = [
            (&self.clear_pipeline, &self.clear_bg, (coarse, coarse)),
            (&self.column_tops_pipeline, &sources.column_tops, (particle_groups, 1)),
            (&self.splat_velocity_pipeline, &sources.splat_velocity, (particle_groups, 1)),
            (&self.resolve_pipeline, &self.resolve_bg, (coarse, coarse)),
            (&self.blur_h_pipeline, &self.blur_h_bg, (coarse, coarse)),
            (&self.blur_v_pipeline, &self.blur_v_bg, (coarse, coarse)),
            (&self.deposit_pipeline, &sources.deposit, (spray_groups, 1)),
            (&self.inject_pipeline, &self.inject_bg, (fine, fine)),
            (&self.forward_pipeline, &self.forward_bg, (fine, fine)),
            (&self.backward_pipeline, &self.backward_bg, (fine, fine)),
            (&self.correct_pipeline, &self.correct_bg, (fine, fine)),
            (&self.coords_pipeline, &self.coords_bg, (fine, fine)),
        ];
        // Separate passes: each step reads the previous one's output
        for (pipeline, bind_group, (x, y)) in steps {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Foam Map"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        // Advected coordinates become the current set
        encoder.copy_texture_to_texture(
            self.coords[1].as_image_copy(),
            self.coords[0].as_image_copy(),
            wgpu::Extent3d { width: FINE_DIM, height: FINE_DIM, depth_or_array_layers: 1 },
        );
        if self.params.flags & FLAG_RESET != 0 {
            self.reset_pending = false;
        }
    }
}
