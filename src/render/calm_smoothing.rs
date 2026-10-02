//! Calm-surface smoothing for the marching-cubes density field.
//!
//! Bulk-gated low-pass (see `mc_calm_smooth.wgsl`): two half-resolution
//! helper fields — a wide blur S (the smoothing target) and a wider blur G
//! (the gate) — feed a combine pass `final = mix(base, S, strength * gate(G))`.
//! Flattens the particle-scale lumps on still water while splash sheets and
//! droplets keep the base field. The gate is read on the smoothed surface
//! (S = iso), so it cannot fade across a crossing. The half-res blurs reuse
//! `mc_blur.wgsl`.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Triangle-filter half width of the smoothing target S, in sim kernel radii
/// (~10 cm at h = 0.035): wider than the 4-8 cm lumps a still surface carries.
/// Converted to half-res cells per frame, so every grid preset smooths the
/// same physical width (radius 3 at High).
const SMOOTH_HALF_WIDTH_H: f32 = 2.75;
/// Half width of the further blur applied on top of S to make the gate G
/// (~7 cm sigma overall): well above splash-sheet thickness, so sheets read
/// low (radius 5 at High).
const GATE_HALF_WIDTH_H: f32 = 4.1;
/// Gate thresholds on G where S crosses the iso value, as fractions of what G
/// reads there on a flat bulk surface (`bulk_surface_gate`: ~0.40 of the
/// interior density at High with the default threshold, where a 4 cm sheet
/// peaks near 0.23 and droplets near 0). The upper one leaves room for convex
/// bulk (crests read ~10% low): at 1.0 the calm surface itself was only just
/// gated.
const GATE_LO: f32 = 0.55;
const GATE_HI: f32 = 0.88;
/// SPH rest spacing as a fraction of the kernel radius (the spawn lattice;
/// PCISPH holds rest density there). With normalized splat kernels the
/// interior field value is the number density 1 / spacing^3.
const REST_SPACING_FACTOR: f32 = 0.6;

const DIRECTIONS: [[i32; 3]; 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuCalmParams {
    full_size: u32,
    half_size: u32,
    gate_lo: f32,
    gate_hi: f32,
    strength: f32,
    iso: f32,
    _pad: [f32; 2],
}

/// Variance of `mc_blur.wgsl`'s triangle filter of radius r, in cells^2
fn triangle_variance(radius: i32) -> f32 {
    let w = (radius + 1) as f32;
    (w * w - 1.0) / 6.0
}

/// G where S crosses `iso`, on a flat bulk surface, as a fraction of the
/// interior density. Across a half-space S and G are the same edge at two
/// widths, so in logit space one is the other scaled by the width ratio
/// (logistic stand-in for the filters' profiles; within 0.01 of the measured
/// value at every grid preset for iso fractions 0.15-0.4).
fn bulk_surface_gate(iso_fraction: f32, smooth_radius: i32, gate_radius: i32) -> f32 {
    // What the raw field brings in (splat kernel, 2x2x2 downsample), half-res cells^2
    const PRE_FILTER_VARIANCE: f32 = 0.1;
    let smooth = PRE_FILTER_VARIANCE + triangle_variance(smooth_radius);
    let width_ratio = (smooth / (smooth + triangle_variance(gate_radius))).sqrt();
    let iso = iso_fraction.clamp(0.02, 0.98);
    1.0 / (1.0 + ((1.0 - iso) / iso).powf(width_ratio))
}

/// Same layout as `BlurParams` in mc_blur.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuHalfBlurParams {
    dir: [i32; 3],
    radius: i32,
    grid_size: u32,
    _pad: [u32; 3],
}

pub struct CalmSmoothing {
    full_size: u32,
    half_size: u32,
    _half_textures: [wgpu::Texture; 3],
    /// G, once `encode_helpers` has run
    gate_view: wgpu::TextureView,
    params_buffer: wgpu::Buffer,
    /// S chain (X, Y, Z) then G chain (X, Y, Z); radii rewritten each update
    blur_params_buffers: [wgpu::Buffer; 6],
    downsample_pipeline: wgpu::ComputePipeline,
    blur_pipeline: wgpu::ComputePipeline,
    combine_pipeline: wgpu::ComputePipeline,
    downsample_bind_group: wgpu::BindGroup,
    /// S chain (X, Y, Z) then G chain (X, Y, Z)
    blur_bind_groups: [wgpu::BindGroup; 6],
    /// [0]: base field in A, writes B; [1]: base field in B, writes A
    combine_bind_groups: [wgpu::BindGroup; 2],
}

impl CalmSmoothing {
    /// `field_a` / `field_b` are the MC density ping-pong textures; the raw
    /// density pass writes A.
    pub fn new(
        device: &wgpu::Device,
        full_size: u32,
        field_a: &wgpu::TextureView,
        field_b: &wgpu::TextureView,
    ) -> Self {
        let half_size = full_size.div_ceil(2);

        let half_textures: [wgpu::Texture; 3] = std::array::from_fn(|i| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("MC Calm Half Field {}", i)),
                size: wgpu::Extent3d {
                    width: half_size,
                    height: half_size,
                    depth_or_array_layers: half_size,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        });
        let half_views: [wgpu::TextureView; 3] =
            std::array::from_fn(|i| half_textures[i].create_view(&wgpu::TextureViewDescriptor::default()));

        let calm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Calm Smoothing Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mc_calm_smooth.wgsl").into()),
        });
        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Calm Half Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mc_blur.wgsl").into()),
        });
        let compute_pipeline = |label: &str, module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let downsample_pipeline = compute_pipeline("MC Calm Downsample", &calm_shader, "downsample");
        let combine_pipeline = compute_pipeline("MC Calm Combine", &calm_shader, "combine");
        let blur_pipeline = compute_pipeline("MC Calm Half Blur", &blur_shader, "main");

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Calm Params"),
            contents: bytemuck::bytes_of(&GpuCalmParams {
                full_size,
                half_size,
                gate_lo: 0.0,
                gate_hi: 1.0,
                strength: 0.0,
                iso: 0.0,
                _pad: [0.0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // S chain then G chain; radii are filled in by `update`
        let blur_params_buffers: [wgpu::Buffer; 6] = std::array::from_fn(|i| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MC Calm Half Blur Params"),
                contents: bytemuck::bytes_of(&GpuHalfBlurParams {
                    dir: DIRECTIONS[i % 3],
                    radius: 1,
                    grid_size: half_size,
                    _pad: [0; 3],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });

        let downsample_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MC Calm Downsample BG"),
            layout: &downsample_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(field_a) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&half_views[0]) },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        // Downsampled raw field lands in H0. S: X H0->H1, Y H1->H2, Z H2->H1.
        // G (from S, leaving S intact): X H1->H0, Y H0->H2, Z H2->H0.
        let blur_layout = blur_pipeline.get_bind_group_layout(0);
        let chain: [(usize, usize); 6] = [(0, 1), (1, 2), (2, 1), (1, 0), (0, 2), (2, 0)];
        let blur_bind_groups: [wgpu::BindGroup; 6] = std::array::from_fn(|i| {
            let (src, dst) = chain[i];
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MC Calm Half Blur BG"),
                layout: &blur_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&half_views[src]) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&half_views[dst]) },
                    wgpu::BindGroupEntry { binding: 2, resource: blur_params_buffers[i].as_entire_binding() },
                ],
            })
        });

        let combine_layout = combine_pipeline.get_bind_group_layout(0);
        let combine_bind_groups: [wgpu::BindGroup; 2] = std::array::from_fn(|i| {
            let (base, out) = if i == 0 { (field_a, field_b) } else { (field_b, field_a) };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MC Calm Combine BG"),
                layout: &combine_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(base) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(out) },
                    wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&half_views[1]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&half_views[0]) },
                ],
            })
        });

        let [gate_view, ..] = half_views;
        Self {
            full_size,
            half_size,
            _half_textures: half_textures,
            gate_view,
            params_buffer,
            blur_params_buffers,
            downsample_pipeline,
            blur_pipeline,
            combine_pipeline,
            downsample_bind_group,
            blur_bind_groups,
            combine_bind_groups,
        }
    }

    /// `strength` 0..1 (0 = off); `kernel_radius` is the sim h, which sets the
    /// rest spacing (so the interior field value the gate is scaled to) and
    /// the filter widths; `cell_size` is the full-res MC voxel size and
    /// `iso_value` the field value the mesh is extracted at.
    pub fn update(&self, queue: &wgpu::Queue, strength: f32, kernel_radius: f32, cell_size: f32, iso_value: f32) {
        // Triangle filter of radius r spans r + 1 cells each side
        let half_cell = 2.0 * cell_size;
        let radius_for = |half_width_h: f32| {
            ((half_width_h * kernel_radius / half_cell).round() as i32 - 1).max(1)
        };
        let radii = [radius_for(SMOOTH_HALF_WIDTH_H), radius_for(GATE_HALF_WIDTH_H)];
        for (i, buffer) in self.blur_params_buffers.iter().enumerate() {
            let blur = GpuHalfBlurParams {
                dir: DIRECTIONS[i % 3],
                radius: radii[i / 3],
                grid_size: self.half_size,
                _pad: [0; 3],
            };
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&blur));
        }

        let spacing = REST_SPACING_FACTOR * kernel_radius;
        let interior = 1.0 / (spacing * spacing * spacing);
        let bulk_gate = bulk_surface_gate(iso_value / interior, radii[0], radii[1]) * interior;
        let params = GpuCalmParams {
            full_size: self.full_size,
            half_size: self.half_size,
            gate_lo: GATE_LO * bulk_gate,
            gate_hi: GATE_HI * bulk_gate,
            strength: strength.clamp(0.0, 1.0),
            iso: iso_value,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// The gate field G (half resolution; outside-container voxels hold -1).
    /// Only current on frames `encode_helpers` ran.
    pub fn gate_view(&self) -> &wgpu::TextureView {
        &self.gate_view
    }

    /// Build S and G from the raw density field. Must run while A still holds
    /// the raw field, i.e. before the base blur ping-pongs through it.
    pub fn encode_helpers(&self, encoder: &mut wgpu::CommandEncoder) {
        let half_groups = self.half_size.div_ceil(4);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MC Calm Downsample"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.downsample_pipeline);
            pass.set_bind_group(0, &self.downsample_bind_group, &[]);
            pass.dispatch_workgroups(half_groups, half_groups, half_groups);
        }
        // Separate passes: each blur reads the previous one's output
        for bind_group in &self.blur_bind_groups {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MC Calm Half Blur"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(half_groups, half_groups, half_groups);
        }
    }

    /// Blend the base field toward S where G marks bulk water. Returns where
    /// the result landed: true = texture B.
    pub fn encode_combine(&self, encoder: &mut wgpu::CommandEncoder, base_in_b: bool) -> bool {
        let groups = self.full_size.div_ceil(4);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MC Calm Combine"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.combine_pipeline);
        pass.set_bind_group(0, &self.combine_bind_groups[base_in_b as usize], &[]);
        pass.dispatch_workgroups(groups, groups, groups);
        !base_in_b
    }
}
