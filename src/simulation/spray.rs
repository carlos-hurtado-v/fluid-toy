//! GPU whitewater particle system (Ihmsen et al. 2012 diffuse particles) —
//! emission from trapped-air / wave-crest potentials over the SPH grid, then
//! per-class simulation (ballistic spray, surface-advected foam, buoyant
//! bubbles). Both passes read the grid-sorted particle buffer, which is
//! consistent with the spatial grid built during the last SPH substep.

use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

use wgpu::util::DeviceExt;

use crate::state::{GpuSprayParams, GpuSprayParticle};

const WORKGROUP_SIZE: u32 = 64;

// Online auto-calibration of the emission potential limits (the runtime
// version of SPlisHSPlasH's whole-simulation pre-scan): asymmetric EMA over
// the per-frame potential maxima reported by the emit shader. The target
// statistic is Bender's AVERAGE of per-frame maxima, not the running peak —
// a peak-tracking limit parks the whole potential distribution at the remap
// floor and starves emission (3x count cut in eval). Mild asymmetry tempers
// the burst when a first big event arrives while the limit is still low.
// Frames with no eligible emitters hold the limit, so calm never collapses it.
const AUTO_LIMIT_RISE: f32 = 0.05;
const AUTO_LIMIT_DECAY: f32 = 0.01;
/// Initial guesses for the volume-weighted potential scale — the EMA
/// converges within a second of the first real event.
pub const AUTO_TA_INIT: f32 = 3.0;
pub const AUTO_WC_INIT: f32 = 1.0;
const AUTO_TA_FLOOR: f32 = 0.2;
const AUTO_WC_FLOOR: f32 = 0.05;

// emit_stats_staging map state, shared with the map_async callback
const STATS_MAP_PENDING: u32 = 0;
const STATS_MAP_READY: u32 = 1;
const STATS_MAP_FAILED: u32 = 2;

fn update_auto_limit(current: f32, frame_max: f32, floor: f32) -> f32 {
    if frame_max <= 1e-6 {
        return current;
    }
    let alpha = if frame_max > current {
        AUTO_LIMIT_RISE
    } else {
        AUTO_LIMIT_DECAY
    };
    (current + alpha * (frame_max - current)).max(floor)
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub struct SpraySystem {
    spray_buffer: wgpu::Buffer,
    write_head_buffer: wgpu::Buffer,
    spray_params_buffer: wgpu::Buffer,
    // Live-particle counts written by the simulate pass: [total, spray, foam, bubble]
    stats_buffer: wgpu::Buffer,
    stats_staging_buffer: wgpu::Buffer,
    // Per-frame potential maxima from the emit pass (f32 bit patterns) and the
    // non-blocking readback used to feed the auto-limit EMA
    emit_stats_buffer: wgpu::Buffer,
    emit_stats_staging: wgpu::Buffer,
    emit_stats_inflight: bool,
    emit_stats_map_state: Arc<AtomicU32>,
    auto_ta_max: f32,
    auto_wc_max: f32,

    emit_pipeline: wgpu::ComputePipeline,
    emit_bind_group: wgpu::BindGroup,

    simulate_pipeline: wgpu::ComputePipeline,
    simulate_bind_group: wgpu::BindGroup,

    max_spray_particles: u32,
}

impl SpraySystem {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        sorted_particle_buffer: &wgpu::Buffer,
        sph_params_buffer: &wgpu::Buffer,
        container_geom_buffer: &wgpu::Buffer,
        cell_starts_buffer: &wgpu::Buffer,
        cell_counts_buffer: &wgpu::Buffer,
        grid_params_buffer: &wgpu::Buffer,
        max_spray: u32,
        initial_params: &GpuSprayParams,
    ) -> Self {
        // Diffuse particle ring buffer (initialized to zeros = all dead)
        let spray_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spray Particle Buffer"),
            size: (max_spray as usize * std::mem::size_of::<GpuSprayParticle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Atomic write head (single u32)
        let write_head_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spray Write Head"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Spray params uniform
        let spray_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spray Params"),
            contents: bytemuck::bytes_of(initial_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Per-kind live counts: [total, spray, foam, bubble]
        let stats_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spray Stats"),
            contents: bytemuck::bytes_of(&[0u32; 4]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let stats_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spray Stats Staging"),
            size: (4 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Per-frame potential maxima: [trapped_air_bits, wave_crest_bits, 2x reserved]
        let emit_stats_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spray Emit Stats"),
            contents: bytemuck::bytes_of(&[0u32; 4]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let emit_stats_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spray Emit Stats Staging"),
            size: (4 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // === Emit Pipeline ===
        let emit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spray Emit Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/spray_emit.wgsl").into()),
        });

        let emit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spray Emit BGL"),
            entries: &[
                storage_entry(0, true),  // sorted SPH particles
                storage_entry(1, false), // spray particles
                storage_entry(2, false), // write head (atomic)
                uniform_entry(3),        // spray params
                uniform_entry(4),        // sph params
                storage_entry(5, true),  // cell starts
                storage_entry(6, true),  // cell counts
                uniform_entry(7),        // grid params
                storage_entry(8, false), // per-frame potential maxima (atomic)
            ],
        });

        let emit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Spray Emit Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Spray Emit Pipeline Layout"),
                bind_group_layouts: &[&emit_bgl],
                push_constant_ranges: &[],
            })),
            module: &emit_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let emit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Spray Emit BG"),
            layout: &emit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sorted_particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spray_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: write_head_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: spray_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sph_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cell_starts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: cell_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: grid_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: emit_stats_buffer.as_entire_binding(),
                },
            ],
        });

        // === Simulate Pipeline ===
        let container_common_wgsl = include_str!("../shaders/container_common.wgsl");
        let simulate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spray Simulate Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}", container_common_wgsl, include_str!("../shaders/spray_simulate.wgsl")).into(),
            ),
        });

        let simulate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spray Simulate BGL"),
            entries: &[
                storage_entry(0, false), // spray particles
                uniform_entry(1),        // spray params
                uniform_entry(2),        // container geometry
                storage_entry(3, true),  // sorted SPH particles
                storage_entry(4, true),  // cell starts
                storage_entry(5, true),  // cell counts
                uniform_entry(6),        // grid params
                uniform_entry(7),        // sph params
                storage_entry(8, false), // live stats (atomic counters)
            ],
        });

        let simulate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Spray Simulate Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Spray Simulate Pipeline Layout"),
                bind_group_layouts: &[&simulate_bgl],
                push_constant_ranges: &[],
            })),
            module: &simulate_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let simulate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Spray Simulate BG"),
            layout: &simulate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spray_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spray_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: container_geom_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sorted_particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cell_starts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cell_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grid_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: sph_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: stats_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            spray_buffer,
            write_head_buffer,
            spray_params_buffer,
            stats_buffer,
            stats_staging_buffer,
            emit_stats_buffer,
            emit_stats_staging,
            emit_stats_inflight: false,
            emit_stats_map_state: Arc::new(AtomicU32::new(STATS_MAP_PENDING)),
            auto_ta_max: AUTO_TA_INIT,
            auto_wc_max: AUTO_WC_INIT,
            emit_pipeline,
            emit_bind_group,
            simulate_pipeline,
            simulate_bind_group,
            max_spray_particles: max_spray,
        }
    }

    pub fn update_params(&self, queue: &wgpu::Queue, params: &GpuSprayParams) {
        queue.write_buffer(&self.spray_params_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, num_sph_particles: u32) {
        // Harvest the previous frame's potential maxima (non-blocking) and
        // advance the auto-limit EMA before params are read this frame
        self.collect_emit_stats(device);

        // Zero live-count stats; the simulate pass re-counts every step
        queue.write_buffer(&self.stats_buffer, 0, bytemuck::bytes_of(&[0u32; 4]));
        queue.write_buffer(&self.emit_stats_buffer, 0, bytemuck::bytes_of(&[0u32; 4]));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Spray Encoder"),
        });

        // Emit pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Spray Emit Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.emit_pipeline);
            pass.set_bind_group(0, &self.emit_bind_group, &[]);
            pass.dispatch_workgroups(num_sph_particles.div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        // Simulate pass (classifies particles emitted this frame too)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Spray Simulate Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.simulate_pipeline);
            pass.set_bind_group(0, &self.simulate_bind_group, &[]);
            pass.dispatch_workgroups(
                self.max_spray_particles.div_ceil(WORKGROUP_SIZE),
                1,
                1,
            );
        }

        // Stage stats for optional CPU readback (16 bytes, negligible)
        encoder.copy_buffer_to_buffer(
            &self.stats_buffer,
            0,
            &self.stats_staging_buffer,
            0,
            (4 * std::mem::size_of::<u32>()) as u64,
        );

        // Stage the potential maxima whenever the staging buffer is free
        // (skipped frames just make the EMA see slightly older maxima)
        let arm_emit_stats = !self.emit_stats_inflight;
        if arm_emit_stats {
            encoder.copy_buffer_to_buffer(
                &self.emit_stats_buffer,
                0,
                &self.emit_stats_staging,
                0,
                (4 * std::mem::size_of::<u32>()) as u64,
            );
        }

        queue.submit(std::iter::once(encoder.finish()));

        if arm_emit_stats {
            let map_state = self.emit_stats_map_state.clone();
            map_state.store(STATS_MAP_PENDING, Ordering::Release);
            self.emit_stats_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let state = if result.is_ok() {
                        STATS_MAP_READY
                    } else {
                        STATS_MAP_FAILED
                    };
                    map_state.store(state, Ordering::Release);
                });
            self.emit_stats_inflight = true;
        }
    }

    /// Non-blocking harvest of the emit pass potential maxima; advances the
    /// auto-calibrated limit EMA when a mapped result is available.
    fn collect_emit_stats(&mut self, device: &wgpu::Device) {
        if !self.emit_stats_inflight {
            return;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        match self.emit_stats_map_state.load(Ordering::Acquire) {
            STATS_MAP_READY => {
                let bits = {
                    let data = self.emit_stats_staging.slice(..).get_mapped_range();
                    *bytemuck::from_bytes::<[u32; 4]>(&data)
                };
                self.emit_stats_staging.unmap();
                self.emit_stats_inflight = false;
                self.auto_ta_max =
                    update_auto_limit(self.auto_ta_max, f32::from_bits(bits[0]), AUTO_TA_FLOOR);
                self.auto_wc_max =
                    update_auto_limit(self.auto_wc_max, f32::from_bits(bits[1]), AUTO_WC_FLOOR);
            }
            STATS_MAP_FAILED => {
                // Map failed (device loss etc.) — buffer is not mapped; re-arm
                self.emit_stats_inflight = false;
            }
            _ => {} // still pending — try again next frame
        }
    }

    /// Current auto-calibrated potential ceilings: (trapped_air, wave_crest)
    pub fn auto_limits(&self) -> (f32, f32) {
        (self.auto_ta_max, self.auto_wc_max)
    }

    /// Read back live particle counts from the last step: [total, spray, foam, bubble].
    /// Blocks until the GPU finishes — intended for stats/automation runs.
    pub fn read_stats(&self, device: &wgpu::Device) -> [u32; 4] {
        let slice = self.stats_staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let counts = {
            let data = slice.get_mapped_range();
            *bytemuck::from_bytes::<[u32; 4]>(&data)
        };
        self.stats_staging_buffer.unmap();
        counts
    }

    pub fn spray_buffer(&self) -> &wgpu::Buffer {
        &self.spray_buffer
    }

    /// Clear spray buffer and reset write head (for simulation reset)
    pub fn reset(&mut self, queue: &wgpu::Queue) {
        // Zero out all spray particles (lifetime=0 means dead)
        let zeros = vec![0u8; self.max_spray_particles as usize * std::mem::size_of::<GpuSprayParticle>()];
        queue.write_buffer(&self.spray_buffer, 0, &zeros);
        queue.write_buffer(&self.write_head_buffer, 0, bytemuck::bytes_of(&0u32));
        // Recalibrate limits for the new scene (converges within ~1 s)
        self.auto_ta_max = AUTO_TA_INIT;
        self.auto_wc_max = AUTO_WC_INIT;
    }
}
