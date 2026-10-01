//! GPU fluid measurement probes — global extents (dam-break front tracking)
//! and per-column surface heights (wave gauges).
//!
//! One tiny compute pass per frame over the post-integrate particle buffer,
//! atomicMax into 16 quantized slots, 64-byte readback. Stats runs read
//! blocking for exact per-frame CSV rows (same policy as the MC vertex
//! count); interactive frames use the non-blocking map pattern from the
//! spray auto-limit readback so the CPU never stalls on the GPU.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::state::{FluidMeasurements, ProbeConfig, MAX_PROBES};

const WORKGROUP_SIZE: u32 = 256;
const RESULT_SLOTS: usize = 16;
const PROBE_BASE: usize = 8;
const QUANT_SCALE: f32 = 1_000_000.0;
const QUANT_OFFSET: f32 = 16.0;

const MAP_PENDING: u32 = 0;
const MAP_READY: u32 = 1;
const MAP_FAILED: u32 = 2;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuProbeParams {
    num_particles: u32,
    num_probes: u32,
    _pad: [u32; 2],
    /// xyzw = probe x, probe z, radius squared, unused
    probes: [[f32; 4]; MAX_PROBES],
}

pub struct ProbeSystem {
    params_buffer: wgpu::Buffer,
    results_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    probes: Vec<ProbeConfig>,
    /// A results→staging copy was recorded this frame (staging is fresh)
    copy_staged: bool,
    /// Staging buffer is mapped or mapping (interactive non-blocking path)
    inflight: bool,
    map_state: Arc<AtomicU32>,
    latest: Option<FluidMeasurements>,
}

impl ProbeSystem {
    pub fn new(
        device: &wgpu::Device,
        particle_buffer: &wgpu::Buffer,
        probes: &[ProbeConfig],
    ) -> Self {
        let probes: Vec<ProbeConfig> = probes.iter().take(MAX_PROBES).cloned().collect();

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Probe Params"),
            contents: bytemuck::bytes_of(&GpuProbeParams {
                num_particles: 0,
                num_probes: probes.len() as u32,
                _pad: [0; 2],
                probes: [[0.0; 4]; MAX_PROBES],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let results_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Probe Results"),
            contents: bytemuck::bytes_of(&[0u32; RESULT_SLOTS]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Probe Results Staging"),
            size: (RESULT_SLOTS * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Probe Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/probes.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Probe BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Probe Bind Group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: results_buffer.as_entire_binding(),
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Probe Pipeline Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Probe Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("probe_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            params_buffer,
            results_buffer,
            staging_buffer,
            pipeline,
            bind_group,
            probes,
            copy_staged: false,
            inflight: false,
            map_state: Arc::new(AtomicU32::new(MAP_PENDING)),
            latest: None,
        }
    }

    /// Harvest a completed non-blocking readback, if any (call before encode).
    pub fn collect(&mut self, device: &wgpu::Device) {
        if !self.inflight {
            return;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        match self.map_state.load(Ordering::Acquire) {
            MAP_READY => {
                let bits = {
                    let data = self.staging_buffer.slice(..).get_mapped_range();
                    *bytemuck::from_bytes::<[u32; RESULT_SLOTS]>(&data)
                };
                self.staging_buffer.unmap();
                self.inflight = false;
                self.latest = Some(self.decode(&bits));
            }
            MAP_FAILED => {
                // Map failed (device loss etc.) — buffer is not mapped; re-arm
                self.inflight = false;
            }
            _ => {} // still pending — try again next frame
        }
    }

    /// Record this frame's measurement pass: clear results, dispatch one
    /// thread per particle, stage the copy for readback when staging is free.
    pub fn encode(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        num_particles: u32,
    ) {
        let params = GpuProbeParams {
            num_particles,
            num_probes: self.probes.len() as u32,
            _pad: [0; 2],
            probes: std::array::from_fn(|i| {
                self.probes.get(i).map_or([0.0; 4], |p| {
                    [p.x, p.z, p.radius * p.radius, 0.0]
                })
            }),
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(
            &self.results_buffer,
            0,
            bytemuck::bytes_of(&[0u32; RESULT_SLOTS]),
        );

        if num_particles > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Probe Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(num_particles.div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        self.copy_staged = !self.inflight;
        if self.copy_staged {
            encoder.copy_buffer_to_buffer(
                &self.results_buffer,
                0,
                &self.staging_buffer,
                0,
                (RESULT_SLOTS * std::mem::size_of::<u32>()) as u64,
            );
        }
    }

    /// Interactive path: start the async map after the encoder is submitted.
    pub fn arm_map(&mut self) {
        if !self.copy_staged || self.inflight {
            return;
        }
        let map_state = self.map_state.clone();
        map_state.store(MAP_PENDING, Ordering::Release);
        self.staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let state = if result.is_ok() { MAP_READY } else { MAP_FAILED };
                map_state.store(state, Ordering::Release);
            });
        self.inflight = true;
        self.copy_staged = false;
    }

    /// Stats path: block for this frame's exact values (deterministic CSV
    /// rows). Only valid when the non-blocking path is unused, so staging is
    /// always free and `encode` staged a copy this frame.
    pub fn read_blocking(&mut self, device: &wgpu::Device) -> Option<FluidMeasurements> {
        if !self.copy_staged || self.inflight {
            return self.latest.clone();
        }
        let slice = self.staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let bits = {
            let data = slice.get_mapped_range();
            *bytemuck::from_bytes::<[u32; RESULT_SLOTS]>(&data)
        };
        self.staging_buffer.unmap();
        self.copy_staged = false;
        self.latest = Some(self.decode(&bits));
        self.latest.clone()
    }

    pub fn latest(&self) -> Option<&FluidMeasurements> {
        self.latest.as_ref()
    }

    fn decode(&self, bits: &[u32; RESULT_SLOTS]) -> FluidMeasurements {
        let dq = |q: u32| (q != 0).then(|| q as f32 / QUANT_SCALE - QUANT_OFFSET);
        FluidMeasurements {
            max_x: dq(bits[0]),
            min_x: dq(bits[1]).map(|v| -v),
            max_y: dq(bits[2]),
            max_z: dq(bits[3]),
            min_z: dq(bits[4]).map(|v| -v),
            probe_heights: (0..self.probes.len())
                .map(|k| dq(bits[PROBE_BASE + k]))
                .collect(),
        }
    }
}
