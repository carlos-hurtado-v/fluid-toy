//! GPU spray particle system — emission from high-energy SPH surface particles
//! and ballistic simulation with gravity + drag.

use wgpu::util::DeviceExt;

use crate::state::{GpuSprayParams, GpuSprayParticle};

const WORKGROUP_SIZE: u32 = 64;

pub struct SpraySystem {
    spray_buffer: wgpu::Buffer,
    write_head_buffer: wgpu::Buffer,
    spray_params_buffer: wgpu::Buffer,

    emit_pipeline: wgpu::ComputePipeline,
    emit_bind_group: wgpu::BindGroup,

    simulate_pipeline: wgpu::ComputePipeline,
    simulate_bind_group: wgpu::BindGroup,

    max_spray_particles: u32,
}

impl SpraySystem {
    pub fn new(
        device: &wgpu::Device,
        sph_particle_buffer: &wgpu::Buffer,
        sph_params_buffer: &wgpu::Buffer,
        container_geom_buffer: &wgpu::Buffer,
        max_spray: u32,
        initial_params: &GpuSprayParams,
    ) -> Self {
        // Spray particle ring buffer (initialized to zeros = all dead)
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

        // === Emit Pipeline ===
        let emit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spray Emit Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/spray_emit.wgsl").into()),
        });

        let emit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spray Emit BGL"),
            entries: &[
                // SPH particles (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Spray particles (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Write head (atomic)
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
                // Spray params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // SPH params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                    resource: sph_particle_buffer.as_entire_binding(),
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
                // Spray particles (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Spray params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bounds params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            ],
        });

        Self {
            spray_buffer,
            write_head_buffer,
            spray_params_buffer,
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

    pub fn step(&self, device: &wgpu::Device, queue: &wgpu::Queue, num_sph_particles: u32) {
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

        // Simulate pass
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

        queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn spray_buffer(&self) -> &wgpu::Buffer {
        &self.spray_buffer
    }

    /// Clear spray buffer and reset write head (for simulation reset)
    pub fn reset(&self, queue: &wgpu::Queue) {
        // Zero out all spray particles (lifetime=0 means dead)
        let zeros = vec![0u8; self.max_spray_particles as usize * std::mem::size_of::<GpuSprayParticle>()];
        queue.write_buffer(&self.spray_buffer, 0, &zeros);
        queue.write_buffer(&self.write_head_buffer, 0, bytemuck::bytes_of(&0u32));
    }
}
