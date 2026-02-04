//! Simulation module - physics computation on GPU

pub mod particle;
pub mod sph;
pub mod sph_3d;
pub mod sph_3d_grid;

pub use particle::{Particle, SphParticle, SphParticle3D};
pub use sph::SphSimulation;
pub use sph_3d::SphSimulation3D;
pub use sph_3d_grid::SphSimulation3DGrid;

use crate::state::GpuSimParams;
use wgpu::util::DeviceExt;

/// Manages GPU-based particle simulation
pub struct Simulation {
    compute_pipeline: wgpu::ComputePipeline,
    particle_buffers: [wgpu::Buffer; 2],
    bind_groups: [wgpu::BindGroup; 2],
    params_buffer: wgpu::Buffer,
    frame: usize,
    num_particles: u32,
    workgroup_count: u32,
}

const WORKGROUP_SIZE: u32 = 64;

impl Simulation {
    pub fn new(device: &wgpu::Device, particles: &[Particle], params: GpuSimParams) -> Self {
        let num_particles = particles.len() as u32;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Simulation Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compute.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Simulation Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<GpuSimParams>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            (num_particles as u64) * std::mem::size_of::<Particle>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            (num_particles as u64) * std::mem::size_of::<Particle>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Simulation Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Simulation Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let particle_buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Particle Buffer A"),
                contents: bytemuck::cast_slice(particles),
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Particle Buffer B"),
                contents: bytemuck::cast_slice(particles),
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            }),
        ];

        let bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Simulation Bind Group A"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_buffers[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: particle_buffers[1].as_entire_binding(),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Simulation Bind Group B"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_buffers[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: particle_buffers[0].as_entire_binding(),
                    },
                ],
            }),
        ];

        let workgroup_count = num_particles.div_ceil(WORKGROUP_SIZE);

        Self {
            compute_pipeline,
            particle_buffers,
            bind_groups,
            params_buffer,
            frame: 0,
            num_particles,
            workgroup_count,
        }
    }

    /// Run one simulation step
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Simulation Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &self.bind_groups[self.frame % 2], &[]);
        compute_pass.dispatch_workgroups(self.workgroup_count, 1, 1);

        drop(compute_pass);
        self.frame += 1;
    }

    /// Update simulation parameters from state
    pub fn update_params(&self, queue: &wgpu::Queue, params: &GpuSimParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Get the current particle buffer for rendering
    pub fn current_particle_buffer(&self) -> &wgpu::Buffer {
        &self.particle_buffers[self.frame % 2]
    }

    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }
}
