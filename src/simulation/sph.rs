//! SPH (Smoothed Particle Hydrodynamics) simulation module

use crate::simulation::particle::SphParticle;
use crate::state::{GpuBoundsParams, GpuSphParams};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 64;

/// SPH simulation with 3-pass compute pipeline
pub struct SphSimulation {
    // Compute pipelines
    density_pipeline: wgpu::ComputePipeline,
    force_pipeline: wgpu::ComputePipeline,
    integrate_pipeline: wgpu::ComputePipeline,

    // Buffers
    particle_buffer: wgpu::Buffer,
    sph_params_buffer: wgpu::Buffer,
    bounds_buffer: wgpu::Buffer,

    // Bind groups
    density_bind_group: wgpu::BindGroup,
    force_bind_group: wgpu::BindGroup,
    integrate_bind_group: wgpu::BindGroup,

    num_particles: u32,
    workgroup_count: u32,
}

impl SphSimulation {
    pub fn new(
        device: &wgpu::Device,
        particles: &[SphParticle],
        sph_params: GpuSphParams,
        bounds_params: GpuBoundsParams,
    ) -> Self {
        let num_particles = particles.len() as u32;
        let workgroup_count = num_particles.div_ceil(WORKGROUP_SIZE);

        // Create shader modules
        let density_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SPH Density Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sph_density.wgsl").into()),
        });

        let force_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SPH Force Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sph_force.wgsl").into()),
        });

        let integrate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SPH Integrate Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sph_integrate.wgsl").into()),
        });

        // Create buffers
        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SPH Particle Buffer"),
            contents: bytemuck::cast_slice(particles),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });

        let sph_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SPH Params Buffer"),
            contents: bytemuck::bytes_of(&sph_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bounds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bounds Params Buffer"),
            contents: bytemuck::bytes_of(&bounds_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bind group layouts

        // Density/Force: params + particles (read-write)
        let density_force_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SPH Density/Force Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<GpuSphParams>() as u64,
                        ),
                    },
                    count: None,
                },
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
            ],
        });

        // Integrate: params + particles + bounds
        let integrate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SPH Integrate Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<GpuSphParams>() as u64,
                        ),
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<GpuBoundsParams>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });

        // Create pipelines
        let density_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SPH Density Pipeline Layout"),
            bind_group_layouts: &[&density_force_layout],
            push_constant_ranges: &[],
        });

        let density_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SPH Density Pipeline"),
            layout: Some(&density_pipeline_layout),
            module: &density_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let force_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SPH Force Pipeline Layout"),
            bind_group_layouts: &[&density_force_layout],
            push_constant_ranges: &[],
        });

        let force_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SPH Force Pipeline"),
            layout: Some(&force_pipeline_layout),
            module: &force_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let integrate_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SPH Integrate Pipeline Layout"),
            bind_group_layouts: &[&integrate_layout],
            push_constant_ranges: &[],
        });

        let integrate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SPH Integrate Pipeline"),
            layout: Some(&integrate_pipeline_layout),
            module: &integrate_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create bind groups
        let density_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SPH Density Bind Group"),
            layout: &density_force_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sph_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particle_buffer.as_entire_binding(),
                },
            ],
        });

        let force_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SPH Force Bind Group"),
            layout: &density_force_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sph_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particle_buffer.as_entire_binding(),
                },
            ],
        });

        let integrate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SPH Integrate Bind Group"),
            layout: &integrate_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sph_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bounds_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            density_pipeline,
            force_pipeline,
            integrate_pipeline,
            particle_buffer,
            sph_params_buffer,
            bounds_buffer,
            density_bind_group,
            force_bind_group,
            integrate_bind_group,
            num_particles,
            workgroup_count,
        }
    }

    /// Run one simulation step (3 compute passes)
    pub fn step(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SPH Compute Pass"),
            timestamp_writes: None,
        });

        // Pass 1: Compute density
        compute_pass.set_pipeline(&self.density_pipeline);
        compute_pass.set_bind_group(0, &self.density_bind_group, &[]);
        compute_pass.dispatch_workgroups(self.workgroup_count, 1, 1);

        // Pass 2: Compute forces (requires density from pass 1)
        compute_pass.set_pipeline(&self.force_pipeline);
        compute_pass.set_bind_group(0, &self.force_bind_group, &[]);
        compute_pass.dispatch_workgroups(self.workgroup_count, 1, 1);

        // Pass 3: Integrate (update position/velocity)
        compute_pass.set_pipeline(&self.integrate_pipeline);
        compute_pass.set_bind_group(0, &self.integrate_bind_group, &[]);
        compute_pass.dispatch_workgroups(self.workgroup_count, 1, 1);
    }

    /// Update SPH parameters
    pub fn update_sph_params(&self, queue: &wgpu::Queue, params: &GpuSphParams) {
        queue.write_buffer(&self.sph_params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Update boundary parameters
    pub fn update_bounds_params(&self, queue: &wgpu::Queue, params: &GpuBoundsParams) {
        queue.write_buffer(&self.bounds_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Get the particle buffer for rendering
    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        &self.particle_buffer
    }

    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }
}
