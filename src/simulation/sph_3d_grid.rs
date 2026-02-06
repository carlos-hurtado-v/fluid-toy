//! 3D SPH simulation with spatial hashing for O(n) neighbor search

use crate::simulation::particle::SphParticle3D;
use crate::state::{GpuBoundsParams3D, GpuGravity, GpuMouseForce, GpuSphParams3D};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 64;

/// Grid parameters for spatial hashing
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGridParams {
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub total_cells: u32,
    pub cell_size: f32,
    pub inv_cell_size: f32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub num_particles: u32,
    pub _padding: [u32; 2],
}

/// Prefix sum parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrefixSumParams {
    pub count: u32,
    pub offset: u32,
    pub _padding: [u32; 2],
}

/// 3D SPH simulation with grid-accelerated neighbor search
pub struct SphSimulation3DGrid {
    // Compute pipelines
    grid_clear_pipeline: wgpu::ComputePipeline,
    grid_build_pipeline: wgpu::ComputePipeline,
    prefix_sum_pipeline: wgpu::ComputePipeline,
    grid_reorder_pipeline: wgpu::ComputePipeline,
    density_pipeline: wgpu::ComputePipeline,
    force_pipeline: wgpu::ComputePipeline,
    integrate_pipeline: wgpu::ComputePipeline,

    // Particle buffers
    particle_buffer: wgpu::Buffer,
    _sorted_particle_buffer: wgpu::Buffer,

    // Grid buffers
    cell_counts_buffer: wgpu::Buffer,
    cell_counts_temp_buffer: wgpu::Buffer, // For prefix sum ping-pong
    cell_starts_buffer: wgpu::Buffer,
    cell_offsets_buffer: wgpu::Buffer, // Reset for reorder atomic counter
    _particle_cell_indices_buffer: wgpu::Buffer,

    // Parameter buffers
    sph_params_buffer: wgpu::Buffer,
    bounds_buffer: wgpu::Buffer,
    mouse_force_buffer: wgpu::Buffer,
    gravity_buffer: wgpu::Buffer,
    grid_params_buffer: wgpu::Buffer,
    prefix_sum_params_buffer: wgpu::Buffer,

    // Bind groups
    grid_clear_bind_group: wgpu::BindGroup,
    grid_build_bind_group: wgpu::BindGroup,
    prefix_sum_bind_groups: Vec<(wgpu::BindGroup, wgpu::BindGroup)>, // Pairs for ping-pong
    grid_reorder_bind_group: wgpu::BindGroup,
    density_bind_group: wgpu::BindGroup,
    force_bind_group: wgpu::BindGroup,
    integrate_bind_group: wgpu::BindGroup,

    // Grid dimensions
    grid_params: GpuGridParams,
    num_particles: u32,
    max_particles: u32,
    num_prefix_sum_passes: u32,
}

impl SphSimulation3DGrid {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &[SphParticle3D],
        sph_params: GpuSphParams3D,
        bounds_params: GpuBoundsParams3D,
        max_particles: u32,
    ) -> Self {
        let num_particles = particles.len() as u32;
        let max_particles = max_particles.max(num_particles); // Ensure at least enough for initial particles

        // Calculate grid dimensions based on bounds and kernel radius
        // Use sqrt(3) multiplier to handle tilted containers (diagonal extends further)
        let cell_size = sph_params.kernel_radius;
        // For Y extent, use max of |floor| and |ceiling| since container can be asymmetric
        let y_extent = bounds_params.floor_y.abs().max(bounds_params.ceiling_y.abs());
        let base_extent = bounds_params.bound_x.max(y_extent).max(bounds_params.bound_z);
        let bounds_extent = base_extent * 1.8 + 0.2;  // ~sqrt(3) for worst-case tilt + margin
        let grid_size = ((2.0 * bounds_extent) / cell_size).ceil() as u32 + 2;
        let total_cells = grid_size * grid_size * grid_size;

        let grid_params = GpuGridParams {
            grid_size_x: grid_size,
            grid_size_y: grid_size,
            grid_size_z: grid_size,
            total_cells,
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            grid_origin_x: -bounds_extent - cell_size,
            grid_origin_y: -bounds_extent - cell_size,
            grid_origin_z: -bounds_extent - cell_size,
            num_particles,
            _padding: [0; 2],
        };

        // Calculate number of prefix sum passes needed
        let num_prefix_sum_passes = (total_cells as f32).log2().ceil() as u32;

        // Create shader modules
        let grid_clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grid Clear Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/grid_clear.wgsl").into()),
        });

        let grid_build_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grid Build Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/grid_build.wgsl").into()),
        });

        let prefix_sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Prefix Sum Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/prefix_sum.wgsl").into()),
        });

        let grid_reorder_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grid Reorder Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/grid_reorder.wgsl").into()),
        });

        let density_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SPH 3D Density Grid Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sph_density_3d_grid.wgsl").into()),
        });

        let force_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SPH 3D Force Grid Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sph_force_3d_grid.wgsl").into()),
        });

        let integrate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SPH 3D Integrate Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sph_integrate_3d.wgsl").into()),
        });

        // Create buffers with max capacity for dynamic spawning
        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Buffer"),
            size: (std::mem::size_of::<SphParticle3D>() * max_particles as usize) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sorted_particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sorted Particle Buffer"),
            size: (std::mem::size_of::<SphParticle3D>() * max_particles as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write initial particles to buffer
        queue.write_buffer(&particle_buffer, 0, bytemuck::cast_slice(particles));

        let cell_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Counts Buffer"),
            size: (4 * total_cells) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cell_counts_temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Counts Temp Buffer"),
            size: (4 * total_cells) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cell_starts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Starts Buffer"),
            size: (4 * total_cells) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cell_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Offsets Buffer"),
            size: (4 * total_cells) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_cell_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Cell Indices Buffer"),
            size: (4 * max_particles) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sph_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SPH Params Buffer"),
            contents: bytemuck::bytes_of(&sph_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bounds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bounds Buffer"),
            contents: bytemuck::bytes_of(&bounds_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mouse_force_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mouse Force Buffer"),
            contents: bytemuck::bytes_of(&GpuMouseForce::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Default gravity pointing down
        let default_gravity = GpuGravity {
            direction: [0.0, -9.8, 0.0],
            _padding: 0.0,
        };
        let gravity_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gravity Buffer"),
            contents: bytemuck::bytes_of(&default_gravity),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let grid_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Params Buffer"),
            contents: bytemuck::bytes_of(&grid_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let prefix_sum_params = PrefixSumParams {
            count: total_cells,
            offset: 1,
            _padding: [0; 2],
        };
        let prefix_sum_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Prefix Sum Params Buffer"),
            contents: bytemuck::bytes_of(&prefix_sum_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layouts and pipelines
        // Grid Clear
        let grid_clear_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Grid Clear Layout"),
            entries: &[
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
            ],
        });

        let grid_clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Grid Clear Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Grid Clear Pipeline Layout"),
                bind_group_layouts: &[&grid_clear_layout],
                push_constant_ranges: &[],
            })),
            module: &grid_clear_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let grid_clear_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grid Clear Bind Group"),
            layout: &grid_clear_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cell_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Grid Build
        let grid_build_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Grid Build Layout"),
            entries: &[
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            ],
        });

        let grid_build_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Grid Build Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Grid Build Pipeline Layout"),
                bind_group_layouts: &[&grid_build_layout],
                push_constant_ranges: &[],
            })),
            module: &grid_build_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let grid_build_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grid Build Bind Group"),
            layout: &grid_build_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: particle_cell_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grid_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Prefix Sum
        let prefix_sum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Prefix Sum Layout"),
            entries: &[
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
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let prefix_sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Prefix Sum Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Prefix Sum Pipeline Layout"),
                bind_group_layouts: &[&prefix_sum_layout],
                push_constant_ranges: &[],
            })),
            module: &prefix_sum_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create prefix sum bind groups for ping-pong (using cell_starts and cell_counts_temp)
        let prefix_sum_bind_groups = vec![
            (
                // cell_starts -> cell_counts_temp
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Prefix Sum starts->temp"),
                    layout: &prefix_sum_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: cell_starts_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: cell_counts_temp_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: prefix_sum_params_buffer.as_entire_binding() },
                    ],
                }),
                // cell_counts_temp -> cell_starts
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Prefix Sum temp->starts"),
                    layout: &prefix_sum_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: cell_counts_temp_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: cell_starts_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: prefix_sum_params_buffer.as_entire_binding() },
                    ],
                }),
            ),
        ];

        // Grid Reorder
        let grid_reorder_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Grid Reorder Layout"),
            entries: &[
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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

        let grid_reorder_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Grid Reorder Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Grid Reorder Pipeline Layout"),
                bind_group_layouts: &[&grid_reorder_layout],
                push_constant_ranges: &[],
            })),
            module: &grid_reorder_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let grid_reorder_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grid Reorder Bind Group"),
            layout: &grid_reorder_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sorted_particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: particle_cell_indices_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cell_offsets_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cell_starts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cell_counts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: grid_params_buffer.as_entire_binding() },
            ],
        });

        // Density (with grid)
        let density_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Density Grid Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let density_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Density Grid Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&density_layout],
                push_constant_ranges: &[],
            })),
            module: &density_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let density_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Density Grid Bind Group"),
            layout: &density_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sph_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sorted_particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cell_starts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cell_counts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: grid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: particle_cell_indices_buffer.as_entire_binding() },
            ],
        });

        // Force (with grid and gravity)
        let force_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Force Grid Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let force_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Force Grid Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&force_layout],
                push_constant_ranges: &[],
            })),
            module: &force_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let force_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Force Grid Bind Group"),
            layout: &force_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sph_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sorted_particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cell_starts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cell_counts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: grid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: gravity_buffer.as_entire_binding() },
            ],
        });

        // Integrate (with mouse force)
        let integrate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Integrate Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let integrate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Integrate Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&integrate_layout],
                push_constant_ranges: &[],
            })),
            module: &integrate_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let integrate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Integrate Bind Group"),
            layout: &integrate_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sph_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bounds_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mouse_force_buffer.as_entire_binding() },
            ],
        });

        Self {
            grid_clear_pipeline,
            grid_build_pipeline,
            prefix_sum_pipeline,
            grid_reorder_pipeline,
            density_pipeline,
            force_pipeline,
            integrate_pipeline,
            particle_buffer,
            _sorted_particle_buffer: sorted_particle_buffer,
            cell_counts_buffer,
            cell_counts_temp_buffer,
            cell_starts_buffer,
            cell_offsets_buffer,
            _particle_cell_indices_buffer: particle_cell_indices_buffer,
            sph_params_buffer,
            bounds_buffer,
            mouse_force_buffer,
            gravity_buffer,
            grid_params_buffer,
            prefix_sum_params_buffer,
            grid_clear_bind_group,
            grid_build_bind_group,
            prefix_sum_bind_groups,
            grid_reorder_bind_group,
            density_bind_group,
            force_bind_group,
            integrate_bind_group,
            grid_params,
            num_particles,
            max_particles,
            num_prefix_sum_passes,
        }
    }

    /// Run one simulation step. This submits multiple command buffers for synchronization.
    pub fn step(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let particle_workgroups = self.num_particles.div_ceil(WORKGROUP_SIZE);
        let cell_workgroups = self.grid_params.total_cells.div_ceil(WORKGROUP_SIZE);

        // Phase 1: Clear and build grid
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Grid Build Encoder"),
            });

            // 1. Clear grid
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Grid Clear Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.grid_clear_pipeline);
                pass.set_bind_group(0, &self.grid_clear_bind_group, &[]);
                pass.dispatch_workgroups(cell_workgroups, 1, 1);
            }

            // Also clear cell_offsets for reorder
            encoder.clear_buffer(&self.cell_offsets_buffer, 0, None);

            // 2. Build grid (count particles per cell)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Grid Build Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.grid_build_pipeline);
                pass.set_bind_group(0, &self.grid_build_bind_group, &[]);
                pass.dispatch_workgroups(particle_workgroups, 1, 1);
            }

            // Copy cell_counts to cell_starts as base for prefix sum
            encoder.copy_buffer_to_buffer(
                &self.cell_counts_buffer,
                0,
                &self.cell_starts_buffer,
                0,
                (4 * self.grid_params.total_cells) as u64,
            );

            queue.submit(std::iter::once(encoder.finish()));
        }

        // Phase 2: Prefix sum (each pass needs separate submit for uniform buffer sync)
        let mut offset = 1u32;
        let mut read_from_starts = true;
        for _pass_idx in 0..self.num_prefix_sum_passes {
            let params = PrefixSumParams {
                count: self.grid_params.total_cells,
                offset,
                _padding: [0; 2],
            };
            queue.write_buffer(&self.prefix_sum_params_buffer, 0, bytemuck::bytes_of(&params));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Prefix Sum Encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Prefix Sum Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.prefix_sum_pipeline);
                let bind_group = if read_from_starts {
                    &self.prefix_sum_bind_groups[0].0 // starts -> temp
                } else {
                    &self.prefix_sum_bind_groups[0].1 // temp -> starts
                };
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(cell_workgroups, 1, 1);
            }

            queue.submit(std::iter::once(encoder.finish()));

            offset *= 2;
            read_from_starts = !read_from_starts;
        }

        // Copy final result to cell_starts if it ended up in temp
        if !read_from_starts {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Prefix Sum Final Copy"),
            });
            encoder.copy_buffer_to_buffer(
                &self.cell_counts_temp_buffer,
                0,
                &self.cell_starts_buffer,
                0,
                (4 * self.grid_params.total_cells) as u64,
            );
            queue.submit(std::iter::once(encoder.finish()));
        }

        // Phase 3: Reorder, density, reorder, force, integrate
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SPH Compute Encoder"),
            });

            // 4. Reorder particles
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Grid Reorder Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.grid_reorder_pipeline);
                pass.set_bind_group(0, &self.grid_reorder_bind_group, &[]);
                pass.dispatch_workgroups(particle_workgroups, 1, 1);
            }

            // 5. Density computation
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Density Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.density_pipeline);
                pass.set_bind_group(0, &self.density_bind_group, &[]);
                pass.dispatch_workgroups(particle_workgroups, 1, 1);
            }

            // 6. Force computation
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Force Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.force_pipeline);
                pass.set_bind_group(0, &self.force_bind_group, &[]);
                pass.dispatch_workgroups(particle_workgroups, 1, 1);
            }

            // 7. Integration
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Integrate Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.integrate_pipeline);
                pass.set_bind_group(0, &self.integrate_bind_group, &[]);
                pass.dispatch_workgroups(particle_workgroups, 1, 1);
            }

            queue.submit(std::iter::once(encoder.finish()));
        }
    }

    pub fn update_sph_params(&mut self, queue: &wgpu::Queue, params: &GpuSphParams3D) {
        queue.write_buffer(&self.sph_params_buffer, 0, bytemuck::bytes_of(params));
        // Keep num_particles in sync between both uniform buffers
        if self.grid_params.num_particles != params.num_particles {
            self.grid_params.num_particles = params.num_particles;
            queue.write_buffer(&self.grid_params_buffer, 0, bytemuck::bytes_of(&self.grid_params));
        }
    }

    pub fn update_bounds_params(&self, queue: &wgpu::Queue, params: &GpuBoundsParams3D) {
        queue.write_buffer(&self.bounds_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_mouse_force(&self, queue: &wgpu::Queue, params: &GpuMouseForce) {
        queue.write_buffer(&self.mouse_force_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_gravity(&self, queue: &wgpu::Queue, params: &GpuGravity) {
        queue.write_buffer(&self.gravity_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        &self.particle_buffer
    }

    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }

    /// Spawn new particles at a given position with some spread
    /// Returns the number of particles actually spawned (may be less if at capacity)
    pub fn spawn_particles(&mut self, queue: &wgpu::Queue, position: [f32; 3], count: u32, spread: f32) -> u32 {
        let available = self.max_particles - self.num_particles;
        let spawn_count = count.min(available);

        if spawn_count == 0 {
            return 0;
        }

        // Create new particles with random spread
        let mut new_particles = Vec::with_capacity(spawn_count as usize);
        for i in 0..spawn_count {
            // Simple deterministic spread pattern (spiral)
            let angle = i as f32 * 2.39996; // Golden angle
            let r = spread * (i as f32 / spawn_count as f32).sqrt();
            let layer = i / 20; // Stack in layers

            let px = position[0] + r * angle.cos();
            let py = position[1] + layer as f32 * spread * 0.3;
            let pz = position[2] + r * angle.sin();

            new_particles.push(SphParticle3D::new(px, py, pz));
        }

        // Write new particles to buffer at offset
        let offset = (self.num_particles as usize) * std::mem::size_of::<SphParticle3D>();
        queue.write_buffer(&self.particle_buffer, offset as u64, bytemuck::cast_slice(&new_particles));

        // Update particle count
        self.num_particles += spawn_count;

        // Update grid params with new count
        self.grid_params.num_particles = self.num_particles;
        queue.write_buffer(&self.grid_params_buffer, 0, bytemuck::bytes_of(&self.grid_params));

        spawn_count
    }
}
