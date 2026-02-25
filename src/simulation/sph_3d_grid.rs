//! 3D SPH simulation with spatial hashing for O(n) neighbor search

use crate::render::mesh_loader::SdfData;
use crate::simulation::particle::SphParticle3D;
use crate::state::{GpuContainerGeometry, GpuGravity, GpuMouseForce, GpuRigidBody, GpuRigidBodyAccum, GpuSphParams3D};
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
    xsph_pipeline: wgpu::ComputePipeline,
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
    container_geom_buffer: wgpu::Buffer,
    mouse_force_buffer: wgpu::Buffer,
    gravity_buffer: wgpu::Buffer,
    grid_params_buffer: wgpu::Buffer,

    // Bind groups
    grid_clear_bind_group: wgpu::BindGroup,
    grid_build_bind_group: wgpu::BindGroup,
    prefix_sum_bind_groups: Vec<(wgpu::BindGroup, wgpu::BindGroup)>, // Pairs for ping-pong
    grid_reorder_bind_group: wgpu::BindGroup,
    density_bind_group: wgpu::BindGroup,
    xsph_bind_group: wgpu::BindGroup,
    force_bind_group: wgpu::BindGroup,
    integrate_bind_group: wgpu::BindGroup,

    // Rigid body buffers
    rigid_body_buffer: wgpu::Buffer,
    rigid_body_accum_buffer: wgpu::Buffer,
    rigid_body_accum_staging: wgpu::Buffer,
    last_accum: GpuRigidBodyAccum,

    // SDF texture for custom mesh collision (kept alive for bind group)
    _sdf_texture: wgpu::Texture,
    _sdf_sampler: wgpu::Sampler,

    // PCISPH buffers and pipelines
    pcisph_predict_pipeline: wgpu::ComputePipeline,
    pcisph_solve_pipeline: wgpu::ComputePipeline,
    pcisph_finalize_pipeline: wgpu::ComputePipeline,
    _sorted_predicted_a_buffer: wgpu::Buffer,
    _sorted_predicted_b_buffer: wgpu::Buffer,
    _pressure_buffer: wgpu::Buffer,
    pcisph_predict_bind_group: wgpu::BindGroup,
    pcisph_solve_bind_group_a: wgpu::BindGroup, // read A, write B
    pcisph_solve_bind_group_b: wgpu::BindGroup, // read B, write A
    pcisph_finalize_bind_group_a: wgpu::BindGroup, // read A
    pcisph_finalize_bind_group_b: wgpu::BindGroup, // read B
    pcisph_iterations: u32,

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
        container_geom: GpuContainerGeometry,
        max_particles: u32,
        sdf_data: Option<&SdfData>,
    ) -> Self {
        let num_particles = particles.len() as u32;
        let max_particles = max_particles.max(num_particles); // Ensure at least enough for initial particles

        // Calculate grid dimensions based on kernel radius.
        // Pre-allocate for the maximum possible container configuration:
        // sliders allow up to 3.0 per axis (half-extent 1.5), full tilt (±π).
        // The diagonal of a 1.5×1.5×1.5 half-box is 1.5*sqrt(3) ≈ 2.6.
        // With center_y offset and margin, 4.0 covers all cases.
        let cell_size = sph_params.kernel_radius;
        let bounds_extent = 4.0f32;
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

        let container_common_wgsl = include_str!("../shaders/container_common.wgsl");

        let density_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SPH 3D Density Grid Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}", container_common_wgsl, include_str!("../shaders/sph_density_3d_grid.wgsl")).into(),
            ),
        });

        let xsph_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("XSPH Velocity Smoothing Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/xsph.wgsl").into()),
        });

        let force_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SPH 3D Force Grid Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sph_force_3d_grid.wgsl").into()),
        });

        let integrate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SPH 3D Integrate Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}", container_common_wgsl, include_str!("../shaders/sph_integrate_3d.wgsl")).into(),
            ),
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

        let container_geom_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Container Geometry Buffer"),
            contents: bytemuck::bytes_of(&container_geom),
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

        let rigid_body_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rigid Body Buffer"),
            contents: bytemuck::bytes_of(&GpuRigidBody::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let rigid_body_accum_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rigid Body Accum Buffer"),
            contents: bytemuck::bytes_of(&GpuRigidBodyAccum::default()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let rigid_body_accum_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rigid Body Accum Staging"),
            size: std::mem::size_of::<GpuRigidBodyAccum>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // SDF 3D texture for custom mesh collision
        let (sdf_texture, sdf_texture_view, sdf_sampler) = if let Some(sdf) = sdf_data {
            let res = sdf.resolution;
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("SDF 3D Texture"),
                size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&sdf.data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * res),
                    rows_per_image: Some(res),
                },
                wgpu::Extent3d { width: res, height: res, depth_or_array_layers: res },
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("SDF Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            (texture, view, sampler)
        } else {
            // Dummy 1x1x1 texture with value 1.0 (= outside, no collision effect)
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("SDF Dummy Texture"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&[1.0f32]),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("SDF Dummy Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            (texture, view, sampler)
        };

        let grid_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Params Buffer"),
            contents: bytemuck::bytes_of(&grid_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Pre-create one uniform buffer per prefix sum pass (avoids per-pass queue.submit)
        let prefix_sum_params_buffers: Vec<wgpu::Buffer> = (0..num_prefix_sum_passes)
            .map(|pass_idx| {
                let params = PrefixSumParams {
                    count: total_cells,
                    offset: 1u32 << pass_idx,
                    _padding: [0; 2],
                };
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Prefix Sum Params {}", pass_idx)),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            })
            .collect();

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

        // Create prefix sum bind groups: one ping-pong pair per pass
        let prefix_sum_bind_groups: Vec<(wgpu::BindGroup, wgpu::BindGroup)> = prefix_sum_params_buffers
            .iter()
            .enumerate()
            .map(|(i, params_buf)| {
                (
                    // cell_starts -> cell_counts_temp
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("Prefix Sum starts->temp {}", i)),
                        layout: &prefix_sum_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: cell_starts_buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: cell_counts_temp_buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
                        ],
                    }),
                    // cell_counts_temp -> cell_starts
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("Prefix Sum temp->starts {}", i)),
                        layout: &prefix_sum_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: cell_counts_temp_buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: cell_starts_buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
                        ],
                    }),
                )
            })
            .collect();

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
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
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
                wgpu::BindGroupEntry { binding: 7, resource: container_geom_buffer.as_entire_binding() },
            ],
        });

        // XSPH velocity smoothing (same layout as density minus sorted_index)
        let xsph_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("XSPH Layout"),
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
            ],
        });

        let xsph_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("XSPH Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&xsph_layout],
                push_constant_ranges: &[],
            })),
            module: &xsph_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let xsph_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("XSPH Bind Group"),
            layout: &xsph_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sph_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sorted_particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cell_starts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cell_counts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: grid_params_buffer.as_entire_binding() },
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

        // Integrate (with mouse force + rigid body)
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
                wgpu::BindGroupEntry { binding: 2, resource: container_geom_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mouse_force_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: rigid_body_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: rigid_body_accum_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&sdf_texture_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(&sdf_sampler) },
            ],
        });

        // === PCISPH Buffers and Pipelines ===

        // Per-particle pressure buffer for warm-starting (persists across frames)
        let pressure_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure Buffer"),
            size: (max_particles as u64) * 4, // f32 per particle
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // PredictedState: 32 bytes per particle (8 x f32)
        let predicted_buf_size = (max_particles as u64) * 32;
        let sorted_predicted_a_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sorted Predicted A"),
            size: predicted_buf_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let sorted_predicted_b_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sorted Predicted B"),
            size: predicted_buf_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // PCISPH Predict shader
        let pcisph_predict_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PCISPH Predict"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pcisph_predict.wgsl").into()),
        });
        let pcisph_predict_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PCISPH Predict Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pcisph_predict_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PCISPH Predict Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&pcisph_predict_layout],
                push_constant_ranges: &[],
            })),
            module: &pcisph_predict_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pcisph_predict_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCISPH Predict Bind Group"),
            layout: &pcisph_predict_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sph_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sorted_predicted_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: particle_cell_indices_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pressure_buffer.as_entire_binding() },
            ],
        });

        // PCISPH Solve shader
        let pcisph_solve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PCISPH Solve"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pcisph_solve.wgsl").into()),
        });
        let pcisph_solve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PCISPH Solve Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pcisph_solve_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PCISPH Solve Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&pcisph_solve_layout],
                push_constant_ranges: &[],
            })),
            module: &pcisph_solve_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        // Bind group A: read from sorted_predicted_a, write to sorted_predicted_b
        let pcisph_solve_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCISPH Solve A"),
            layout: &pcisph_solve_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sph_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sorted_predicted_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sorted_predicted_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cell_starts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cell_counts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: grid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: particle_cell_indices_buffer.as_entire_binding() },
            ],
        });
        // Bind group B: read from sorted_predicted_b, write to sorted_predicted_a
        let pcisph_solve_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCISPH Solve B"),
            layout: &pcisph_solve_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sph_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sorted_predicted_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sorted_predicted_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cell_starts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cell_counts_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: grid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: particle_cell_indices_buffer.as_entire_binding() },
            ],
        });

        // PCISPH Finalize shader
        let pcisph_finalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PCISPH Finalize"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pcisph_finalize.wgsl").into()),
        });
        let pcisph_finalize_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PCISPH Finalize Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let pcisph_finalize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PCISPH Finalize Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&pcisph_finalize_layout],
                push_constant_ranges: &[],
            })),
            module: &pcisph_finalize_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        // Finalize bind group A: read from sorted_predicted_a
        let pcisph_finalize_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCISPH Finalize A"),
            layout: &pcisph_finalize_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sph_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sorted_predicted_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: particle_cell_indices_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pressure_buffer.as_entire_binding() },
            ],
        });
        // Finalize bind group B: read from sorted_predicted_b
        let pcisph_finalize_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PCISPH Finalize B"),
            layout: &pcisph_finalize_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sph_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sorted_predicted_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: particle_cell_indices_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pressure_buffer.as_entire_binding() },
            ],
        });

        Self {
            grid_clear_pipeline,
            grid_build_pipeline,
            prefix_sum_pipeline,
            grid_reorder_pipeline,
            density_pipeline,
            xsph_pipeline,
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
            container_geom_buffer,
            mouse_force_buffer,
            gravity_buffer,
            grid_params_buffer,
            grid_clear_bind_group,
            grid_build_bind_group,
            prefix_sum_bind_groups,
            grid_reorder_bind_group,
            density_bind_group,
            xsph_bind_group,
            force_bind_group,
            integrate_bind_group,
            rigid_body_buffer,
            rigid_body_accum_buffer,
            rigid_body_accum_staging,
            last_accum: GpuRigidBodyAccum::default(),
            _sdf_texture: sdf_texture,
            _sdf_sampler: sdf_sampler,
            pcisph_predict_pipeline,
            pcisph_solve_pipeline,
            pcisph_finalize_pipeline,
            _sorted_predicted_a_buffer: sorted_predicted_a_buffer,
            _sorted_predicted_b_buffer: sorted_predicted_b_buffer,
            _pressure_buffer: pressure_buffer,
            pcisph_predict_bind_group,
            pcisph_solve_bind_group_a,
            pcisph_solve_bind_group_b,
            pcisph_finalize_bind_group_a,
            pcisph_finalize_bind_group_b,
            pcisph_iterations: 5,
            grid_params,
            num_particles,
            max_particles,
            num_prefix_sum_passes,
        }
    }

    /// Run one simulation step (single encoder, single submit).
    pub fn step(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let particle_workgroups = self.num_particles.div_ceil(WORKGROUP_SIZE);
        let cell_workgroups = self.grid_params.total_cells.div_ceil(WORKGROUP_SIZE);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SPH Step Encoder"),
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

        // 3. Prefix sum (all passes in single encoder, separate compute passes for barriers)
        let mut read_from_starts = true;
        for pass_idx in 0..self.num_prefix_sum_passes as usize {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Prefix Sum Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.prefix_sum_pipeline);
                let bind_group = if read_from_starts {
                    &self.prefix_sum_bind_groups[pass_idx].0 // starts -> temp
                } else {
                    &self.prefix_sum_bind_groups[pass_idx].1 // temp -> starts
                };
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(cell_workgroups, 1, 1);
            }
            read_from_starts = !read_from_starts;
        }

        // Copy final result to cell_starts if it ended up in temp
        if !read_from_starts {
            encoder.copy_buffer_to_buffer(
                &self.cell_counts_temp_buffer,
                0,
                &self.cell_starts_buffer,
                0,
                (4 * self.grid_params.total_cells) as u64,
            );
        }

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

        // 6. XSPH velocity smoothing (damps jitter while preserving bulk flow)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("XSPH Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.xsph_pipeline);
            pass.set_bind_group(0, &self.xsph_bind_group, &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }

        // 7. Force computation
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Force Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.force_pipeline);
            pass.set_bind_group(0, &self.force_bind_group, &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }

        // 7. PCISPH Predict (compute predicted state from non-pressure forces)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PCISPH Predict Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pcisph_predict_pipeline);
            pass.set_bind_group(0, &self.pcisph_predict_bind_group, &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }

        // 8. PCISPH Solve iterations (ping-pong between predicted buffers)
        for iter in 0..self.pcisph_iterations {
            let bind_group = if iter % 2 == 0 {
                &self.pcisph_solve_bind_group_a // read A, write B
            } else {
                &self.pcisph_solve_bind_group_b // read B, write A
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PCISPH Solve Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pcisph_solve_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }

        // 9. PCISPH Finalize (write corrected velocity, zero force field)
        {
            // After N iterations: if N is even, last write went to A (via solve_b reading B, writing A)
            // Wait — predict writes to A. Solve iter 0 reads A, writes B. Iter 1 reads B, writes A.
            // So for N iters: if N is even, final result is in A. If N is odd, final result is in B.
            let finalize_bg = if self.pcisph_iterations % 2 == 0 {
                &self.pcisph_finalize_bind_group_a
            } else {
                &self.pcisph_finalize_bind_group_b
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PCISPH Finalize Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pcisph_finalize_pipeline);
            pass.set_bind_group(0, finalize_bg, &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }

        // 10. Integration (+ rigid body penalty forces)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Integrate Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.integrate_pipeline);
            pass.set_bind_group(0, &self.integrate_bind_group, &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }

        // Copy accumulator to staging for CPU readback
        encoder.copy_buffer_to_buffer(
            &self.rigid_body_accum_buffer,
            0,
            &self.rigid_body_accum_staging,
            0,
            std::mem::size_of::<GpuRigidBodyAccum>() as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn update_sph_params(&mut self, queue: &wgpu::Queue, params: &GpuSphParams3D) {
        queue.write_buffer(&self.sph_params_buffer, 0, bytemuck::bytes_of(params));
        // Keep num_particles in sync between both uniform buffers
        if self.grid_params.num_particles != params.num_particles {
            self.grid_params.num_particles = params.num_particles;
            queue.write_buffer(&self.grid_params_buffer, 0, bytemuck::bytes_of(&self.grid_params));
        }
    }

    pub fn update_container_geometry(&self, queue: &wgpu::Queue, geom: &GpuContainerGeometry) {
        queue.write_buffer(&self.container_geom_buffer, 0, bytemuck::bytes_of(geom));
    }

    pub fn update_mouse_force(&self, queue: &wgpu::Queue, params: &GpuMouseForce) {
        queue.write_buffer(&self.mouse_force_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_gravity(&self, queue: &wgpu::Queue, params: &GpuGravity) {
        queue.write_buffer(&self.gravity_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_rigid_body(&self, queue: &wgpu::Queue, params: &GpuRigidBody) {
        queue.write_buffer(&self.rigid_body_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn set_pcisph_iterations(&mut self, iterations: u32) {
        self.pcisph_iterations = iterations.max(1);
    }

    pub fn clear_rigid_body_accum(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.rigid_body_accum_buffer,
            0,
            bytemuck::bytes_of(&GpuRigidBodyAccum::default()),
        );
    }

    pub fn read_rigid_body_accum(&mut self, device: &wgpu::Device) {
        let slice = self.rigid_body_accum_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        {
            let data = slice.get_mapped_range();
            let accum: &GpuRigidBodyAccum = bytemuck::from_bytes(&data);
            self.last_accum = *accum;
        }
        self.rigid_body_accum_staging.unmap();
    }

    pub fn rigid_body_accum(&self) -> &GpuRigidBodyAccum {
        &self.last_accum
    }

    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        &self.particle_buffer
    }

    pub fn sorted_particle_buffer(&self) -> &wgpu::Buffer {
        &self._sorted_particle_buffer
    }

    pub fn cell_starts_buffer(&self) -> &wgpu::Buffer {
        &self.cell_starts_buffer
    }

    pub fn cell_counts_buffer(&self) -> &wgpu::Buffer {
        &self.cell_counts_buffer
    }

    pub fn grid_params_buffer(&self) -> &wgpu::Buffer {
        &self.grid_params_buffer
    }

    pub fn sph_params_buffer(&self) -> &wgpu::Buffer {
        &self.sph_params_buffer
    }

    pub fn container_geom_buffer(&self) -> &wgpu::Buffer {
        &self.container_geom_buffer
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
