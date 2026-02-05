//! Wireframe renderer for container visualization

use wgpu::util::DeviceExt;

use crate::render::camera::GpuCameraParams;
use crate::state::ContainerConfig;

/// GPU-compatible container parameters for wireframe rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuContainerParams {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub min_z: f32,
    pub max_z: f32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    pub color_a: f32,
    pub _padding: [f32; 2],
    // Rotation matrix (3x3 stored as 3 vec4s for alignment)
    pub rotation_row0: [f32; 4],
    pub rotation_row1: [f32; 4],
    pub rotation_row2: [f32; 4],
}

impl GpuContainerParams {
    pub fn from_config(config: &ContainerConfig) -> Self {
        // Compute rotation matrix from tilt angles (same as physics bounds)
        let (sin_x, cos_x) = config.tilt_x.sin_cos();
        let (sin_z, cos_z) = config.tilt_z.sin_cos();

        // This is the INVERSE rotation (transpose) - transforms container-local to world space
        // We want to rotate the wireframe corners FROM local TO world
        let rotation_row0 = [cos_z, sin_z, 0.0, 0.0];
        let rotation_row1 = [-sin_z * cos_x, cos_z * cos_x, sin_x, 0.0];
        let rotation_row2 = [sin_z * sin_x, -cos_z * sin_x, cos_x, 0.0];

        Self {
            min_x: -config.half_width(),
            max_x: config.half_width(),
            min_y: config.floor_y,
            max_y: config.ceiling_y(),
            min_z: -config.half_depth(),
            max_z: config.half_depth(),
            // Light blue/cyan color for visibility
            color_r: 0.3,
            color_g: 0.7,
            color_b: 1.0,
            color_a: 0.8,
            _padding: [0.0; 2],
            rotation_row0,
            rotation_row1,
            rotation_row2,
        }
    }
}

pub struct WireframeRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    container_buffer: wgpu::Buffer,
}

impl WireframeRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_params: &GpuCameraParams,
        container_params: &GpuContainerParams,
    ) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Wireframe Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/wireframe.wgsl").into()),
        });

        // Create buffers
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wireframe Camera Buffer"),
            contents: bytemuck::bytes_of(camera_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let container_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wireframe Container Buffer"),
            contents: bytemuck::bytes_of(container_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Wireframe Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Wireframe Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: container_buffer.as_entire_binding(),
                },
            ],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Wireframe Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Wireframe Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group,
            camera_buffer,
            container_buffer,
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, params: &GpuCameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_container(&self, queue: &wgpu::Queue, params: &GpuContainerParams) {
        queue.write_buffer(&self.container_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        // 12 edges * 2 vertices = 24 vertices
        render_pass.draw(0..24, 0..1);
    }
}
