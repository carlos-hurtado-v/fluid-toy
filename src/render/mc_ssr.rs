//! Screen-space reflections for the marching-cubes water (`ssr.wgsl`): a
//! compute pass that marches the reflected ray of each water pixel against
//! the background depth and writes the hit color + confidence.

use super::mc_background::Background;
use super::mc_faces::FaceBuffers;
use crate::gpu::bind::{entry, layout, COMPUTE};
use crate::state::GpuSsrParams;
use wgpu::util::DeviceExt;

fn create_ssr_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let ssr_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("SSR Texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let ssr_view = ssr_texture.create_view(&wgpu::TextureViewDescriptor::default());
    (ssr_texture, ssr_view)
}

pub struct Ssr {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    color_sampler: wgpu::Sampler,
}

impl Ssr {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        camera_buffer: &wgpu::Buffer,
        faces: &FaceBuffers,
        background: &Background,
    ) -> Self {
        let (ssr_texture, ssr_view) = create_ssr_texture(device, width, height);

        let ssr_params = GpuSsrParams::default();
        let ssr_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SSR Params"),
            contents: bytemuck::bytes_of(&ssr_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let ssr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSR Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ssr.wgsl").into()),
        });

        // === SSR Compute Pipeline ===
        let ssr_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR BGL"),
            entries: &[
                // Camera uniform
                layout::uniform(0, COMPUTE),
                // Front depth (water surface)
                layout::texture_depth(1, COMPUTE),
                // Background depth (scene without water)
                layout::texture_depth(2, COMPUTE),
                // Background color
                layout::texture_2d(3, COMPUTE),
                // Depth sampler
                layout::sampler(4, COMPUTE),
                // Color sampler
                layout::sampler(5, COMPUTE),
                // SSR params
                layout::uniform(6, COMPUTE),
                // SSR output
                layout::storage_texture_2d(7, COMPUTE, wgpu::TextureFormat::Rgba16Float),
                // Normal G-buffer (smooth world normals from front face pass)
                layout::texture_2d_unfilterable(8, COMPUTE),
            ],
        });

        let ssr_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR Pipeline Layout"),
            bind_group_layouts: &[&ssr_bind_group_layout],
            push_constant_ranges: &[],
        });

        let ssr_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SSR Pipeline"),
            layout: Some(&ssr_pipeline_layout),
            module: &ssr_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let color_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSR Color Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group = Self::create_bind_group(
            device, &ssr_bind_group_layout, camera_buffer, faces, background, &color_sampler, &ssr_params_buffer,
            &ssr_view,
        );

        Self {
            _texture: ssr_texture,
            view: ssr_view,
            pipeline: ssr_pipeline,
            bind_group_layout: ssr_bind_group_layout,
            bind_group,
            params_buffer: ssr_params_buffer,
            color_sampler,
        }
    }

    fn create_bind_group(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        camera_buffer: &wgpu::Buffer,
        faces: &FaceBuffers,
        background: &Background,
        color_sampler: &wgpu::Sampler,
        params_buffer: &wgpu::Buffer,
        ssr_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR BG"),
            layout: bind_group_layout,
            entries: &[
                entry::buffer(0, camera_buffer),
                entry::view(1, faces.front_depth_view()),
                entry::view(2, background.depth_view()),
                entry::view(3, background.color_view()),
                entry::sampler(4, faces.depth_sampler()),
                entry::sampler(5, color_sampler),
                entry::buffer(6, params_buffer),
                entry::view(7, ssr_view),
                entry::view(8, faces.front_normal_view()),
            ],
        })
    }

    /// New output texture, rebound over the (already resized) G-buffers and background
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        camera_buffer: &wgpu::Buffer,
        faces: &FaceBuffers,
        background: &Background,
    ) {
        let (ssr_texture, ssr_view) = create_ssr_texture(device, width, height);
        self.bind_group = Self::create_bind_group(
            device, &self.bind_group_layout, camera_buffer, faces, background, &self.color_sampler,
            &self.params_buffer, &ssr_view,
        );
        self._texture = ssr_texture;
        self.view = ssr_view;
    }

    /// Reflected color (rgb) + confidence (a), full resolution
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Set whether SSR is enabled and update GPU params
    pub fn set_enabled(&self, queue: &wgpu::Queue, enabled: bool) {
        let params = GpuSsrParams {
            enabled: if enabled { 1 } else { 0 },
            ..GpuSsrParams::default()
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Always dispatched: the shader checks the enabled flag and writes
    /// zeros when disabled
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) {
        let mut ssr_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SSR Pass"),
            timestamp_writes: None,
        });
        ssr_pass.set_pipeline(&self.pipeline);
        ssr_pass.set_bind_group(0, &self.bind_group, &[]);
        let wg_x = width.div_ceil(8);
        let wg_y = height.div_ceil(8);
        ssr_pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}
