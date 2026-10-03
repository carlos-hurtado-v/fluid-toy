//! The scene behind the water (backdrop, container, bodies, spray) as a
//! texture: what the MC water shader's refraction and SSR read.
//!
//! Level 0 is the render attachment; `mip_downsample.wgsl` regenerates the
//! levels below it every frame for filtered refraction lookups. The depth
//! target next to it is what refracted rays land on.

use super::mc_faces::create_samplable_depth_texture;
use crate::gpu::bind::{entry, layout, FRAGMENT};

/// Levels in the refraction background's mip chain (level 5 = a 32-texel
/// footprint; anisotropic filtering stretches that 16x along one axis)
const BACKGROUND_MIPS: u32 = 6;

/// The scene behind the water (backdrop, container, bodies, spray), read by
/// screen-space refraction and SSR. Drawn at level 0; the levels below are
/// regenerated every frame so that lookups which shrink the image (grazing
/// mirrors, strong lensing) read a footprint instead of skipping texels.
struct BackgroundTexture {
    _texture: wgpu::Texture,
    /// Level 0 alone: the render attachment, and what SSR reads
    view: wgpu::TextureView,
    /// Every level: the water shader's refraction lookups
    mip_view: wgpu::TextureView,
    /// Per generated level: its view, and a bind group reading the level above
    mip_passes: Vec<(wgpu::TextureView, wgpu::BindGroup)>,
}

fn create_background_texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    mip_layout: &wgpu::BindGroupLayout,
    mip_sampler: &wgpu::Sampler,
) -> BackgroundTexture {
    let (width, height) = (width.max(1), height.max(1));
    let mip_level_count = BACKGROUND_MIPS.min(width.min(height).ilog2() + 1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MC Background Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let level_view = |level: u32| {
        texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: level,
            mip_level_count: Some(1),
            ..Default::default()
        })
    };
    let view = level_view(0);
    let mip_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mip_passes = (1..mip_level_count)
        .map(|level| {
            let source = level_view(level - 1);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MC Background Mip BG"),
                layout: mip_layout,
                entries: &[
                    entry::view(0, &source),
                    entry::sampler(1, mip_sampler),
                ],
            });
            (level_view(level), bind_group)
        })
        .collect();
    BackgroundTexture { _texture: texture, view, mip_view, mip_passes }
}

pub struct Background {
    texture: BackgroundTexture,
    // Single-sampled scene depth of the background pass (samplable)
    _depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    mip_pipeline: wgpu::RenderPipeline,
    mip_bind_group_layout: wgpu::BindGroupLayout,
    mip_sampler: wgpu::Sampler,
    /// Clamped trilinear + anisotropic: the water's filtered lookups
    sampler: wgpu::Sampler,
    format: wgpu::TextureFormat,
}

impl Background {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let mip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mip Downsample Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mip_downsample.wgsl").into()),
        });
        let mip_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mip Downsample BGL"),
            entries: &[
                layout::texture_2d(0, FRAGMENT),
                layout::sampler(1, FRAGMENT),
            ],
        });
        let mip_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mip Downsample Pipeline Layout"),
            bind_group_layouts: &[&mip_bind_group_layout],
            push_constant_ranges: &[],
        });
        let mip_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mip Downsample Pipeline"),
            layout: Some(&mip_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mip_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &mip_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let mip_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Mip Downsample Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("MC Background Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });
        let texture = create_background_texture(device, format, width, height, &mip_bind_group_layout, &mip_sampler);
        let (depth_texture, depth_view) = create_samplable_depth_texture(device, width, height);

        Self {
            texture,
            _depth_texture: depth_texture,
            depth_view,
            mip_pipeline,
            mip_bind_group_layout,
            mip_sampler,
            sampler,
            format,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (depth_texture, depth_view) = create_samplable_depth_texture(device, width, height);
        self._depth_texture = depth_texture;
        self.depth_view = depth_view;
        self.texture = create_background_texture(
            device, self.format, width, height, &self.mip_bind_group_layout, &self.mip_sampler,
        );
    }

    /// Level 0 alone: the render attachment, and what SSR reads
    pub fn color_view(&self) -> &wgpu::TextureView {
        &self.texture.view
    }

    /// Every level: the water shader's refraction lookups
    pub fn mip_view(&self) -> &wgpu::TextureView {
        &self.texture.mip_view
    }

    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// Begin the pass that draws the scene behind the water (color level 0 +
    /// depth, both cleared); the caller draws the backdrop and scene objects
    pub fn begin_pass<'e>(&self, encoder: &'e mut wgpu::CommandEncoder) -> wgpu::RenderPass<'e> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("MC Environment Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.texture.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        })
    }

    /// Regenerate the mip chain from level 0
    pub fn encode_mips(&self, encoder: &mut wgpu::CommandEncoder) {
        for (view, bind_group) in &self.texture.mip_passes {
            let mut mip_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("MC Background Mip Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            mip_pass.set_pipeline(&self.mip_pipeline);
            mip_pass.set_bind_group(0, bind_group, &[]);
            mip_pass.draw(0..3, 0..1);
        }
    }
}
