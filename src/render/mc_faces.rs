//! Water surface G-buffers of the marching-cubes renderer.
//!
//! Front faces: nearest depth (GTAO's input, SSR's start point, front-face
//! refraction exits) + smooth world normals. Back faces: depth + outward
//! normals, the exit interface of two-interface refraction (and the legacy
//! thickness). Both are drawn from the MC mesh by `mc_back_depth.wgsl`.

use super::{ContainerRenderer, RigidBodyRenderer};
use crate::gpu::bind::{entry, layout, FRAGMENT, VERTEX, VERTEX_FRAGMENT};

/// Create a depth texture that can be sampled (for back-face depth / thickness calculation)
pub(super) fn create_samplable_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MC Back Depth Texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create a normal G-buffer texture for SSR (smooth world-space normals, Rgba16Float)
fn create_normal_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MC Normal G-Buffer"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Depth + Rgba16Float normal pipeline over the MC mesh (`vs_normal`); the
/// two face passes differ in fragment entry point and culling
fn create_face_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    fragment_entry: &str,
    cull_mode: Option<wgpu::Face>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_normal"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Cw, // MC triangles are clockwise
            cull_mode,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

/// The four screen-sized targets (texture kept alive next to its view)
struct Targets {
    front_depth: (wgpu::Texture, wgpu::TextureView),
    front_normal: (wgpu::Texture, wgpu::TextureView),
    back_depth: (wgpu::Texture, wgpu::TextureView),
    back_normal: (wgpu::Texture, wgpu::TextureView),
}

impl Targets {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        Self {
            front_depth: create_samplable_depth_texture(device, width, height),
            front_normal: create_normal_texture(device, width, height),
            back_depth: create_samplable_depth_texture(device, width, height),
            back_normal: create_normal_texture(device, width, height),
        }
    }
}

pub struct FaceBuffers {
    // Front faces: cull none, writes depth + normals
    front_face_pipeline: wgpu::RenderPipeline,
    // Back faces: cull front, writes depth + outward normals (w = 1 where present)
    back_face_pipeline: wgpu::RenderPipeline,
    /// Camera, mesh vertices, container clip: shared by both pipelines
    bind_group: wgpu::BindGroup,
    targets: Targets,
    /// Nearest, clamped: how the water shader and SSR read these depths
    depth_sampler: wgpu::Sampler,
}

impl FaceBuffers {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        camera_buffer: &wgpu::Buffer,
        vertex_buffer: &wgpu::Buffer,
        container_geom_buffer: &wgpu::Buffer,
    ) -> Self {
        let back_depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MC Back Depth Shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../shaders/container_common.wgsl"),
                    include_str!("../shaders/mc_back_depth.wgsl")
                )
                .into(),
            ),
        });

        let back_face_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC Back Face BGL"),
            entries: &[
                // Camera: fragment too, to orient back-face normals away from the eye
                layout::uniform(0, VERTEX_FRAGMENT),
                layout::storage(1, VERTEX),
                // Container clip params
                layout::uniform(2, FRAGMENT),
            ],
        });

        let back_face_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC Back Face Pipeline Layout"),
            bind_group_layouts: &[&back_face_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Back faces (for thickness and the refraction exit interface):
        // cull front faces, write outward normals
        let back_face_pipeline = create_face_pipeline(
            device,
            &back_face_pipeline_layout,
            &back_depth_shader,
            "MC Back Face Pipeline",
            "fs_back_normal",
            Some(wgpu::Face::Front),
        );

        // Front faces (depth + normal G-buffer for GTAO + SSR). For GTAO input
        // we need nearest visible depth regardless of winding. Marching-cubes
        // output can have local winding inconsistencies, so culling here
        // creates missing depth regions and unstable AO.
        let front_face_pipeline = create_face_pipeline(
            device,
            &back_face_pipeline_layout,
            &back_depth_shader,
            "MC Front Face Pipeline",
            "fs_normal",
            None,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MC Back Face BG"),
            layout: &back_face_bind_group_layout,
            entries: &[
                entry::buffer(0, camera_buffer),
                entry::buffer(1, vertex_buffer),
                entry::buffer(2, container_geom_buffer),
            ],
        });

        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("MC Back Depth Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            front_face_pipeline,
            back_face_pipeline,
            bind_group,
            targets: Targets::new(device, width, height),
            depth_sampler,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.targets = Targets::new(device, width, height);
    }

    /// Nearest water surface depth (GTAO input, front-face exits)
    pub fn front_depth_view(&self) -> &wgpu::TextureView {
        &self.targets.front_depth.1
    }

    /// Smooth world normals of the nearest water surface
    pub fn front_normal_view(&self) -> &wgpu::TextureView {
        &self.targets.front_normal.1
    }

    pub fn back_depth_view(&self) -> &wgpu::TextureView {
        &self.targets.back_depth.1
    }

    pub fn back_normal_view(&self) -> &wgpu::TextureView {
        &self.targets.back_normal.1
    }

    pub fn depth_sampler(&self) -> &wgpu::Sampler {
        &self.depth_sampler
    }

    /// Front faces into the depth + normal G-buffer, then rigid bodies and
    /// the container into that depth (GTAO and SSR see them too)
    pub fn encode_front(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        indirect_buffer: &wgpu::Buffer,
        rigid_body: Option<&RigidBodyRenderer>,
        container: Option<&ContainerRenderer>,
    ) {
        // Pass 0a: Render water front faces to depth + normal G-buffer (for GTAO + SSR)
        {
            let mut front_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("MC Front Face + Normal Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.front_normal_view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.front_depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            front_pass.set_pipeline(&self.front_face_pipeline);
            front_pass.set_bind_group(0, &self.bind_group, &[]);
            front_pass.draw_indirect(indirect_buffer, 0);
        }

        // Pass 0b: Render rigid body + container into front depth (depth-only, no color targets)
        if rigid_body.is_some() || container.is_some() {
            let mut depth_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("MC Front Depth (RB+Container)"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.front_depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if let Some(rb) = rigid_body {
                rb.render_depth_only(&mut depth_pass);
            }
            if let Some(ct) = container {
                ct.render_depth_only(&mut depth_pass);
            }
        }
    }

    /// Back faces: depth (thickness, in-water tests) and their normals
    /// (refraction exit interface; w = 0 where there is no back face)
    pub fn encode_back(&self, encoder: &mut wgpu::CommandEncoder, indirect_buffer: &wgpu::Buffer) {
        let mut back_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("MC Back Face Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.back_normal_view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: self.back_depth_view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        back_pass.set_pipeline(&self.back_face_pipeline);
        back_pass.set_bind_group(0, &self.bind_group, &[]);
        // Use indirect draw - vertex count comes from GPU buffer
        back_pass.draw_indirect(indirect_buffer, 0);
    }
}
