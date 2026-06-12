//! Screen-space fluid renderer
//!
//! Renders SPH particles as billboard quads, smooths depth via narrow-range filter
//! (Truong & Yuksel i3D 2018), reconstructs normals, then composites with PBR water shading.
//!
//! Pipeline: depth splat → thickness splat (half-res) → thickness blur →
//! narrow-range filter → normals → opaque scene to background (env + container +
//! rigid body + spray, with depth) → depth-aware composite (water or scene per pixel)

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::render::camera::GpuCameraParams;
use crate::render::container_renderer::ContainerRenderer;
use crate::render::marching_cubes::GpuWaterParams;
use crate::render::rigid_body_renderer::RigidBodyRenderer;
use crate::render::spray_renderer::SprayRenderer;
use crate::state::{GpuLightParams, GpuShCoefficients};

// ─── GPU uniform structs ───────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuSsParams {
    particle_radius: f32,
    num_particles: u32,
    screen_width: f32,
    screen_height: f32,
    thickness_scale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuFilterParams {
    projected_particle_constant: f32,
    max_filter_size: f32,
    mu: f32,
    depth_threshold: f32,
    screen_width: u32,
    screen_height: u32,
    blur_2d: u32,
    direction: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuThicknessBlurParams {
    screen_width: u32,
    screen_height: u32,
    radius: u32,
    direction: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuNormalParams {
    screen_width: u32,
    screen_height: u32,
    _pad0: u32,
    _pad1: u32,
}

// ─── Helper: create a 2D texture ───────────────────────────────────────────

fn create_texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

// ─── Renderer ──────────────────────────────────────────────────────────────

pub struct ScreenSpaceFluidRenderer {
    width: u32,
    height: u32,
    surface_format: wgpu::TextureFormat,

    // Textures
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    hw_depth_texture: wgpu::Texture,
    hw_depth_view: wgpu::TextureView,
    filtered_depth_texture: wgpu::Texture,
    filtered_depth_view: wgpu::TextureView,
    thickness_texture: wgpu::Texture,
    thickness_view: wgpu::TextureView,
    filtered_thickness_a: wgpu::Texture,
    filtered_thickness_a_view: wgpu::TextureView,
    filtered_thickness_b: wgpu::Texture,
    filtered_thickness_b_view: wgpu::TextureView,
    normal_texture: wgpu::Texture,
    normal_view: wgpu::TextureView,
    background_texture: wgpu::Texture,
    background_view: wgpu::TextureView,
    background_depth_texture: wgpu::Texture,
    background_depth_view: wgpu::TextureView,

    // Uniform buffers
    ss_params_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    water_params_buffer: wgpu::Buffer,
    light_params_buffer: wgpu::Buffer,
    sh_coefficients_buffer: wgpu::Buffer,
    filter_params_h_buffer: wgpu::Buffer,
    filter_params_v_buffer: wgpu::Buffer,
    filter_params_2d_buffer: wgpu::Buffer,
    filter_params_2d_back_buffer: wgpu::Buffer,
    thickness_blur_h_buffer: wgpu::Buffer,
    thickness_blur_v_buffer: wgpu::Buffer,
    normal_params_buffer: wgpu::Buffer,

    // Pipelines
    depth_pipeline: wgpu::RenderPipeline,
    thickness_pipeline: wgpu::RenderPipeline,
    filter_pipeline: wgpu::ComputePipeline,
    thickness_blur_pipeline: wgpu::ComputePipeline,
    normal_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::RenderPipeline,
    env_pipeline: wgpu::RenderPipeline,

    // Bind group layouts (needed for rebuilding bind groups on resize)
    splat_bgl: wgpu::BindGroupLayout,
    filter_bgl: wgpu::BindGroupLayout,
    thickness_blur_bgl: wgpu::BindGroupLayout,
    normal_bgl: wgpu::BindGroupLayout,
    composite_uniform_bgl: wgpu::BindGroupLayout,
    composite_texture_bgl: wgpu::BindGroupLayout,
    env_bgl: wgpu::BindGroupLayout,

    // Bind groups (rebuilt on resize)
    filter_h_bg: wgpu::BindGroup,
    filter_v_bg: wgpu::BindGroup,
    filter_2d_bg: wgpu::BindGroup,
    filter_2d_back_bg: wgpu::BindGroup,
    thickness_blur_h_bg: wgpu::BindGroup,
    thickness_blur_v_bg: wgpu::BindGroup,
    normal_bg: wgpu::BindGroup,
    composite_uniform_bg: wgpu::BindGroup,
    composite_texture_bg: wgpu::BindGroup,
    env_bg: wgpu::BindGroup,
    env_params_buffer: wgpu::Buffer,

    // Sampler
    sampler: wgpu::Sampler,
}

impl ScreenSpaceFluidRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        env_view: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
        camera_params: &GpuCameraParams,
        light_params: &GpuLightParams,
        water_params: &GpuWaterParams,
        sh: &GpuShCoefficients,
        width: u32,
        height: u32,
    ) -> Self {
        // ── Textures ──────────────────────────────────────────────────────
        let tex_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING;
        let storage_tex_usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING;

        let depth_texture = create_texture(device, "SS Depth", width, height,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
        let depth_view = depth_texture.create_view(&Default::default());

        let hw_depth_texture = create_texture(device, "SS HW Depth", width, height,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let hw_depth_view = hw_depth_texture.create_view(&Default::default());

        let filtered_depth_texture = create_texture(device, "SS Filtered Depth", width, height,
            wgpu::TextureFormat::R32Float, storage_tex_usage | wgpu::TextureUsages::RENDER_ATTACHMENT);
        let filtered_depth_view = filtered_depth_texture.create_view(&Default::default());

        // Thickness is splatted and blurred at half resolution (Splash parity) —
        // it's naturally low-frequency, and this quarters the splat fill cost.
        let (thick_w, thick_h) = ((width / 2).max(1), (height / 2).max(1));
        let thickness_texture = create_texture(device, "SS Thickness", thick_w, thick_h,
            wgpu::TextureFormat::Rgba16Float, tex_usage);
        let thickness_view = thickness_texture.create_view(&Default::default());

        let filtered_thickness_a = create_texture(device, "SS Filt Thick A", thick_w, thick_h,
            wgpu::TextureFormat::Rgba16Float, storage_tex_usage);
        let filtered_thickness_a_view = filtered_thickness_a.create_view(&Default::default());

        let filtered_thickness_b = create_texture(device, "SS Filt Thick B", thick_w, thick_h,
            wgpu::TextureFormat::Rgba16Float, storage_tex_usage);
        let filtered_thickness_b_view = filtered_thickness_b.create_view(&Default::default());

        let normal_texture = create_texture(device, "SS Normals", width, height,
            wgpu::TextureFormat::Rgba16Float, storage_tex_usage);
        let normal_view = normal_texture.create_view(&Default::default());

        let background_texture = create_texture(device, "SS Background", width, height,
            surface_format, tex_usage);
        let background_view = background_texture.create_view(&Default::default());

        let background_depth_texture = create_texture(device, "SS Background Depth", width, height,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let background_depth_view = background_depth_texture.create_view(&Default::default());

        // ── Sampler ───────────────────────────────────────────────────────
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SS Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ── Uniform Buffers ───────────────────────────────────────────────
        let ss_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Params"),
            contents: bytemuck::bytes_of(&GpuSsParams {
                particle_radius: 0.02,
                num_particles: 0,
                screen_width: width as f32,
                screen_height: height as f32,
                thickness_scale: 1.0,
                _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Camera"),
            contents: bytemuck::bytes_of(camera_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let water_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Water Params"),
            contents: bytemuck::bytes_of(water_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Light Params"),
            contents: bytemuck::bytes_of(light_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sh_coefficients_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS SH Coefficients"),
            contents: bytemuck::bytes_of(sh),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let default_filter = GpuFilterParams {
            projected_particle_constant: 100.0,
            max_filter_size: 50.0,
            mu: 0.63,
            depth_threshold: 2.1,
            screen_width: width,
            screen_height: height,
            blur_2d: 0,
            direction: 0,
            _pad0: 0.0, _pad1: 0.0, _pad2: 0.0, _pad3: 0.0,
        };
        let filter_params_h_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Filter H"),
            contents: bytemuck::bytes_of(&default_filter),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let mut filter_v = default_filter;
        filter_v.direction = 1;
        let filter_params_v_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Filter V"),
            contents: bytemuck::bytes_of(&filter_v),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let mut filter_2d = default_filter;
        filter_2d.blur_2d = 1;
        let filter_params_2d_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Filter 2D"),
            contents: bytemuck::bytes_of(&filter_2d),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let filter_params_2d_back_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Filter 2D Back"),
            contents: bytemuck::bytes_of(&filter_2d),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let thickness_blur_h_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Thick Blur H"),
            contents: bytemuck::bytes_of(&GpuThicknessBlurParams {
                screen_width: thick_w, screen_height: thick_h, radius: 10, direction: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let thickness_blur_v_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Thick Blur V"),
            contents: bytemuck::bytes_of(&GpuThicknessBlurParams {
                screen_width: thick_w, screen_height: thick_h, radius: 10, direction: 1,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let normal_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Normal Params"),
            contents: bytemuck::bytes_of(&GpuNormalParams {
                screen_width: width, screen_height: height, _pad0: 0, _pad1: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let env_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SS Env Params"),
            contents: bytemuck::bytes_of(&crate::state::GpuEnvironmentParams {
                use_env_background: 1,
                background_r: 0.02,
                background_g: 0.02,
                background_b: 0.05,
                env_intensity: 1.0,
                _pad: [0.0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ── Bind Group Layouts ────────────────────────────────────────────

        // Splat BGL: camera (uniform) + particles (storage) + ss_params (uniform)
        let splat_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SS Splat BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Filter BGL: params (uniform) + input_depth (texture) + output_depth (storage_texture)
        let filter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SS Filter BGL"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Thickness blur BGL: same shape as filter BGL but with Rgba16Float storage
        let thickness_blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SS Thick Blur BGL"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Normal BGL: params + camera + depth_tex + output_normal
        let normal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SS Normal BGL"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Composite uniform BGL (group 0): camera, water, light, sh_coeffs
        let composite_uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SS Composite Uniform BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Composite texture BGL (group 1): filtered_depth, filtered_thickness (half-res,
        // sampled), normals, background color, env, sampler, background depth
        let composite_texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SS Composite Texture BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Env BGL: camera + env_tex + env_sampler + env_params
        let env_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SS Env BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ── Pipelines ─────────────────────────────────────────────────────

        let depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Depth Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ss_depth.wgsl").into()),
        });
        let thickness_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Thickness Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ss_thickness.wgsl").into()),
        });
        let filter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Filter Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ss_filter.wgsl").into()),
        });
        let thickness_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Thickness Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ss_thickness_blur.wgsl").into()),
        });
        let normal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Normal Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ss_normal.wgsl").into()),
        });
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ss_composite.wgsl").into()),
        });
        let env_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Env Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mc_environment.wgsl").into()),
        });

        // Depth pipeline
        let depth_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SS Depth PL"),
            bind_group_layouts: &[&splat_bgl],
            push_constant_ranges: &[],
        });
        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SS Depth Pipeline"),
            layout: Some(&depth_pl),
            vertex: wgpu::VertexState {
                module: &depth_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &depth_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        // Thickness pipeline (additive blending, no depth test)
        let thickness_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SS Thickness PL"),
            bind_group_layouts: &[&splat_bgl],
            push_constant_ranges: &[],
        });
        let thickness_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SS Thickness Pipeline"),
            layout: Some(&thickness_pl),
            vertex: wgpu::VertexState {
                module: &thickness_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &thickness_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        // Compute pipelines
        let filter_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SS Filter PL"),
            bind_group_layouts: &[&filter_bgl],
            push_constant_ranges: &[],
        });
        let filter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SS Filter Pipeline"),
            layout: Some(&filter_pl),
            module: &filter_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let thickness_blur_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SS Thick Blur PL"),
            bind_group_layouts: &[&thickness_blur_bgl],
            push_constant_ranges: &[],
        });
        let thickness_blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SS Thick Blur Pipeline"),
            layout: Some(&thickness_blur_pl),
            module: &thickness_blur_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let normal_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SS Normal PL"),
            bind_group_layouts: &[&normal_bgl],
            push_constant_ranges: &[],
        });
        let normal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SS Normal Pipeline"),
            layout: Some(&normal_pl),
            module: &normal_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Composite pipeline
        let composite_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SS Composite PL"),
            bind_group_layouts: &[&composite_uniform_bgl, &composite_texture_bgl],
            push_constant_ranges: &[],
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SS Composite Pipeline"),
            layout: Some(&composite_pl),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        // Environment background pipeline (reuses mc_environment.wgsl).
        // Draws into the background pass (color + depth) at the far plane,
        // before container/rigid body/spray — same depth state as MC's env pass.
        let env_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SS Env PL"),
            bind_group_layouts: &[&env_bgl],
            push_constant_ranges: &[],
        });
        let env_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SS Env Pipeline"),
            layout: Some(&env_pl),
            vertex: wgpu::VertexState {
                module: &env_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &env_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual, // fullscreen at far plane
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        // ── Bind Groups ───────────────────────────────────────────────────

        // 1D filter: H reads depth → writes filtered, V reads filtered → writes depth
        let filter_h_bg = Self::create_filter_bg(device, &filter_bgl, &filter_params_h_buffer, &depth_view, &filtered_depth_view);
        let filter_v_bg = Self::create_filter_bg(device, &filter_bgl, &filter_params_v_buffer, &filtered_depth_view, &depth_view);
        // 2D filter: reads depth → writes filtered (forward), reads filtered → writes depth (back)
        let filter_2d_bg = Self::create_filter_bg(device, &filter_bgl, &filter_params_2d_buffer, &depth_view, &filtered_depth_view);
        let filter_2d_back_bg = Self::create_filter_bg(device, &filter_bgl, &filter_params_2d_back_buffer, &filtered_depth_view, &depth_view);

        let thickness_blur_h_bg = Self::create_blur_bg(device, &thickness_blur_bgl, &thickness_blur_h_buffer, &thickness_view, &filtered_thickness_a_view);
        let thickness_blur_v_bg = Self::create_blur_bg(device, &thickness_blur_bgl, &thickness_blur_v_buffer, &filtered_thickness_a_view, &filtered_thickness_b_view);

        // Normal and composite read from depth_view (final result after 6 filter passes)
        let normal_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Normal BG"),
            layout: &normal_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: normal_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&depth_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&normal_view) },
            ],
        });

        let composite_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Composite Uniform BG"),
            layout: &composite_uniform_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: water_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: light_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sh_coefficients_buffer.as_entire_binding() },
            ],
        });

        let composite_texture_bg = Self::create_composite_texture_bg(
            device, &composite_texture_bgl, &depth_view, &filtered_thickness_b_view,
            &normal_view, &background_view, env_view, &sampler, &background_depth_view,
        );

        let env_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Env BG"),
            layout: &env_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(env_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(env_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: env_params_buffer.as_entire_binding() },
            ],
        });

        Self {
            width, height, surface_format,
            depth_texture, depth_view,
            hw_depth_texture, hw_depth_view,
            filtered_depth_texture, filtered_depth_view,
            thickness_texture, thickness_view,
            filtered_thickness_a, filtered_thickness_a_view,
            filtered_thickness_b, filtered_thickness_b_view,
            normal_texture, normal_view,
            background_texture, background_view,
            background_depth_texture, background_depth_view,
            ss_params_buffer, camera_buffer, water_params_buffer,
            light_params_buffer, sh_coefficients_buffer,
            filter_params_h_buffer, filter_params_v_buffer,
            filter_params_2d_buffer, filter_params_2d_back_buffer,
            thickness_blur_h_buffer, thickness_blur_v_buffer,
            normal_params_buffer, env_params_buffer,
            depth_pipeline, thickness_pipeline,
            filter_pipeline, thickness_blur_pipeline,
            normal_pipeline, composite_pipeline, env_pipeline,
            splat_bgl, filter_bgl, thickness_blur_bgl, normal_bgl,
            composite_uniform_bgl, composite_texture_bgl, env_bgl,
            filter_h_bg, filter_v_bg,
            filter_2d_bg, filter_2d_back_bg,
            thickness_blur_h_bg, thickness_blur_v_bg,
            normal_bg, composite_uniform_bg, composite_texture_bg,
            env_bg, sampler,
        }
    }

    // ── Bind group helpers ────────────────────────────────────────────────

    fn create_filter_bg(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        params_buf: &wgpu::Buffer,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Filter BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(output_view) },
            ],
        })
    }

    fn create_blur_bg(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        params_buf: &wgpu::Buffer,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Blur BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(output_view) },
            ],
        })
    }

    fn create_composite_texture_bg(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        depth_view: &wgpu::TextureView,
        thickness_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        background_view: &wgpu::TextureView,
        env_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        background_depth_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Composite Texture BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(thickness_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(normal_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(background_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(env_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(background_depth_view) },
            ],
        })
    }

    // ── Public API ────────────────────────────────────────────────────────

    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.hw_depth_view
    }

    pub fn front_depth_view(&self) -> &wgpu::TextureView {
        &self.hw_depth_view
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, params: &GpuCameraParams) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_water_params(&self, queue: &wgpu::Queue, params: &GpuWaterParams) {
        queue.write_buffer(&self.water_params_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_light_params(&self, queue: &wgpu::Queue, params: &GpuLightParams) {
        queue.write_buffer(&self.light_params_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_sh_coefficients(&self, queue: &wgpu::Queue, sh: &GpuShCoefficients) {
        queue.write_buffer(&self.sh_coefficients_buffer, 0, bytemuck::bytes_of(sh));
    }

    pub fn update_env_params(&self, queue: &wgpu::Queue, params: &crate::state::GpuEnvironmentParams) {
        queue.write_buffer(&self.env_params_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        env_view: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
        width: u32,
        height: u32,
    ) {
        if width == 0 || height == 0 { return; }
        self.width = width;
        self.height = height;
        self.recreate_textures_and_bind_groups(device, env_view, env_sampler);
    }

    pub fn rebuild_env_bind_groups(
        &mut self,
        device: &wgpu::Device,
        env_view: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
    ) {
        // Composite reads from depth_view (filter result), not filtered_depth_view
        self.composite_texture_bg = Self::create_composite_texture_bg(
            device, &self.composite_texture_bgl,
            &self.depth_view, &self.filtered_thickness_b_view,
            &self.normal_view, &self.background_view, env_view, &self.sampler,
            &self.background_depth_view,
        );
        self.env_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Env BG"),
            layout: &self.env_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(env_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(env_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.env_params_buffer.as_entire_binding() },
            ],
        });
    }

    fn recreate_textures_and_bind_groups(
        &mut self,
        device: &wgpu::Device,
        env_view: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
    ) {
        let w = self.width;
        let h = self.height;
        let tex_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let storage_tex_usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;

        self.depth_texture = create_texture(device, "SS Depth", w, h, wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING);
        self.depth_view = self.depth_texture.create_view(&Default::default());

        self.hw_depth_texture = create_texture(device, "SS HW Depth", w, h, wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        self.hw_depth_view = self.hw_depth_texture.create_view(&Default::default());

        self.filtered_depth_texture = create_texture(device, "SS Filtered Depth", w, h, wgpu::TextureFormat::R32Float,
            storage_tex_usage | wgpu::TextureUsages::RENDER_ATTACHMENT);
        self.filtered_depth_view = self.filtered_depth_texture.create_view(&Default::default());

        let (thick_w, thick_h) = ((w / 2).max(1), (h / 2).max(1));
        self.thickness_texture = create_texture(device, "SS Thickness", thick_w, thick_h, wgpu::TextureFormat::Rgba16Float, tex_usage);
        self.thickness_view = self.thickness_texture.create_view(&Default::default());

        self.filtered_thickness_a = create_texture(device, "SS Filt Thick A", thick_w, thick_h, wgpu::TextureFormat::Rgba16Float, storage_tex_usage);
        self.filtered_thickness_a_view = self.filtered_thickness_a.create_view(&Default::default());

        self.filtered_thickness_b = create_texture(device, "SS Filt Thick B", thick_w, thick_h, wgpu::TextureFormat::Rgba16Float, storage_tex_usage);
        self.filtered_thickness_b_view = self.filtered_thickness_b.create_view(&Default::default());

        self.normal_texture = create_texture(device, "SS Normals", w, h, wgpu::TextureFormat::Rgba16Float, storage_tex_usage);
        self.normal_view = self.normal_texture.create_view(&Default::default());

        self.background_texture = create_texture(device, "SS Background", w, h, self.surface_format, tex_usage);
        self.background_view = self.background_texture.create_view(&Default::default());

        self.background_depth_texture = create_texture(device, "SS Background Depth", w, h,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        self.background_depth_view = self.background_depth_texture.create_view(&Default::default());

        // Rebuild all bind groups that reference textures
        self.filter_h_bg = Self::create_filter_bg(device, &self.filter_bgl, &self.filter_params_h_buffer, &self.depth_view, &self.filtered_depth_view);
        self.filter_v_bg = Self::create_filter_bg(device, &self.filter_bgl, &self.filter_params_v_buffer, &self.filtered_depth_view, &self.depth_view);
        self.filter_2d_bg = Self::create_filter_bg(device, &self.filter_bgl, &self.filter_params_2d_buffer, &self.depth_view, &self.filtered_depth_view);
        self.filter_2d_back_bg = Self::create_filter_bg(device, &self.filter_bgl, &self.filter_params_2d_back_buffer, &self.filtered_depth_view, &self.depth_view);
        self.thickness_blur_h_bg = Self::create_blur_bg(device, &self.thickness_blur_bgl, &self.thickness_blur_h_buffer, &self.thickness_view, &self.filtered_thickness_a_view);
        self.thickness_blur_v_bg = Self::create_blur_bg(device, &self.thickness_blur_bgl, &self.thickness_blur_v_buffer, &self.filtered_thickness_a_view, &self.filtered_thickness_b_view);
        // Normal and composite read from depth_view (final filter result)
        self.normal_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Normal BG"),
            layout: &self.normal_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.normal_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.depth_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.normal_view) },
            ],
        });
        self.composite_texture_bg = Self::create_composite_texture_bg(
            device, &self.composite_texture_bgl,
            &self.depth_view, &self.filtered_thickness_b_view,
            &self.normal_view, &self.background_view, env_view, &self.sampler,
            &self.background_depth_view,
        );
        self.env_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Env BG"),
            layout: &self.env_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(env_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(env_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: self.env_params_buffer.as_entire_binding() },
            ],
        });
    }

    /// Main render method: runs all passes.
    /// `rigid_body`/`spray`/`container` are drawn into the background (with depth)
    /// so the water refracts and occludes against them, mirroring the MC renderer.
    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
        particle_buffer: &wgpu::Buffer,
        num_particles: u32,
        camera_params: &GpuCameraParams,
        particle_radius: f32,
        particle_spacing: f32,
        filter_size: u32,
        filter_iterations: u32,
        fov_y: f32,
        rigid_body: Option<&RigidBodyRenderer>,
        spray: Option<&SprayRenderer>,
        container: Option<&ContainerRenderer>,
    ) {
        if num_particles == 0 { return; }

        // Each thickness splat adds its chord length, so a ray's accumulated
        // thickness ≈ splat volume fraction × true water depth. Dividing by that
        // fraction ((4/3)πr³ per spacing³ cell) keeps thickness ≈ world-unit depth
        // regardless of the radius-scale slider (absorption stays consistent).
        let splat_volume_fraction = (4.0 / 3.0) * std::f32::consts::PI
            * particle_radius.powi(3) / particle_spacing.powi(3).max(1e-9);
        let thickness_scale = (1.0 / splat_volume_fraction).clamp(0.05, 4.0);

        // Update per-frame uniforms
        queue.write_buffer(&self.ss_params_buffer, 0, bytemuck::bytes_of(&GpuSsParams {
            particle_radius,
            num_particles,
            screen_width: self.width as f32,
            screen_height: self.height as f32,
            thickness_scale,
            _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
        }));
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera_params));

        // Compute projectedParticleConstant (Splash formula)
        // projectedParticleConstant = (blurFilterSize * diameter * 0.05 * height/2) / tan(fov/2)
        let diameter = 2.0 * particle_radius;
        let blur_filter_size = filter_size.max(1) as f32; // GUI "Filter Size" (Splash default: 12)
        let projected_particle_constant = (blur_filter_size * diameter * 0.05
            * (self.height as f32 / 2.0)) / (fov_y / 2.0).tan();
        let max_filter_size = 50.0_f32;
        let mu = 3.0 * particle_radius;
        let depth_threshold = 10.0 * particle_radius;

        // 1D filter params (H direction)
        let filter_h = GpuFilterParams {
            projected_particle_constant,
            max_filter_size,
            mu,
            depth_threshold,
            screen_width: self.width,
            screen_height: self.height,
            blur_2d: 0,
            direction: 0,
            _pad0: 0.0, _pad1: 0.0, _pad2: 0.0, _pad3: 0.0,
        };
        queue.write_buffer(&self.filter_params_h_buffer, 0, bytemuck::bytes_of(&filter_h));

        // 1D filter params (V direction)
        let mut filter_v = filter_h;
        filter_v.direction = 1;
        queue.write_buffer(&self.filter_params_v_buffer, 0, bytemuck::bytes_of(&filter_v));

        // 2D filter params (forward: depth → filtered)
        let mut filter_2d = filter_h;
        filter_2d.blur_2d = 1;
        queue.write_buffer(&self.filter_params_2d_buffer, 0, bytemuck::bytes_of(&filter_2d));
        queue.write_buffer(&self.filter_params_2d_back_buffer, 0, bytemuck::bytes_of(&filter_2d));

        // Update thickness blur params (thickness runs at half resolution)
        let (thick_w, thick_h) = ((self.width / 2).max(1), (self.height / 2).max(1));
        queue.write_buffer(&self.thickness_blur_h_buffer, 0, bytemuck::bytes_of(&GpuThicknessBlurParams {
            screen_width: thick_w, screen_height: thick_h, radius: 10, direction: 0,
        }));
        queue.write_buffer(&self.thickness_blur_v_buffer, 0, bytemuck::bytes_of(&GpuThicknessBlurParams {
            screen_width: thick_w, screen_height: thick_h, radius: 10, direction: 1,
        }));

        // Update normal params
        queue.write_buffer(&self.normal_params_buffer, 0, bytemuck::bytes_of(&GpuNormalParams {
            screen_width: self.width, screen_height: self.height, _pad0: 0, _pad1: 0,
        }));

        // Create per-frame splat bind group (references particle_buffer which may change)
        let splat_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SS Splat BG"),
            layout: &self.splat_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.ss_params_buffer.as_entire_binding() },
            ],
        });

        let wg_x = (self.width + 15) / 16;
        let wg_y = (self.height + 15) / 16;
        let wg_tx = (thick_w + 15) / 16;
        let wg_ty = (thick_h + 15) / 16;

        // ── Pass 1: Depth splatting ───────────────────────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SS Depth Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.depth_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.hw_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.depth_pipeline);
            pass.set_bind_group(0, &splat_bg, &[]);
            pass.draw(0..6, 0..num_particles);
        }

        // ── Pass 2: Thickness splatting ───────────────────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SS Thickness Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.thickness_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.thickness_pipeline);
            pass.set_bind_group(0, &splat_bg, &[]);
            pass.draw(0..6, 0..num_particles);
        }

        // ── Pass 3: Thickness blur (H then V, half resolution) ───────────
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SS Thick Blur H"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.thickness_blur_pipeline);
            pass.set_bind_group(0, &self.thickness_blur_h_bg, &[]);
            pass.dispatch_workgroups(wg_tx, wg_ty, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SS Thick Blur V"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.thickness_blur_pipeline);
            pass.set_bind_group(0, &self.thickness_blur_v_bg, &[]);
            pass.dispatch_workgroups(wg_tx, wg_ty, 1);
        }

        // ── Pass 4: Narrow-range filter (Truong & Yuksel, matching Splash) ──
        // 2 iterations of 1D separable (4 passes: H,V,H,V)
        // + 1 iteration of 2D diamond refinement (2 passes)
        // Total: 6 passes. Ping-pong: depth ↔ filtered_depth.
        // After 6 passes result is in depth_texture.
        let num_1d_iters = filter_iterations.max(1);
        for _ in 0..num_1d_iters {
            // H: depth → filtered
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SS Filter 1D H"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.filter_pipeline);
                pass.set_bind_group(0, &self.filter_h_bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            // V: filtered → depth
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SS Filter 1D V"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.filter_pipeline);
                pass.set_bind_group(0, &self.filter_v_bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }
        // 2D refinement: depth → filtered
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SS Filter 2D"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.filter_pipeline);
            pass.set_bind_group(0, &self.filter_2d_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        // 2D refinement: filtered → depth
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SS Filter 2D back"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.filter_pipeline);
            pass.set_bind_group(0, &self.filter_2d_back_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // ── Pass 5: Normal reconstruction ─────────────────────────────────
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SS Normal Recon"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.normal_pipeline);
            pass.set_bind_group(0, &self.normal_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // ── Background pass: opaque scene to background_texture (+ depth) ─
        // Environment at the far plane, then container, rigid body, and spray
        // with depth testing — mirrors the MC renderer's background pass. The
        // composite refracts this texture and depth-tests the water against it.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SS Scene Background"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.background_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.background_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.env_pipeline);
            pass.set_bind_group(0, &self.env_bg, &[]);
            pass.draw(0..3, 0..1);

            if let Some(rb) = rigid_body {
                rb.render(&mut pass);
            }
            if let Some(ct) = container {
                ct.render(&mut pass);
            }
            if let Some(sp) = spray {
                sp.render(&mut pass);
            }
        }

        // ── Pass 6: Composite (sole writer of the output view) ────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SS Composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.hw_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &self.composite_uniform_bg, &[]);
            pass.set_bind_group(1, &self.composite_texture_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
