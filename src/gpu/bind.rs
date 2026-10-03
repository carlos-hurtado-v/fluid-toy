//! Short forms for bind group layouts and bind groups.
//!
//! A `wgpu::BindGroupLayoutEntry` spelled out is nine lines and a
//! `wgpu::BindGroupEntry` four; a pass with a dozen bindings buries its
//! pipeline under them. These cover the shapes this project uses (no dynamic
//! offsets, no binding arrays, filtering samplers, write-only storage
//! textures). Anything else is still written out in full at its call site.
//!
//! ```ignore
//! use crate::gpu::bind::{entry, layout, COMPUTE};
//! let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
//!     label: Some("Blur BGL"),
//!     entries: &[
//!         layout::texture_3d_unfilterable(0, COMPUTE),
//!         layout::storage_texture_3d(1, COMPUTE, wgpu::TextureFormat::R32Float),
//!         layout::uniform(2, COMPUTE),
//!     ],
//! });
//! ```

pub const VERTEX: wgpu::ShaderStages = wgpu::ShaderStages::VERTEX;
pub const FRAGMENT: wgpu::ShaderStages = wgpu::ShaderStages::FRAGMENT;
pub const VERTEX_FRAGMENT: wgpu::ShaderStages = wgpu::ShaderStages::VERTEX_FRAGMENT;
pub const COMPUTE: wgpu::ShaderStages = wgpu::ShaderStages::COMPUTE;

/// `wgpu::BindGroupLayoutEntry` constructors: `(binding, visibility)`
pub mod layout {
    use wgpu::{BindGroupLayoutEntry, BindingType, ShaderStages, TextureSampleType, TextureViewDimension};

    fn entry(binding: u32, visibility: ShaderStages, ty: BindingType) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry { binding, visibility, ty, count: None }
    }

    fn buffer(binding: u32, visibility: ShaderStages, ty: wgpu::BufferBindingType) -> BindGroupLayoutEntry {
        entry(binding, visibility, BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None })
    }

    fn texture(
        binding: u32,
        visibility: ShaderStages,
        sample_type: TextureSampleType,
        view_dimension: TextureViewDimension,
    ) -> BindGroupLayoutEntry {
        entry(binding, visibility, BindingType::Texture { sample_type, view_dimension, multisampled: false })
    }

    /// Uniform buffer
    pub fn uniform(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        buffer(binding, visibility, wgpu::BufferBindingType::Uniform)
    }

    /// Read-only storage buffer
    pub fn storage(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        buffer(binding, visibility, wgpu::BufferBindingType::Storage { read_only: true })
    }

    /// Read-write storage buffer
    pub fn storage_rw(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        buffer(binding, visibility, wgpu::BufferBindingType::Storage { read_only: false })
    }

    /// 2D float texture a filtering sampler may read
    pub fn texture_2d(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        texture(binding, visibility, TextureSampleType::Float { filterable: true }, TextureViewDimension::D2)
    }

    /// 2D float texture read with `textureLoad` (or a format that cannot be filtered)
    pub fn texture_2d_unfilterable(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        texture(binding, visibility, TextureSampleType::Float { filterable: false }, TextureViewDimension::D2)
    }

    /// 2D depth texture
    pub fn texture_depth(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        texture(binding, visibility, TextureSampleType::Depth, TextureViewDimension::D2)
    }

    /// 3D float texture a filtering sampler may read (R32Float needs FLOAT32_FILTERABLE)
    pub fn texture_3d(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        texture(binding, visibility, TextureSampleType::Float { filterable: true }, TextureViewDimension::D3)
    }

    /// 3D float texture read with `textureLoad`
    pub fn texture_3d_unfilterable(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        texture(binding, visibility, TextureSampleType::Float { filterable: false }, TextureViewDimension::D3)
    }

    /// 3D unsigned-integer texture
    pub fn texture_3d_uint(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        texture(binding, visibility, TextureSampleType::Uint, TextureViewDimension::D3)
    }

    /// Filtering sampler
    pub fn sampler(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
        entry(binding, visibility, BindingType::Sampler(wgpu::SamplerBindingType::Filtering))
    }

    fn storage_texture(
        binding: u32,
        visibility: ShaderStages,
        format: wgpu::TextureFormat,
        view_dimension: TextureViewDimension,
    ) -> BindGroupLayoutEntry {
        entry(
            binding,
            visibility,
            BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension },
        )
    }

    /// Write-only 2D storage texture
    pub fn storage_texture_2d(binding: u32, visibility: ShaderStages, format: wgpu::TextureFormat) -> BindGroupLayoutEntry {
        storage_texture(binding, visibility, format, TextureViewDimension::D2)
    }

    /// Write-only 3D storage texture
    pub fn storage_texture_3d(binding: u32, visibility: ShaderStages, format: wgpu::TextureFormat) -> BindGroupLayoutEntry {
        storage_texture(binding, visibility, format, TextureViewDimension::D3)
    }
}

/// `wgpu::BindGroupEntry` constructors: `(binding, resource)`
pub mod entry {
    use wgpu::{BindGroupEntry, BindingResource};

    /// A whole buffer
    pub fn buffer(binding: u32, buffer: &wgpu::Buffer) -> BindGroupEntry<'_> {
        BindGroupEntry { binding, resource: buffer.as_entire_binding() }
    }

    pub fn view(binding: u32, view: &wgpu::TextureView) -> BindGroupEntry<'_> {
        BindGroupEntry { binding, resource: BindingResource::TextureView(view) }
    }

    pub fn sampler(binding: u32, sampler: &wgpu::Sampler) -> BindGroupEntry<'_> {
        BindGroupEntry { binding, resource: BindingResource::Sampler(sampler) }
    }
}
