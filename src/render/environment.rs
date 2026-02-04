//! Environment map loading and GPU texture creation
//! Supports equirectangular HDR images (Radiance .hdr format)

use half::f16;
use std::io::Cursor;

/// Load the embedded environment map (compile-time included)
pub fn load_embedded_environment_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(wgpu::Texture, wgpu::TextureView, wgpu::Sampler), String> {
    // Include the HDR file at compile time
    let hdr_bytes = include_bytes!("../assets/farmland.hdr");

    // Use the high-level image API to load HDR
    let reader = image::ImageReader::new(Cursor::new(hdr_bytes))
        .with_guessed_format()
        .map_err(|e| format!("Failed to guess image format: {}", e))?;

    let dynamic_image = reader.decode()
        .map_err(|e| format!("Failed to decode HDR image: {}", e))?;

    // Convert to Rgb32F (HDR format)
    let rgb32f = dynamic_image.to_rgb32f();
    let width = rgb32f.width();
    let height = rgb32f.height();

    log::info!("Loading embedded environment map: {}x{}", width, height);

    // Convert to RGBA f16 for GPU (filterable format)
    // Store as u16 (the bit representation of f16) for bytemuck compatibility
    let mut rgba_data: Vec<u16> = Vec::with_capacity((width * height * 4) as usize);
    for pixel in rgb32f.pixels() {
        rgba_data.push(f16::from_f32(pixel.0[0]).to_bits());
        rgba_data.push(f16::from_f32(pixel.0[1]).to_bits());
        rgba_data.push(f16::from_f32(pixel.0[2]).to_bits());
        rgba_data.push(f16::from_f32(1.0).to_bits());
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Environment Map"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Rgba16Float is filterable (supports linear sampling) and has good HDR precision
        format: wgpu::TextureFormat::Rgba16Float,
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
        bytemuck::cast_slice(&rgba_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4 * 2), // 4 channels * 2 bytes per f16
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Environment Sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    log::info!("Environment map loaded successfully");

    Ok((texture, view, sampler))
}
