//! Environment map loading and GPU texture creation
//! Supports equirectangular HDR images (Radiance .hdr format)

use half::f16;
use std::io::Cursor;

use crate::state::HdrEnvironment;

/// Order-2 spherical harmonics coefficients (9 coefficients × RGB + padding)
/// Pre-convolved with cosine lobe for direct irradiance evaluation.
/// Layout: each coefficient is [R, G, B, pad] for vec4 alignment in WGSL.
#[derive(Debug, Clone, Copy)]
pub struct ShCoefficients {
    pub coeffs: [[f32; 4]; 9],
}

impl Default for ShCoefficients {
    fn default() -> Self {
        Self { coeffs: [[0.0; 4]; 9] }
    }
}

/// Compute order-2 spherical harmonics irradiance coefficients from an equirectangular HDR map.
/// Coefficients are pre-multiplied with cosine lobe convolution (Ramamoorthi & Hanrahan 2001).
pub fn compute_sh_irradiance(pixels: &[f32], width: u32, height: u32) -> ShCoefficients {
    let pi = std::f32::consts::PI;
    let mut sh = [[0.0f64; 3]; 9]; // accumulate in f64 for precision
    let mut weight_sum = 0.0f64;

    for y in 0..height {
        let theta = pi * (y as f32 + 0.5) / height as f32; // 0..PI
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        // solid angle weight for equirectangular projection
        let solid_angle = sin_theta as f64;

        for x in 0..width {
            let phi = 2.0 * pi * (x as f32 + 0.5) / width as f32; // 0..2PI
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            // Direction on unit sphere
            let dx = sin_theta * cos_phi;
            let dy = cos_theta; // Y-up
            let dz = sin_theta * sin_phi;

            let idx = ((y * width + x) * 3) as usize;
            let r = pixels[idx] as f64;
            let g = pixels[idx + 1] as f64;
            let b = pixels[idx + 2] as f64;

            let dx64 = dx as f64;
            let dy64 = dy as f64;
            let dz64 = dz as f64;

            // SH basis functions (real, orthonormal)
            let y00 = 0.282095;               // 1/(2*sqrt(pi))
            let y1m1 = 0.488603 * dy64;       // sqrt(3)/(2*sqrt(pi)) * y
            let y10  = 0.488603 * dz64;        // sqrt(3)/(2*sqrt(pi)) * z
            let y1p1 = 0.488603 * dx64;        // sqrt(3)/(2*sqrt(pi)) * x
            let y2m2 = 1.092548 * dx64 * dy64; // sqrt(15)/(2*sqrt(pi)) * xy
            let y2m1 = 1.092548 * dy64 * dz64; // sqrt(15)/(2*sqrt(pi)) * yz
            let y20  = 0.315392 * (3.0 * dz64 * dz64 - 1.0); // sqrt(5)/(4*sqrt(pi)) * (3z²-1)
            let y2p1 = 1.092548 * dx64 * dz64; // sqrt(15)/(2*sqrt(pi)) * xz
            let y2p2 = 0.546274 * (dx64 * dx64 - dy64 * dy64); // sqrt(15)/(4*sqrt(pi)) * (x²-y²)

            let basis = [y00, y1m1, y10, y1p1, y2m2, y2m1, y20, y2p1, y2p2];
            let color = [r, g, b];

            for (i, &b_val) in basis.iter().enumerate() {
                for c in 0..3 {
                    sh[i][c] += color[c] * b_val * solid_angle;
                }
            }
            weight_sum += solid_angle;
        }
    }

    // Normalize by total solid angle (should be ~4*pi for full sphere)
    let norm = 4.0 * pi as f64 / weight_sum;

    // Cosine lobe convolution constants (Ramamoorthi & Hanrahan 2001)
    // A_l coefficients: A0=pi, A1=2pi/3, A2=pi/4
    let a_hat = [pi as f64, 2.0 * pi as f64 / 3.0, pi as f64 / 4.0];
    let band_idx = [0usize, 1, 1, 1, 2, 2, 2, 2, 2]; // which band each coeff belongs to

    let mut result = ShCoefficients::default();
    for i in 0..9 {
        let a = a_hat[band_idx[i]];
        result.coeffs[i][0] = (sh[i][0] * norm * a) as f32;
        result.coeffs[i][1] = (sh[i][1] * norm * a) as f32;
        result.coeffs[i][2] = (sh[i][2] * norm * a) as f32;
        result.coeffs[i][3] = 0.0;
    }

    result
}

/// Load the embedded environment map (compile-time included)
/// Returns (texture, view, sampler, sh_coefficients)
pub fn load_embedded_environment_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    selection: HdrEnvironment,
) -> Result<(wgpu::Texture, wgpu::TextureView, wgpu::Sampler, ShCoefficients), String> {
    // Include both HDR files at compile time
    let hdr_bytes: &[u8] = match selection {
        HdrEnvironment::Farmland => include_bytes!("../assets/farmland.hdr"),
        HdrEnvironment::PureSky => include_bytes!("../assets/puresky.hdr"),
    };

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

    // Compute SH irradiance from the full-precision f32 data (before f16 conversion)
    let raw_pixels: Vec<f32> = rgb32f.pixels().flat_map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let sh_coefficients = compute_sh_irradiance(&raw_pixels, width, height);
    log::info!("SH irradiance computed (band 0 RGB: [{:.3}, {:.3}, {:.3}])",
        sh_coefficients.coeffs[0][0], sh_coefficients.coeffs[0][1], sh_coefficients.coeffs[0][2]);

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

    Ok((texture, view, sampler, sh_coefficients))
}
