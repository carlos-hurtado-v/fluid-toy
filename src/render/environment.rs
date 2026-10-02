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

impl ShCoefficients {
    /// Coefficients for a uniform environment of the given radiance: exactly
    /// what `compute_sh_irradiance` would return for an HDR map filled with
    /// `color`. Only the L0 band is non-zero:
    ///   c00 = color * Y00 * 4pi (projection) * A0 (cosine lobe, = pi)
    /// so the shader's `evaluate_sh_irradiance` yields `pi * color` for every
    /// normal, matching the convention of the HDR path.
    pub fn uniform(color: [f32; 3]) -> Self {
        let pi = std::f32::consts::PI;
        let c00 = 0.282095 * 4.0 * pi * pi;
        let mut coeffs = [[0.0; 4]; 9];
        coeffs[0] = [color[0] * c00, color[1] * c00, color[2] * c00, 0.0];
        Self { coeffs }
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

/// The sun as found in an HDR map, split out so it can be lit analytically:
/// direction toward it (same equirect convention as `compute_sh_irradiance`),
/// the irradiance it delivers on a surface facing it, and SH irradiance of
/// the map with the sun disk painted over by the surrounding sky (so ambient
/// lighting does not count the sun a second time).
#[derive(Debug, Clone, Copy)]
pub struct SunEstimate {
    pub direction: [f32; 3],
    pub irradiance: [f32; 3],
    pub sky_sh: ShCoefficients,
}

/// Peak-to-median luminance below which the brightest spot is treated as
/// overcast glow, not a sun disk worth splitting out.
const SUN_MIN_PEAK_RATIO: f32 = 30.0;
/// Angular radius around the peak searched for the sun's core.
const SUN_CORE_RADIUS_DEG: f32 = 5.0;
/// Pixels this many times brighter than the median sky (inside the core
/// radius) belong to the sun disk + tight corona.
const SUN_CORE_RATIO: f32 = 100.0;
/// The ring just outside the core whose mean radiance repaints the core in
/// the sky-only map.
const SUN_RING_RADIUS_DEG: f32 = 8.0;

/// Locate the sun in an equirectangular HDR map (upper hemisphere only) and
/// separate it from the sky. Returns None when no peak stands clearly above
/// the sky (overcast maps keep the plain SH and the manual sun).
pub fn estimate_sun(pixels: &[f32], width: u32, height: u32) -> Option<SunEstimate> {
    let pi = std::f32::consts::PI;
    let dir_of = |x: u32, y: u32| {
        let theta = pi * (y as f32 + 0.5) / height as f32;
        let phi = 2.0 * pi * (x as f32 + 0.5) / width as f32;
        [theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()]
    };
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let luminance = |x: u32, y: u32| {
        let i = ((y * width + x) * 3) as usize;
        0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2]
    };

    let sky_rows = height / 2;
    let mut sky: Vec<f32> = Vec::with_capacity((sky_rows * width) as usize);
    let (mut peak, mut peak_xy) = (0.0f32, (0, 0));
    for y in 0..sky_rows {
        for x in 0..width {
            let l = luminance(x, y);
            sky.push(l);
            if l > peak {
                peak = l;
                peak_xy = (x, y);
            }
        }
    }
    let mid = sky.len() / 2;
    let median = *sky.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
    let peak_ratio = peak / median.max(1e-6);
    if peak_ratio < SUN_MIN_PEAK_RATIO {
        log::info!("HDR sun: peak/median ratio {:.0} - too diffuse, keeping the manual sun", peak_ratio);
        return None;
    }

    // Core pixels (around the peak, far above the sky) and the ring around them
    let peak_dir = dir_of(peak_xy.0, peak_xy.1);
    let cos_core = SUN_CORE_RADIUS_DEG.to_radians().cos();
    let cos_ring = SUN_RING_RADIUS_DEG.to_radians().cos();
    let core_level = SUN_CORE_RATIO * median;
    let d_theta = pi / height as f32;
    let d_phi = 2.0 * pi / width as f32;
    let mut core: Vec<usize> = Vec::new();
    let mut centroid = [0.0f32; 3];
    let mut ring_sum = [0.0f64; 3];
    let mut ring_weight = 0.0f64;
    for y in 0..sky_rows {
        let sin_theta = (pi * (y as f32 + 0.5) / height as f32).sin();
        for x in 0..width {
            let d = dir_of(x, y);
            let c = dot(d, peak_dir);
            if c < cos_ring {
                continue;
            }
            let i = ((y * width + x) * 3) as usize;
            let l = luminance(x, y);
            if c >= cos_core && l > core_level {
                core.push(i);
                // Solid-angle weight of an equirect pixel goes as sin(theta)
                let w = l * sin_theta;
                for k in 0..3 {
                    centroid[k] += w * d[k];
                }
            } else if c < cos_core {
                for k in 0..3 {
                    ring_sum[k] += pixels[i + k] as f64 * sin_theta as f64;
                }
                ring_weight += sin_theta as f64;
            }
        }
    }
    let len = dot(centroid, centroid).sqrt();
    let direction = if len > 0.0 { centroid.map(|v| v / len) } else { peak_dir };
    let ring = ring_sum.map(|v| (v / ring_weight.max(1e-9)) as f32);

    // Irradiance the core delivers on a plane facing the sun, and a sky-only
    // copy of the map with the core repainted by the surrounding ring
    let mut irradiance = [0.0f64; 3];
    let mut sky_pixels = pixels.to_vec();
    for &i in &core {
        let px = i / 3;
        let (x, y) = ((px as u32) % width, (px as u32) / width);
        let theta = pi * (y as f32 + 0.5) / height as f32;
        let d_omega = (theta.sin() * d_theta * d_phi) as f64;
        let cos_n = dot(dir_of(x, y), direction).max(0.0) as f64;
        for k in 0..3 {
            irradiance[k] += (pixels[i + k] - ring[k]).max(0.0) as f64 * cos_n * d_omega;
            sky_pixels[i + k] = ring[k];
        }
    }
    let irradiance = irradiance.map(|v| v as f32);
    let sky_sh = compute_sh_irradiance(&sky_pixels, width, height);

    log::info!(
        "HDR sun: peak/median {:.0}, {} core px, direction [{:.3}, {:.3}, {:.3}] (elevation {:.1} deg), irradiance [{:.2}, {:.2}, {:.2}]",
        peak_ratio,
        core.len(),
        direction[0],
        direction[1],
        direction[2],
        direction[1].asin().to_degrees(),
        irradiance[0],
        irradiance[1],
        irradiance[2],
    );
    Some(SunEstimate { direction, irradiance, sky_sh })
}

/// Upper bound on environment texels uploaded to the GPU (see the upload loop)
const ENV_RADIANCE_CAP: f32 = 64.0;
/// Levels in the environment map's mip chain (2048x1024 down to 16x8)
const ENV_MAX_MIPS: u32 = 8;

/// Box-filter an RGB f32 image to (nw, nh), each a floor-halving of (w, h)
fn downsample_rgb(src: &[f32], w: u32, h: u32, nw: u32, nh: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((nw * nh * 3) as usize);
    for y in 0..nh {
        for x in 0..nw {
            let (x0, y0) = ((2 * x).min(w - 1), (2 * y).min(h - 1));
            let (x1, y1) = ((2 * x + 1).min(w - 1), (2 * y + 1).min(h - 1));
            for c in 0..3 {
                let at = |px: u32, py: u32| src[((py * w + px) * 3 + c) as usize];
                out.push(0.25 * (at(x0, y0) + at(x1, y0) + at(x0, y1) + at(x1, y1)));
            }
        }
    }
    out
}

/// Load the embedded environment map (compile-time included)
/// Returns (texture, view, sampler, sh_coefficients, sun)
#[allow(clippy::type_complexity)]
pub fn load_embedded_environment_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    selection: HdrEnvironment,
) -> Result<(wgpu::Texture, wgpu::TextureView, wgpu::Sampler, ShCoefficients, Option<SunEstimate>), String> {
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
    let sun = estimate_sun(&raw_pixels, width, height);
    log::info!("SH irradiance computed (band 0 RGB: [{:.3}, {:.3}, {:.3}])",
        sh_coefficients.coeffs[0][0], sh_coefficients.coeffs[0][1], sh_coefficients.coeffs[0][2]);

    // Convert to RGBA f16 for GPU (filterable format)
    // Store as u16 (the bit representation of f16) for bytemuck compatibility.
    // Sun disks exceed the f16 range (Farmland peaks near 69k, PureSky 99k):
    // they would become inf, and inf texels turn into NaN under filtering.
    // Capped instead — the sun's light is measured from the full-precision data
    // above, and the shown/reflected disk only needs to read far above white.
    // Mip chain (box filter of the capped radiance): refraction and mirror
    // lookups that shrink the map read a level that matches their footprint
    // instead of skipping texels. Direct views sample level 0, as before.
    let mut level: Vec<f32> = rgb32f
        .pixels()
        .flat_map(|p| [p.0[0].min(ENV_RADIANCE_CAP), p.0[1].min(ENV_RADIANCE_CAP), p.0[2].min(ENV_RADIANCE_CAP)])
        .collect();
    let mip_level_count = (width.min(height).max(1).ilog2() + 1).min(ENV_MAX_MIPS);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Environment Map"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Rgba16Float is filterable (supports linear sampling) and has good HDR precision
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let (mut w, mut h) = (width, height);
    for mip_level in 0..mip_level_count {
        let one = f16::from_f32(1.0).to_bits();
        let rgba_data: Vec<u16> = level
            .chunks_exact(3)
            .flat_map(|p| [f16::from_f32(p[0]).to_bits(), f16::from_f32(p[1]).to_bits(), f16::from_f32(p[2]).to_bits(), one])
            .collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&rgba_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4 * 2), // 4 channels * 2 bytes per f16
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        level = downsample_rgb(&level, w, h, nw, nh);
        (w, h) = (nw, nh);
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Environment Sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        // Only gradient lookups (the water's filtered refraction) use it;
        // explicit level-0 samples are unaffected
        anisotropy_clamp: 16,
        ..Default::default()
    });

    log::info!("Environment map loaded successfully");

    Ok((texture, view, sampler, sh_coefficients, sun))
}
