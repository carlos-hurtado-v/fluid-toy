//! GLB mesh loader — extracts vertices, indices, and texture from embedded GLB files

/// A single mesh vertex: position + normal + UV + color (48 bytes, GPU-compatible)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3], // 12 bytes
    pub normal: [f32; 3],   // 12 bytes
    pub uv: [f32; 2],       // 8 bytes
    pub color: [f32; 4],    // 16 bytes → 48 total
}

/// CPU-side mesh data extracted from a GLB file, normalized to [-1, 1]^3
pub struct LoadedMesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub texture_rgba: Vec<u8>,
    pub texture_width: u32,
    pub texture_height: u32,
}

/// Load the embedded duck.glb asset
pub fn load_embedded_duck() -> Result<LoadedMesh, Box<dyn std::error::Error>> {
    let glb_bytes = include_bytes!("../assets/duck.glb");
    load_glb_from_bytes(glb_bytes)
}

/// Parse a GLB file from raw bytes, merging all meshes and primitives
pub fn load_glb_from_bytes(bytes: &[u8]) -> Result<LoadedMesh, Box<dyn std::error::Error>> {
    let (document, buffers, images) = gltf::import_slice(bytes)?;

    // First pass: collect all raw positions across all primitives for AABB computation
    let mut all_positions: Vec<[f32; 3]> = Vec::new();
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            if let Some(pos_iter) = reader.read_positions() {
                all_positions.extend(pos_iter);
            }
        }
    }

    if all_positions.is_empty() {
        return Err("No vertex positions found in GLB".into());
    }

    // Compute AABB for normalization
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for p in &all_positions {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }

    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let half_extents = [
        (max[0] - min[0]) * 0.5,
        (max[1] - min[1]) * 0.5,
        (max[2] - min[2]) * 0.5,
    ];
    let max_half = half_extents[0]
        .max(half_extents[1])
        .max(half_extents[2])
        .max(0.001);

    // Find the first texture across all primitives (used as the shared texture)
    let mut texture_rgba: Option<Vec<u8>> = None;
    let mut texture_width = 1u32;
    let mut texture_height = 1u32;

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if texture_rgba.is_some() {
                break;
            }
            let material = primitive.material();
            let pbr = material.pbr_metallic_roughness();
            if let Some(tex_info) = pbr.base_color_texture() {
                let img_index = tex_info.texture().source().index();
                if img_index < images.len() {
                    let img_data = &images[img_index];
                    texture_rgba = Some(convert_to_rgba8(img_data));
                    texture_width = img_data.width;
                    texture_height = img_data.height;
                }
            }
        }
    }

    // Fallback: 1x1 white texture if no texture found
    let texture_rgba = texture_rgba.unwrap_or_else(|| vec![255, 255, 255, 255]);

    // Second pass: merge all primitives into one vertex/index buffer
    let mut vertices: Vec<MeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(iter) => iter.collect(),
                None => continue,
            };

            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|tc| tc.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            // Determine per-vertex color for this primitive:
            // - If primitive has its own texture, use white (texture provides color)
            // - If no texture, bake the material's base_color_factor as vertex color
            let material = primitive.material();
            let pbr = material.pbr_metallic_roughness();
            let has_texture = pbr.base_color_texture().is_some();
            let base_color = if has_texture {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                pbr.base_color_factor()
            };

            // Offset indices by current vertex count
            let base_vertex = vertices.len() as u32;

            if let Some(idx_reader) = reader.read_indices() {
                for idx in idx_reader.into_u32() {
                    indices.push(base_vertex + idx);
                }
            } else {
                // No index buffer: generate sequential indices
                for i in 0..positions.len() as u32 {
                    indices.push(base_vertex + i);
                }
            }

            // Build normalized vertices
            for i in 0..positions.len() {
                let pos = positions[i];
                vertices.push(MeshVertex {
                    position: [
                        (pos[0] - center[0]) / max_half,
                        (pos[1] - center[1]) / max_half,
                        (pos[2] - center[2]) / max_half,
                    ],
                    normal: normals[i],
                    uv: uvs[i],
                    color: base_color,
                });
            }
        }
    }

    log::info!(
        "Loaded GLB: {} vertices, {} indices across all primitives, texture {}x{}",
        vertices.len(),
        indices.len(),
        texture_width,
        texture_height,
    );

    Ok(LoadedMesh {
        vertices,
        indices,
        texture_rgba,
        texture_width,
        texture_height,
    })
}

/// Convert gltf image data to RGBA8 bytes
fn convert_to_rgba8(img_data: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format;

    match img_data.format {
        Format::R8G8B8A8 => img_data.pixels.clone(),
        Format::R8G8B8 => {
            let pixel_count = img_data.pixels.len() / 3;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for chunk in img_data.pixels.chunks(3) {
                rgba.push(chunk[0]);
                rgba.push(chunk[1]);
                rgba.push(chunk[2]);
                rgba.push(255);
            }
            rgba
        }
        _ => {
            if let Ok(img) = image::load_from_memory(&img_data.pixels) {
                img.to_rgba8().into_raw()
            } else {
                log::warn!("Unknown image format {:?}, using white fallback", img_data.format);
                vec![255, 255, 255, 255]
            }
        }
    }
}
