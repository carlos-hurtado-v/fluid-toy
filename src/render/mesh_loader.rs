//! GLB mesh loader — extracts vertices, indices, texture, and SDF from embedded GLB files

/// A single mesh vertex: position + normal + UV + color (48 bytes, GPU-compatible)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3], // 12 bytes
    pub normal: [f32; 3],   // 12 bytes
    pub uv: [f32; 2],       // 8 bytes
    pub color: [f32; 4],    // 16 bytes → 48 total
}

/// Voxelized signed distance field for mesh collision
pub struct SdfData {
    pub data: Vec<f32>,
    pub resolution: u32,
}

/// CPU-side mesh data extracted from a GLB file, normalized to [-1, 1]^3
pub struct LoadedMesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub texture_rgba: Vec<u8>,
    pub texture_width: u32,
    pub texture_height: u32,
    pub sdf: Option<SdfData>,
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

    let sdf = if !indices.is_empty() {
        let start = std::time::Instant::now();
        let sdf = voxelize_sdf(&vertices, &indices, 32);
        log::info!("SDF voxelization (32³): {:.2?}", start.elapsed());
        Some(sdf)
    } else {
        None
    };

    Ok(LoadedMesh {
        vertices,
        indices,
        texture_rgba,
        texture_width,
        texture_height,
        sdf,
    })
}

/// Voxelize a mesh into a 3D signed distance field.
/// Grid spans [-1, 1]³ matching the normalized mesh coordinate space.
/// Negative = inside mesh, positive = outside.
/// Uses ray-casting for robust inside/outside determination.
pub fn voxelize_sdf(vertices: &[MeshVertex], indices: &[u32], resolution: u32) -> SdfData {
    let res = resolution as usize;
    let mut data = vec![f32::MAX; res * res * res];

    // Build triangle list
    let num_tris = indices.len() / 3;
    let mut tri_verts: Vec<[[f32; 3]; 3]> = Vec::with_capacity(num_tris);
    for t in 0..num_tris {
        let i0 = indices[t * 3] as usize;
        let i1 = indices[t * 3 + 1] as usize;
        let i2 = indices[t * 3 + 2] as usize;
        tri_verts.push([vertices[i0].position, vertices[i1].position, vertices[i2].position]);
    }

    // For each voxel, find closest triangle distance + ray-cast for sign
    for zi in 0..res {
        for yi in 0..res {
            for xi in 0..res {
                // Voxel center in [-1, 1]³
                let p = [
                    -1.0 + (xi as f32 + 0.5) * 2.0 / resolution as f32,
                    -1.0 + (yi as f32 + 0.5) * 2.0 / resolution as f32,
                    -1.0 + (zi as f32 + 0.5) * 2.0 / resolution as f32,
                ];

                let mut best_dist_sq = f32::MAX;
                // Ray-cast in +X direction for inside/outside test
                let mut intersections = 0u32;

                for t in 0..num_tris {
                    let [v0, v1, v2] = tri_verts[t];

                    // Unsigned distance to closest point on triangle
                    let (_, dist_sq) = closest_point_on_triangle(p, v0, v1, v2);
                    if dist_sq < best_dist_sq {
                        best_dist_sq = dist_sq;
                    }

                    // Möller–Trumbore ray intersection (ray from p in +X direction)
                    if ray_hits_triangle(p, v0, v1, v2) {
                        intersections += 1;
                    }
                }

                let sign = if intersections % 2 == 1 { -1.0 } else { 1.0 };
                let idx = xi + yi * res + zi * res * res;
                data[idx] = sign * best_dist_sq.sqrt();
            }
        }
    }

    SdfData { data, resolution }
}

/// Möller–Trumbore ray-triangle intersection test.
/// Ray origin = `p`, direction = +X axis `[1, 0, 0]`.
/// Returns true if the ray hits the triangle at t > 0.
fn ray_hits_triangle(p: [f32; 3], v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> bool {
    let e1 = sub3(v1, v0);
    let e2 = sub3(v2, v0);
    // h = cross(dir, e2) where dir = [1, 0, 0]
    let h = [0.0, -e2[2], e2[1]];
    let a = dot3(e1, h);
    if a.abs() < 1e-10 {
        return false;
    }
    let f = 1.0 / a;
    let s = sub3(p, v0);
    let u = f * dot3(s, h);
    if u < 0.0 || u > 1.0 {
        return false;
    }
    let q = cross3(s, e1);
    // v = f * dot(dir, q) where dir = [1, 0, 0]
    let v = f * q[0];
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    // t = f * dot(e2, q)
    let t = f * dot3(e2, q);
    t > 1e-10
}

/// Closest point on triangle (v0, v1, v2) to point p. Returns (closest_point, distance_squared).
fn closest_point_on_triangle(p: [f32; 3], v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> ([f32; 3], f32) {
    let ab = sub3(v1, v0);
    let ac = sub3(v2, v0);
    let ap = sub3(p, v0);

    let d1 = dot3(ab, ap);
    let d2 = dot3(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (v0, dist_sq3(p, v0));
    }

    let bp = sub3(p, v1);
    let d3 = dot3(ab, bp);
    let d4 = dot3(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (v1, dist_sq3(p, v1));
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let pt = [v0[0] + ab[0] * v, v0[1] + ab[1] * v, v0[2] + ab[2] * v];
        return (pt, dist_sq3(p, pt));
    }

    let cp = sub3(p, v2);
    let d5 = dot3(ab, cp);
    let d6 = dot3(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (v2, dist_sq3(p, v2));
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let pt = [v0[0] + ac[0] * w, v0[1] + ac[1] * w, v0[2] + ac[2] * w];
        return (pt, dist_sq3(p, pt));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let pt = [
            v1[0] + (v2[0] - v1[0]) * w,
            v1[1] + (v2[1] - v1[1]) * w,
            v1[2] + (v2[2] - v1[2]) * w,
        ];
        return (pt, dist_sq3(p, pt));
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let pt = [
        v0[0] + ab[0] * v + ac[0] * w,
        v0[1] + ab[1] * v + ac[1] * w,
        v0[2] + ab[2] * v + ac[2] * w,
    ];
    (pt, dist_sq3(p, pt))
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dist_sq3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = sub3(a, b);
    dot3(d, d)
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
