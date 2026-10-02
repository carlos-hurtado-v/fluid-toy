//! Simulation state files (`.state`): everything the simulation carries from
//! one frame to the next, so a moment can be reloaded exactly — rendered
//! frozen (`--hold`) for frame-exact A/B of render changes, or resumed.
//!
//! Runs are not bit-reproducible (the counting sort's atomics reorder float
//! sums, so trajectories drift from frame 1); a state file sidesteps that by
//! restoring the GPU buffers themselves instead of replaying the simulation.
//!
//! Layout: 8-byte magic, u32 LE header length, JSON `StateHeader`, then the
//! raw blobs in header order. The config (camera, tunables, rigid body poses)
//! is NOT in here: it travels as the matching JSON written next to it.
//!
//! Contents: canonical + grid-sorted SPH particles and the cell table
//! (renderers read the sorted set between steps, and rebuilding it would
//! reorder it), the spray ring buffer + write head, and the foam map's
//! persistent textures, plus the CPU-side clocks and accumulators in the
//! header. Per-frame scratch (PCISPH predicted/pressure, grid build offsets,
//! foam map accumulators) and render temporal history are not saved.

use std::path::Path;

use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 8] = b"FTSTATE\0";
pub const STATE_VERSION: u32 = 1;

/// Foam map scalars that live on the CPU
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default)]
pub struct FoamMapScalars {
    pub flow_time: f32,
    pub was_active: bool,
    pub reset_pending: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BlobEntry {
    pub name: String,
    pub len: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StateHeader {
    pub version: u32,
    pub particle_count: u32,
    /// Spatial grid the sorted particles + cell table belong to (cell size =
    /// the kernel radius the simulation was built with)
    pub grid_cell_size: f32,
    pub grid_total_cells: u32,
    pub spray_capacity: u32,
    pub sim_frame_index: u64,
    pub sim_time: f64,
    /// Visual clock (ripple phase)
    pub time_elapsed: f32,
    /// Spray emission RNG seed
    pub spray_frame_count: u32,
    /// Auto-calibrated emission ceilings (trapped air, wave crest)
    pub spray_auto_limits: [f32; 2],
    pub foam_map: FoamMapScalars,
    pub scenario_fired: Vec<bool>,
    pub scenario_events_fired: u32,
    /// Kinematic spin angle per rigid body (serde-skipped in the config)
    pub spin_angles: Vec<f32>,
    pub blobs: Vec<BlobEntry>,
}

pub struct SimState {
    pub header: StateHeader,
    data: Vec<Vec<u8>>,
}

impl SimState {
    /// `header.blobs` is filled from `blobs`
    pub fn new(mut header: StateHeader, blobs: Vec<(String, Vec<u8>)>) -> Self {
        header.blobs = blobs
            .iter()
            .map(|(name, data)| BlobEntry { name: name.clone(), len: data.len() as u64 })
            .collect();
        Self { header, data: blobs.into_iter().map(|(_, d)| d).collect() }
    }

    pub fn blob(&self, name: &str) -> Result<&[u8], String> {
        self.header
            .blobs
            .iter()
            .position(|b| b.name == name)
            .map(|i| self.data[i].as_slice())
            .ok_or_else(|| format!("state file has no '{name}' data"))
    }

    pub fn total_bytes(&self) -> u64 {
        self.data.iter().map(|d| d.len() as u64).sum()
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        let header = serde_json::to_vec(&self.header)
            .map_err(|e| format!("state header to json failed: {e}"))?;
        let mut out = Vec::with_capacity(12 + header.len() + self.total_bytes() as usize);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(header.len() as u32).to_le_bytes());
        out.extend_from_slice(&header);
        for d in &self.data {
            out.extend_from_slice(d);
        }
        std::fs::write(path, out).map_err(|e| format!("cannot write {}: {e}", path.display()))
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let bad = |what: &str| format!("{} is not a fluid-toy state file ({what})", path.display());
        if bytes.len() < 12 || &bytes[..8] != MAGIC {
            return Err(bad("bad magic"));
        }
        let header_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let header_end = 12 + header_len;
        if bytes.len() < header_end {
            return Err(bad("truncated header"));
        }
        let header: StateHeader = serde_json::from_slice(&bytes[12..header_end])
            .map_err(|e| format!("{}: bad state header: {e}", path.display()))?;
        if header.version != STATE_VERSION {
            return Err(format!(
                "{}: state version {} (this build reads {STATE_VERSION}); re-save it",
                path.display(),
                header.version
            ));
        }
        let mut data = Vec::with_capacity(header.blobs.len());
        let mut at = header_end;
        for b in &header.blobs {
            let end = at + b.len as usize;
            if end > bytes.len() {
                return Err(bad("truncated data"));
            }
            data.push(bytes[at..end].to_vec());
            at = end;
        }
        Ok(Self { header, data })
    }
}

/// A GPU resource to dump: a buffer prefix, or a whole single-mip 2D texture
pub enum GpuSource<'a> {
    Buffer { buffer: &'a wgpu::Buffer, size: u64 },
    Texture { texture: &'a wgpu::Texture, bytes_per_texel: u32 },
}

/// Copy every source to mappable staging in one submit, wait, and return
/// the bytes (texture row padding stripped). Sources need COPY_SRC.
pub fn read_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sources: Vec<(&'static str, GpuSource<'_>)>,
) -> Vec<(String, Vec<u8>)> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("State Readback"),
    });
    // (name, staging, unpadded row bytes, padded row bytes, rows)
    let mut stagings = Vec::with_capacity(sources.len());
    for (name, source) in sources {
        match source {
            GpuSource::Buffer { buffer, size } => {
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("State Readback Staging"),
                    size: size.max(4),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                if size > 0 {
                    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
                }
                stagings.push((name, staging, size as u32, size as u32, 1u32));
            }
            GpuSource::Texture { texture, bytes_per_texel } => {
                let size = texture.size();
                let row = size.width * bytes_per_texel;
                let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                    * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("State Readback Staging"),
                    size: padded as u64 * size.height as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                encoder.copy_texture_to_buffer(
                    texture.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &staging,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(padded),
                            rows_per_image: Some(size.height),
                        },
                    },
                    wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
                );
                stagings.push((name, staging, row, padded, size.height));
            }
        }
    }
    queue.submit(Some(encoder.finish()));
    for (_, staging, ..) in &stagings {
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    }
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    stagings
        .into_iter()
        .map(|(name, staging, row, padded, rows)| {
            let mut bytes = Vec::with_capacity(row as usize * rows as usize);
            {
                let data = staging.slice(..).get_mapped_range();
                for r in 0..rows as usize {
                    let start = r * padded as usize;
                    bytes.extend_from_slice(&data[start..start + row as usize]);
                }
            }
            staging.unmap();
            (name.to_string(), bytes)
        })
        .collect()
}

/// Upload a whole single-mip 2D texture saved by `read_gpu` (needs COPY_DST)
pub fn write_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    bytes_per_texel: u32,
    data: &[u8],
    name: &str,
) -> Result<(), String> {
    let size = texture.size();
    let expected = size.width as usize * size.height as usize * bytes_per_texel as usize;
    if data.len() != expected {
        return Err(format!(
            "state '{name}' holds {} bytes, this build's texture needs {expected}",
            data.len()
        ));
    }
    queue.write_texture(
        texture.as_image_copy(),
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.width * bytes_per_texel),
            rows_per_image: Some(size.height),
        },
        wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
    );
    Ok(())
}
