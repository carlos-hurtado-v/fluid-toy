//! Pixel probe (`--probe`): the record buffer the probe variant of the
//! water shader (`mc_probe_on.wgsl`) writes its refraction events into, and
//! its readback. Decoded by scripts/probe_decode.py.

/// Pixel probe (--probe) record layout: mirrors ProbeBuffer in
/// mc_probe_on.wgsl (header, PROBE_MAX_PIXELS pixel slots, then fragment
/// slots of 2 + PROBE_EVENT_VEC4S * PROBE_EVENTS_PER_SLOT vec4s)
pub const PROBE_MAX_PIXELS: usize = 64;
const PROBE_EVENTS_PER_SLOT: u32 = 256;
/// vec4s per event: (a.xyz, tag), (b.xyz, c), d
const PROBE_EVENT_VEC4S: u64 = 3;
/// Fragment slots: every fragment shading a probed pixel takes one (occluded
/// ones and MSAA edge triangles included)
const PROBE_MAX_SLOTS: u64 = 4 * PROBE_MAX_PIXELS as u64;
const PROBE_HEADER_BYTES: u64 = 16 + 16 * PROBE_MAX_PIXELS as u64;

fn create_probe_buffer(device: &wgpu::Device, slots: u64) -> wgpu::Buffer {
    let slot_bytes = 16 * (2 + PROBE_EVENT_VEC4S * PROBE_EVENTS_PER_SLOT as u64);
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MC Probe Records"),
        size: PROBE_HEADER_BYTES + slots * slot_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// One fragment that shaded a probed pixel: its recorded refraction events,
/// each [tag, a.x, a.y, a.z, b.x, b.y, b.z, c, d.x, d.y, d.z, d.w]
/// (scripts/probe_decode.py names
/// them; tags are the PRB_* constants in mc_render/debug_records.wgsl)
#[derive(serde::Serialize)]
pub struct ProbeFragment {
    pub pixel: [u32; 2],
    pub frag_xy: [f32; 2],
    pub depth: f32,
    pub overflow: bool,
    pub events: Vec<[f32; 12]>,
}

#[derive(serde::Serialize)]
pub struct ProbeDump {
    pub pixels: Vec<[u32; 2]>,
    pub events_per_slot: u32,
    pub slots_dropped: u32,
    pub fragments: Vec<ProbeFragment>,
}

pub struct PixelProbe {
    /// Record buffer (a one-slot placeholder until enabled: the normal
    /// shader never touches it, but the bind group needs a buffer)
    buffer: wgpu::Buffer,
    /// Probed pixels (capture-PNG coordinates)
    pixels: Vec<[u32; 2]>,
    enabled: bool,
}

impl PixelProbe {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            buffer: create_probe_buffer(device, 1),
            pixels: Vec::new(),
            enabled: false,
        }
    }

    /// Record at these pixels (at most PROBE_MAX_PIXELS). Recreates the
    /// buffer: bind groups over the old one must be rebuilt.
    pub fn enable(&mut self, device: &wgpu::Device, pixels: &[[u32; 2]]) {
        self.pixels = pixels.iter().copied().take(PROBE_MAX_PIXELS).collect();
        self.buffer = create_probe_buffer(device, PROBE_MAX_SLOTS);
        self.enabled = true;
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Empty the records before this frame's water pass (queue writes land
    /// ahead of the next submit)
    pub fn reset(&self, queue: &wgpu::Queue) {
        if !self.enabled {
            return;
        }
        let mut header = vec![0u32; PROBE_HEADER_BYTES as usize / 4];
        header[0] = self.pixels.len() as u32;
        header[2] = PROBE_EVENTS_PER_SLOT;
        for (i, px) in self.pixels.iter().enumerate() {
            header[4 + 4 * i] = px[0];
            header[4 + 4 * i + 1] = px[1];
        }
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&header));
    }

    /// Read the last frame's records (blocking)
    pub fn read(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> ProbeDump {
        use crate::simulation::snapshot::{read_gpu, GpuSource};
        let size = self.buffer.size();
        let bytes = read_gpu(device, queue, vec![("probe", GpuSource::Buffer { buffer: &self.buffer, size })])
            .remove(0)
            .1;
        let words: &[u32] = bytemuck::cast_slice(&bytes);
        let floats: &[f32] = bytemuck::cast_slice(&bytes);
        let slots_used = words[1] as usize;
        let events_per_slot = words[2] as usize;
        let stride = 4 * (2 + PROBE_EVENT_VEC4S as usize * events_per_slot);
        let data = &floats[PROBE_HEADER_BYTES as usize / 4..];
        let capacity = data.len() / stride;
        let mut fragments = Vec::new();
        for slot in 0..slots_used.min(capacity) {
            let d = &data[slot * stride..(slot + 1) * stride];
            let count = (d[4] as usize).min(events_per_slot);
            let events = (0..count)
                .map(|e| {
                    let v = &d[8 + 12 * e..20 + 12 * e];
                    // [tag, a.xyz, b.xyz, c, d]
                    [v[3], v[0], v[1], v[2], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]]
                })
                .collect();
            fragments.push(ProbeFragment {
                pixel: self.pixels.get(d[0] as usize).copied().unwrap_or([0, 0]),
                frag_xy: [d[1], d[2]],
                depth: d[3],
                overflow: d[5] > 0.5,
                events,
            });
        }
        ProbeDump {
            pixels: self.pixels.clone(),
            events_per_slot: events_per_slot as u32,
            slots_dropped: slots_used.saturating_sub(capacity) as u32,
            fragments,
        }
    }
}
