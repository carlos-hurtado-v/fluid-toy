//! Per-phase CPU and GPU timings of a run (`--profile <file.json>`).
//!
//! Wall-clock frame time says how slow a configuration is, not why: a blocking
//! readback, a CPU-bound encode and a heavy pass all look the same from
//! outside. With `--profile`, the frame phases call `cpu(label)` and
//! `gpu(encoder, label)` at their boundaries; the averages over the run are
//! printed at exit and written as JSON.
//!
//! - `cpu(label)`: the CPU time since the previous CPU mark belongs to
//!   `label` (so the labels of one frame add up to the frame's wall time).
//! - `gpu(encoder, label)`: writes a GPU timestamp; the GPU time between the
//!   previous mark and this one belongs to `label`. The first mark of a frame
//!   closes the interval that began at the previous frame's last mark: time
//!   the GPU spent on nothing this profiler marks, i.e. mostly idle, waiting
//!   for the CPU. `gpu_submit` does the same from a one-command encoder, for
//!   phases that submit their own command buffers (the SPH substeps).
//!
//! The profiler lives in a thread-local, so passes mark themselves without it
//! being threaded through every signature; without `--profile` each call is a
//! `None` check. GPU marks need the TIMESTAMP_QUERY and
//! TIMESTAMP_QUERY_INSIDE_ENCODERS features (requested only with `--profile`);
//! without them the report holds the CPU side alone. Readback is asynchronous
//! (a ring of staging buffers), so profiling adds no CPU-GPU sync of its own;
//! a frame that finds no free buffer goes unmeasured.

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// GPU marks one frame may write
const MAX_MARKS: u32 = 64;
/// Frames whose timestamps may be in flight at once
const RING: usize = 6;
/// Frames left out of the averages: pipelines compile and temporal history
/// fills during the first ones
const WARMUP_FRAMES: u64 = 20;

const SLOT_FREE: u8 = 0;
const SLOT_PENDING: u8 = 1;
const SLOT_READY: u8 = 2;

struct Slot {
    staging: wgpu::Buffer,
    state: Arc<AtomicU8>,
    labels: Vec<&'static str>,
    frame: u64,
}

/// Running total of one label, in first-seen order
struct Total {
    label: &'static str,
    ms: f64,
}

fn add(totals: &mut Vec<Total>, label: &'static str, ms: f64) {
    match totals.iter_mut().find(|t| t.label == label) {
        Some(t) => t.ms += ms,
        None => totals.push(Total { label, ms }),
    }
}

struct Profiler {
    path: PathBuf,
    query_set: Option<wgpu::QuerySet>,
    resolve: Option<wgpu::Buffer>,
    slots: Vec<Slot>,
    period_ns: f64,
    /// Frames begun, and the slot this frame's marks go to (None: unmeasured)
    frame: u64,
    slot: Option<usize>,
    labels: Vec<&'static str>,
    /// Last mark of the latest frame read back: (frame, ticks)
    last_end: Option<(u64, u64)>,
    cpu_last: Instant,
    started: Option<Instant>,
    cpu: Vec<Total>,
    gpu: Vec<Total>,
    gpu_frames: u64,
}

thread_local! {
    static PROFILER: RefCell<Option<Profiler>> = const { RefCell::new(None) };
}

/// Features `--profile` asks the device for, where the adapter has them
pub fn wanted_features() -> wgpu::Features {
    wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
}

/// Start profiling this run; the report goes to `path` at `report()`.
pub fn init(device: &wgpu::Device, queue: &wgpu::Queue, path: PathBuf) {
    let gpu_marks = device.features().contains(wanted_features());
    if !gpu_marks {
        eprintln!("--profile: no GPU timestamp queries on this adapter, CPU phases only");
    }
    let bytes = u64::from(MAX_MARKS) * 8;
    let (query_set, resolve, slots) = if gpu_marks {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Profile Timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: MAX_MARKS,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Profile Resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = (0..RING)
            .map(|_| Slot {
                staging: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Profile Staging"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(SLOT_FREE)),
                labels: Vec::new(),
                frame: 0,
            })
            .collect();
        (Some(query_set), Some(resolve), slots)
    } else {
        (None, None, Vec::new())
    };
    PROFILER.with_borrow_mut(|p| {
        *p = Some(Profiler {
            path,
            query_set,
            resolve,
            slots,
            period_ns: f64::from(queue.get_timestamp_period()),
            frame: 0,
            slot: None,
            labels: Vec::new(),
            last_end: None,
            cpu_last: Instant::now(),
            started: None,
            cpu: Vec::new(),
            gpu: Vec::new(),
            gpu_frames: 0,
        });
    });
}

/// Start of a frame: collect finished timestamp readbacks and pick this
/// frame's slot. The CPU time since the last mark of the previous frame
/// (event loop, the wait for the next redraw) is booked as "between frames".
pub fn begin_frame(device: &wgpu::Device) {
    PROFILER.with_borrow_mut(|p| {
        let Some(p) = p.as_mut() else { return };
        p.mark_cpu("between frames");
        p.frame += 1;
        if p.frame == WARMUP_FRAMES + 1 {
            // The averages start here
            p.cpu.clear();
            p.started = Some(Instant::now());
        }
        let _ = device.poll(wgpu::PollType::Poll);
        p.collect();
        p.slot = p.slots.iter().position(|s| s.state.load(Ordering::Acquire) == SLOT_FREE);
        p.labels.clear();
    });
}

/// The CPU time since the previous CPU mark belongs to `label`
pub fn cpu(label: &'static str) {
    PROFILER.with_borrow_mut(|p| {
        if let Some(p) = p.as_mut() {
            p.mark_cpu(label);
        }
    });
}

/// The GPU time between the previous GPU mark and this point of `encoder`
/// belongs to `label`
pub fn gpu(encoder: &mut wgpu::CommandEncoder, label: &'static str) {
    PROFILER.with_borrow_mut(|p| {
        if let Some(p) = p.as_mut() {
            p.mark_gpu(encoder, label);
        }
    });
}

/// `gpu` for work submitted outside the frame's encoder: the mark is
/// submitted now, in its own command buffer
pub fn gpu_submit(device: &wgpu::Device, queue: &wgpu::Queue, label: &'static str) {
    PROFILER.with_borrow_mut(|p| {
        let Some(p) = p.as_mut() else { return };
        if p.slot.is_none() || p.query_set.is_none() {
            return;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Profile Mark"),
        });
        p.mark_gpu(&mut encoder, label);
        queue.submit(std::iter::once(encoder.finish()));
    });
}

/// End of the frame's encoder: resolve this frame's marks for readback
pub fn end_frame(encoder: &mut wgpu::CommandEncoder) {
    PROFILER.with_borrow_mut(|p| {
        let Some(p) = p.as_mut() else { return };
        let (Some(slot), Some(query_set), Some(resolve)) = (p.slot, &p.query_set, &p.resolve) else {
            return;
        };
        let count = p.labels.len() as u32;
        if count == 0 {
            p.slot = None;
            return;
        }
        encoder.resolve_query_set(query_set, 0..count, resolve, 0);
        encoder.copy_buffer_to_buffer(resolve, 0, &p.slots[slot].staging, 0, u64::from(count) * 8);
    });
}

/// After the frame's submit: ask for this frame's timestamps
pub fn after_submit() {
    PROFILER.with_borrow_mut(|p| {
        let Some(p) = p.as_mut() else { return };
        let Some(index) = p.slot.take() else { return };
        let slot = &mut p.slots[index];
        slot.labels.clone_from(&p.labels);
        slot.frame = p.frame;
        slot.state.store(SLOT_PENDING, Ordering::Release);
        let state = slot.state.clone();
        let bytes = slot.labels.len() as u64 * 8;
        slot.staging.slice(..bytes).map_async(wgpu::MapMode::Read, move |result| {
            state.store(if result.is_ok() { SLOT_READY } else { SLOT_FREE }, Ordering::Release);
        });
    });
}

/// Print the averages and write them to the `--profile` file
pub fn report(device: &wgpu::Device) {
    PROFILER.with_borrow_mut(|p| {
        let Some(p) = p.as_mut() else { return };
        p.mark_cpu("between frames");
        // Whatever is still in flight
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        p.collect();
        p.write_report();
    });
}

impl Profiler {
    fn mark_cpu(&mut self, label: &'static str) {
        let now = Instant::now();
        add(&mut self.cpu, label, now.duration_since(self.cpu_last).as_secs_f64() * 1000.0);
        self.cpu_last = now;
    }

    fn mark_gpu(&mut self, encoder: &mut wgpu::CommandEncoder, label: &'static str) {
        let (Some(_), Some(query_set)) = (self.slot, &self.query_set) else {
            return;
        };
        let index = self.labels.len() as u32;
        if index >= MAX_MARKS {
            return;
        }
        encoder.write_timestamp(query_set, index);
        self.labels.push(label);
    }

    /// Fold every finished readback into the totals, oldest frame first
    fn collect(&mut self) {
        let mut ready: Vec<usize> = (0..self.slots.len())
            .filter(|&i| self.slots[i].state.load(Ordering::Acquire) == SLOT_READY)
            .collect();
        ready.sort_by_key(|&i| self.slots[i].frame);
        for i in ready {
            let slot = &self.slots[i];
            let bytes = slot.labels.len() as u64 * 8;
            let ticks: Vec<u64> = {
                let data = slot.staging.slice(..bytes).get_mapped_range();
                data.chunks_exact(8).map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect()
            };
            slot.staging.unmap();
            slot.state.store(SLOT_FREE, Ordering::Release);
            let measured = slot.frame > WARMUP_FRAMES;
            let to_ms = self.period_ns / 1.0e6;
            for (k, &label) in slot.labels.iter().enumerate() {
                if !measured {
                    break;
                }
                let from = if k > 0 {
                    Some(ticks[k - 1])
                } else {
                    // Only across directly consecutive frames
                    self.last_end.filter(|&(frame, _)| frame + 1 == slot.frame).map(|(_, t)| t)
                };
                if let Some(from) = from {
                    add(&mut self.gpu, label, ticks[k].saturating_sub(from) as f64 * to_ms);
                }
            }
            if let Some(&end) = ticks.last() {
                self.last_end = Some((slot.frame, end));
            }
            self.gpu_frames += u64::from(measured);
        }
    }

    fn write_report(&self) {
        let measured = self.frame.saturating_sub(WARMUP_FRAMES);
        if measured == 0 {
            eprintln!("--profile: the run ended within the {WARMUP_FRAMES} warm-up frames, nothing measured");
            return;
        }
        let frames = measured as f64;
        let wall = self.started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0) / frames;
        println!("profile: {measured} frames, {wall:.2} ms/frame wall ({:.1} fps)", 1000.0 / wall.max(1e-6));
        println!("  CPU phase                    ms/frame");
        let mut cpu_sum = 0.0;
        for t in &self.cpu {
            println!("    {:<26} {:8.3}", t.label, t.ms / frames);
            cpu_sum += t.ms / frames;
        }
        println!("    {:<26} {:8.3}", "(sum)", cpu_sum);
        let gpu_frames = self.gpu_frames.max(1) as f64;
        let mut gpu_sum = 0.0;
        if !self.gpu.is_empty() {
            println!("  GPU interval                 ms/frame   ({} frames measured)", self.gpu_frames);
            for t in &self.gpu {
                println!("    {:<26} {:8.3}", t.label, t.ms / gpu_frames);
                gpu_sum += t.ms / gpu_frames;
            }
            println!("    {:<26} {:8.3}", "(sum)", gpu_sum);
        }
        let rows = |totals: &[Total], n: f64| {
            totals
                .iter()
                .map(|t| serde_json::json!({ "label": t.label, "ms": t.ms / n }))
                .collect::<Vec<_>>()
        };
        let json = serde_json::json!({
            "frames": measured,
            "wall_ms": wall,
            "cpu": rows(&self.cpu, frames),
            "gpu_frames": self.gpu_frames,
            "gpu": rows(&self.gpu, gpu_frames),
        });
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match std::fs::write(&self.path, serde_json::to_string_pretty(&json).unwrap_or_default()) {
            Ok(()) => println!("profile written to {}", self.path.display()),
            Err(e) => eprintln!("--profile: could not write {}: {e}", self.path.display()),
        }
    }
}
