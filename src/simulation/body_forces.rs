//! The fluid's forces on the Dynamic rigid bodies: the accumulator the
//! integrate shader adds into over a frame's substeps, and its readback.
//!
//! The read is the one CPU-GPU sync of a frame with Dynamic bodies, and where
//! it sits in the frame matters. `request` goes right after the substeps are
//! submitted; `collect` waits for them, and for nothing submitted after them.
//!
//! Reading the forces a frame late instead (no wait at all) was tried and is
//! not an option: bodies of low density go unstable on the extra frame of lag
//! (relative density 0.1: 570-880 mm/s rms vertical speed where the same-frame
//! read rests at 2.8).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::state::{GpuRigidBodyAccum, MAX_RIGID_BODIES};

const ACCUM_BYTES: u64 = (std::mem::size_of::<GpuRigidBodyAccum>() * MAX_RIGID_BODIES) as u64;

pub struct BodyForces {
    /// What the integrate shader accumulates into
    accum: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// A map of the staging buffer has been requested and not yet read: the
    /// submission whose copy it waits for, and whether the map has completed
    pending: Option<(wgpu::SubmissionIndex, Arc<AtomicBool>)>,
    latest: [GpuRigidBodyAccum; MAX_RIGID_BODIES],
}

impl BodyForces {
    pub fn new(device: &wgpu::Device) -> Self {
        let accum = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rigid Body Accum Buffer"),
            contents: bytemuck::cast_slice(&[GpuRigidBodyAccum::default(); MAX_RIGID_BODIES]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rigid Body Accum Staging"),
            size: ACCUM_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            accum,
            staging,
            pending: None,
            latest: [GpuRigidBodyAccum::default(); MAX_RIGID_BODIES],
        }
    }

    /// The accumulator, for the integrate shader's bind group
    pub fn accum_buffer(&self) -> &wgpu::Buffer {
        &self.accum
    }

    /// Zero the accumulator (once per frame, before the substeps)
    pub fn clear(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.accum,
            0,
            bytemuck::cast_slice(&[GpuRigidBodyAccum::default(); MAX_RIGID_BODIES]),
        );
    }

    /// End of a substep: copy the running totals to the staging buffer
    pub fn encode_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.accum, 0, &self.staging, 0, ACCUM_BYTES);
    }

    /// Ask for the totals as of `last_step` (the frame's last substep's
    /// submission) without waiting for them
    pub fn request(&mut self, last_step: wgpu::SubmissionIndex) {
        if self.pending.is_some() {
            return;
        }
        let mapped = Arc::new(AtomicBool::new(false));
        let flag = mapped.clone();
        self.staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            flag.store(result.is_ok(), Ordering::Release);
        });
        self.pending = Some((last_step, mapped));
    }

    /// Wait for the requested totals and read them into `latest`. Returns
    /// false if none were requested. Must run before the next substep, which
    /// copies into the staging buffer.
    pub fn collect(&mut self, device: &wgpu::Device) -> bool {
        let Some((last_step, mapped)) = self.pending.take() else {
            return false;
        };
        device
            .poll(wgpu::PollType::Wait { submission_index: Some(last_step), timeout: None })
            .ok();
        if !mapped.load(Ordering::Acquire) {
            // The map resolves on a later poll on some backends
            device.poll(wgpu::PollType::wait_indefinitely()).ok();
        }
        if !mapped.load(Ordering::Acquire) {
            return false;
        }
        {
            let data = self.staging.slice(..).get_mapped_range();
            let accums: &[GpuRigidBodyAccum] = bytemuck::cast_slice(&data);
            self.latest.copy_from_slice(&accums[..MAX_RIGID_BODIES]);
        }
        self.staging.unmap();
        true
    }

    /// The totals last read by `collect`
    pub fn latest(&self) -> &[GpuRigidBodyAccum; MAX_RIGID_BODIES] {
        &self.latest
    }
}
