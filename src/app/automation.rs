//! Automation and capture: `--capture` / `--snapshot` / F12 frame captures,
//! field and probe dumps, `.state` files, `--stats` rows, auto-exit, config
//! export.

use std::io::Write as _;

use super::App;
use crate::state::FluidRenderMode;

/// What this frame has to capture, decided before rendering
pub(super) struct PendingCapture {
    /// A `--capture` frame is due
    frame_capture: bool,
    /// A `--snapshot` frame is due
    scheduled_snapshot: bool,
    /// F12 was pressed
    snapshot: bool,
    /// Swapchain readback buffer + its padded row size, if anything is due
    readback: Option<(wgpu::Buffer, u32)>,
}

impl App {
    /// Decide what is due this frame and prepare the swapchain readback.
    /// Also empties the pixel probe's records ahead of the water pass.
    pub(super) fn begin_capture(&mut self) -> PendingCapture {
        let gpu = self.gpu.as_ref().unwrap();

        // Prepare a swapchain readback if a capture is due this frame
        let mut frame_capture = false;
        while self
            .pending_captures
            .front()
            .is_some_and(|&f| self.milestone_frame >= f)
        {
            self.pending_captures.pop_front();
            frame_capture = true;
        }
        let mut scheduled_snapshot = false;
        while self
            .pending_snapshots
            .front()
            .is_some_and(|&f| self.milestone_frame >= f)
        {
            self.pending_snapshots.pop_front();
            scheduled_snapshot = true;
        }
        let snapshot = std::mem::take(&mut self.snapshot_requested);
        let capture_due = frame_capture || scheduled_snapshot || snapshot;
        // Pixel probe: fresh records for this frame's water pass
        if let Some(mc) = &self.mc_renderer {
            mc.reset_probe(&gpu.queue);
        }
        let capture = if capture_due {
            let unpadded_bytes_per_row = gpu.config.width * 4;
            let padded_bytes_per_row = unpadded_bytes_per_row
                .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Capture Readback"),
                size: padded_bytes_per_row as u64 * gpu.config.height as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            Some((buffer, padded_bytes_per_row))
        } else {
            None
        };
        PendingCapture {
            frame_capture,
            scheduled_snapshot,
            snapshot,
            readback: capture,
        }
    }

    pub(super) fn encode_capture_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        surface_texture: &wgpu::Texture,
        capture: &PendingCapture,
    ) {
        let gpu = self.gpu.as_ref().unwrap();

        // Capture the scene as rendered so far (everything except the GUI)
        if let Some((buffer, padded_bytes_per_row)) = &capture.readback {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: surface_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(*padded_bytes_per_row),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: gpu.config.width,
                    height: gpu.config.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    pub(super) fn finish_capture(&mut self, capture: PendingCapture) {
        // Finish any pending capture (map readback, write PNG)
        if let Some((buffer, padded_bytes_per_row)) = capture.readback {
            if capture.frame_capture {
                let path = self
                    .launch
                    .out_dir
                    .join(format!("frame_{:05}.png", self.milestone_frame));
                self.save_capture(&buffer, padded_bytes_per_row, &path);
                let probe_path = self
                    .launch
                    .out_dir
                    .join(format!("frame_{:05}_probe.json", self.milestone_frame));
                self.save_probe(&probe_path);
                if self.launch.dump_field {
                    self.save_field(&self.launch.out_dir.join(format!("frame_{:05}_field", self.milestone_frame)));
                }
            }
            if capture.snapshot {
                self.save_snapshot(&buffer, padded_bytes_per_row, false);
            }
            if capture.scheduled_snapshot {
                self.save_snapshot(&buffer, padded_bytes_per_row, true);
            }
        }
    }

    pub(super) fn write_stats_row(&mut self, stepped: bool) {
        // Append a stats row for every simulated frame
        if stepped && self.stats_file.is_some() {
            let spray_counts = if self.state.spray.enabled {
                let device = &self.gpu.as_ref().unwrap().device;
                self.spray_system
                    .as_ref()
                    .map_or([0; 4], |s| s.read_stats(device))
            } else {
                [0; 4]
            };
            let mc_vertices = if self.state.rendering.render_mode == FluidRenderMode::MarchingCubes
            {
                self.mc_renderer.as_ref().map_or(0, |mc| mc.vertex_count())
            } else {
                0
            };
            let mut row = format!(
                "{},{:.4},{},{},{},{},{},{},{:.1},{:.4},{:.4}",
                self.sim_frame_index,
                self.sim_time,
                self.state.runtime.particle_count,
                mc_vertices,
                spray_counts[0],
                spray_counts[1],
                spray_counts[2],
                spray_counts[3],
                self.state.runtime.fps,
                self.state.runtime.spray_ta_limit,
                self.state.runtime.spray_wc_limit,
            );
            // Measurement columns (empty cell = no fluid in range)
            let fmt = |v: Option<f32>| v.map_or(String::new(), |v| format!("{v:.5}"));
            let m = self.state.runtime.measurements.as_ref();
            row.push_str(&format!(
                ",{},{},{}",
                fmt(m.and_then(|m| m.max_x)),
                fmt(m.and_then(|m| m.min_x)),
                fmt(m.and_then(|m| m.max_y)),
            ));
            for k in 0..self.state.scenario.probes.len().min(crate::state::MAX_PROBES) {
                let h = m.and_then(|m| m.probe_heights.get(k).copied().flatten());
                row.push_str(&format!(",{}", fmt(h)));
            }
            row.push('\n');
            if let Some(file) = self.stats_file.as_mut() {
                let _ = file.write_all(row.as_bytes());
                let _ = file.flush();
            }
        }
    }

    pub(super) fn check_automation_exit(&mut self) {
        // Automation exit: leave once every requested milestone is reached
        if self.launch.is_automated() && !self.launch.stay && !self.should_exit {
            let captures_done =
                self.pending_captures.is_empty() && self.pending_snapshots.is_empty();
            let exit_frame_reached = self
                .launch
                .exit_after
                .is_none_or(|n| self.milestone_frame >= n);
            let has_milestone = self.had_captures || self.launch.exit_after.is_some();
            if has_milestone && captures_done && exit_frame_reached {
                println!(
                    "Automation milestones reached at frame {} — exiting",
                    self.milestone_frame
                );
                self.should_exit = true;
            }
        }
    }

    /// Map a completed swapchain readback and write it out as a PNG named
    /// after the current simulation frame.
    pub(super) fn save_capture(&self, buffer: &wgpu::Buffer, padded_bytes_per_row: u32, path: &std::path::Path) -> bool {
        let gpu = self.gpu.as_ref().unwrap();
        let (width, height) = (gpu.config.width, gpu.config.height);

        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        {
            let data = slice.get_mapped_range();
            for row in 0..height {
                let start = (row * padded_bytes_per_row) as usize;
                pixels.extend_from_slice(&data[start..start + (width * 4) as usize]);
            }
        }
        buffer.unmap();

        // Swapchain is BGRA on most Windows backends; PNG expects RGBA.
        // Alpha is meaningless post-composite, so force opaque.
        let swap_bgra = matches!(
            gpu.config.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        for px in pixels.chunks_exact_mut(4) {
            if swap_bgra {
                px.swap(0, 2);
            }
            px[3] = 255;
        }

        match image::RgbaImage::from_raw(width, height, pixels) {
            Some(img) => match img.save(path) {
                Ok(()) => {
                    println!("Captured {}", path.display());
                    true
                }
                Err(e) => {
                    eprintln!("error: failed to save capture {}: {e}", path.display());
                    false
                }
            },
            None => {
                eprintln!("error: capture buffer size mismatch");
                false
            }
        }
    }

    /// Snapshot: this frame as a GUI-free PNG, the exact config (live camera
    /// included) as JSON and the simulation state as `.state`, all from the
    /// same instant. F12 writes captures/snapshots/snap_NNN_WxH.*; scheduled
    /// (`--snapshot`) ones go to --out as snap_fNNNNN_WxH.* (milestone frame).
    /// Repro: `--config <json> --load-state <state> --size WxH [--hold]`.
    pub(super) fn save_snapshot(&mut self, buffer: &wgpu::Buffer, padded_bytes_per_row: u32, scheduled: bool) {
        let (width, height) = {
            let gpu = self.gpu.as_ref().unwrap();
            (gpu.config.width, gpu.config.height)
        };
        let (dir, stem) = if scheduled {
            let stem = format!("snap_f{:05}_{width}x{height}", self.milestone_frame);
            (self.launch.out_dir.clone(), stem)
        } else {
            let dir = std::path::PathBuf::from("captures").join("snapshots");
            if let Err(e) = std::fs::create_dir_all(&dir) {
                eprintln!("error: cannot create {}: {e}", dir.display());
                return;
            }
            let mut n = 1u32;
            let stem = loop {
                let stem = format!("snap_{n:03}_{width}x{height}");
                if !dir.join(format!("{stem}.png")).exists() {
                    break stem;
                }
                n += 1;
            };
            (dir, stem)
        };
        let png = dir.join(format!("{stem}.png"));
        let json = dir.join(format!("{stem}.json"));
        let sim_state = dir.join(format!("{stem}.state"));
        if !self.save_capture(buffer, padded_bytes_per_row, &png) {
            return;
        }
        let mut state = self.state.clone();
        state.camera.distance = self.camera.distance;
        state.camera.yaw = self.camera.yaw;
        state.camera.pitch = self.camera.pitch;
        state.camera.target = self.camera.target;
        state.camera.fov = self.camera.fov;
        let message = match std::fs::write(&json, crate::launch::config_to_json(&state))
            .map_err(|e| format!("snapshot config failed: {e}"))
            .and_then(|()| self.save_state(&sim_state))
        {
            Ok(()) => format!(
                "snapshot {} (repro: --config {} --load-state {} --size {width}x{height}, \
                 add --hold to freeze it)",
                png.display(),
                json.display(),
                sim_state.display(),
            ),
            Err(e) => format!("snapshot failed: {e}"),
        };
        println!("{message}");
        self.state.runtime.last_export = Some(message);
    }

    /// Field dump (--dump-field): `<stem>.bin` = the MC density field the mesh
    /// was extracted from (grid_size^3 f32, x fastest), `<stem>.json` = grid +
    /// container parameters; read by scripts/field_profile.py
    pub(super) fn save_field(&self, stem: &std::path::Path) {
        let (Some(gpu), Some(mc)) = (self.gpu.as_ref(), self.mc_renderer.as_ref()) else {
            return;
        };
        let (grid, bytes) = mc.read_field(&gpu.device, &gpu.queue);
        let meta = serde_json::json!({
            "frame": self.milestone_frame,
            "grid_size": grid.grid_size,
            "grid_min": grid.grid_min,
            "cell_size": grid.cell_size,
            "kernel_radius": grid.kernel_radius,
            "iso_value": grid.iso_value,
            "container": &self.state.container,
        });
        let result = std::fs::write(stem.with_extension("bin"), bytes)
            .and_then(|()| std::fs::write(stem.with_extension("json"), meta.to_string()));
        match result {
            Ok(()) => println!("field: {}^3 -> {}.bin", grid.grid_size, stem.display()),
            Err(e) => println!("field dump failed: {e}"),
        }
    }

    /// Pixel probe: this frame's refraction records as JSON (no-op unless
    /// --probe); decoded by scripts/probe_decode.py
    pub(super) fn save_probe(&self, path: &std::path::Path) {
        let (Some(gpu), Some(mc)) = (self.gpu.as_ref(), self.mc_renderer.as_ref()) else {
            return;
        };
        if !mc.probe_enabled() {
            return;
        }
        let dump = mc.read_probe(&gpu.device, &gpu.queue);
        let probed: std::collections::HashSet<[u32; 2]> =
            dump.fragments.iter().map(|f| f.pixel).collect();
        let overflowed = dump.fragments.iter().filter(|f| f.overflow).count();
        let mut value = serde_json::to_value(&dump).unwrap_or_default();
        if let Some(obj) = value.as_object_mut() {
            obj.insert("frame".into(), self.milestone_frame.into());
            obj.insert("size".into(), serde_json::json!([gpu.config.width, gpu.config.height]));
            obj.insert(
                "render_mode".into(),
                format!("{:?}", self.state.rendering.render_mode).into(),
            );
        }
        let text = serde_json::to_string(&value).unwrap_or_default();
        match std::fs::write(path, text) {
            Ok(()) => {
                println!(
                    "probe: {} fragment(s) on {}/{} pixel(s) -> {}",
                    dump.fragments.len(),
                    probed.len(),
                    dump.pixels.len(),
                    path.display()
                );
                if self.state.rendering.render_mode != FluidRenderMode::MarchingCubes {
                    println!("probe: warning: only the MarchingCubes water shader is instrumented");
                }
                if overflowed > 0 || dump.slots_dropped > 0 {
                    println!(
                        "probe: warning: {overflowed} fragment(s) ran out of event slots, {} fragment(s) dropped",
                        dump.slots_dropped
                    );
                }
            }
            Err(e) => eprintln!("error: cannot write {}: {e}", path.display()),
        }
    }

    /// Write the simulation state (GPU buffers + CPU clocks) to a `.state`
    /// file; see simulation/snapshot.rs for what is in it and why.
    pub(super) fn save_state(&self, path: &std::path::Path) -> Result<(), String> {
        use crate::simulation::snapshot::{read_gpu, SimState, StateHeader, STATE_VERSION};
        let gpu = self.gpu.as_ref().ok_or("no GPU")?;
        let sph_sim = self.sph_simulation.as_ref().ok_or("no simulation")?;
        let spray = self.spray_system.as_ref().ok_or("no spray system")?;
        let foam_map = self.foam_map.as_ref().ok_or("no foam map")?;

        let mut sources = sph_sim.snapshot_sources();
        sources.extend(spray.snapshot_sources());
        sources.extend(foam_map.snapshot_sources());
        let blobs = read_gpu(&gpu.device, &gpu.queue, sources);

        let (grid_cell_size, grid_total_cells) = sph_sim.grid_layout();
        let (ta, wc) = spray.auto_limits();
        let header = StateHeader {
            version: STATE_VERSION,
            particle_count: sph_sim.num_particles(),
            grid_cell_size,
            grid_total_cells,
            spray_capacity: spray.capacity(),
            sim_frame_index: self.sim_frame_index,
            sim_time: self.sim_time,
            time_elapsed: self.state.runtime.time_elapsed,
            spray_frame_count: self.state.runtime.frame_count,
            spray_auto_limits: [ta, wc],
            foam_map: foam_map.snapshot_scalars(),
            scenario_fired: self.scenario_fired.clone(),
            scenario_events_fired: self.state.runtime.scenario_events_fired,
            spin_angles: self.state.rigid_bodies.iter().map(|b| b.spin_angle).collect(),
            blobs: Vec::new(),
        };
        SimState::new(header, blobs).write(path)
    }

    /// Restore a `.state` file over the freshly initialized simulation. The
    /// config (camera, tunables, rigid body poses) comes from the snapshot's
    /// JSON via --config; this restores what the config cannot express.
    pub(super) fn load_state(&mut self, path: &std::path::Path) -> Result<(), String> {
        let state = crate::simulation::snapshot::SimState::read(path)?;
        let h = &state.header;
        let substep_dt = self.simulation_substep_dt();
        let gpu = self.gpu.as_ref().ok_or("no GPU")?;
        let sph_sim = self.sph_simulation.as_mut().ok_or("no simulation")?;
        let spray = self.spray_system.as_mut().ok_or("no spray system")?;
        let foam_map = self.foam_map.as_mut().ok_or("no foam map")?;

        // Layout checks first: a mismatched grid or ring buffer would load as
        // garbage, not fail
        let (cell_size, total_cells) = sph_sim.grid_layout();
        if h.grid_cell_size != cell_size || h.grid_total_cells != total_cells {
            return Err(format!(
                "saved on a kernel_radius {} grid ({} cells), this config builds {} ({} cells); \
                 load it with the snapshot's --config",
                h.grid_cell_size, h.grid_total_cells, cell_size, total_cells
            ));
        }
        if h.spray_capacity != spray.capacity() {
            return Err(format!(
                "saved with spray.max_particles {}, this config has {}",
                h.spray_capacity,
                spray.capacity()
            ));
        }
        sph_sim.restore_snapshot(&gpu.queue, &state)?;
        spray.restore_snapshot(&gpu.queue, &state)?;
        foam_map.restore_snapshot(&gpu.queue, &state)?;

        self.state.runtime.particle_count = h.particle_count;
        let sph_params = self.state.sph.to_gpu_params_3d(h.particle_count, substep_dt);
        sph_sim.update_sph_params(&gpu.queue, &sph_params);

        self.sim_frame_index = h.sim_frame_index;
        self.sim_time = h.sim_time;
        self.state.runtime.time_elapsed = h.time_elapsed;
        self.state.runtime.frame_count = h.spray_frame_count;
        // Event flags realign to the config's list in pump_scenario_events
        self.scenario_fired = h.scenario_fired.clone();
        self.state.runtime.scenario_events_fired = h.scenario_events_fired;
        if h.spin_angles.len() != self.state.rigid_bodies.len() {
            eprintln!(
                "warning: state has {} rigid bodies, config has {}; spin angles matched by index",
                h.spin_angles.len(),
                self.state.rigid_bodies.len()
            );
        }
        for (body, &angle) in self.state.rigid_bodies.iter_mut().zip(&h.spin_angles) {
            body.spin_angle = angle;
        }
        // The loaded state is already whitewater-live: no re-enable reset
        self.spray_prev_enabled = self.state.spray.enabled;

        println!(
            "loaded state {} (sim frame {}, t={:.3}s, {} particles){}",
            path.display(),
            h.sim_frame_index,
            h.sim_time,
            h.particle_count,
            if self.launch.hold { ", held" } else { "" },
        );
        Ok(())
    }

    /// Write the current state (including the live camera pose) to
    /// configs/export_NNN.json.
    pub(super) fn export_config(&mut self) {
        // Sync the live camera back into the serializable config
        self.state.camera.distance = self.camera.distance;
        self.state.camera.yaw = self.camera.yaw;
        self.state.camera.pitch = self.camera.pitch;
        self.state.camera.target = self.camera.target;
        self.state.camera.fov = self.camera.fov;

        match crate::launch::export_config(&self.state) {
            Ok(path) => {
                let display = path.display().to_string();
                println!("Exported config to {display}");
                self.state.runtime.last_export = Some(display);
            }
            Err(e) => {
                eprintln!("error: config export failed: {e}");
                self.state.runtime.last_export = Some(format!("export failed: {e}"));
            }
        }
    }
}
