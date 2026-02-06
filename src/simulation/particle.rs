//! Particle data structure for simulation

/// Extended particle for 3D SPH simulation (64 bytes)
/// Uses double density relaxation (density + near_density)
/// Layout matches WGSL vec3 alignment (16-byte aligned)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SphParticle3D {
    pub position: [f32; 3],     // 12 bytes @ 0
    pub _pad0: f32,             // 4 bytes @ 12 (vec3 padding)
    pub velocity: [f32; 3],     // 12 bytes @ 16
    pub _pad1: f32,             // 4 bytes @ 28 (vec3 padding)
    pub force: [f32; 3],        // 12 bytes @ 32
    pub density: f32,           // 4 bytes @ 44
    pub near_density: f32,      // 4 bytes @ 48
    pub _padding: [f32; 3],     // 12 bytes @ 52 (pad to 64)
}

impl SphParticle3D {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: [x, y, z],
            _pad0: 0.0,
            velocity: [0.0, 0.0, 0.0],
            _pad1: 0.0,
            force: [0.0, 0.0, 0.0],
            density: 0.0,
            near_density: 0.0,
            _padding: [0.0, 0.0, 0.0],
        }
    }
}

impl Default for SphParticle3D {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

/// Create a cube of particles (N×N×N) centered horizontally, starting above center vertically.
pub fn create_particle_block(spacing: f32, cube_size: u32) -> Vec<SphParticle3D> {
    let mut particles = Vec::new();

    let size = (cube_size as f32 - 1.0) * spacing;
    let half = size / 2.0;

    for y in 0..cube_size {
        for z in 0..cube_size {
            for x in 0..cube_size {
                let px = -half + (x as f32) * spacing;
                let py = 0.2 + (y as f32) * spacing; // Start above center
                let pz = -half + (z as f32) * spacing;
                // Small jitter to prevent perfectly aligned particles
                let jitter = 0.0005 * rand_f32();
                particles.push(SphParticle3D::new(px + jitter, py + jitter, pz + jitter));
            }
        }
    }

    particles
}

/// Simple pseudo-random float (not cryptographic, just for jitter)
fn rand_f32() -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    static mut SEED: u64 = 0;
    unsafe {
        SEED = SEED.wrapping_add(1);
        let mut hasher = DefaultHasher::new();
        (SEED, SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()).hash(&mut hasher);
        (hasher.finish() % 1000) as f32 / 1000.0
    }
}
