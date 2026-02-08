// Spray particle emission — dispatched over SPH particles
// For each SPH particle with high acceleration and low density (surface),
// emit spray particles into a ring buffer.

struct SphParticle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    force: vec3<f32>,       // acceleration (force shader outputs accel)
    density: f32,
    near_density: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
};

struct SprayParticle {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    lifetime: f32,
    vel_x: f32,
    vel_y: f32,
    vel_z: f32,
    max_lifetime: f32,
};

struct SprayParams {
    emission_threshold: f32,
    spray_count: u32,
    lifetime: f32,
    lifetime_variation: f32,
    drag: f32,
    speed_multiplier: f32,
    velocity_jitter: f32,
    dt: f32,
    max_particles: u32,
    num_sph_particles: u32,
    frame_count: u32,
    gravity_y: f32,
};

struct SphParams {
    kernel_radius: f32,
    kernel_radius_sq: f32,
    kernel_radius_pow5: f32,
    kernel_radius_pow6: f32,
    kernel_radius_pow9: f32,
    mass: f32,
    rest_density: f32,
    stiffness: f32,
    near_stiffness: f32,
    viscosity: f32,
    dt: f32,
    num_particles: u32,
    surface_tension: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> sph_particles: array<SphParticle>;
@group(0) @binding(1) var<storage, read_write> spray_particles: array<SprayParticle>;
@group(0) @binding(2) var<storage, read_write> write_head: atomic<u32>;
@group(0) @binding(3) var<uniform> params: SprayParams;
@group(0) @binding(4) var<uniform> sph_params: SphParams;

// Simple hash for pseudo-random numbers
fn hash(seed: u32) -> u32 {
    var x = seed;
    x = x ^ (x >> 16u);
    x = x * 0x45d9f3bu;
    x = x ^ (x >> 16u);
    x = x * 0x45d9f3bu;
    x = x ^ (x >> 16u);
    return x;
}

fn hash_float(seed: u32) -> f32 {
    return f32(hash(seed) & 0xFFFFu) / 65535.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.num_sph_particles) {
        return;
    }

    let p = sph_particles[idx];
    let accel = p.force;  // force shader outputs acceleration
    let accel_mag = length(accel);
    let speed = length(p.velocity);

    // Surface detection: density below rest density (surface particles have ~60-90% of rest)
    let is_surface = p.density < sph_params.rest_density * 0.85;

    // Require both: high acceleration AND meaningful velocity (not just static pressure)
    if (!is_surface || accel_mag < params.emission_threshold || speed < 0.5) {
        return;
    }

    // Probabilistic thinning: hash-based chance per particle per frame
    let thin_seed = hash(idx * 17u + params.frame_count * 7919u);
    let thin_chance = f32(thin_seed & 0xFFu) / 255.0;
    // ~50% of qualifying particles actually emit
    if (thin_chance > 0.5) {
        return;
    }

    let emit_count = params.spray_count;
    let base_seed = idx + params.frame_count * 7919u;

    for (var i = 0u; i < emit_count; i = i + 1u) {
        let slot = atomicAdd(&write_head, 1u) % params.max_particles;

        let seed = base_seed + i * 3571u;
        let r0 = hash_float(seed) * 2.0 - 1.0;
        let r1 = hash_float(seed + 1u) * 2.0 - 1.0;
        let r2 = hash_float(seed + 2u) * 2.0 - 1.0;
        let r3 = hash_float(seed + 3u);

        // Velocity: parent velocity scaled + jitter + upward bias
        let jitter = vec3<f32>(r0, r1, r2) * params.velocity_jitter;
        let upward_kick = vec3<f32>(0.0, speed * 0.5, 0.0);
        let vel = p.velocity * params.speed_multiplier + jitter + upward_kick;

        // Lifetime with variation
        let lt = params.lifetime * (1.0 + (r3 * 2.0 - 1.0) * params.lifetime_variation);

        spray_particles[slot] = SprayParticle(
            p.position.x,
            p.position.y,
            p.position.z,
            lt,
            vel.x,
            vel.y,
            vel.z,
            lt,
        );
    }
}
