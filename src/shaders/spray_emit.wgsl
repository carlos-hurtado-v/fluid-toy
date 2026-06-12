// Whitewater emission — Ihmsen et al. 2012 "Unified Spray, Foam and Bubbles
// for Particle-Based Fluids", with the operational changes from SPlisHSPlasH's
// FoamGenerator (Bender et al. 2019): volume-weighted potentials, emitter
// embeddedness gate, auto-calibrated clamp limits, energy-scaled lifetimes.
//
// Dispatched over the grid-sorted SPH particles (positions consistent with the
// spatial grid built at the start of the last substep). For each fast-moving
// fluid particle, two potentials are gathered from grid neighbors:
//   - trapped air:  converging relative velocities (air entrainment at impacts)
//   - wave crest:   normal divergence on convex, outward-moving surface particles
// Neighbor terms are volume-weighted (V_j * W_poly6) so the sums are proper
// SPH field estimates rather than raw neighbor-count sums. The clamp ceilings
// arrive per frame in the uniform (asymmetric EMA of per-frame maxima, run on
// the CPU from this shader's atomicMax stats); the floor is a fixed fraction
// of the ceiling. Both potentials are remapped to [0,1], weighted, scaled by
// the kinetic speed gate, and converted to an emission rate (particles/s).
// Spawned particles go into a ring buffer; classification into
// spray/foam/bubble happens in spray_simulate the same frame.

struct SphParticle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    force: vec3<f32>,
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
    kind: u32,
    age: f32,
    _pad1: f32,
    _pad2: f32,
};

struct SprayParams {
    min_speed: f32,
    emission_rate: f32,
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
    k_trapped_air: f32,
    k_wave_crest: f32,
    ta_limit: f32,
    bubble_buoyancy: f32,
    bubble_drag: f32,
    wc_limit: f32,
    _pad0: f32,
    _pad1: f32,
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
    pcisph_delta: f32,
    xsph_epsilon: f32,
    _pad_st2: f32,
};

struct GridParams {
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,
    total_cells: u32,
    cell_size: f32,
    inv_cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    num_particles: u32,
    _padding: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> sorted_particles: array<SphParticle>;
@group(0) @binding(1) var<storage, read_write> spray_particles: array<SprayParticle>;
@group(0) @binding(2) var<storage, read_write> write_head: atomic<u32>;
@group(0) @binding(3) var<uniform> params: SprayParams;
@group(0) @binding(4) var<uniform> sph_params: SphParams;
@group(0) @binding(5) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(6) var<storage, read> cell_counts: array<u32>;
@group(0) @binding(7) var<uniform> grid: GridParams;
// Per-frame maxima of the raw potentials, zeroed by the CPU each frame:
// [trapped_air_bits, wave_crest_bits, reserved, reserved]. Non-negative f32
// bit patterns order like u32, so atomicMax on the bitcast is a float max.
@group(0) @binding(8) var<storage, read_write> emit_stats: array<atomic<u32>, 4>;

const TWO_PI: f32 = 6.28318530718;
const PI: f32 = 3.14159265359;

// Potential clamp floor as a fraction of the auto-calibrated ceiling
// (SPlisHSPlasH uses [0.1*avgMax, avgMax])
const AUTO_LIMIT_MIN_FRAC: f32 = 0.1;
// Embeddedness gate: only particles with a healthy fluid neighborhood emit.
// Kills emission from dispersed droplets and thin tendrils, where normals and
// potentials are noise. SPlisHSPlasH uses 15, but their offline sims run
// millions of particles and keep splash sheets dense; at our counts the crown
// curtain lives in the 8-15 neighbor band (15 cut peak spray 8x in eval), so
// the gate sits below it.
const EMIT_MIN_NEIGHBORS: u32 = 10u;
// Kinetic gate saturation speed. The gate remaps SPEED linearly (not energy):
// an energy remap is quadratic in v and crushes mid-energy events — mouse
// churn at 1-2 m/s scored 3-14% of full rate while the dam-break impact
// (3.5+ m/s) saturated, which is why interactive forces barely emitted.
// Speed-linear with a 2.5 m/s ceiling gives churn 12-71% and leaves
// impact-scale events saturated exactly as before.
const KINETIC_MAX_SPEED: f32 = 2.5;
// Surface gate for the crest potential (interior particles have ~rest density)
const CREST_DENSITY_FRAC: f32 = 0.9;
// Crest counts only when moving with the surface normal (Ihmsen delta_vn)
const CREST_VEL_NORMAL_MIN: f32 = 0.6;
// Crest stretching gate, remapped to [0,1]. The crest term is otherwise pure
// geometry x absolute speed, so a rigidly translating surface (the initial
// block in freefall) emits from its whole bottom face. Real crest spray forms
// where the sheet STRETCHES (positive velocity divergence, droplets pinching
// off); the falling block's surface-tension clumping is contraction (negative
// divergence), so the sign separates them robustly — magnitude alone does not
// (coherent crown sheets have small relative speeds too). Trapped air needs
// no gate: it is built from relative velocities, naturally zero in freefall.
// Lower edge at exactly 0: the sign alone separates the cases (freefall
// clumping is strictly negative), and crown sheets stretch weakly — surface
// tension pulls rims back even as they spread — so any positive stretching
// should count and saturate fast.
const CREST_DIVERGENCE_MIN: f32 = 0.0;
const CREST_DIVERGENCE_MAX: f32 = 0.5;
// Hard cap on spawns per fluid particle per frame
const MAX_EMIT_PER_FRAME: u32 = 16u;

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

fn position_to_cell(pos: vec3<f32>) -> vec3<i32> {
    let local_pos = pos - vec3<f32>(grid.grid_origin_x, grid.grid_origin_y, grid.grid_origin_z);
    return vec3<i32>(floor(local_pos * grid.inv_cell_size));
}

fn cell_to_index(cell: vec3<i32>) -> u32 {
    return u32(cell.x) + u32(cell.y) * grid.grid_size_x + u32(cell.z) * grid.grid_size_x * grid.grid_size_y;
}

fn is_valid_cell(cell: vec3<i32>) -> bool {
    return cell.x >= 0i && cell.x < i32(grid.grid_size_x) &&
           cell.y >= 0i && cell.y < i32(grid.grid_size_y) &&
           cell.z >= 0i && cell.z < i32(grid.grid_size_z);
}

fn remap_clamp(x: f32, lo: f32, hi: f32) -> f32 {
    return clamp((x - lo) / (hi - lo), 0.0, 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.num_sph_particles) {
        return;
    }

    let p = sorted_particles[i];
    let vel = p.velocity;
    let speed = length(vel);

    // Kinetic gate doubles as an early-out: calm fluid does no neighbor work.
    if (speed < params.min_speed) {
        return;
    }

    let h = sph_params.kernel_radius;
    let h_sq = sph_params.kernel_radius_sq;

    let normal_i = vec3<f32>(p.normal_x, p.normal_y, p.normal_z);
    let n_len_i = length(normal_i);
    // Crest needs a usable surface normal AND a surface-ish density
    let crest_eligible = n_len_i > 1e-4 && p.density < sph_params.rest_density * CREST_DENSITY_FRAC;
    var n_hat_i = vec3<f32>(0.0, 1.0, 0.0);
    if (crest_eligible) {
        n_hat_i = normal_i / n_len_i;
    }

    var trapped_air = 0.0;
    var crest = 0.0;
    var divergence = 0.0;
    var n_neighbors = 0u;

    // Volume-weighted poly6 normalization (315 / (64 pi h^9))
    let poly6_norm = 315.0 / (64.0 * PI * sph_params.kernel_radius_pow9);

    let cell_i = position_to_cell(p.position);
    for (var dz = -1i; dz <= 1i; dz++) {
        for (var dy = -1i; dy <= 1i; dy++) {
            for (var dx = -1i; dx <= 1i; dx++) {
                let neighbor_cell = cell_i + vec3<i32>(dx, dy, dz);
                if (!is_valid_cell(neighbor_cell)) {
                    continue;
                }
                let cell_idx = cell_to_index(neighbor_cell);
                let count = cell_counts[cell_idx];
                let end = cell_starts[cell_idx];
                let start = end - count;

                for (var k = 0u; k < count; k++) {
                    let pj = sorted_particles[start + k];
                    let r_vec = p.position - pj.position;
                    let r_sq = dot(r_vec, r_vec);
                    if (r_sq >= h_sq || r_sq < 1e-12) {
                        continue;
                    }
                    let r = sqrt(r_sq);
                    let w = 1.0 - r / h;
                    let x_hat = r_vec / r;
                    n_neighbors += 1u;

                    // SPH volume weight: V_j * W_poly6 (dimensionless). Makes
                    // the potentials field estimates — dense clumps no longer
                    // outweigh sparse regions just by neighbor count.
                    let d2 = h_sq - r_sq;
                    let w_vol = (sph_params.mass / max(pj.density, 1e-6))
                        * poly6_norm * d2 * d2 * d2;

                    // Trapped air: relative velocity magnitude, weighted by how
                    // head-on the approach is (1 - cos > 1 when converging)
                    let v_ij = vel - pj.velocity;
                    let v_ij_len = length(v_ij);
                    if (v_ij_len > 1e-6) {
                        trapped_air += v_ij_len * (1.0 - dot(v_ij / v_ij_len, x_hat)) * w_vol;
                    }

                    // Local stretching for the crest gate (> 0 when separating).
                    // Keeps the hat weight: its 0-0.5 remap below is tuned to
                    // that scale, and it is a gate, not an Ihmsen potential.
                    divergence += dot(v_ij, x_hat) * w;

                    // Wave crest: normal divergence, counted only for neighbors
                    // on the inside of the curve (convex check, Ihmsen eq. 7)
                    if (crest_eligible && dot(-x_hat, n_hat_i) < 0.0) {
                        let normal_j = vec3<f32>(pj.normal_x, pj.normal_y, pj.normal_z);
                        let n_len_j = length(normal_j);
                        if (n_len_j > 1e-4) {
                            crest += (1.0 - dot(n_hat_i, normal_j / n_len_j)) * w_vol;
                        }
                    }
                }
            }
        }
    }

    // Embeddedness gate: dispersed droplets and thin tendrils never emit
    if (n_neighbors < EMIT_MIN_NEIGHBORS) {
        return;
    }

    // Crest only counts when the particle moves along its surface normal
    if (dot(vel / speed, n_hat_i) < CREST_VEL_NORMAL_MIN) {
        crest = 0.0;
    }

    // Report per-frame maxima for the CPU-side limit calibration. Gated
    // particles only — this is exactly the population the remap below sees.
    if (trapped_air > 0.0) {
        atomicMax(&emit_stats[0], bitcast<u32>(trapped_air));
    }
    if (crest > 0.0) {
        atomicMax(&emit_stats[1], bitcast<u32>(crest));
    }

    let phi_div = remap_clamp(divergence, CREST_DIVERGENCE_MIN, CREST_DIVERGENCE_MAX);

    let ta_hi = max(params.ta_limit, 1e-4);
    let wc_hi = max(params.wc_limit, 1e-4);
    let phi_ta = remap_clamp(trapped_air, AUTO_LIMIT_MIN_FRAC * ta_hi, ta_hi);
    let phi_wc = remap_clamp(crest, AUTO_LIMIT_MIN_FRAC * wc_hi, wc_hi) * phi_div;
    let phi_k = remap_clamp(speed, params.min_speed, KINETIC_MAX_SPEED);

    // Particles per second, converted to a per-frame count with a
    // probabilistic fractional remainder so low rates still emit.
    let rate = phi_k * (params.k_trapped_air * phi_ta + params.k_wave_crest * phi_wc) * params.emission_rate;
    let expected = rate * params.dt;
    let base_seed = i + params.frame_count * 7919u;
    var n_emit = u32(floor(expected));
    if (hash_float(base_seed * 13u + 5u) < fract(expected)) {
        n_emit = n_emit + 1u;
    }
    n_emit = min(n_emit, MAX_EMIT_PER_FRAME);
    if (n_emit == 0u) {
        return;
    }

    // Spawn inside a cylinder around the particle, oriented along velocity
    let v_dir = vel / speed;
    var ref_axis = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(v_dir.y) > 0.99) {
        ref_axis = vec3<f32>(1.0, 0.0, 0.0);
    }
    let e1 = normalize(cross(v_dir, ref_axis));
    let e2 = cross(v_dir, e1);

    for (var e = 0u; e < n_emit; e = e + 1u) {
        let slot = atomicAdd(&write_head, 1u) % params.max_particles;

        let seed = base_seed + e * 3571u;
        let r0 = hash_float(seed);
        let r1 = hash_float(seed + 1u);
        let r2 = hash_float(seed + 2u);
        let r3 = hash_float(seed + 3u);
        let r4 = hash_float(seed + 4u) * 2.0 - 1.0;
        let r5 = hash_float(seed + 5u) * 2.0 - 1.0;
        let r6 = hash_float(seed + 6u) * 2.0 - 1.0;

        let disc_r = h * 0.5 * sqrt(r0);
        let theta = TWO_PI * r1;
        let along = r2 * params.dt * speed;
        let pos = p.position
            + e1 * (disc_r * cos(theta))
            + e2 * (disc_r * sin(theta))
            + v_dir * along;

        let jitter = vec3<f32>(r4, r5, r6) * params.velocity_jitter;
        let spawn_vel = vel * params.speed_multiplier + jitter;

        // Energy-scaled lifetime (SPlisHSPlasH): violent events leave
        // longer-lived foam. Only foam decays, so this is the foam budget.
        let lt_min = params.lifetime * (1.0 - params.lifetime_variation);
        let lt_span = 2.0 * params.lifetime * params.lifetime_variation;
        let lt = lt_min + phi_k * r3 * lt_span;

        spray_particles[slot] = SprayParticle(
            pos.x, pos.y, pos.z,
            lt,
            spawn_vel.x, spawn_vel.y, spawn_vel.z,
            lt,
            0u,        // kind: spray until spray_simulate reclassifies this frame
            0.0,       // age
            0.0, 0.0,
        );
    }
}
