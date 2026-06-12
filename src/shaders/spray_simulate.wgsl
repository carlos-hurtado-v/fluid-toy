// Whitewater simulation — classification + per-class dynamics (Ihmsen et al. 2012).
// Dispatched over all diffuse buffer slots. Each live particle counts fluid
// neighbors via the spatial grid and is reclassified every step:
//   spray  (few neighbors, airborne)  — ballistic: gravity + air drag
//   foam   (at the surface)           — advects with the local fluid velocity
//   bubble (submerged)                — buoyancy + drag toward fluid velocity
// Near container walls the neighbor count is extrapolated by the truncated
// sphere-cap volume before the foam/bubble decision, so submerged particles
// with wall-cut neighborhoods do not misread as surface foam (SPlisHSPlasH).
// Lifetime decays for FOAM ONLY: spray persists until it lands and bubbles
// until they surface (they die by becoming foam, leaving the container, or
// ring-buffer overwrite). Expects container_common.wgsl prepended
// (ContainerGeometry, world_to_local, is_inside_box).

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
    _pad0_p: f32,
    _pad1_p: f32,
};

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

@group(0) @binding(0) var<storage, read_write> spray_particles: array<SprayParticle>;
@group(0) @binding(1) var<uniform> params: SprayParams;
@group(0) @binding(2) var<uniform> container: ContainerGeometry;
@group(0) @binding(3) var<storage, read> sorted_particles: array<SphParticle>;
@group(0) @binding(4) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(5) var<storage, read> cell_counts: array<u32>;
@group(0) @binding(6) var<uniform> grid: GridParams;
@group(0) @binding(7) var<uniform> sph_params: SphParams;
// Live-particle counts, zeroed each step by the CPU: [total, spray, foam, bubble]
@group(0) @binding(8) var<storage, read_write> live_stats: array<atomic<u32>, 4>;

const KIND_SPRAY: u32 = 0u;
const KIND_FOAM: u32 = 1u;
const KIND_BUBBLE: u32 = 2u;

// Fluid-neighbor count thresholds (Ihmsen classification)
const SPRAY_MAX_NEIGHBORS: u32 = 6u;
const BUBBLE_MIN_NEIGHBORS: u32 = 20u;

// Newborn grace window: diffuse particles spawn inside their emitter's fluid
// neighborhood, so immediate classification would mark them foam and snap
// their velocity to the fluid's — wiping the inherited launch velocity and
// jitter before they ever fly. For their first moments, would-be-foam stays
// ballistic spray so crest/splash particles actually detach; genuinely
// submerged newborns still become bubbles right away.
const NEWBORN_SPRAY_TIME: f32 = 0.15;

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

// Volume fraction of a sphere cut off by a plane, as a function of the cap
// height fraction t in [0,1]: cap_vol / sphere_vol = t^2 (3 - t) / 4
// (t = 1 - d/r where d is the center-to-plane distance).
fn cap_volume_fraction(t: f32) -> f32 {
    let tc = clamp(t, 0.0, 1.0);
    return tc * tc * (3.0 - tc) * 0.25;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.max_particles) {
        return;
    }

    var p = spray_particles[idx];
    if (p.lifetime <= 0.0) {
        return;
    }

    let dt = params.dt;
    let pos = vec3<f32>(p.pos_x, p.pos_y, p.pos_z);
    var vel = vec3<f32>(p.vel_x, p.vel_y, p.vel_z);

    // Gather fluid neighborhood: count + radially-weighted average velocity
    let h = sph_params.kernel_radius;
    let h_sq = sph_params.kernel_radius_sq;
    var neighbor_count = 0u;
    var w_sum = 0.0;
    var v_avg = vec3<f32>(0.0);

    let cell_i = position_to_cell(pos);
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
                    let r_vec = pos - pj.position;
                    let r_sq = dot(r_vec, r_vec);
                    if (r_sq < h_sq) {
                        let w = 1.0 - sqrt(r_sq) / h;
                        neighbor_count = neighbor_count + 1u;
                        w_sum = w_sum + w;
                        v_avg = v_avg + pj.velocity * w;
                    }
                }
            }
        }
    }
    if (w_sum > 1e-6) {
        v_avg = v_avg / w_sum;
    }

    // Reclassify from the fluid neighborhood
    var kind = KIND_FOAM;
    if (neighbor_count <= SPRAY_MAX_NEIGHBORS) {
        kind = KIND_SPRAY;
    } else if (neighbor_count >= BUBBLE_MIN_NEIGHBORS) {
        kind = KIND_BUBBLE;
    }

    // Wall-truncated neighborhoods undercount: a submerged particle hugging a
    // wall reads foam-range counts. Extrapolate by the missing sphere-cap
    // volume and re-test only the bubble threshold — the free-surface deficit
    // IS the foam signal and must stay uncorrected (SPlisHSPlasH).
    if (kind == KIND_FOAM) {
        let local_pos = world_to_local(container, pos);
        let half_ext = vec3<f32>(container.half_width, container.half_height, container.half_depth);
        let dist_lo = half_ext + local_pos;
        let dist_hi = half_ext - local_pos;
        var missing = 0.0;
        for (var a = 0; a < 3; a++) {
            if (dist_lo[a] < h) {
                missing += cap_volume_fraction(1.0 - dist_lo[a] / h);
            }
            if (dist_hi[a] < h) {
                missing += cap_volume_fraction(1.0 - dist_hi[a] / h);
            }
        }
        // Corner caps overlap, so bound the extrapolation at 4x
        let visible = max(1.0 - missing, 0.25);
        if (u32(round(f32(neighbor_count) / visible)) >= BUBBLE_MIN_NEIGHBORS) {
            kind = KIND_BUBBLE;
        }
    }

    // Newborns at the surface keep flying instead of becoming foam instantly
    if (kind == KIND_FOAM && p.age < NEWBORN_SPRAY_TIME) {
        kind = KIND_SPRAY;
    }

    let gravity = vec3<f32>(0.0, params.gravity_y, 0.0);
    // Only foam ages out (SPlisHSPlasH rule): spray persists until it lands
    // and bubbles until they surface — both convert to foam and decay then.
    var lifetime_decay = 0.0;

    if (kind == KIND_SPRAY) {
        // Ballistic: gravity + air drag
        vel = vel + dt * (gravity - params.drag * vel);
    } else if (kind == KIND_FOAM) {
        // Foam rides the surface: take the local fluid velocity directly
        vel = v_avg;
        lifetime_decay = 1.0;
    } else {
        // Bubble: buoyancy opposes gravity, drag blends toward fluid velocity
        vel = vel + dt * (-params.bubble_buoyancy * gravity);
        vel = mix(vel, v_avg, clamp(params.bubble_drag, 0.0, 1.0));
    }

    let new_pos = pos + dt * vel;
    p.pos_x = new_pos.x;
    p.pos_y = new_pos.y;
    p.pos_z = new_pos.z;
    p.vel_x = vel.x;
    p.vel_y = vel.y;
    p.vel_z = vel.z;
    p.kind = kind;
    p.age = p.age + dt;
    p.lifetime = p.lifetime - dt * lifetime_decay;

    // Kill particles that exit the container bounds
    let local_pos = world_to_local(container, new_pos);
    if (!is_inside_box(container, local_pos, 0.0)) {
        p.lifetime = 0.0;
    }

    // Count particles still alive after this step, by kind
    if (p.lifetime > 0.0) {
        atomicAdd(&live_stats[0], 1u);
        atomicAdd(&live_stats[1u + kind], 1u);
    }

    spray_particles[idx] = p;
}
