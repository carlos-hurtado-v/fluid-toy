// Surface foam map: an Eulerian 2D foam layer over the container.
//
// Particle foam can't be the persistent surface layer at our particle counts:
// ~50k foam particles leave every splat individually visible ("snow"), and
// long-lived particle foam advected at the SPH kernel scale collapses onto
// the fluid particles themselves (a polka-dot lattice). Here diffuse particles
// that settle on the top surface deposit into a top-down map (container-local
// XZ) and retire; the map is then advected by a surface velocity field
// smoothed above the particle scale. Shear stretches patches into filaments,
// converging flow gathers foam into lines (conservative form), and foam decays
// with a persistence half-life. Particles keep everything off the top
// surface (overhangs, walls, airborne spray, bubbles).
//
// Grids: coarse (column tops + surface velocity) and fine (foam), both square
// over the container's larger horizontal extent. Pass order per frame:
//   clear_coarse -> column_tops -> splat_velocity -> resolve -> blur_h -> blur_v
//   -> deposit -> inject -> advect_forward -> advect_backward -> correct
// (container_common.wgsl is prepended at module creation.)

struct FoamMapParams {
    // Container-local XZ of the map's (0,0) corner
    origin_x: f32,
    origin_z: f32,
    fine_cell: f32,
    coarse_cell: f32,
    fine_dim: u32,
    coarse_dim: u32,
    num_particles: u32,
    max_spray: u32,
    dt: f32,
    // Per-frame persistence multiplier, exp(-ln2 dt / half-life)
    decay: f32,
    // Depth below a column's top fluid particle that counts as its surface
    surface_band: f32,
    // Foam (map units x m^2) one settling particle deposits, and its spread
    deposit_amount: f32,
    deposit_sigma: f32,
    // Newborn diffuse particles stay particles this long (spray grace)
    grace_age: f32,
    // Surface velocity smoothing, in coarse cells
    blur_sigma: f32,
    // bit 0: map active (deposit + render), bit 1: reset to empty this frame,
    // bit 2 / 3: restart flow-map phase A / B this frame
    flags: u32,
    // Flow-map phase A in [0,1) (phase B = phase A + 0.5); the renderer
    // crossfades the two so each restart happens at zero weight
    flow_phase: f32,
    // Foam lost per frame to bursting regardless of thickness: thin foam
    // dies quickly while thick foam rides the half-life
    burst: f32,
    _pad0: f32,
    _pad1: f32,
}

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
}

@group(0) @binding(0) var<uniform> params: FoamMapParams;
@group(0) @binding(1) var<uniform> container: ContainerGeometry;
// SPH particle = 4 vec4s: [i*4].xyz position, [i*4+1].xyz velocity
@group(0) @binding(2) var<storage, read> particles: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> col_top: array<atomic<u32>>;
// Fixed-point velocity accumulation per coarse cell: [3i] vx, [3i+1] vz, [3i+2] weight
@group(0) @binding(4) var<storage, read_write> vel_accum: array<atomic<i32>>;
@group(0) @binding(5) var surface_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(6) var surface_in: texture_2d<f32>;
@group(0) @binding(7) var<storage, read_write> spray: array<SprayParticle>;
@group(0) @binding(8) var<storage, read_write> deposit_accum: array<atomic<u32>>;
@group(0) @binding(9) var foam_in: texture_2d<f32>;
@group(0) @binding(10) var foam_out: texture_storage_2d<r32float, write>;
@group(0) @binding(11) var foam_orig: texture_2d<f32>;
@group(0) @binding(12) var foam_back: texture_2d<f32>;
// Flow-map coordinates: (A.xy, B.xy) = map position the material at this
// texel occupied when phase A / B last restarted. The renderer evaluates the
// lace pattern there, so the pattern moves (and shears) with the foam.
@group(0) @binding(13) var coords_in: texture_2d<f32>;
@group(0) @binding(14) var coords_out: texture_storage_2d<rgba32float, write>;

const KIND_FOAM: u32 = 1u;
// Column top quantization: order-preserving, 0 = empty column, 10 um steps
const TOP_OFFSET: f32 = 16.0;
const TOP_SCALE: f32 = 100000.0;
// Surface texel .a for a column with no fluid
const NO_FLUID: f32 = -100.0;
const VEL_FIXED: f32 = 4096.0;
const DEPOSIT_FIXED: f32 = 65536.0;
// Densest foam the map holds (keeps convergence zones from blowing up, and
// past ~3 the compositor is fully white anyway: more is just lingering)
const FOAM_MAX: f32 = 3.0;
// Bound on per-frame compression/expansion from surface divergence
const MAX_COMPRESSION: f32 = 1.5;

fn flag_active() -> bool {
    return (params.flags & 1u) != 0u;
}

fn flag_reset() -> bool {
    return (params.flags & 2u) != 0u;
}

fn flag_restart_a() -> bool {
    return (params.flags & 4u) != 0u;
}

fn flag_restart_b() -> bool {
    return (params.flags & 8u) != 0u;
}

fn coarse_index(c: vec2<i32>) -> u32 {
    return u32(c.y) * params.coarse_dim + u32(c.x);
}

fn in_coarse(c: vec2<i32>) -> bool {
    return all(c >= vec2<i32>(0)) && all(c < vec2<i32>(i32(params.coarse_dim)));
}

fn map_coord(local_xz: vec2<f32>) -> vec2<f32> {
    return local_xz - vec2<f32>(params.origin_x, params.origin_z);
}

fn decode_top(q: u32) -> f32 {
    if (q == 0u) {
        return NO_FLUID;
    }
    return f32(q) / TOP_SCALE - TOP_OFFSET;
}

// ---------------------------------------------------------------------------
// Coarse surface grid
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8)
fn clear_coarse(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = vec2<i32>(id.xy);
    if (!in_coarse(c)) {
        return;
    }
    let i = coarse_index(c);
    atomicStore(&col_top[i], 0u);
    atomicStore(&vel_accum[3u * i], 0);
    atomicStore(&vel_accum[3u * i + 1u], 0);
    atomicStore(&vel_accum[3u * i + 2u], 0);
}

// Highest fluid particle per column (container-local)
@compute @workgroup_size(256)
fn column_tops(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.num_particles) {
        return;
    }
    let local = world_to_local(container, particles[id.x * 4u].xyz);
    let c = vec2<i32>(floor(map_coord(local.xz) / params.coarse_cell));
    if (!in_coarse(c)) {
        return;
    }
    let q = u32(clamp((local.y + TOP_OFFSET) * TOP_SCALE, 1.0, 4.0e9));
    atomicMax(&col_top[coarse_index(c)], q);
}

// Horizontal velocity of the surface layer (particles within the band below
// their column's top), bilinearly splatted to cell centers
@compute @workgroup_size(256)
fn splat_velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.num_particles) {
        return;
    }
    let local = world_to_local(container, particles[id.x * 4u].xyz);
    let m = map_coord(local.xz) / params.coarse_cell;
    let own = vec2<i32>(floor(m));
    if (!in_coarse(own)) {
        return;
    }
    if (local.y < decode_top(atomicLoad(&col_top[coarse_index(own)])) - params.surface_band) {
        return;
    }
    let v = world_dir_to_local(container, particles[id.x * 4u + 1u].xyz).xz;
    let g = m - 0.5;
    let c0 = vec2<i32>(floor(g));
    let f = g - floor(g);
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1);
        let c = c0 + o;
        if (!in_coarse(c)) {
            continue;
        }
        let wv = select(1.0 - f, f, o == vec2<i32>(1));
        let w = wv.x * wv.y;
        let i = coarse_index(c);
        atomicAdd(&vel_accum[3u * i], i32(round(v.x * w * VEL_FIXED)));
        atomicAdd(&vel_accum[3u * i + 1u], i32(round(v.y * w * VEL_FIXED)));
        atomicAdd(&vel_accum[3u * i + 2u], i32(round(w * VEL_FIXED)));
    }
}

// Surface texel: rgb = (vx w, vz w, w) premultiplied for normalized
// convolution, a = column top height (NO_FLUID when empty)
@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = vec2<i32>(id.xy);
    if (!in_coarse(c)) {
        return;
    }
    let i = coarse_index(c);
    let vx = f32(atomicLoad(&vel_accum[3u * i])) / VEL_FIXED;
    let vz = f32(atomicLoad(&vel_accum[3u * i + 1u])) / VEL_FIXED;
    let w = f32(atomicLoad(&vel_accum[3u * i + 2u])) / VEL_FIXED;
    let top = decode_top(atomicLoad(&col_top[i]));
    textureStore(surface_out, c, vec4<f32>(vx, vz, w, top));
}

// Separable gaussian on the premultiplied velocity (empty cells weigh zero,
// so the smoothed velocity is an average of fluid only); top passes through
fn blur(c: vec2<i32>, dir: vec2<i32>) {
    if (!in_coarse(c)) {
        return;
    }
    let r = i32(ceil(2.5 * params.blur_sigma));
    let inv = 1.0 / (2.0 * params.blur_sigma * params.blur_sigma);
    var sum = vec3<f32>(0.0);
    for (var k = -r; k <= r; k++) {
        let s = c + dir * k;
        if (!in_coarse(s)) {
            continue;
        }
        sum += textureLoad(surface_in, s, 0).rgb * exp(-f32(k * k) * inv);
    }
    textureStore(surface_out, c, vec4<f32>(sum, textureLoad(surface_in, c, 0).a));
}

@compute @workgroup_size(8, 8)
fn blur_h(@builtin(global_invocation_id) id: vec3<u32>) {
    blur(vec2<i32>(id.xy), vec2<i32>(1, 0));
}

@compute @workgroup_size(8, 8)
fn blur_v(@builtin(global_invocation_id) id: vec3<u32>) {
    blur(vec2<i32>(id.xy), vec2<i32>(0, 1));
}

// Smoothed surface velocity at a map position (m, container-local XZ units),
// bilinear over the blurred coarse grid
fn surface_velocity(m: vec2<f32>) -> vec2<f32> {
    let g = m / params.coarse_cell - 0.5;
    let c0 = vec2<i32>(floor(g));
    let f = g - floor(g);
    var sum = vec3<f32>(0.0);
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1);
        let c = clamp(c0 + o, vec2<i32>(0), vec2<i32>(i32(params.coarse_dim) - 1));
        let wv = select(1.0 - f, f, o == vec2<i32>(1));
        sum += textureLoad(surface_in, c, 0).rgb * (wv.x * wv.y);
    }
    if (sum.z < 1e-4) {
        return vec2<f32>(0.0);
    }
    return sum.xy / sum.z;
}

fn column_top_at(m: vec2<f32>) -> f32 {
    let c = clamp(vec2<i32>(floor(m / params.coarse_cell)), vec2<i32>(0), vec2<i32>(i32(params.coarse_dim) - 1));
    return textureLoad(surface_in, c, 0).a;
}

// ---------------------------------------------------------------------------
// Fine foam map
// ---------------------------------------------------------------------------

// Settled foam particles on a column's top surface deposit a gaussian parcel
// and retire. Foam below the top (overhangs, wall films) stays a particle.
@compute @workgroup_size(256)
fn deposit(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.max_spray || !flag_active() || flag_reset()) {
        return;
    }
    var p = spray[id.x];
    if (p.lifetime <= 0.0 || p.kind != KIND_FOAM || p.age < params.grace_age) {
        return;
    }
    let local = world_to_local(container, vec3<f32>(p.pos_x, p.pos_y, p.pos_z));
    let m = map_coord(local.xz);
    let top = column_top_at(m);
    if (top <= NO_FLUID || local.y < top - params.surface_band) {
        return;
    }
    let center = m / params.fine_cell;
    let sigma_t = params.deposit_sigma / params.fine_cell;
    let r = i32(ceil(2.5 * sigma_t));
    let inv = 1.0 / (2.0 * sigma_t * sigma_t);
    let peak = params.deposit_amount / (6.2831853 * params.deposit_sigma * params.deposit_sigma);
    let c0 = vec2<i32>(floor(center));
    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            let t = c0 + vec2<i32>(dx, dy);
            if (any(t < vec2<i32>(0)) || any(t >= vec2<i32>(i32(params.fine_dim)))) {
                continue;
            }
            let d = vec2<f32>(t) + 0.5 - center;
            let amount = peak * exp(-dot(d, d) * inv);
            atomicAdd(&deposit_accum[u32(t.y) * params.fine_dim + u32(t.x)], u32(amount * DEPOSIT_FIXED));
        }
    }
    p.lifetime = 0.0;
    spray[id.x] = p;
}

fn in_fine(t: vec2<i32>) -> bool {
    return all(t >= vec2<i32>(0)) && all(t < vec2<i32>(i32(params.fine_dim)));
}

// Bilinear read of a fine map at a map position (m); outside = no foam
fn sample_fine(tex: texture_2d<f32>, m: vec2<f32>) -> f32 {
    let g = m / params.fine_cell - 0.5;
    let t0 = vec2<i32>(floor(g));
    let f = g - floor(g);
    var sum = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1);
        let t = t0 + o;
        if (!in_fine(t)) {
            continue;
        }
        let wv = select(1.0 - f, f, o == vec2<i32>(1));
        sum += textureLoad(tex, t, 0).r * (wv.x * wv.y);
    }
    return sum;
}

fn texel_center(t: vec2<i32>) -> vec2<f32> {
    return (vec2<f32>(t) + 0.5) * params.fine_cell;
}

// Add this frame's deposits (and clear the accumulator)
@compute @workgroup_size(8, 8)
fn inject(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = vec2<i32>(id.xy);
    if (!in_fine(t)) {
        return;
    }
    let i = u32(t.y) * params.fine_dim + u32(t.x);
    let added = f32(atomicLoad(&deposit_accum[i])) / DEPOSIT_FIXED;
    atomicStore(&deposit_accum[i], 0u);
    var foam = textureLoad(foam_in, t, 0).r + added;
    if (flag_reset()) {
        foam = 0.0;
    }
    textureStore(foam_out, t, vec4<f32>(foam, 0.0, 0.0, 0.0));
}

// MacCormack advection (low diffusion keeps filaments sharp): forward
// semi-Lagrangian step, backward step, then error correction + limiter
@compute @workgroup_size(8, 8)
fn advect_forward(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = vec2<i32>(id.xy);
    if (!in_fine(t)) {
        return;
    }
    let m = texel_center(t);
    let back = m - surface_velocity(m) * params.dt;
    textureStore(foam_out, t, vec4<f32>(sample_fine(foam_in, back), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn advect_backward(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = vec2<i32>(id.xy);
    if (!in_fine(t)) {
        return;
    }
    let m = texel_center(t);
    let fwd = m + surface_velocity(m) * params.dt;
    textureStore(foam_out, t, vec4<f32>(sample_fine(foam_in, fwd), 0.0, 0.0, 0.0));
}

// foam_in = forward result, foam_orig = pre-advection field, foam_back =
// backward result. Also: compression by surface divergence (foam is a
// conserved surface density, so converging flow piles it into lines and
// upwelling clears it), persistence decay, and removal over dry columns.
@compute @workgroup_size(8, 8)
fn correct(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = vec2<i32>(id.xy);
    if (!in_fine(t)) {
        return;
    }
    let m = texel_center(t);
    let v = surface_velocity(m);
    let forward = textureLoad(foam_in, t, 0).r;
    var foam = forward + 0.5 * (textureLoad(foam_orig, t, 0).r - textureLoad(foam_back, t, 0).r);

    // Limiter: stay within the source neighborhood the forward step read
    let g = (m - v * params.dt) / params.fine_cell - 0.5;
    let t0 = vec2<i32>(floor(g));
    var lo = 1e9;
    var hi = 0.0;
    for (var k = 0; k < 4; k++) {
        let s = t0 + vec2<i32>(k & 1, k >> 1);
        if (!in_fine(s)) {
            lo = min(lo, 0.0);
            continue;
        }
        let f = textureLoad(foam_orig, s, 0).r;
        lo = min(lo, f);
        hi = max(hi, f);
    }
    foam = clamp(foam, lo, hi);

    // Surface divergence (central differences over one coarse cell)
    let e = params.coarse_cell;
    let div = (surface_velocity(m + vec2<f32>(e, 0.0)).x - surface_velocity(m - vec2<f32>(e, 0.0)).x
        + surface_velocity(m + vec2<f32>(0.0, e)).y - surface_velocity(m - vec2<f32>(0.0, e)).y) / (2.0 * e);
    foam *= clamp(exp(-div * params.dt), 1.0 / MAX_COMPRESSION, MAX_COMPRESSION);

    foam = clamp(foam * params.decay - params.burst, 0.0, FOAM_MAX);
    if (column_top_at(m) <= NO_FLUID || flag_reset()) {
        foam = 0.0;
    }
    textureStore(foam_out, t, vec4<f32>(foam, 0.0, 0.0, 0.0));
}

// Advect the flow-map coordinates with the same surface velocity (plain
// semi-Lagrangian: they are smooth), restarting a phase to identity when its
// cycle wraps
@compute @workgroup_size(8, 8)
fn advect_coords(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = vec2<i32>(id.xy);
    if (!in_fine(t)) {
        return;
    }
    let m = texel_center(t);
    let back = m - surface_velocity(m) * params.dt;
    let g = back / params.fine_cell - 0.5;
    let t0 = vec2<i32>(floor(g));
    let f = g - floor(g);
    var c = vec4<f32>(0.0);
    var w_sum = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1);
        let s = t0 + o;
        if (!in_fine(s)) {
            continue;
        }
        let wv = select(1.0 - f, f, o == vec2<i32>(1));
        c += textureLoad(coords_in, s, 0) * (wv.x * wv.y);
        w_sum += wv.x * wv.y;
    }
    if (w_sum < 0.5) {
        c = vec4<f32>(m, m);
    } else {
        c /= w_sum;
    }
    if (flag_restart_a() || flag_reset()) {
        c = vec4<f32>(m, c.zw);
    }
    if (flag_restart_b() || flag_reset()) {
        c = vec4<f32>(c.xy, m);
    }
    textureStore(coords_out, t, c);
}

