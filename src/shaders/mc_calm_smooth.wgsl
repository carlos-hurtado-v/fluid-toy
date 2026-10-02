// Marching Cubes - Calm-surface smoothing
//
// Bulk-gated smoothing of the density field. Thick, calm water gets a wide
// low-pass that flattens the particle-scale lumps a still SPH surface carries
// ("orbeez"), while thin sheets and droplets keep the base field. One global
// blur can't do both: wide enough to flatten the lumps, it erases droplets
// and collapses sheets.
//
// Two helper fields live at half resolution (downsample, then mc_blur.wgsl):
//   S - the smoothing target (wide blur of the raw field)
//   G - the gate (S blurred further)
// The discriminator is a half-space property: at the edge of bulk water G
// reads ~half the interior density however wide the filter, while a sheet
// thinner than the filter reads ~thickness/width and a droplet ~0. So the
// gate smoothstep(lo, hi, G) is ~1 on bulk surfaces and ~0 on splash detail.
//
// The gate is read ON the smoothed surface (S = iso), not per voxel: see
// surface_gate.

struct CalmParams {
    full_size: u32,
    half_size: u32,
    // Gate thresholds on G, absolute field units (fractions of what G reads
    // on a bulk surface, scaled on the CPU)
    gate_lo: f32,
    gate_hi: f32,
    // 0 = off (base field untouched), 1 = full replacement where gated
    strength: f32,
    // The marching-cubes iso value: S = iso is the smoothed surface
    iso: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var src_field: texture_3d<f32>;
@group(0) @binding(1) var dst_field: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: CalmParams;
@group(0) @binding(3) var smooth_field: texture_3d<f32>;
@group(0) @binding(4) var gate_field: texture_3d<f32>;

fn corner(k: i32) -> vec3<i32> {
    return vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
}

// Full -> half resolution: mean of each 2x2x2 block's in-container voxels.
// Negative values are the outside-container sentinel (see mc_blur.wgsl); a
// block with no inside voxel stays sentinel.
@compute @workgroup_size(4, 4, 4)
fn downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(params.half_size))) {
        return;
    }
    let base = vec3<i32>(id) * 2;
    let full_max = vec3<i32>(i32(params.full_size) - 1);
    var sum = 0.0;
    var count = 0.0;
    for (var k = 0; k < 8; k++) {
        let v = textureLoad(src_field, min(base + corner(k), full_max), 0).r;
        if (v >= 0.0) {
            sum += v;
            count += 1.0;
        }
    }
    let out = select(-1.0, sum / max(count, 1.0), count > 0.0);
    textureStore(dst_field, vec3<i32>(id), vec4<f32>(out, 0.0, 0.0, 0.0));
}

// Voxels closer than NEAR cells to the smoothed surface read the gate on it,
// fading to their own G by FAR (distances along the gradient of S, linear
// estimate). Marching cubes reads the voxels on either side of a crossing and
// their neighbours for the normals: two cells each way.
const PROJECT_NEAR: f32 = 3.0;
const PROJECT_FAR: f32 = 5.0;

struct HalfSample {
    value: f32,
    // Per full-res cell
    grad: vec3<f32>,
}

// Trilinear read of a half-res field at a full-res grid position, skipping
// outside-container taps (value -1 if every tap is outside), with the
// gradient of the interpolant. Outside taps take the value for the gradient,
// which then has no component across a wall.
fn sample_half(field: texture_3d<f32>, p: vec3<f32>) -> HalfSample {
    // Full voxel center p + 0.5 sits at half-res coordinate (p + 0.5) / 2 - 0.5
    let q = (p + 0.5) * 0.5 - 0.5;
    let q0 = vec3<i32>(floor(q));
    let f = q - floor(q);
    let half_max = vec3<i32>(i32(params.half_size) - 1);
    var v: array<f32, 8>;
    var sum = 0.0;
    var w_sum = 0.0;
    for (var k = 0; k < 8; k++) {
        let o = corner(k);
        v[k] = textureLoad(field, clamp(q0 + o, vec3<i32>(0), half_max), 0).r;
        let w3 = select(1.0 - f, f, o == vec3<i32>(1));
        let w = w3.x * w3.y * w3.z;
        if (v[k] >= 0.0) {
            sum += v[k] * w;
            w_sum += w;
        }
    }
    if (w_sum <= 1e-6) {
        return HalfSample(-1.0, vec3<f32>(0.0));
    }
    let value = sum / w_sum;
    for (var k = 0; k < 8; k++) {
        if (v[k] < 0.0) {
            v[k] = value;
        }
    }
    let dx = mix(mix(v[1] - v[0], v[3] - v[2], f.y), mix(v[5] - v[4], v[7] - v[6], f.y), f.z);
    let dy = mix(mix(v[2] - v[0], v[3] - v[1], f.x), mix(v[6] - v[4], v[7] - v[5], f.x), f.z);
    let dz = mix(mix(v[4] - v[0], v[5] - v[1], f.x), mix(v[6] - v[2], v[7] - v[3], f.x), f.y);
    return HalfSample(value, 0.5 * vec3<f32>(dx, dy, dz));
}

// The gate value for the voxel at p: G at the nearest point of the smoothed
// surface when p is close to it, its own G otherwise.
//
// The gate has to be a property of the surface, not of the voxel. G falls
// off across a bulk surface, so gating each voxel on its own G lets some of
// the base field back in on the air side of every crossing - where the base
// field, a much narrower blur, is far below S - and how much depends on how
// far above the surface that voxel happens to sit. The surface dips by a
// sawtooth, one tooth per voxel layer it climbs through (contour-line stripes
// on any tilted calm surface, period = cell / slope): well under a millimetre
// of height, but slope enough for a grazing mirror to draw as streaks
// (snap_004: lookups of neighbouring pixel rows 20 px apart, 4 px without
// it). Read on the surface, the gate is the same for every voxel along the
// normal and the crossing is that of one fixed blend.
fn surface_gate(p: vec3<f32>, s: HalfSample, g: f32) -> f32 {
    let slope = max(length(s.grad), 1e-6 * params.iso);
    let reach = 1.0 - smoothstep(PROJECT_NEAR, PROJECT_FAR, abs(params.iso - s.value) / slope);
    if (reach <= 0.0) {
        return g;
    }
    // Two Newton steps onto S = iso: the profile flattens away from the
    // surface, so the first one overshoots
    var q = p + s.grad * ((params.iso - s.value) / (slope * slope));
    let s1 = sample_half(smooth_field, q);
    if (s1.value >= 0.0) {
        let slope1 = max(length(s1.grad), 1e-6 * params.iso);
        let step = clamp((params.iso - s1.value) / slope1, -PROJECT_FAR, PROJECT_FAR);
        q += s1.grad * (step / slope1);
    }
    // max: never less gated than the voxel's own G (an outside-container
    // read is -1)
    return mix(g, max(g, sample_half(gate_field, q).value), reach);
}

// final = mix(base, S, strength * smoothstep(lo, hi, G on the surface))
@compute @workgroup_size(4, 4, 4)
fn combine(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(params.full_size))) {
        return;
    }
    let p = vec3<i32>(id);
    let base = textureLoad(src_field, p, 0).r;
    var out = base;
    if (base >= 0.0) {
        let s = sample_half(smooth_field, vec3<f32>(id));
        let g = sample_half(gate_field, vec3<f32>(id)).value;
        if (s.value >= 0.0 && g >= 0.0) {
            let bulk = smoothstep(params.gate_lo, params.gate_hi, surface_gate(vec3<f32>(id), s, g)) * params.strength;
            out = mix(base, s.value, bulk);
        }
    }
    textureStore(dst_field, p, vec4<f32>(out, 0.0, 0.0, 0.0));
}
