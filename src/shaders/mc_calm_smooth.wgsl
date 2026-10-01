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

struct CalmParams {
    full_size: u32,
    half_size: u32,
    // Gate thresholds on G, absolute field units (fractions of the interior
    // density, scaled on the CPU)
    gate_lo: f32,
    gate_hi: f32,
    // 0 = off (base field untouched), 1 = full replacement where gated
    strength: f32,
    _pad0: f32,
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

// Trilinear read of a half-res field at a full-res voxel, skipping
// outside-container taps. Returns -1 if every tap is outside.
fn sample_half(field: texture_3d<f32>, p: vec3<i32>) -> f32 {
    // Full voxel center p + 0.5 sits at half-res coordinate (p + 0.5) / 2 - 0.5
    let q = (vec3<f32>(p) + 0.5) * 0.5 - 0.5;
    let q0 = vec3<i32>(floor(q));
    let f = q - floor(q);
    let half_max = vec3<i32>(i32(params.half_size) - 1);
    var sum = 0.0;
    var w_sum = 0.0;
    for (var k = 0; k < 8; k++) {
        let o = corner(k);
        let v = textureLoad(field, clamp(q0 + o, vec3<i32>(0), half_max), 0).r;
        let w3 = select(1.0 - f, f, o == vec3<i32>(1));
        let w = w3.x * w3.y * w3.z;
        if (v >= 0.0) {
            sum += v * w;
            w_sum += w;
        }
    }
    return select(-1.0, sum / max(w_sum, 1e-6), w_sum > 1e-6);
}

// final = mix(base, S, strength * smoothstep(lo, hi, G))
@compute @workgroup_size(4, 4, 4)
fn combine(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(params.full_size))) {
        return;
    }
    let p = vec3<i32>(id);
    let base = textureLoad(src_field, p, 0).r;
    var out = base;
    if (base >= 0.0) {
        let s = sample_half(smooth_field, p);
        let g = sample_half(gate_field, p);
        if (s >= 0.0 && g >= 0.0) {
            let bulk = smoothstep(params.gate_lo, params.gate_hi, g) * params.strength;
            out = mix(base, s, bulk);
        }
    }
    textureStore(dst_field, p, vec4<f32>(out, 0.0, 0.0, 0.0));
}
