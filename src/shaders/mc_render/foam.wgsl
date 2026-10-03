// Water shader (mc_render), part: whitewater compositing (density field foam, surface foam map raft)

// The whitewater field's calibration constants (FOAM_*, AERATION_*) are in
// water_common.wgsl: ss_composite reads the same field and shares them.

// Coarse surface grid .a for a column without fluid (foam_map.wgsl NO_FLUID)
const MAP_NO_FLUID: f32 = -99.0;

// === Surface foam (map) appearance ===
// Real surface foam is a raft of bubbles, not a cream: it thins by bursting,
// which opens holes until only a lace network of bubble strings is left. The
// map's density sets how much of the surface the raft covers; WHERE it covers
// is a cellular lace pattern (Voronoi cell walls) carried by the flow map, so
// thick foam is a near-solid raft, thinning foam opens growing holes, and the
// last of it is strings along the cell walls.
// Lace cell sizes (m): large holes + finer secondary network
const LACE_CELL: f32 = 0.045;
const LACE_CELL_FINE: f32 = 0.017;
// Bubble grain inside the raft (m)
const BUBBLE_CELL: f32 = 0.0045;
// Uneven bursting: coverage jitter frequency (1/m, ~12 cm patches)
const LACE_PATCH_FREQ: f32 = 8.0;
// Map density below which no raft is drawn (sparse leftover bubbles)
const RAFT_DENSITY_LO: f32 = 0.2;
// String breakup frequency (1/m, ~3 cm fragments)
const LACE_SNAP_FREQ: f32 = 33.0;
// Coverage response to map density: fraction of surface the raft covers
const RAFT_COVERAGE_K: f32 = 0.9;
// Opacity of the raft itself: a thin monolayer is see-through, a thick
// multilayer raft nearly opaque
const RAFT_ALPHA_THIN: f32 = 0.35;
const RAFT_ALPHA_THICK: f32 = 0.92;
const RAFT_ALPHA_K: f32 = 0.7;
// Lace geometry
const LACE_WARP_CELLS: f32 = 3.0;
const LACE_WARP_AMP: f32 = 0.3;
const LACE_FEATURE_WEIGHT: f32 = 0.3;
const LACE_ROUND_DENSE: f32 = 1.9;
const LACE_ROUND_COV_LO: f32 = 0.45;
const LACE_ROUND_COV_HI: f32 = 0.9;
const LACE_METRIC_OFFSET: f32 = 0.3;
const LACE_WIDTH_NOISE: f32 = 0.5;

struct MapFoam {
    density: f32,
    on_top: f32,
    // Flow-map coordinates of this point (phase A xy, phase B xy)
    coords: vec4<f32>,
}

// Surface-map foam at a fragment: bilinear map reads (density + flow-map
// coordinates), weighted by whether the fragment is the top surface of its
// column (overhang undersides, wave and body sides keep particle foam only).
// `n_local_y`: camera-facing normal's container-local up component.
fn sample_map_foam(local: vec3<f32>, n_local_y: f32) -> MapFoam {
    var out: MapFoam;
    out.density = 0.0;
    out.on_top = 0.0;
    out.coords = vec4<f32>(local.xz, local.xz);
    if ((foam_map.flags & 1u) == 0u) {
        return out;
    }
    let m = local.xz - vec2<f32>(foam_map.origin_x, foam_map.origin_z);
    // Column top, bilinear over the columns that hold fluid (a nearest-cell
    // top switches the test on and off in cell-sized blocks on rough water)
    let cd = i32(foam_map.coarse_dim);
    let gc = m / foam_map.coarse_cell - 0.5;
    let c0 = vec2<i32>(floor(gc));
    let fc = gc - floor(gc);
    var top_sum = 0.0;
    var top_w = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1);
        let c = clamp(c0 + o, vec2<i32>(0), vec2<i32>(cd - 1));
        let h = textureLoad(foam_surface_tex, c, 0).a;
        let w = select(1.0 - fc, fc, o == vec2<i32>(1));
        if (h > MAP_NO_FLUID) {
            top_sum += h * w.x * w.y;
            top_w += w.x * w.y;
        }
    }
    if (top_w < 1e-4) {
        return out;
    }
    let top = top_sum / top_w;
    let band = foam_map.surface_band;
    out.on_top = smoothstep(top - band, top - 0.5 * band, local.y) * smoothstep(0.15, 0.45, n_local_y);
    if (out.on_top <= 0.0) {
        return out;
    }
    let g = m / foam_map.fine_cell - 0.5;
    let t0 = vec2<i32>(floor(g));
    let f = g - floor(g);
    let fd = i32(foam_map.fine_dim);
    var foam = 0.0;
    var coords = vec4<f32>(0.0);
    for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1);
        let t = clamp(t0 + o, vec2<i32>(0), vec2<i32>(fd - 1));
        let w = select(1.0 - f, f, o == vec2<i32>(1));
        foam += textureLoad(foam_map_tex, t, 0).r * (w.x * w.y);
        coords += textureLoad(foam_coords_tex, t, 0) * (w.x * w.y);
    }
    out.density = foam;
    out.coords = coords;
    return out;
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(hash2(p), hash2(p + vec2<f32>(19.19, 73.31)));
}

// Worley distances in cell units: x = F1 (nearest feature), y = F2 - F1
// (distance-to-cell-wall proxy: 0 on the Voronoi walls)
fn worley(p: vec2<f32>) -> vec2<f32> {
    let cell = floor(p);
    let fr = p - cell;
    var f1 = 8.0;
    var f2 = 8.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let o = vec2<f32>(f32(x), f32(y));
            let d = length(o + hash22(cell + o) - fr);
            if (d < f1) {
                f2 = f1;
                f1 = d;
            } else if (d < f2) {
                f2 = d;
            }
        }
    }
    return vec2<f32>(f1, f2 - f1);
}

// Worley with additively weighted features (each feature's distance is
// reduced by a random share of LACE_FEATURE_WEIGHT cells): the cells form an
// Apollonius diagram - curved walls, varied cell sizes. Returns (F1, F2) in
// cell units. The weight stays well under a cell so the 3x3 search still
// finds the two nearest.
fn worley_weighted(p: vec2<f32>) -> vec2<f32> {
    let cell = floor(p);
    let fr = p - cell;
    var f1 = 8.0;
    var f2 = 8.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let o = vec2<f32>(f32(x), f32(y));
            let h = hash22(cell + o);
            let d = length(o + h - fr) - LACE_FEATURE_WEIGHT * fract(h.x * 7.0 + h.y * 13.0);
            if (d < f1) {
                f2 = f1;
                f1 = d;
            } else if (d < f2) {
                f2 = d;
            }
        }
    }
    return vec2<f32>(f1, f2);
}

// Divergence-free displacement of the lace domain: rotated value-noise
// gradients, two octaves (LACE_WARP_CELLS and half that, in lace cells),
// LACE_WARP_AMP cells rms - the two-octave rotated gradient has an rms of
// ~0.68 per noise cell, which the scale divides out.
fn lace_warp(p: vec2<f32>) -> vec2<f32> {
    let f = 1.0 / (LACE_WARP_CELLS * LACE_CELL);
    let g0 = value_noise_grad(p * f);
    let g1 = value_noise_grad(p * (2.0 * f) + vec2<f32>(7.1, 3.3));
    let w = vec2<f32>(g0.z, -g0.y) + 0.5 * vec2<f32>(g1.z, -g1.y);
    return p + w * (LACE_WARP_AMP * LACE_CELL / 0.68);
}

// Raft mask at one flow-map position: covered where the wall metric of the
// lace network is under a threshold that grows with coverage (thick foam:
// everything; thinning: holes open from the cell centres; last: strings).
// Bursting is uneven, so the local coverage is jittered at the patch scale:
// some areas hold a raft while neighbours are already down to strings.
// `px`: pixel footprint (m) for antialiasing / distance fade.
fn raft_mask(p: vec2<f32>, coverage: f32, px: f32) -> f32 {
    let patch_noise = value_noise_grad(p * LACE_PATCH_FREQ).x;
    let cov = clamp(coverage * (0.45 + 1.1 * patch_noise), 0.0, 1.0);
    let q = lace_warp(p);
    // Wall metric F2 - c F1 (+ offset): 0 on the walls, negative inside them
    // (most negative at the junctions), positive toward the hole centres
    let c = mix(1.0, LACE_ROUND_DENSE, smoothstep(LACE_ROUND_COV_LO, LACE_ROUND_COV_HI, cov));
    let off = (c - 1.0) * LACE_METRIC_OFFSET;
    let wc = worley_weighted(q / LACE_CELL);
    let wf = worley_weighted(q / LACE_CELL_FINE + vec2<f32>(5.3, 1.7));
    let coarse = wc.y - c * wc.x + off;
    let fine = wf.y - c * wf.x + off;
    // The fine network only subdivides holes while the raft is still dense:
    // thin foam is a few coarse strings, not a uniform net
    let fine_weight = mix(3.5, 1.4, smoothstep(0.3, 0.8, cov));
    let wall = min(coarse, fine * fine_weight);
    // String-scale noise: wall width along the string, and the snap gate
    let string_noise = value_noise_grad(q * LACE_SNAP_FREQ + vec2<f32>(3.1, 7.9)).x;
    let threshold = -log(max(1.0 - cov * 0.985, 1e-3)) * 0.22
        * (1.0 - LACE_WIDTH_NOISE + 2.0 * LACE_WIDTH_NOISE * string_noise);
    // Antialiasing width in metric units: the metric's gradient is up to
    // 1 + c (2 for F2 - F1), so the footprint scales with it
    let soft = max(px / LACE_CELL_FINE * 1.5, 0.03) * (0.5 * (1.0 + c));
    var mask = 1.0 - smoothstep(threshold - soft, threshold + soft, wall);
    // Thin lace is broken, not a connected net: strings snap into fragments
    // as the foam thins (gate the string noise by coverage; a string pinches
    // where the noise is low, then breaks there)
    let keep = clamp(cov * 1.8, 0.0, 1.0);
    mask *= smoothstep(1.0 - keep - 0.12, 1.0 - keep + 0.12, string_noise);
    // Below a pixel the lace can't resolve: converge to its mean coverage
    return mix(mask, cov, smoothstep(0.25, 0.8, px / LACE_CELL_FINE));
}

// Bubble grain: bright bubble walls, darker cell interiors; fades to its mean
// when bubbles shrink below a pixel
fn bubble_grain(p: vec2<f32>, px: f32) -> f32 {
    let w = worley(p / BUBBLE_CELL + vec2<f32>(11.1, 3.7));
    let grain = 0.85 + 0.3 * (1.0 - smoothstep(0.0, 0.25, w.y));
    return mix(grain, 0.93, smoothstep(0.3, 1.0, px / BUBBLE_CELL));
}
