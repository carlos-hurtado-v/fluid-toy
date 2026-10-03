// Marching Cubes - Voxel normals
//
// The surface normal at every voxel, written once per frame for the two
// readers that have to agree: mc_generate (vertex normals) and mc_render's
// water_normal (refraction exit normals, re-triangulating the cell).
//
// A voxel's normal is the field's central-difference gradient. On calm water
// it is then DENOISED. The calm-smoothed field still carries the particles as
// a ripple: well under a millimetre of height, but 0.35-0.4 deg of slope on a
// fully settled tank, and a mirror seen at a grazing angle shows slope, not
// height (marbled reflections under the surface). The calm gate field G
// (mc_calm_smooth.wgsl) is a much wider blur of the same water: its gradient
// is three times quieter, but it rounds off real shape. So the normal takes
// G's direction where the two agree to within about `denoise` (noise) and
// keeps the field's own where they differ by more (waves, crater walls,
// sheets, droplets, the mesh's sides on the walls):
//
//   n = wide + d * |d|^2 / (|d|^2 + denoise^2),   d = n_field - wide
//
// which never moves a normal by more than denoise / 2. Widening the smoothing
// instead reaches the same noise only by moving the surface (it sits another
// centimetre high) and flattening real slopes by several degrees.

struct GridParams {
    grid_min: vec3<f32>,
    grid_size: u32,
    grid_max: vec3<f32>,
    cell_size: f32,
    kernel_radius: f32,
    iso_value: f32,
    num_particles: u32,
    max_vertices: u32,
}

struct NormalParams {
    // Radians; 0 = the field's own normals (calm smoothing off: G is stale)
    denoise: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var field: texture_3d<f32>;
@group(0) @binding(1) var normals: texture_storage_3d<r32uint, write>;
@group(0) @binding(2) var<uniform> params: GridParams;
@group(0) @binding(3) var<uniform> normal_params: NormalParams;
// Calm gate field G, half resolution
@group(0) @binding(4) var wide_field: texture_3d<f32>;

// Only voxels this close to the surface (cells, linear estimate) are
// denoised: a cell the mesh crosses has its corners within sqrt(3)
const DENOISE_REACH: f32 = 4.0;

fn sample_field(pos: vec3<i32>) -> f32 {
    return textureLoad(field, clamp(pos, vec3<i32>(0), vec3<i32>(i32(params.grid_size) - 1)), 0).r;
}

// Trilinear read of the half-res wide field at a full-res grid position,
// skipping outside-container taps (-1 if every tap is outside), as
// mc_calm_smooth.wgsl reads it
fn sample_wide(p: vec3<f32>) -> f32 {
    let q = (p + 0.5) * 0.5 - 0.5;
    let q0 = vec3<i32>(floor(q));
    let f = q - floor(q);
    let half_max = vec3<i32>(textureDimensions(wide_field)) - 1;
    var sum = 0.0;
    var w_sum = 0.0;
    for (var k = 0; k < 8; k++) {
        let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let v = textureLoad(wide_field, clamp(q0 + o, vec3<i32>(0), half_max), 0).r;
        let w3 = select(1.0 - f, f, o == vec3<i32>(1));
        let w = w3.x * w3.y * w3.z;
        if (v >= 0.0) {
            sum += v * w;
            w_sum += w;
        }
    }
    return select(-1.0, sum / max(w_sum, 1e-6), w_sum > 1e-6);
}

// oct_encode(): octahedral_common.wgsl, prepended at module creation

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(params.grid_size))) {
        return;
    }
    let p = vec3<i32>(id);
    // A 1-cell central difference preserves local wave detail and avoids
    // over-smoothed "waxy" shading on dynamic surfaces.
    let grad = vec3<f32>(
        sample_field(p + vec3<i32>(1, 0, 0)) - sample_field(p - vec3<i32>(1, 0, 0)),
        sample_field(p + vec3<i32>(0, 1, 0)) - sample_field(p - vec3<i32>(0, 1, 0)),
        sample_field(p + vec3<i32>(0, 0, 1)) - sample_field(p - vec3<i32>(0, 0, 1)),
    );
    let len = length(grad);
    var n = vec3<f32>(0.0, 1.0, 0.0);
    if (len > 0.0001) {
        n = -grad / len;  // Point outward from surface
        // grad spans two cells
        let dist = abs(sample_field(p) - params.iso_value) / (0.5 * len);
        if (normal_params.denoise > 0.0 && dist < DENOISE_REACH) {
            let pf = vec3<f32>(id);
            let x0 = sample_wide(pf - vec3<f32>(1.0, 0.0, 0.0));
            let x1 = sample_wide(pf + vec3<f32>(1.0, 0.0, 0.0));
            let y0 = sample_wide(pf - vec3<f32>(0.0, 1.0, 0.0));
            let y1 = sample_wide(pf + vec3<f32>(0.0, 1.0, 0.0));
            let z0 = sample_wide(pf - vec3<f32>(0.0, 0.0, 1.0));
            let z1 = sample_wide(pf + vec3<f32>(0.0, 0.0, 1.0));
            let wide_grad = vec3<f32>(x1 - x0, y1 - y0, z1 - z0);
            let wide_len = length(wide_grad);
            if (min(min(x0, x1), min(min(y0, y1), min(z0, z1))) >= 0.0 && wide_len > 1e-6 * params.iso_value) {
                let wide = -wide_grad / wide_len;
                let d = n - wide;
                let m2 = dot(d, d);
                n = normalize(wide + d * (m2 / (m2 + normal_params.denoise * normal_params.denoise)));
            }
        }
    }
    textureStore(normals, p, vec4<u32>(oct_encode(n), 0u, 0u, 0u));
}
