// Marching Cubes - Anisotropic kernel estimation (Yu & Turk 2013)
// One thread per sorted particle: fits a weighted covariance ellipsoid to the
// local particle distribution via the SPH spatial hash grid, eigendecomposes it
// (cyclic Jacobi), and emits a world->kernel-space transform G plus a
// Laplacian-smoothed splat center. mc_density.wgsl consumes the output to splat
// ellipsoids instead of spheres. Render-only: the simulation never reads this.

struct SphParticle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    force: vec3<f32>,
    density: f32,
    near_density: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
}

struct SphGridParams {
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
}

struct AnisoParams {
    enabled: u32,
    strength: f32,       // 0 = isotropic, 1 = full Yu & Turk anisotropy
    support_radius: f32, // covariance neighborhood radius (2 * sim kernel radius)
    h_mc: f32,           // MC density kernel radius (sim h * mc_density_radius_scale)
    kr: f32,             // max stddev ratio between largest/smallest axis (Yu & Turk k_r)
    lambda: f32,         // center smoothing toward weighted neighbor mean
    max_stretch: f32,    // hard cap on axis scale; bounds the density pass search radius
    max_shift: f32,      // hard cap on center smoothing shift, world units
}

struct ParticleAniso {
    q0: vec4<f32>, // (Gxx, Gxy, Gxz, center.x)
    q1: vec4<f32>, // (Gyy, Gyz, Gzz, center.y)
    q2: vec4<f32>, // (center.z, reach, amplitude, 0)
}

@group(0) @binding(0) var<storage, read> sorted_particles: array<SphParticle3D>;
@group(0) @binding(1) var<storage, read> cell_starts: array<u32>;
@group(0) @binding(2) var<storage, read> cell_counts: array<u32>;
@group(0) @binding(3) var<uniform> sph_grid: SphGridParams;
@group(0) @binding(4) var<uniform> params: AnisoParams;
@group(0) @binding(5) var<storage, read_write> aniso_out: array<ParticleAniso>;

// Neighbor counts (within support_radius) where anisotropy fades in.
// Below N_LO features stay spherical (isolated droplets); a one-layer sheet
// at 0.6h spacing has ~35 neighbors at radius 2h, a particle string ~7.
const N_LO: f32 = 4.0;
const N_HI: f32 = 12.0;
// Floor for axis scales (a flat sheet's thin axis is kr^(-2/3) ~ 0.4 at kr=4).
const MIN_STRETCH: f32 = 0.25;
const PI: f32 = 3.14159265359;

fn position_to_sph_cell(pos: vec3<f32>) -> vec3<i32> {
    let local = pos - vec3<f32>(sph_grid.grid_origin_x, sph_grid.grid_origin_y, sph_grid.grid_origin_z);
    return vec3<i32>(floor(local * sph_grid.inv_cell_size));
}

fn sph_cell_to_index(cell: vec3<i32>) -> u32 {
    return u32(cell.x) + u32(cell.y) * sph_grid.grid_size_x + u32(cell.z) * sph_grid.grid_size_x * sph_grid.grid_size_y;
}

fn is_valid_sph_cell(cell: vec3<i32>) -> bool {
    return cell.x >= 0 && cell.x < i32(sph_grid.grid_size_x) &&
           cell.y >= 0 && cell.y < i32(sph_grid.grid_size_y) &&
           cell.z >= 0 && cell.z < i32(sph_grid.grid_size_z);
}

// One Jacobi rotation in the (p, q) plane, zeroing A[p][q]; k is the remaining
// index. Accumulates the rotation into V (columns converge to eigenvectors).
fn jacobi_rotate(A: ptr<function, mat3x3<f32>>, V: ptr<function, mat3x3<f32>>, p: i32, q: i32, k: i32) {
    let apq = (*A)[p][q];
    if (abs(apq) < 1e-12) {
        return;
    }
    let app = (*A)[p][p];
    let aqq = (*A)[q][q];
    let tau = (aqq - app) / (2.0 * apq);
    let sign_tau = select(1.0, -1.0, tau < 0.0);
    let t = sign_tau / (abs(tau) + sqrt(1.0 + tau * tau));
    let c = inverseSqrt(1.0 + t * t);
    let s = t * c;

    (*A)[p][p] = app - t * apq;
    (*A)[q][q] = aqq + t * apq;
    (*A)[p][q] = 0.0;
    (*A)[q][p] = 0.0;
    let akp = (*A)[k][p];
    let akq = (*A)[k][q];
    (*A)[k][p] = c * akp - s * akq;
    (*A)[p][k] = (*A)[k][p];
    (*A)[k][q] = s * akp + c * akq;
    (*A)[q][k] = (*A)[k][q];

    let vp = (*V)[p];
    let vq = (*V)[q];
    (*V)[p] = c * vp - s * vq;
    (*V)[q] = s * vp + c * vq;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sph_grid.num_particles) {
        return;
    }

    let xi = sorted_particles[i].position;
    let r_support = params.support_radius;
    let h = params.h_mc;
    // Poly6 peak at r=0 for the MC kernel; the field calibration anchor.
    let peak = 315.0 / (64.0 * PI * h * h * h);

    // Gather weighted neighborhood moments. Positions are taken relative to xi
    // so the one-pass covariance has no catastrophic cancellation.
    var w_sum = 0.0;
    var n_count = 0.0;
    var m1 = vec3<f32>(0.0);
    var cxx = 0.0;
    var cyy = 0.0;
    var czz = 0.0;
    var cxy = 0.0;
    var cxz = 0.0;
    var cyz = 0.0;

    let center_cell = position_to_sph_cell(xi);
    let cell_radius = i32(ceil(r_support * sph_grid.inv_cell_size));

    for (var dz = -cell_radius; dz <= cell_radius; dz++) {
        for (var dy = -cell_radius; dy <= cell_radius; dy++) {
            for (var dx = -cell_radius; dx <= cell_radius; dx++) {
                let cell = center_cell + vec3<i32>(dx, dy, dz);
                if (!is_valid_sph_cell(cell)) {
                    continue;
                }
                let cell_idx = sph_cell_to_index(cell);
                let count = cell_counts[cell_idx];
                if (count == 0u) {
                    continue;
                }
                // cell_starts contains inclusive prefix sum, so start = end - count
                let end = cell_starts[cell_idx];
                let start = end - count;
                for (var n = 0u; n < count; n++) {
                    let d = sorted_particles[start + n].position - xi;
                    let r = length(d);
                    if (r < r_support) {
                        let u = r / r_support;
                        let w = 1.0 - u * u * u;
                        w_sum += w;
                        n_count += 1.0;
                        m1 += w * d;
                        cxx += w * d.x * d.x;
                        cyy += w * d.y * d.y;
                        czz += w * d.z * d.z;
                        cxy += w * d.x * d.y;
                        cxz += w * d.x * d.z;
                        cyz += w * d.y * d.z;
                    }
                }
            }
        }
    }

    // Fade anisotropy (and center smoothing) out for sparse neighborhoods.
    let t_n = smoothstep(N_LO, N_HI, n_count);
    let s_eff = params.strength * t_n;

    let inv_w = 1.0 / max(w_sum, 1e-9);
    let mu = m1 * inv_w;

    // Weighted covariance C = E[d d^T] - mu mu^T (symmetric).
    var A = mat3x3<f32>(
        vec3<f32>(cxx * inv_w - mu.x * mu.x, cxy * inv_w - mu.x * mu.y, cxz * inv_w - mu.x * mu.z),
        vec3<f32>(cxy * inv_w - mu.x * mu.y, cyy * inv_w - mu.y * mu.y, cyz * inv_w - mu.y * mu.z),
        vec3<f32>(cxz * inv_w - mu.x * mu.z, cyz * inv_w - mu.y * mu.z, czz * inv_w - mu.z * mu.z),
    );
    var V = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    for (var sweep = 0; sweep < 5; sweep++) {
        jacobi_rotate(&A, &V, 0, 1, 2);
        jacobi_rotate(&A, &V, 0, 2, 1);
        jacobi_rotate(&A, &V, 1, 2, 0);
    }

    // Sort eigenpairs descending (V columns are the eigenvectors).
    var e = vec3<f32>(A[0][0], A[1][1], A[2][2]);
    var v0 = V[0];
    var v1 = V[1];
    var v2 = V[2];
    if (e.x < e.y) {
        let te = e.x; e.x = e.y; e.y = te;
        let tv = v0; v0 = v1; v1 = tv;
    }
    if (e.x < e.z) {
        let te = e.x; e.x = e.z; e.z = te;
        let tv = v0; v0 = v2; v2 = tv;
    }
    if (e.y < e.z) {
        let te = e.y; e.y = e.z; e.z = te;
        let tv = v1; v1 = v2; v2 = tv;
    }

    // Shape-gated Laplacian center smoothing: full strength for flat/linear
    // neighborhoods (bumpy calm surfaces, sheets, strings — where smoothing
    // kills lattice texture), suppressed for round neighborhoods so small
    // droplet clumps don't condense into inflated merged blobs.
    let s1_raw = sqrt(max(e.x, 0.0));
    let s3_raw = sqrt(max(e.z, 0.0));
    var flatness = 0.0;
    if (s1_raw > 1e-7) {
        let roundness = s3_raw / s1_raw; // 1 = isotropic clump, 0 = perfectly flat/linear
        flatness = 1.0 - roundness * roundness;
    }
    // Shift is clamped so the density pass search radius bound
    // (max_stretch * h + max_shift) stays valid.
    var shift = params.lambda * s_eff * flatness * mu;
    let shift_len = length(shift);
    if (shift_len > params.max_shift) {
        shift *= params.max_shift / shift_len;
    }
    let center = xi + shift;

    // Axis scales: stddevs ratio-clamped (Yu & Turk), volume-normalized so
    // a.x * a.y * a.z = 1 (interior field and iso calibration unchanged),
    // shaped by strength, then hard-capped.
    var a = vec3<f32>(1.0);
    let s1 = s1_raw;
    if (s1 > 1e-7 && s_eff > 0.0) {
        let s_min = s1 / params.kr;
        let s2 = max(sqrt(max(e.y, 0.0)), s_min);
        let s3 = max(s3_raw, s_min);
        let norm = pow(s1 * s2 * s3, 1.0 / 3.0);
        a = vec3<f32>(s1, s2, s3) / norm;
        a = pow(a, vec3<f32>(s_eff));
        a = clamp(a, vec3<f32>(MIN_STRETCH), vec3<f32>(params.max_stretch));
    }

    // G maps world offsets to unit kernel space: G = sum_k v_k v_k^T / (a_k * h).
    // Symmetric, so 6 unique entries.
    let inv_r = 1.0 / (a * h);
    let gxx = inv_r.x * v0.x * v0.x + inv_r.y * v1.x * v1.x + inv_r.z * v2.x * v2.x;
    let gyy = inv_r.x * v0.y * v0.y + inv_r.y * v1.y * v1.y + inv_r.z * v2.y * v2.y;
    let gzz = inv_r.x * v0.z * v0.z + inv_r.y * v1.z * v1.z + inv_r.z * v2.z * v2.z;
    let gxy = inv_r.x * v0.x * v0.y + inv_r.y * v1.x * v1.y + inv_r.z * v2.x * v2.y;
    let gxz = inv_r.x * v0.x * v0.z + inv_r.y * v1.x * v1.z + inv_r.z * v2.x * v2.z;
    let gyz = inv_r.x * v0.y * v0.z + inv_r.y * v1.y * v1.z + inv_r.z * v2.y * v2.z;

    // Volume normalization keeps a.x*a.y*a.z = 1, so amplitude = peak and the
    // field calibration matches the isotropic kernel exactly. When the hard
    // caps break the product (extreme strings), cap the amplitude at peak
    // rather than boosting it — a brighter peak would inflate the apparent
    // size of exactly the sparse features that should stay small.
    let amplitude = peak / max(a.x * a.y * a.z, 1.0);
    let reach = h * max(a.x, max(a.y, a.z));

    aniso_out[i] = ParticleAniso(
        vec4<f32>(gxx, gxy, gxz, center.x),
        vec4<f32>(gyy, gyz, gzz, center.y),
        vec4<f32>(center.z, reach, amplitude, 0.0),
    );
}
