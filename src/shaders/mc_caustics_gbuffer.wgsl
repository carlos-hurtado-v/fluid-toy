// Caustics - Light-Space G-Buffer Pass
// Rasterizes the MC water mesh from the sun (orthographic) and captures the
// front-most surface's world position + normal per texel. The splat pass then
// refracts one photon per texel and deposits it on the pool floor.
// Concatenated with container_common.wgsl (clip test mirrors mc_back_depth).

struct CausticsParams {
    light_view_proj: mat4x4<f32>,
    // Direction toward the sun (world space, normalized)
    sun_dir: vec3<f32>,
    // Energy carried by one light-space texel (m^2 of beam cross-section)
    flux_area: f32,
    // Per-channel index of refraction (dispersion pre-applied on CPU)
    ior_rgb: vec3<f32>,
    // Gaussian splat sigma on the floor (container-local meters)
    sigma: f32,
    // Beer-Lambert absorption coefficients (matches mc_render.wgsl)
    absorb_rgb: vec3<f32>,
    inv_two_sigma_sq: f32,
    // 1 / (2 pi sigma^2) - normalizes the splat kernel to unit integral
    splat_norm: f32,
    // Splat quad half-extent on the floor (meters, ~2.5 sigma)
    splat_radius: f32,
    // Optical density from water clarity (matches mc_render.wgsl)
    optical_density: f32,
    // Light-space raster resolution (texels per side)
    light_res: u32,
    // Sim time driving the procedural ripple animation (matches surface)
    time: f32,
    // Ripple normal perturbation gain for the caustics path (decoupled from
    // the surface's ripple_strength: caustic focusing needs more than specular)
    ripple_strength: f32,
    // Photon kinds per texel: 4 = R/G/B/shadow (chromatic), 2 = white/shadow
    kinds: u32,
    _pad0: f32,
}

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: CausticsParams;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<uniform> container: ContainerGeometry;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let vertex = vertices[vertex_index];

    var output: VertexOutput;
    output.world_position = vertex.position;
    output.world_normal = vertex.normal;
    output.clip_position = params.light_view_proj * vec4<f32>(vertex.position, 1.0);
    return output;
}

struct FragmentOutput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
}

// --- Procedural micro-ripples (mirrors mc_render.wgsl exactly) ---
// The MC mesh only carries wave detail down to grid-cell scale; the fine
// caustic filaments come from these sub-mesh ripples, animated by sim time so
// the floor dapple stays coherent with the surface sparkle.

fn hash2(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smooth value noise with analytic gradient (returns: vec3(noise, dN/dx, dN/dz))
fn value_noise_grad(p: vec2<f32>) -> vec3<f32> {
    let i = floor(p);
    let f = fract(p);
    // Quintic Hermite interpolation (C2 continuous - no grid artifacts)
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    let a = hash2(i + vec2<f32>(0.0, 0.0));
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));

    let val = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    let dx = du.x * ((b - a) + (a - b - c + d) * u.y);
    let dy = du.y * ((c - a) + (a - b - c + d) * u.x);
    return vec3<f32>(val, dx, dy);
}

// Multi-octave noise normal perturbation with analytic derivatives.
// Coarser spectrum than the surface ripple (3 octaves from 14cm wavelength,
// steeper falloff): at tank depth the fine octaves are past their focal
// distance and only add speckle, while the long wavelengths focus into the
// big sinuous webs that read as caustics.
fn ripple_normal(world_pos: vec3<f32>, t: f32) -> vec3<f32> {
    var grad = vec2<f32>(0.0);
    var amp = 1.0;
    var freq = 7.0;

    // Octaves at different time offsets to avoid coherent drift
    for (var oct = 0u; oct < 3u; oct++) {
        let time_offset = t * (0.3 + f32(oct) * 0.15);
        // Rotate sample coords per octave to break axis alignment
        let angle = f32(oct) * 1.8;
        let cs = cos(angle);
        let sn = sin(angle);
        let p = vec2<f32>(
            world_pos.x * cs - world_pos.z * sn,
            world_pos.x * sn + world_pos.z * cs,
        );
        let n = value_noise_grad(p * freq + vec2<f32>(time_offset, -time_offset * 0.7));
        // Rotate gradient back to world XZ
        grad += amp * vec2<f32>(
            n.y * cs + n.z * sn,
            -n.y * sn + n.z * cs,
        );
        freq *= 2.0;
        amp *= 0.45;
    }

    return vec3<f32>(grad.x, 0.0, grad.y);
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    if (container.clip_enabled != 0u) {
        let local = world_to_local(container, in.world_position);
        if (!is_inside_box(container, local, container.clip_margin)) {
            discard;
        }
    }

    var normal = normalize(in.world_normal);
    if (params.ripple_strength > 0.0) {
        let ripple_grad = ripple_normal(in.world_position, params.time);
        normal = normalize(normal + ripple_grad * params.ripple_strength);
    }

    var out: FragmentOutput;
    out.position = vec4<f32>(in.world_position, 1.0);
    out.normal = vec4<f32>(normal, 0.0);
    return out;
}

// === Container occluder (depth-only; color writes masked in the pipeline) ===
// Drawn into the light-space depth before the water mesh, so water sitting in
// rim shadow fails the depth test and emits neither photons nor shadow splats.
// Consistent with the analytic rim_visibility test in container.wgsl.

struct OccluderVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_occluder(@location(0) position: vec3<f32>) -> OccluderVertexOutput {
    var out: OccluderVertexOutput;
    let world_pos = local_to_world(container, position);
    out.clip_position = params.light_view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

@fragment
fn fs_occluder() -> FragmentOutput {
    var out: FragmentOutput;
    out.position = vec4<f32>(0.0);
    out.normal = vec4<f32>(0.0);
    return out;
}
