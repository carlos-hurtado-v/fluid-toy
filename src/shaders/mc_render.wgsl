// Marching Cubes - Mesh Rendering
// Renders the generated triangle mesh with water shading

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    near: f32,
    far: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct WaterParams {
    water_color: vec3<f32>,
    roughness: f32,
    ior: f32,
    refraction_strength: f32,
    env_intensity: f32,
    use_env_background: u32,
    background_r: f32,
    background_g: f32,
    background_b: f32,
    time: f32,
    deep_color_r: f32,
    deep_color_g: f32,
    deep_color_b: f32,
    ripple_strength: f32,
    clarity: f32,
    // 1 = physical (Snell, two-interface) refraction, 0 = legacy UV offset
    physical_refraction: f32,
    foam_coverage: f32,
    aeration_strength: f32,
    // 1 = physical water medium (absorption + single scattering), 0 = legacy
    physical_medium: f32,
    _pad_m0: f32,
    _pad_m1: f32,
    _pad_m2: f32,
}

struct LightParams {
    sun_direction: vec3<f32>,
    sun_enabled: u32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    _pad2: f32,
    _padding: vec3<f32>,
}

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> water: WaterParams;
@group(0) @binding(2) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(3) var env_tex: texture_2d<f32>;
@group(0) @binding(4) var env_sampler: sampler;
@group(0) @binding(5) var back_depth_tex: texture_depth_2d;
@group(0) @binding(6) var depth_sampler: sampler;
@group(0) @binding(7) var background_tex: texture_2d<f32>;
@group(0) @binding(8) var<uniform> light: LightParams;
@group(0) @binding(9) var<uniform> sh_coeffs: array<vec4<f32>, 9>;
@group(0) @binding(10) var ssr_tex: texture_2d<f32>;
@group(0) @binding(12) var foam_density_tex: texture_2d<f32>;
// Exit interface for refraction: outward normal of the nearest back face (w = 1
// where one exists), written by the back-face pass alongside back_depth_tex
@group(0) @binding(13) var back_normal_tex: texture_2d<f32>;
// Depth of everything behind the water (backdrop, container, bodies)
@group(0) @binding(14) var background_depth_tex: texture_depth_2d;

// Surface foam map (foam_map.wgsl): advected 2D foam layer in container-local
// XZ, plus the coarse surface grid whose .a is each column's top fluid height
struct FoamMapParams {
    origin_x: f32,
    origin_z: f32,
    fine_cell: f32,
    coarse_cell: f32,
    fine_dim: u32,
    coarse_dim: u32,
    num_particles: u32,
    max_spray: u32,
    dt: f32,
    decay: f32,
    surface_band: f32,
    deposit_amount: f32,
    deposit_sigma: f32,
    grace_age: f32,
    blur_sigma: f32,
    flags: u32,
    flow_phase: f32,
    burst: f32,
    _pad0: f32,
    _pad1: f32,
}
@group(0) @binding(15) var foam_map_tex: texture_2d<f32>;
@group(0) @binding(16) var foam_surface_tex: texture_2d<f32>;
@group(0) @binding(17) var<uniform> foam_map: FoamMapParams;
// Flow-map coordinates: (phase A xy, phase B xy) = where the foam at this
// map texel was when that phase restarted (the lace pattern rides the flow)
@group(0) @binding(18) var foam_coords_tex: texture_2d<f32>;

@group(0) @binding(11) var<uniform> container: ContainerGeometry;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @builtin(front_facing) front_facing: bool,
}

const PI: f32 = 3.14159265359;
// Ceiling on written radiance (far above display white, well inside f16)
const HDR_OUTPUT_MAX: f32 = 4096.0;

// === PBR: GGX/Cook-Torrance BRDF ===

// GGX (Trowbridge-Reitz) Normal Distribution Function
fn D_GGX(NdotH: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Schlick-GGX geometry term (one direction)
fn G_SchlickGGX(NdotX: f32, k: f32) -> f32 {
    return NdotX / (NdotX * (1.0 - k) + k);
}

// Smith's method: combined geometry for both view and light directions
fn G_Smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return G_SchlickGGX(NdotV, k) * G_SchlickGGX(NdotL, k);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let vertex = vertices[vertex_index];

    var output: VertexOutput;
    output.world_position = vertex.position;
    output.world_normal = vertex.normal;

    let world_pos = vec4<f32>(vertex.position, 1.0);
    let view_pos = camera.view * world_pos;
    output.clip_position = camera.projection * view_pos;

    return output;
}

// Sample equirectangular environment map. Same convention as the CPU SH
// projection (compute_sh_irradiance in environment.rs): row 0 = +Y, and
// u = phi / 2pi for dir = (sin t cos phi, cos t, sin t sin phi). (The old
// mapping, v = 1 - t/pi with u offset by pi, sampled the antipode -dir.)
fn sample_environment(dir: vec3<f32>) -> vec3<f32> {
    let phi = atan2(dir.z, dir.x);
    let theta = acos(clamp(dir.y, -1.0, 1.0));
    let u = fract(phi / (2.0 * PI) + 1.0);
    let v = theta / PI;
    // Explicit LOD 0 (the map has a single mip) so refraction can call this
    // from per-pixel branches
    return textureSampleLevel(env_tex, env_sampler, vec2<f32>(u, v), 0.0).rgb;
}

// Evaluate order-2 spherical harmonics irradiance
// Coefficients are pre-convolved with cosine lobe on CPU
fn evaluate_sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    // Band 0 (constant)
    var irradiance = sh_coeffs[0].rgb * 0.282095;
    // Band 1 (linear)
    irradiance += sh_coeffs[1].rgb * 0.488603 * n.y;
    irradiance += sh_coeffs[2].rgb * 0.488603 * n.z;
    irradiance += sh_coeffs[3].rgb * 0.488603 * n.x;
    // Band 2 (quadratic)
    irradiance += sh_coeffs[4].rgb * 1.092548 * n.x * n.y;
    irradiance += sh_coeffs[5].rgb * 1.092548 * n.y * n.z;
    irradiance += sh_coeffs[6].rgb * 0.315392 * (3.0 * n.z * n.z - 1.0);
    irradiance += sh_coeffs[7].rgb * 1.092548 * n.x * n.z;
    irradiance += sh_coeffs[8].rgb * 0.546274 * (n.x * n.x - n.y * n.y);
    return max(irradiance, vec3<f32>(0.0));
}

// Linearize depth from depth buffer (reverse-Z or standard)
fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

// === Physical water medium (keep in sync: mc_render.wgsl / ss_composite.wgsl) ===
// Single scattering in a homogeneous medium along the in-water view path:
//   interior = background * exp(-sigma_t d) + in-scattered sun + sky
// Absorption is pure water at representative R/G/B wavelengths (Pope & Fry
// 1997, ~620 / 550 / 460 nm, per metre). Scattering (turbidity) comes from
// the Clarity slider, its spectral shape from the water color. The body
// color is not set by hand: it emerges as (sigma_s / sigma_t) x light x phase.
const WATER_ABSORPTION: vec3<f32> = vec3<f32>(0.30, 0.055, 0.015);
// Clarity 0 -> 3 /m (murky), 1 -> 0.02 /m (very clear pool), log-mapped;
// the 0.65 default is ~0.1 /m, a real swimming pool
const TURBIDITY_MAX: f32 = 3.0;
const TURBIDITY_MIN: f32 = 0.02;
// Particle scattering is forward-peaked (Henyey-Greenstein g), with a small
// isotropic lobe so skylight and the sun still backscatter a little
const PHASE_G: f32 = 0.85;
const PHASE_ISOTROPIC: f32 = 0.2;
// Diffuse skylight under water: transmission through the surface and the
// mean cosine of the downwelling light field
const SKY_TRANSMISSION: f32 = 0.93;
const SKY_MEAN_COSINE: f32 = 0.8;

struct Medium {
    transmittance: vec3<f32>,
    inscatter: vec3<f32>,
}

fn medium_scattering() -> vec3<f32> {
    let turbidity = TURBIDITY_MAX * pow(TURBIDITY_MIN / TURBIDITY_MAX, water.clarity);
    let c = water.water_color;
    return turbidity * c / max(max(c.r, c.g), max(c.b, 1e-4));
}

fn medium_phase(cos_theta: f32) -> f32 {
    let g2 = PHASE_G * PHASE_G;
    let hg = (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * PHASE_G * cos_theta, 1.5));
    return mix(hg, 1.0 / (4.0 * PI), PHASE_ISOTROPIC);
}

// `d`: path length in water along the view ray. `view_in`: refracted view
// direction inside the water (travelling away from the camera). `sun_rgb`:
// sun irradiance (zero when off or shadowed). `sky_down`: sky irradiance on a
// horizontal surface. Light enters through the mean (flat) surface and the
// path is taken to start at the surface (side faces start deeper: their sun
// in-scatter is overestimated). In-scattered radiance leaves the water
// scaled by 1/n^2; the exit Fresnel is applied by the caller's mix.
fn water_medium(d: f32, view_in: vec3<f32>, sun_dir: vec3<f32>, sun_rgb: vec3<f32>, sky_down: vec3<f32>) -> Medium {
    let sigma_s = medium_scattering();
    let sigma_t = WATER_ABSORPTION + sigma_s;
    var m: Medium;
    m.transmittance = exp(-sigma_t * d);
    // Depth gained per unit of path (0 for a horizontal ray)
    let cos_v = max(-view_in.y, 0.0);

    // Skylight: diffuse downwelling field, dimming with depth
    let k_sky = sigma_t * (1.0 + cos_v / SKY_MEAN_COSINE);
    let sky_scalar = sky_down * (SKY_TRANSMISSION / SKY_MEAN_COSINE);
    var inscatter = sigma_s * sky_scalar / (4.0 * PI) * (1.0 - exp(-k_sky * d)) / k_sky;

    // Sun: a refracted beam, dimming along its own (steeper) path with depth
    if (sun_dir.y > 0.0) {
        let s_in = refract(-sun_dir, vec3<f32>(0.0, 1.0, 0.0), 1.0 / water.ior);
        let cos_s = max(-s_in.y, 0.05);
        let f0 = pow((water.ior - 1.0) / (water.ior + 1.0), 2.0);
        let entry = 1.0 - (f0 + (1.0 - f0) * pow(1.0 - sun_dir.y, 5.0));
        // Irradiance across the beam: refraction narrows it by cos_L / cos_s
        let beam = sun_rgb * entry * sun_dir.y / cos_s;
        let k_sun = sigma_t * (1.0 + cos_v / cos_s);
        inscatter += sigma_s * medium_phase(dot(s_in, -view_in)) * beam * (1.0 - exp(-k_sun * d)) / k_sun;
    }
    m.inscatter = inscatter / (water.ior * water.ior);
    return m;
}

// === Physical screen-space refraction ===
// Raw depth at or past this is the environment backdrop (drawn at 0.9999) or
// nothing at all: whatever lies behind is infinitely far.
const BACKDROP_DEPTH: f32 = 0.9998;

fn texel_at(uv: vec2<f32>, dims: vec2<u32>) -> vec2<i32> {
    return clamp(vec2<i32>(uv * vec2<f32>(dims)), vec2<i32>(0), vec2<i32>(dims) - 1);
}

fn background_depth_at(uv: vec2<f32>) -> f32 {
    return textureLoad(background_depth_tex, texel_at(uv, textureDimensions(background_depth_tex)), 0);
}

// World point seen at screen uv with raw hardware depth. Goes through the real
// inverse matrices, so it holds whatever depth convention the projection uses.
fn screen_to_world(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let view = camera.inv_projection * ndc;
    return (camera.inv_view * vec4<f32>(view.xyz / view.w, 1.0)).xyz;
}

// Background at a refracted uv. If what sits there is in front of the water
// (e.g. the pool's near wall), the refracted ray can't have reached it — keep
// the straight-through sample instead of leaking the occluder into the water.
fn background_at(uv: vec2<f32>, front_depth_raw: f32, straight: vec3<f32>) -> vec3<f32> {
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))
        || background_depth_at(uv) < front_depth_raw) {
        return straight;
    }
    return textureSampleLevel(background_tex, env_sampler, uv, 0.0).rgb;
}

// Radiance from infinitely far along a world direction: the background texture
// where that direction lands on screen (matches the displayed backdrop exactly),
// else the backdrop evaluated directly.
fn backdrop_along(dir: vec3<f32>) -> vec3<f32> {
    let clip = camera.projection * camera.view * vec4<f32>(dir, 0.0);
    if (clip.w > 1e-4) {
        let ndc = clip.xy / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0))) {
            return textureSampleLevel(background_tex, env_sampler, uv, 0.0).rgb;
        }
    }
    if (water.use_env_background == 0u) {
        return vec3<f32>(water.background_r, water.background_g, water.background_b);
    }
    // Same radiance as the backdrop pass (mc_environment.wgsl)
    return max(sample_environment(dir) * water.env_intensity, vec3<f32>(0.0));
}

// March a ray from `origin` (in or leaving the water) until it passes
// behind the background depth buffer, then bisect to the crossing. Linear
// steps reach several times the straight-line distance to the surface first
// seen at `uv0`: a refracted ray skimming a floor lands far behind that first
// guess, which a fixed-point iteration overshoots toward the horizon.
const MARCH_STEPS: i32 = 20;
const MARCH_REFINE: i32 = 5;
const MARCH_REACH: f32 = 4.0;

// Screen uv (y down) and raw depth of a world point
fn screen_point(p: vec3<f32>) -> vec3<f32> {
    let clip = camera.projection * camera.view * vec4<f32>(p, 1.0);
    let w = max(clip.w, 1e-4);
    return vec3<f32>(clip.x / w * 0.5 + 0.5, 0.5 - clip.y / w * 0.5, clip.z / w);
}

fn march_to_background(origin: vec3<f32>, dir: vec3<f32>, uv0: vec2<f32>, depth0: f32) -> vec2<f32> {
    let reach = MARCH_REACH * distance(origin, screen_to_world(uv0, depth0)) + 0.1;
    var lo = 0.0;
    var hi = -1.0;
    for (var k = 1; k <= MARCH_STEPS; k++) {
        let s = reach * f32(k) / f32(MARCH_STEPS);
        let q = screen_point(origin + dir * s);
        if (any(q.xy < vec2<f32>(0.0)) || any(q.xy > vec2<f32>(1.0))) {
            break;  // left the screen
        }
        if (q.z >= background_depth_at(q.xy)) {
            hi = s;
            break;
        }
        lo = s;
    }
    if (hi < 0.0) {
        // No crossing: the ray left the screen, or rose toward the water
        // surface from below (side faces near the rounded top edge), where it
        // would reflect back down rather than reach anything far away. The
        // surface first seen is the safe answer; the furthest point reached
        // would paint the horizon into the water.
        return uv0;
    }
    for (var k = 0; k < MARCH_REFINE; k++) {
        let mid = 0.5 * (lo + hi);
        let q = screen_point(origin + dir * mid);
        if (q.z >= background_depth_at(q.xy)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return screen_point(origin + dir * hi).xy;
}

// Two-interface image-space refraction (after Wyman 2005). Snell-refract the
// view ray at the front surface. If an opaque surface (floor, wall, body) sits
// inside the water behind this pixel, the ray ends on it — this is the
// apparent-depth shift that makes a pool look shallower than it is. Otherwise
// the ray crosses the body (front-to-back distance), refracts out through the
// back-face normal where it lands, and we look up what the exit ray reaches.
// Curved bodies (drops, crests) bend the exit ray: they act as lenses.
// Returns the radiance arriving from behind, before absorption.
fn refract_scene(
    p: vec3<f32>,
    n: vec3<f32>,
    view_dir: vec3<f32>,
    screen_uv: vec2<f32>,
    front_depth_raw: f32,
    back_depth_raw: f32,
) -> vec3<f32> {
    let straight = textureSampleLevel(background_tex, env_sampler, screen_uv, 0.0).rgb;
    // n faces the camera; entering the denser medium never totally reflects
    let t1 = refract(-view_dir, n, 1.0 / water.ior);
    let bg_depth = background_depth_at(screen_uv);

    // Opaque surface inside the water: the ray ends on it
    if (bg_depth < BACKDROP_DEPTH && bg_depth <= back_depth_raw) {
        return background_at(march_to_background(p, t1, screen_uv, bg_depth), front_depth_raw, straight);
    }
    // No back face behind this pixel (mesh clipped open): treat the body as
    // deep and let the refracted ray run out to the backdrop
    if (back_depth_raw >= 1.0) {
        return backdrop_along(t1);
    }

    // Cross the body to the back face, refract out through its normal there
    let p_exit = p + t1 * distance(p, screen_to_world(screen_uv, back_depth_raw));
    let exit_uv = screen_point(p_exit).xy;
    var back_n = textureLoad(back_normal_tex, texel_at(exit_uv, textureDimensions(back_normal_tex)), 0);
    if (back_n.w < 0.5) {
        // Landed outside the body's silhouette: use the exit face behind this pixel
        back_n = textureLoad(back_normal_tex, texel_at(screen_uv, textureDimensions(back_normal_tex)), 0);
    }
    let n_exit = normalize(back_n.xyz);
    let t2 = refract(t1, -n_exit, water.ior);
    if (dot(t2, t2) < 0.5) {
        // Total internal reflection (typically a flat ray that came in through a
        // side face meeting the bottom past the 48.6 deg critical angle). In a
        // slab the ray then zig-zags between the parallel top and bottom like a
        // light guide and leaves through the far side close to its original
        // heading, so straight-through is the right picture — following the
        // single mirrored bounce instead paints the sky over the whole face.
        return straight;
    }

    // What the exit ray reaches: the surface behind the exit point, if any,
    // else the backdrop far along it
    let bg_exit_depth = background_depth_at(exit_uv);
    if (bg_exit_depth < BACKDROP_DEPTH) {
        return background_at(march_to_background(p_exit, t2, exit_uv, bg_exit_depth), front_depth_raw, straight);
    }
    return backdrop_along(t2);
}

// === Foam field compositing ===
// Density maps to coverage through an asymptotic curve (1 - exp(-k*d)) rather
// than a clipped smoothstep window: a hard window turns any dense carpet into
// a single flat white slab (every pixel past the top is identical), which is
// exactly the "soap" look. With the asymptote, density differences stay
// visible at any thickness, and in-slab variation comes from albedo texture
// noise, which never saturates.
const FOAM_DENSITY_LO: f32 = 0.07;
const FOAM_COVERAGE_K: f32 = 1.1;
// World-anchored value-noise. Coarse octave carves patch edges into lacework
// (coverage, dies at saturation by design); both octaves drive the in-slab
// brightness texture (bubble clusters vs interstices, survives saturation).
const FOAM_NOISE_SCALE: f32 = 50.0;
const FOAM_NOISE_SCALE_FINE: f32 = 187.0;
const FOAM_NOISE_BREAKUP: f32 = 0.9;
const FOAM_TEX_CONTRAST: f32 = 0.22;
const FOAM_TEX_CONTRAST_FINE: f32 = 0.13;
// Thin foam reads as a translucent gray-blue veil over the water; thick foam
// dries toward bright white (ramped on coverage).
const FOAM_ALBEDO: vec3<f32> = vec3<f32>(0.34, 0.36, 0.37);
const FOAM_VEIL_ALBEDO: vec3<f32> = vec3<f32>(0.22, 0.26, 0.29);
const FOAM_THICK_LO: f32 = 0.45;
const FOAM_THICK_HI: f32 = 0.85;
// Aeration (G channel): entrained-air milkiness inside the water volume —
// vortex cores, plunge plumes. Mixed into the refraction path only, so
// reflections and specular survive on top. Slightly bluer than surface foam.
const AERATION_K: f32 = 0.15;
const AERATION_ALBEDO: vec3<f32> = vec3<f32>(0.22, 0.27, 0.31);

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

// Raft mask at one flow-map position: covered where the distance to the lace
// network's walls is under a threshold that grows with coverage (thick foam:
// everything; thinning: holes open from the cell centres; last: strings).
// Bursting is uneven, so the local coverage is jittered at the patch scale:
// some areas hold a raft while neighbours are already down to strings.
// `px`: pixel footprint (m) for antialiasing / distance fade.
fn raft_mask(p: vec2<f32>, coverage: f32, px: f32) -> f32 {
    let patch_noise = value_noise_grad(p * LACE_PATCH_FREQ).x;
    let cov = clamp(coverage * (0.45 + 1.1 * patch_noise), 0.0, 1.0);
    let coarse = worley(p / LACE_CELL).y;
    let fine = worley(p / LACE_CELL_FINE + vec2<f32>(5.3, 1.7)).y;
    // The fine network only subdivides holes while the raft is still dense:
    // thin foam is a few coarse strings, not a uniform net
    let fine_weight = mix(3.5, 1.4, smoothstep(0.3, 0.8, cov));
    let wall = min(coarse, fine * fine_weight);
    let threshold = -log(max(1.0 - cov * 0.985, 1e-3)) * 0.22;
    let soft = max(px / LACE_CELL_FINE * 1.5, 0.03);
    var mask = 1.0 - smoothstep(threshold - soft, threshold + soft, wall);
    // Thin lace is broken, not a connected net: strings snap into fragments
    // as the foam thins (gate a string-scale noise by coverage)
    let snap = value_noise_grad(p * LACE_SNAP_FREQ + vec2<f32>(3.1, 7.9)).x;
    let keep = clamp(cov * 1.8, 0.0, 1.0);
    mask *= smoothstep(1.0 - keep - 0.12, 1.0 - keep + 0.12, snap);
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

// GPU-friendly hash → pseudo-random [0,1]
fn hash2(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smooth value noise with analytic gradient (returns: vec3(noise, dN/dx, dN/dz))
fn value_noise_grad(p: vec2<f32>) -> vec3<f32> {
    let i = floor(p);
    let f = fract(p);
    // Quintic Hermite interpolation (C2 continuous — no grid artifacts)
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
// Each octave doubles frequency and halves amplitude (fBm).
fn ripple_normal(world_pos: vec3<f32>, t: f32) -> vec3<f32> {
    var grad = vec2<f32>(0.0);
    var amp = 1.0;
    var freq = 10.0;

    // 4 octaves at different time offsets to avoid coherent drift
    for (var oct = 0u; oct < 4u; oct++) {
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
        amp *= 0.5;
    }

    return vec3<f32>(grad.x, 0.0, grad.y);
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Clip to container bounds with margin (MC interpolation can place vertices
    // slightly outside the container; clip_margin ≈ 1.5× MC cell_size).
    if (container.clip_enabled != 0u) {
        let local = world_to_local(container, input.world_position);
        if (!is_inside_box(container, local, container.clip_margin)) {
            discard;
        }
    }

    let view_dir = normalize(camera.camera_pos - input.world_position);

    // Get the surface normal - ensure it faces toward the camera
    var normal = normalize(input.world_normal);
    // If normal points away from camera, flip it (ensures correct reflection)
    if (dot(normal, view_dir) < 0.0) {
        normal = -normal;
    }

    if (water.physical_refraction != 0.0) { let lp = world_to_local(container, input.world_position); return vec4<f32>(0.5 + 2.0 * normal.y, 0.5 + 2.0 * view_dir.y, 0.5 + (lp.z - container.half_depth) * 10.0, 1.0); } // DEBUG
    // Mesh normal (camera-facing) before ripples: the foam map's top test
    let surface_up = world_dir_to_local(container, normal).y;
    // Pixel footprint on the surface (m), for the foam lace's antialiasing.
    // Taken here, in uniform control flow, where derivatives are valid.
    let local_pos = world_to_local(container, input.world_position);
    let foam_px = max(length(dpdx(local_pos.xz)), length(dpdy(local_pos.xz)));

    // Micro-ripple perturbation: adds small-scale surface detail the MC mesh can't capture.
    let ripple_grad = ripple_normal(input.world_position, water.time);
    normal = normalize(normal + ripple_grad * water.ripple_strength);

    // === THICKNESS CALCULATION ===
    // Sample back face depth at this screen position
    let screen_size = vec2<f32>(textureDimensions(back_depth_tex));
    let screen_uv = input.clip_position.xy / screen_size;
    let back_depth_raw = textureSample(back_depth_tex, depth_sampler, screen_uv);
    let front_depth_raw = input.clip_position.z;

    // Convert to linear depth using actual camera near/far planes
    let front_linear = linearize_depth(front_depth_raw, camera.near, camera.far);
    let back_linear = linearize_depth(back_depth_raw, camera.near, camera.far);

    // Thickness in world units (clamped to reasonable range). Legacy: this
    // D3D-style linearization runs on a GL-style projection, so it reads
    // ~0.5x the true distance; the legacy look was tuned on it, so it stays.
    var thickness = max(0.0, back_linear - front_linear);
    thickness = min(thickness, 5.0);  // Cap at 5 units

    // True in-water path for the physical medium: to the back face, or to an
    // opaque surface inside the water (floor, wall, body) if that is nearer
    let path_end = min(back_depth_raw, background_depth_at(screen_uv));
    var path_length = 10.0;  // nothing behind: treat as deep
    if (path_end < BACKDROP_DEPTH) {
        path_length = min(distance(input.world_position, screen_to_world(screen_uv, path_end)), 10.0);
    }

    // === ABSORPTION (Beer's Law) ===
    // Light attenuates exponentially through water
    // Different wavelengths absorb at different rates (red absorbs fastest)
    // Clarity controls optical density: 0 = murky (dense), 1 = crystal clear (sparse)
    let absorption_coeffs = vec3<f32>(0.30, 0.08, 0.02);  // RGB absorption rates
    let optical_density = (1.0 - water.clarity) * 2.5 + 0.05;
    let transmittance = exp(-absorption_coeffs * optical_density * thickness);

    // Reflection — solid color or HDR environment
    let reflect_dir = reflect(-view_dir, normal);
    let roughness_sq = water.roughness * water.roughness;
    var reflection_color: vec3<f32>;
    var below_horizon = 0.0;
    if (water.use_env_background == 0u) {
        reflection_color = vec3<f32>(water.background_r, water.background_g, water.background_b);
    } else {
        // Roughness-blurred environment reflection:
        // Sharp env sample at roughness=0 (mirror), SH irradiance at roughness=1 (fully diffuse).
        // Squared roughness maps perceptual roughness to GGX lobe width more naturally.
        let sharp_env = sample_environment(reflect_dir) * water.env_intensity;
        let diffuse_env = evaluate_sh_irradiance(reflect_dir) * water.env_intensity;
        var env_reflection = mix(sharp_env, diffuse_env, roughness_sq);

        // Fade env reflection when reflect direction points below horizon.
        // The env map is a distant panorama — it can't represent nearby scene
        // geometry (walls, floor, other water). Downward reflections would show
        // the far-off ground of the HDRI instead.
        // SSR handles these directions; without SSR, the faded share is filled
        // with the water's own body color once it's known (below).
        let horizon_fade = smoothstep(-0.15, 0.1, reflect_dir.y);
        reflection_color = env_reflection * horizon_fade;
        below_horizon = 1.0 - horizon_fade;
    }

    // Screen-space reflections — blend with env map based on SSR confidence
    // Reduce SSR contribution for rough surfaces (sharp reflections look wrong on rough water)
    let ssr_dims = textureDimensions(ssr_tex);
    let ssr_coord = vec2<i32>(screen_uv * vec2<f32>(f32(ssr_dims.x), f32(ssr_dims.y)));
    let ssr_sample = textureLoad(ssr_tex, ssr_coord, 0);
    let ssr_confidence = ssr_sample.a * (1.0 - roughness_sq);
    reflection_color = mix(reflection_color, ssr_sample.rgb, ssr_confidence);

    // === SCREEN-SPACE REFRACTION ===
    var refracted_background: vec3<f32>;
    if (water.physical_refraction != 0.0) {
        refracted_background = refract_scene(
            input.world_position, normal, view_dir, screen_uv, front_depth_raw, back_depth_raw,
        );
    } else {
        // Legacy: offset by the normal's deviation from flat up (a flat surface
        // shows no shift at all), scaled by the Refraction slider
        let flat_normal_view = normalize((camera.view * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz);
        let normal_view = normalize((camera.view * vec4<f32>(normal, 0.0)).xyz);
        let normal_deviation = normal_view - flat_normal_view;

        let refract_strength = water.refraction_strength * (1.0 + thickness * 0.5);
        let uv_offset = normal_deviation.xy * refract_strength;

        // Sample background with distorted UVs (clamp to avoid sampling outside)
        let refract_uv = clamp(screen_uv + uv_offset, vec2<f32>(0.001), vec2<f32>(0.999));
        refracted_background = textureSampleLevel(background_tex, env_sampler, refract_uv, 0.0).rgb;
    }

    let refracted_scene = refracted_background;
    // Apply absorption to refracted light (Beer-Lambert)
    refracted_background = refracted_background * transmittance;

    // Deep water color (what you see when looking deep)
    let deep_color = vec3<f32>(water.deep_color_r, water.deep_color_g, water.deep_color_b);

    // Blend between refracted background and deep water based on thickness
    // Clarity scales the depth blend rate — clearer water shows background longer
    let depth_blend = 1.0 - exp(-thickness * optical_density * 0.5);
    let water_interior = mix(refracted_background, deep_color, depth_blend);

    // Add water's own color contribution (subsurface scattering approximation)
    let scatter_strength = 0.12 * (1.0 - exp(-thickness * optical_density * 1.2));
    let scatter_color = water.water_color * scatter_strength;
    let interior_with_scatter = water_interior + scatter_color * (1.0 - transmittance);

    // Fresnel (Schlick approximation) - controls reflection vs transmission
    // F0 = ((n1 - n2) / (n1 + n2))^2 where n1=1.0 (air), n2=IOR (water)
    let cos_theta = max(0.0, dot(normal, view_dir));
    let F0 = pow((water.ior - 1.0) / (water.ior + 1.0), 2.0);  // ~0.02 for water
    var fresnel = clamp(F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0), 0.0, 1.0);

    // Total internal reflection — at extreme grazing angles, all light reflects
    let sin_theta_sq = 1.0 - cos_theta * cos_theta;
    let sin_refracted_sq = sin_theta_sq / (water.ior * water.ior);
    if (sin_refracted_sq > 1.0) {
        fresnel = 1.0;
    }

    // === DIRECTIONAL LIGHT (SUN) ===
    // Analytic rim shadow: pool walls block direct sun on the water surface,
    // matching the floor's rim shadowing and the caustics light raster (which
    // draws the container as an occluder). 1.0 outside pool mode.
    let sun_dir_ws = normalize(light.sun_direction);
    let rim_vis = rim_visibility(
        container,
        world_to_local(container, input.world_position),
        world_dir_to_local(container, sun_dir_ws),
    );

    var sun_specular = vec3<f32>(0.0);
    var sun_subsurface = vec3<f32>(0.0);
    if (light.sun_enabled == 1u) {
        let light_dir = sun_dir_ws;
        let NdotL = max(0.0, dot(normal, light_dir));
        let NdotV = max(dot(normal, view_dir), 0.001);

        // Cook-Torrance specular BRDF (GGX distribution)
        let alpha = water.roughness * water.roughness;
        let half_vec = normalize(light_dir + view_dir);
        let NdotH = max(dot(normal, half_vec), 0.0);
        let HdotV = max(dot(half_vec, view_dir), 0.0);

        let D = D_GGX(NdotH, alpha);
        let G = G_Smith(NdotV, max(NdotL, 0.001), water.roughness);
        // Fresnel at half-vector angle (physically correct for microfacet model)
        let F_spec = F0 + (1.0 - F0) * pow(1.0 - HdotV, 5.0);

        let denom = 4.0 * NdotV * max(NdotL, 0.001);
        let specular_brdf = (D * G * F_spec) / max(denom, 0.001);

        sun_specular = light.sun_color * light.sun_intensity * specular_brdf * NdotL * rim_vis;

        // Subsurface illumination — light enters water, scatters, exits toward viewer.
        // Driven by the mean (flat) surface, not the facet: refraction squeezes all
        // transmitted light into the ~49 deg Snell cone and it travels far past the
        // wave scale before scattering back, so body radiance is volumetric. A facet
        // NdotL here is a Lambert lobe — Lambert + sharp GGX is the CG plastic look.
        let light_entering = max(sun_dir_ws.y, 0.0) * (1.0 - F_spec);
        let interior_glow = water.water_color * transmittance;
        sun_subsurface = interior_glow * light_entering * light.sun_color * light.sun_intensity * 0.18;

        // Forward scattering — thin areas glow when backlit (translucency)
        let VdotL = max(0.0, dot(-view_dir, light_dir));
        let forward_scatter = pow(VdotL, 4.0) * exp(-thickness * optical_density * 1.5);
        sun_subsurface += water.water_color * forward_scatter * light.sun_color * light.sun_intensity * 0.10;

        // Both subsurface paths are fed by direct sun at this surface point
        sun_subsurface *= rim_vis;
    }

    // IBL diffuse irradiance from spherical harmonics
    // Light enters the water (1-F), travels through the volume (transmittance),
    // and scatters back (scatter_strength) — same physics as subsurface scattering,
    // so it also sees the mean surface (sky irradiance onto a flat water plane)
    let ambient_irradiance = evaluate_sh_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * water.env_intensity;
    let ambient_subsurface = ambient_irradiance * water.water_color * transmittance * scatter_strength * 0.6;

    // Add sun subsurface (weighted by 1-fresnel for energy conservation) and ambient irradiance
    var lit_interior = interior_with_scatter
        + sun_subsurface * (1.0 - fresnel)
        + ambient_subsurface * (1.0 - fresnel);
    if (water.physical_medium != 0.0) {
        var sun_rgb = vec3<f32>(0.0);
        if (light.sun_enabled == 1u) {
            sun_rgb = light.sun_color * light.sun_intensity * rim_vis;
        }
        let medium = water_medium(
            path_length,
            refract(-view_dir, normal, 1.0 / water.ior),
            sun_dir_ws,
            sun_rgb,
            evaluate_sh_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * water.env_intensity,
        );
        lit_interior = refracted_scene * medium.transmittance + medium.inscatter;
    }

    // === AERATION (submerged whitewater) ===
    // Entrained-air density along this ray (G channel of the whitewater
    // field): mix the interior toward a lit milky tone. White IN the water,
    // as opposed to the surface foam composited after the Fresnel combine.
    let whitewater_field = textureSampleLevel(foam_density_tex, env_sampler, screen_uv, 0.0).rg;
    let aeration = 1.0 - exp(-AERATION_K * water.aeration_strength * whitewater_field.g);
    if (aeration > 0.002) {
        // Bubble clouds sit in the volume: lit through the mean surface like
        // the body light above, not by the facet they're seen through
        var aeration_light = evaluate_sh_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * water.env_intensity;
        if (light.sun_enabled == 1u) {
            aeration_light += light.sun_color * light.sun_intensity
                * max(sun_dir_ws.y, 0.0) * 0.6 * rim_vis;
        }
        lit_interior = mix(lit_interior, AERATION_ALBEDO * aeration_light, aeration);
    }

    // Below-horizon reflection rays mostly hit more water, so the faded env
    // share takes the water's own color. (Black painted dark creases on every
    // wave back at grazing Fresnel; a blurred-env fallback overshoots into white
    // creases since the sky's lower hemisphere is bright haze.)
    reflection_color += lit_interior * below_horizon * (1.0 - ssr_confidence);

    // Combine reflection and refraction based on Fresnel
    // At grazing angles (high fresnel): more reflection
    // Looking straight on (low fresnel): more refraction/transmission
    var color = mix(lit_interior, reflection_color, fresnel);

    // Add specular on top (pure surface reflection, independent of interior)
    color += sun_specular;

    // === FOAM OVERLAY ===
    // Screen-space foam density (splatted half-res by the spray system):
    // whiten the surface where foam accumulates. Foam is rough and diffuse,
    // so it replaces the specular water response rather than adding to it.
    let foam_density = whitewater_field.r;
    if (foam_density > 0.01) {
        let n_coarse = value_noise_grad(input.world_position.xz * FOAM_NOISE_SCALE).x;
        let n_fine = value_noise_grad(
            input.world_position.xz * FOAM_NOISE_SCALE_FINE + vec2<f32>(37.42, 11.18),
        ).x;
        // Coarse noise raggedizes patch edges into stringy breakup
        let breakup = (n_coarse - 0.5) * FOAM_NOISE_BREAKUP;
        let d_eff = max(foam_density * (1.0 + breakup) - FOAM_DENSITY_LO, 0.0);
        let coverage = 1.0 - exp(-FOAM_COVERAGE_K * water.foam_coverage * d_eff);
        if (coverage > 0.002) {
            var foam_light = evaluate_sh_irradiance(normal) * water.env_intensity;
            if (light.sun_enabled == 1u) {
                foam_light += light.sun_color * light.sun_intensity
                    * max(dot(normal, sun_dir_ws), 0.0) * rim_vis;
            }
            // Thin veil -> dry white crest, plus saturation-proof brightness
            // texture so thick carpets keep internal structure
            let thick = smoothstep(FOAM_THICK_LO, FOAM_THICK_HI, coverage);
            let albedo = mix(FOAM_VEIL_ALBEDO, FOAM_ALBEDO, thick);
            let tex = 1.0 + (n_coarse - 0.5) * FOAM_TEX_CONTRAST
                + (n_fine - 0.5) * FOAM_TEX_CONTRAST_FINE;
            let foam_color = albedo * tex * foam_light;
            color = mix(color, foam_color, coverage);
        }
    }

    // Surface foam from the map: a bubble raft whose lace pattern rides the
    // flow (two flow-map phases crossfaded so each restart is invisible)
    let map = sample_map_foam(local_pos, surface_up);
    if (map.density > 0.01 && map.on_top > 0.0) {
        let raft_cover = 1.0 - exp(-RAFT_COVERAGE_K * water.foam_coverage * max(map.density - RAFT_DENSITY_LO, 0.0));
        let w_a = 1.0 - abs(2.0 * foam_map.flow_phase - 1.0);
        let mask = w_a * raft_mask(map.coords.xy, raft_cover, foam_px)
            + (1.0 - w_a) * raft_mask(map.coords.zw, raft_cover, foam_px);
        let grain = w_a * bubble_grain(map.coords.xy, foam_px)
            + (1.0 - w_a) * bubble_grain(map.coords.zw, foam_px);
        let raft_alpha = mix(RAFT_ALPHA_THIN, RAFT_ALPHA_THICK, 1.0 - exp(-RAFT_ALPHA_K * map.density));
        let alpha = mask * raft_alpha * map.on_top;
        if (alpha > 0.002) {
            var raft_light = evaluate_sh_irradiance(normal) * water.env_intensity;
            if (light.sun_enabled == 1u) {
                raft_light += light.sun_color * light.sun_intensity
                    * max(dot(normal, sun_dir_ws), 0.0) * rim_vis;
            }
            let albedo = mix(FOAM_VEIL_ALBEDO, FOAM_ALBEDO, smoothstep(0.3, 2.0, map.density));
            color = mix(color, albedo * grain * raft_light, alpha);
        }
    }

    // Output linear HDR — post-process pipeline handles tone mapping + gamma.
    // Bounded: a grazing sun glint off a near-mirror surface can exceed the
    // f16 scene buffer's range
    return vec4<f32>(min(color, vec3<f32>(HDR_OUTPUT_MAX)), 1.0);
}
