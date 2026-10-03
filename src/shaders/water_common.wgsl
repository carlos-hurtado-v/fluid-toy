// Shared by the two water shaders: prepended to the marching-cubes water
// shader (mc_render/) and to ss_composite.wgsl, so both modes shade water
// with one copy of the BRDF, the physical medium and the whitewater
// calibration.
//
// Needs from the including shader: `const PI` and the `water` uniform
// (WaterParams: ior, clarity, water_color).

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

// === Physical water medium ===
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

// === Whitewater field compositing (same field, same calibration: the two
// modes must read foam identically) ===
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
