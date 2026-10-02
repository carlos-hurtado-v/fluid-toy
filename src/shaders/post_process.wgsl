// Post-processing shader
// Applies various effects to the rendered scene

struct PostProcessParams {
    // Exposure
    exposure: f32,

    // Color grading
    saturation: f32,
    contrast: f32,
    brightness: f32,
    temperature: f32,

    // Vignette
    vignette_enabled: u32,
    vignette_intensity: f32,
    vignette_smoothness: f32,

    // Chromatic aberration
    chromatic_aberration_enabled: u32,
    chromatic_aberration_intensity: f32,

    // Bloom
    bloom_enabled: u32,
    bloom_intensity: f32,
    bloom_threshold: f32,

    // Tonemapping
    tonemapping_enabled: u32,

    // Anamorphic streaks
    streaks_enabled: u32,
    streaks_intensity: f32,
    streaks_threshold: f32,
    // Streak tint color (RGB)
    streaks_tint_r: f32,
    streaks_tint_g: f32,
    streaks_tint_b: f32,

    // Ambient Occlusion
    ao_enabled: u32,
    ao_debug_mode: u32,
    ao_intensity: f32,
    _padding: f32,
}

@group(0) @binding(0) var scene_texture: texture_2d<f32>;
@group(0) @binding(1) var bloom_texture: texture_2d<f32>;
@group(0) @binding(2) var streak_texture: texture_2d<f32>;
@group(0) @binding(3) var texture_sampler: sampler;
@group(0) @binding(4) var<uniform> params: PostProcessParams;

@group(1) @binding(0) var ao_texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Full-screen triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );

    let pos = positions[vertex_index];

    var output: VertexOutput;
    output.position = vec4<f32>(pos, 0.0, 1.0);
    // UV: flip Y for correct orientation
    output.uv = vec2<f32>(pos.x * 0.5 + 0.5, 1.0 - (pos.y * 0.5 + 0.5));
    return output;
}

// === Effect Functions ===

// Convert RGB to luminance
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Saturation adjustment
fn apply_saturation(color: vec3<f32>, saturation: f32) -> vec3<f32> {
    let luma = luminance(color);
    return mix(vec3<f32>(luma), color, saturation);
}

// Contrast adjustment (centered around 0.5)
fn apply_contrast(color: vec3<f32>, contrast: f32) -> vec3<f32> {
    return (color - 0.5) * contrast + 0.5;
}

// Temperature shift (blue <-> orange)
fn apply_temperature(color: vec3<f32>, temperature: f32) -> vec3<f32> {
    // Simple temperature adjustment
    // Positive = warmer (more red/yellow), Negative = cooler (more blue)
    let warm = vec3<f32>(1.0, 0.9, 0.7);
    let cool = vec3<f32>(0.7, 0.9, 1.0);

    if (temperature > 0.0) {
        return mix(color, color * warm, temperature);
    } else {
        return mix(color, color * cool, -temperature);
    }
}

// Vignette effect
fn apply_vignette(color: vec3<f32>, uv: vec2<f32>, intensity: f32, smoothness: f32) -> vec3<f32> {
    let center = vec2<f32>(0.5, 0.5);
    let dist = distance(uv, center);
    let vignette = smoothstep(0.8 - smoothness * 0.5, 1.2 - smoothness, dist * (1.0 + intensity));
    return color * (1.0 - vignette * intensity);
}

// Chromatic aberration
fn apply_chromatic_aberration(uv: vec2<f32>, intensity: f32) -> vec3<f32> {
    let center = vec2<f32>(0.5, 0.5);
    let dir = uv - center;

    let r = textureSample(scene_texture, texture_sampler, uv + dir * intensity).r;
    let g = textureSample(scene_texture, texture_sampler, uv).g;
    let b = textureSample(scene_texture, texture_sampler, uv - dir * intensity).b;

    return vec3<f32>(r, g, b);
}

fn sample_ao(uv: vec2<f32>) -> f32 {
    let ao_size = vec2<i32>(textureDimensions(ao_texture));
    let ao_coord = clamp(vec2<i32>(uv * vec2<f32>(ao_size)), vec2<i32>(0), ao_size - vec2<i32>(1));
    return clamp(textureLoad(ao_texture, ao_coord, 0).r, 0.0, 1.0);
}

// ACES Filmic Tonemapping (Stephen Hill's fit)
// More accurate than the simple Narkowicz approximation
// Includes proper sRGB -> ACES -> RRT+ODT -> sRGB transforms

// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
fn aces_input_matrix(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(color, vec3<f32>(0.59719, 0.35458, 0.04823)),
        dot(color, vec3<f32>(0.07600, 0.90834, 0.01566)),
        dot(color, vec3<f32>(0.02840, 0.13383, 0.83777))
    );
}

// ODT_SAT => XYZ => D60_2_D65 => sRGB
fn aces_output_matrix(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(color, vec3<f32>( 1.60475, -0.53108, -0.07367)),
        dot(color, vec3<f32>(-0.10208,  1.10813, -0.00605)),
        dot(color, vec3<f32>(-0.00327, -0.07276,  1.07602))
    );
}

// RRT and ODT fit
fn rrt_odt_fit(v: vec3<f32>) -> vec3<f32> {
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

// Full ACES fitted tonemapping
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    var c = aces_input_matrix(color);
    c = rrt_odt_fit(c);
    c = aces_output_matrix(c);
    return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec3<f32>;

    // Sample scene (with optional chromatic aberration)
    if (params.chromatic_aberration_enabled == 1u) {
        color = apply_chromatic_aberration(input.uv, params.chromatic_aberration_intensity);
    } else {
        color = textureSample(scene_texture, texture_sampler, input.uv).rgb;
    }

    let ao = sample_ao(input.uv);
    // max(ao, 1e-4) avoids pow(0, 0) which is undefined in WGSL.
    let ao_factor = max(pow(max(ao, 1e-4), params.ao_intensity), 0.05);

    // AO debug views bypass all other grading/tonemapping to inspect AO directly.
    if (params.ao_debug_mode == 1u) {
        return vec4<f32>(vec3<f32>(ao), 1.0);
    }
    if (params.ao_debug_mode == 2u) {
        return vec4<f32>(vec3<f32>(ao_factor), 1.0);
    }

    // Apply ambient occlusion
    if (params.ao_enabled == 1u) {
        // Floor at 0.05 prevents total blackout in tight concavities.
        color *= ao_factor;
    }

    // Add bloom if enabled
    if (params.bloom_enabled == 1u) {
        let bloom = textureSample(bloom_texture, texture_sampler, input.uv).rgb;
        color = color + bloom * params.bloom_intensity;
    }

    // Add anamorphic streaks if enabled
    if (params.streaks_enabled == 1u) {
        let streak = textureSample(streak_texture, texture_sampler, input.uv).rgb;
        let tint = vec3<f32>(params.streaks_tint_r, params.streaks_tint_g, params.streaks_tint_b);
        color = color + streak * tint * params.streaks_intensity;
    }

    // Apply exposure (in linear/HDR space)
    color = color * params.exposure;

    // Apply ACES tonemapping (HDR -> LDR conversion)
    // This should come after exposure/bloom, before color grading
    if (params.tonemapping_enabled == 1u) {
        color = aces_tonemap(color);
    }

    // Apply color grading (in LDR space)
    color = apply_saturation(color, params.saturation);
    color = apply_contrast(color, params.contrast);
    color = color + params.brightness;
    color = apply_temperature(color, params.temperature);

    // Apply vignette
    if (params.vignette_enabled == 1u) {
        color = apply_vignette(color, input.uv, params.vignette_intensity, params.vignette_smoothness);
    }

    // Final clamp
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

    return vec4<f32>(color, 1.0);
}

// === Bloom extraction shader ===
// Separate entry point for bloom threshold pass

// Scene luminance is unclipped HDR: a sun glint or the sun disk can sit
// thousands of times above white. Capping what feeds the blur keeps a single
// glint from flooding the frame (and f16 from overflowing in the blur).
const BLOOM_MAX_LUMINANCE: f32 = 32.0;

// Bright-pass with a soft knee (half the threshold wide): the part of each
// pixel above the threshold, so bloom grows smoothly with brightness instead
// of switching on whole surfaces
fn bright_pass(uv: vec2<f32>, threshold: f32) -> vec3<f32> {
    var color = textureSampleLevel(scene_texture, texture_sampler, uv, 0.0).rgb;
    let luma = luminance(color);
    if (luma > BLOOM_MAX_LUMINANCE) {
        color *= BLOOM_MAX_LUMINANCE / luma;
    }
    let l = min(luma, BLOOM_MAX_LUMINANCE);
    let knee = 0.5 * threshold;
    let soft = clamp(l - threshold + knee, 0.0, 2.0 * knee);
    let contribution = max(soft * soft / (4.0 * knee + 1e-4), l - threshold);
    return color * (contribution / max(l, 1e-4));
}

// Half-res target: the tap at the pixel centre averages its 2x2 scene pixels
@fragment
fn fs_bloom_threshold(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(bright_pass(input.uv, params.bloom_threshold), 1.0);
}

// Streak target is 1/16 x 1/4 of the scene. Every scene pixel under a texel
// has to be read (a single tap sees 2x2 of them: glints elsewhere went
// missing and streaks popped in and out as they moved): 8 bilinear taps
// across, and 4 rows with tent weights reaching half a texel into the
// neighbours above and below, so a streak slides between rows instead of
// jumping. Thresholded per tap, like the bloom source.
@fragment
fn fs_streak_threshold(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(scene_texture));
    var sum = vec3<f32>(0.0);
    for (var j = 0; j < 4; j++) {
        let row_weight = select(0.125, 0.375, j == 1 || j == 2);
        for (var i = 0; i < 8; i++) {
            let offset = vec2<f32>(f32(2 * i - 7), f32(2 * j - 3)) * texel;
            sum += bright_pass(input.uv + offset, params.streaks_threshold) * row_weight;
        }
    }
    return vec4<f32>(sum / 8.0, 1.0);
}

// === Blur shaders ===

struct BlurParams {
    direction: vec2<f32>,  // (1,0) for horizontal, (0,1) for vertical
    _padding: vec2<f32>,
}

@group(0) @binding(5) var<uniform> blur_params: BlurParams;

// One-dimensional Gaussian over every texel within 2 * pairs of the centre.
// Neighbouring texels are read two at a time through one bilinear tap placed
// between them, weighted so the pair gets its two Gaussian weights. Taps must
// not skip texels: spaced further apart they make a comb, and a one-pixel HDR
// glint then shows the tap pattern itself (a square grid of dots in the
// bloom, a dashed line in the streaks) instead of a glow.
fn gaussian_blur(uv: vec2<f32>, texel_step: vec2<f32>, sigma: f32, pairs: i32) -> vec3<f32> {
    let k = -0.5 / (sigma * sigma);
    var sum = textureSampleLevel(scene_texture, texture_sampler, uv, 0.0).rgb;
    var weight_sum = 1.0;
    for (var i = 0; i < pairs; i++) {
        let near = f32(2 * i + 1);
        let w_near = exp(k * near * near);
        let w_far = exp(k * (near + 1.0) * (near + 1.0));
        let w = w_near + w_far;
        let offset = texel_step * (near + w_far / w);
        sum += w * (textureSampleLevel(scene_texture, texture_sampler, uv + offset, 0.0).rgb
            + textureSampleLevel(scene_texture, texture_sampler, uv - offset, 0.0).rgb);
        weight_sum += 2.0 * w;
    }
    return sum / weight_sum;
}

// Bloom: sigma 3.4 texels of the half-res target (~7 scene pixels), called
// twice (horizontal, then vertical)
const BLOOM_SIGMA: f32 = 3.4;
const BLOOM_PAIRS: i32 = 5;

@fragment
fn fs_bloom_blur(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(scene_texture));
    return vec4<f32>(gaussian_blur(input.uv, blur_params.direction * texel, BLOOM_SIGMA, BLOOM_PAIRS), 1.0);
}

// Anamorphic streak: horizontal only, called twice. Sigma 6.85 texels of the
// 1/16-width target per pass = ~155 scene pixels after both. The gain keeps
// the streak energy of the 13-tap kernel this replaced (its weights summed to
// 1.14), so intensity settings carry over.
const STREAK_SIGMA: f32 = 6.85;
const STREAK_PAIRS: i32 = 9;
const STREAK_PASS_GAIN: f32 = 1.14;

@fragment
fn fs_streak_blur(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(scene_texture));
    let step = vec2<f32>(blur_params.direction.x * texel.x, 0.0);
    return vec4<f32>(gaussian_blur(input.uv, step, STREAK_SIGMA, STREAK_PAIRS) * STREAK_PASS_GAIN, 1.0);
}
