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

@fragment
fn fs_bloom_threshold(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(scene_texture, texture_sampler, input.uv).rgb;
    let luma = luminance(color);

    // Extract bright pixels above threshold
    let bloom_color = color * smoothstep(params.bloom_threshold, params.bloom_threshold + 0.2, luma);

    return vec4<f32>(bloom_color, 1.0);
}

// === Bloom blur shader ===
// Gaussian blur for bloom (called twice: horizontal then vertical)

struct BlurParams {
    direction: vec2<f32>,  // (1,0) for horizontal, (0,1) for vertical
    _padding: vec2<f32>,
}

@group(0) @binding(5) var<uniform> blur_params: BlurParams;

@fragment
fn fs_bloom_blur(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(scene_texture));
    let texel = 1.0 / tex_size;

    // 9-tap Gaussian blur
    let offsets = array<f32, 5>(0.0, 1.0, 2.0, 3.0, 4.0);
    let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    var result = textureSample(scene_texture, texture_sampler, input.uv).rgb * weights[0];

    for (var i = 1; i < 5; i++) {
        let offset = blur_params.direction * texel * offsets[i] * 2.0;
        result += textureSample(scene_texture, texture_sampler, input.uv + offset).rgb * weights[i];
        result += textureSample(scene_texture, texture_sampler, input.uv - offset).rgb * weights[i];
    }

    return vec4<f32>(result, 1.0);
}

// === Anamorphic streak blur shader ===
// Very wide horizontal blur for cinematic lens streaks

@fragment
fn fs_streak_blur(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(scene_texture));
    let texel = 1.0 / tex_size;

    // Wide 13-tap blur with extended reach for streak effect
    // Offsets go much further than bloom for that stretched look
    let offsets = array<f32, 7>(0.0, 1.5, 3.5, 6.0, 9.0, 13.0, 18.0);
    let weights = array<f32, 7>(0.14, 0.13, 0.12, 0.10, 0.08, 0.05, 0.02);

    // Horizontal direction only for anamorphic effect
    let dir = blur_params.direction;

    var result = textureSample(scene_texture, texture_sampler, input.uv).rgb * weights[0];

    for (var i = 1; i < 7; i++) {
        let offset = dir * texel * offsets[i] * 4.0; // 4x multiplier for extra width
        result += textureSample(scene_texture, texture_sampler, input.uv + offset).rgb * weights[i];
        result += textureSample(scene_texture, texture_sampler, input.uv - offset).rgb * weights[i];
    }

    return vec4<f32>(result, 1.0);
}
