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
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
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

// Sample equirectangular environment map
fn sample_environment(dir: vec3<f32>) -> vec3<f32> {
    let phi = atan2(dir.z, dir.x);
    let theta = acos(clamp(dir.y, -1.0, 1.0));
    let u = (phi + PI) / (2.0 * PI);
    let v = 1.0 - theta / PI;
    return textureSample(env_tex, env_sampler, vec2<f32>(u, v)).rgb;
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

    // Thickness in world units (clamped to reasonable range)
    var thickness = max(0.0, back_linear - front_linear);
    thickness = min(thickness, 5.0);  // Cap at 5 units

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
        // The env map only contains sky — it can't represent nearby scene geometry
        // (walls, floor). Downward reflections would show incorrect sky colors.
        // SSR handles these directions; without SSR, fade to the interior instead.
        let horizon_fade = smoothstep(-0.15, 0.1, reflect_dir.y);
        reflection_color = env_reflection * horizon_fade;
    }

    // Screen-space reflections — blend with env map based on SSR confidence
    // Reduce SSR contribution for rough surfaces (sharp reflections look wrong on rough water)
    let ssr_dims = textureDimensions(ssr_tex);
    let ssr_coord = vec2<i32>(screen_uv * vec2<f32>(f32(ssr_dims.x), f32(ssr_dims.y)));
    let ssr_sample = textureLoad(ssr_tex, ssr_coord, 0);
    let ssr_confidence = ssr_sample.a * (1.0 - roughness_sq);
    reflection_color = mix(reflection_color, ssr_sample.rgb, ssr_confidence);

    // === SCREEN-SPACE REFRACTION ===
    // Use normal deviation from a flat surface — a perfectly flat water surface
    // should have zero screen-space distortion (you see straight through).
    // Only waves and ripples create visible refraction distortion.
    let flat_normal_view = normalize((camera.view * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz);
    let normal_view = normalize((camera.view * vec4<f32>(normal, 0.0)).xyz);
    let normal_deviation = normal_view - flat_normal_view;

    let refract_strength = water.refraction_strength * (1.0 + thickness * 0.5);
    let uv_offset = normal_deviation.xy * refract_strength;

    // Sample background with distorted UVs (clamp to avoid sampling outside)
    let refract_uv = clamp(screen_uv + uv_offset, vec2<f32>(0.001), vec2<f32>(0.999));
    var refracted_background = textureSample(background_tex, env_sampler, refract_uv).rgb;

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
    var sun_specular = vec3<f32>(0.0);
    var sun_subsurface = vec3<f32>(0.0);
    if (light.sun_enabled == 1u) {
        let light_dir = normalize(light.sun_direction);
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

        sun_specular = light.sun_color * light.sun_intensity * specular_brdf * NdotL;

        // Subsurface illumination — light enters water, scatters, exits toward viewer
        let light_entering = NdotL * (1.0 - F_spec);
        let interior_glow = water.water_color * transmittance;
        sun_subsurface = interior_glow * light_entering * light.sun_color * light.sun_intensity * 0.18;

        // Forward scattering — thin areas glow when backlit (translucency)
        let VdotL = max(0.0, dot(-view_dir, light_dir));
        let forward_scatter = pow(VdotL, 4.0) * exp(-thickness * optical_density * 1.5);
        sun_subsurface += water.water_color * forward_scatter * light.sun_color * light.sun_intensity * 0.10;
    }

    // IBL diffuse irradiance from spherical harmonics
    // Light enters the water (1-F), travels through the volume (transmittance),
    // and scatters back (scatter_strength) — same physics as subsurface scattering
    let ambient_irradiance = evaluate_sh_irradiance(normal) * water.env_intensity;
    let ambient_subsurface = ambient_irradiance * water.water_color * transmittance * scatter_strength * 0.6;

    // Add sun subsurface (weighted by 1-fresnel for energy conservation) and ambient irradiance
    let lit_interior = interior_with_scatter
        + sun_subsurface * (1.0 - fresnel)
        + ambient_subsurface * (1.0 - fresnel);

    // Combine reflection and refraction based on Fresnel
    // At grazing angles (high fresnel): more reflection
    // Looking straight on (low fresnel): more refraction/transmission
    var color = mix(lit_interior, reflection_color, fresnel);

    // Add specular on top (pure surface reflection, independent of interior)
    color += sun_specular;

    // Output linear HDR — post-process pipeline handles tone mapping + gamma

    return vec4<f32>(color, 1.0);
}
