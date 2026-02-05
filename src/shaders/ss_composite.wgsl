// Screen-Space Fluid - Composite Pass
// Reconstructs normals from depth, applies water shading with environment map

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct WaterParams {
    texel_size: vec2<f32>,
    specular_power: f32,
    fresnel_bias: f32,
    inv_projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    // Surface detail parameters
    ripple_scale: f32,      // Frequency of ripples (higher = more dense)
    ripple_strength: f32,   // How much ripples perturb normals
    time: f32,              // For animated ripples
    _padding2: f32,
}

@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var thickness_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> camera: CameraParams;
@group(0) @binding(3) var<uniform> water: WaterParams;
@group(0) @binding(4) var env_tex: texture_2d<f32>;
@group(0) @binding(5) var env_sampler: sampler;

const PI: f32 = 3.14159265359;

// Simple hash function for procedural noise
fn hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

// Smooth noise (value noise with interpolation)
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    // Cubic interpolation for smoothness
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash(i + vec2<f32>(0.0, 0.0));
    let b = hash(i + vec2<f32>(1.0, 0.0));
    let c = hash(i + vec2<f32>(0.0, 1.0));
    let d = hash(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal Brownian Motion - layered noise for natural look
fn fbm(p: vec2<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;

    for (var i = 0; i < octaves; i++) {
        value += amplitude * noise(pos * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

// Compute normal perturbation from procedural ripples
fn compute_ripple_normal(world_pos: vec3<f32>, scale: f32, strength: f32, time: f32) -> vec3<f32> {
    // Use XZ plane for ripple pattern (horizontal surface)
    let p = world_pos.xz * scale;

    // Animated offset
    let t = time * 0.5;

    // Multiple overlapping wave patterns for organic look
    let wave1 = sin(p.x * 1.0 + t * 0.7) * sin(p.y * 1.2 - t * 0.5);
    let wave2 = sin(p.x * 2.3 - t * 0.3) * sin(p.y * 1.8 + t * 0.4) * 0.5;
    let wave3 = fbm(p * 0.5 + vec2<f32>(t * 0.1, -t * 0.15), 3) * 2.0 - 1.0;

    // Combine waves
    let height = (wave1 + wave2 + wave3 * 0.3) * strength;

    // Compute gradient for normal (numerical derivative)
    let eps = 0.1;
    let p_dx = world_pos.xz * scale + vec2<f32>(eps, 0.0);
    let p_dy = world_pos.xz * scale + vec2<f32>(0.0, eps);

    let wave1_dx = sin(p_dx.x * 1.0 + t * 0.7) * sin(p_dx.y * 1.2 - t * 0.5);
    let wave2_dx = sin(p_dx.x * 2.3 - t * 0.3) * sin(p_dx.y * 1.8 + t * 0.4) * 0.5;
    let wave3_dx = fbm(p_dx * 0.5 + vec2<f32>(t * 0.1, -t * 0.15), 3) * 2.0 - 1.0;
    let height_dx = (wave1_dx + wave2_dx + wave3_dx * 0.3) * strength;

    let wave1_dy = sin(p_dy.x * 1.0 + t * 0.7) * sin(p_dy.y * 1.2 - t * 0.5);
    let wave2_dy = sin(p_dy.x * 2.3 - t * 0.3) * sin(p_dy.y * 1.8 + t * 0.4) * 0.5;
    let wave3_dy = fbm(p_dy * 0.5 + vec2<f32>(t * 0.1, -t * 0.15), 3) * 2.0 - 1.0;
    let height_dy = (wave1_dy + wave2_dy + wave3_dy * 0.3) * strength;

    // Normal from height gradient
    let dx = (height_dx - height) / eps;
    let dy = (height_dy - height) / eps;

    return normalize(vec3<f32>(-dx, 1.0, -dy));
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

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
    output.uv = pos * 0.5 + 0.5;
    return output;
}

// Sample equirectangular environment map from world-space direction
fn sample_environment(dir: vec3<f32>) -> vec3<f32> {
    // Convert direction to spherical coordinates
    // phi = atan2(z, x), theta = acos(y)
    let phi = atan2(dir.z, dir.x);
    let theta = acos(clamp(dir.y, -1.0, 1.0));

    // Convert to UV coordinates
    // U: phi goes from -PI to PI, map to 0..1
    // V: theta goes from 0 (top) to PI (bottom), map to 0..1
    //    Flip V because texture has Y=0 at top
    let u = (phi + PI) / (2.0 * PI);
    let v = 1.0 - theta / PI;

    let color = textureSample(env_tex, env_sampler, vec2<f32>(u, v));
    return color.rgb;
}

// Reconstruct view-space position from UV and depth
fn compute_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    var ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, 0.0, 1.0);
    ndc.z = -camera.projection[2][2] + camera.projection[3][2] / depth;
    var view_pos = water.inv_projection * ndc;
    return view_pos.xyz / view_pos.w;
}

fn get_view_pos_at(uv: vec2<f32>, iuv: vec2<i32>) -> vec3<f32> {
    let depth = abs(textureLoad(depth_tex, iuv, 0).r);
    return compute_view_pos(uv, depth);
}

// sRGB gamma correction (linear -> sRGB)
fn gamma_correct(color: vec3<f32>) -> vec3<f32> {
    return pow(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

// Attempt to linearize if data was stored with gamma (sRGB -> linear)
fn gamma_to_linear(color: vec3<f32>) -> vec3<f32> {
    return pow(color, vec3<f32>(2.2));
}

// HDR to SDR conversion
// Set USE_GAMMA_CORRECT to see which looks better
fn hdr_to_sdr(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    let exposed = color * exposure;
    // The HDR data from the image crate appears to already have gamma baked in
    // So we DON'T apply additional gamma correction
    return clamp(exposed, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(depth_tex));
    let flipped_uv = vec2<f32>(input.uv.x, 1.0 - input.uv.y);
    let iuv = vec2<i32>(flipped_uv * tex_size);

    let depth = abs(textureLoad(depth_tex, iuv, 0).r);

    // Compute world-space ray direction for background
    let ndc = vec2<f32>(input.uv.x * 2.0 - 1.0, 1.0 - 2.0 * input.uv.y);
    let view_ray = normalize((water.inv_projection * vec4<f32>(ndc, 1.0, 1.0)).xyz);
    let world_ray = normalize((water.inv_view * vec4<f32>(view_ray, 0.0)).xyz);

    // Background - sample environment map
    if (depth == 0.0 || depth >= 1e4) {
        let env_color = sample_environment(world_ray);
        // Apply tone mapping and gamma for HDR (exposure = 1.0 for neutral)
        let mapped = hdr_to_sdr(env_color, 1.0);
        return vec4<f32>(mapped, 1.0);
    }

    // Get view-space position
    let view_pos = compute_view_pos(flipped_uv, depth);

    // === NORMAL RECONSTRUCTION WITH STRIDE ===
    let stride = 4.0;
    let stride_i = i32(stride);

    let ddx1 = get_view_pos_at(
        flipped_uv + vec2<f32>(water.texel_size.x * stride, 0.0),
        iuv + vec2<i32>(stride_i, 0)
    ) - view_pos;

    let ddy1 = get_view_pos_at(
        flipped_uv + vec2<f32>(0.0, water.texel_size.y * stride),
        iuv + vec2<i32>(0, stride_i)
    ) - view_pos;

    let ddx2 = view_pos - get_view_pos_at(
        flipped_uv - vec2<f32>(water.texel_size.x * stride, 0.0),
        iuv - vec2<i32>(stride_i, 0)
    );

    let ddy2 = view_pos - get_view_pos_at(
        flipped_uv - vec2<f32>(0.0, water.texel_size.y * stride),
        iuv - vec2<i32>(0, stride_i)
    );

    var ddx = ddx1;
    if (abs(ddx2.z) < abs(ddx1.z)) {
        ddx = ddx2;
    }
    var ddy = ddy1;
    if (abs(ddy2.z) < abs(ddy1.z)) {
        ddy = ddy2;
    }

    var normal = -normalize(cross(ddx, ddy));

    // Ray direction (view space)
    let ray_dir = normalize(view_pos);

    // Transform normal to world space for environment lookup
    var normal_world = normalize((water.inv_view * vec4<f32>(normal, 0.0)).xyz);

    // === SURFACE DETAIL: Procedural ripple perturbation ===
    // This adds micro-surface variation to break up flat areas
    if (water.ripple_strength > 0.001) {
        // Get world position of this surface point
        let world_pos = (water.inv_view * vec4<f32>(view_pos, 1.0)).xyz;

        // Compute ripple normal in world space
        let ripple_normal = compute_ripple_normal(
            world_pos,
            water.ripple_scale,
            water.ripple_strength,
            water.time
        );

        // Blend ripple normal with geometric normal using TBN-style perturbation
        // The ripple_normal is in tangent space (Y-up), we blend it with world normal
        let up = vec3<f32>(0.0, 1.0, 0.0);
        let tangent = normalize(cross(up, normal_world));
        let bitangent = cross(normal_world, tangent);

        // Transform ripple from tangent to world space and blend
        let perturbed = tangent * ripple_normal.x + normal_world * ripple_normal.y + bitangent * ripple_normal.z;
        normal_world = normalize(mix(normal_world, perturbed, water.ripple_strength));
    }
    let ray_world = normalize((water.inv_view * vec4<f32>(ray_dir, 0.0)).xyz);

    // Reflection direction in world space
    let reflect_world = reflect(ray_world, normal_world);

    // Sample environment for reflection
    let reflection_color = sample_environment(reflect_world);

    // Light direction (from environment - approximate sun direction)
    // For "puresky" type HDRIs, sun is typically near the horizon
    let light_dir = normalize(vec3<f32>(0.5, 0.3, 0.8));

    // Specular highlight (Blinn-Phong)
    let H = normalize(light_dir - ray_world);
    let specular = pow(max(0.0, dot(H, normal_world)), water.specular_power) * 2.0;

    // Thickness for absorption
    let thickness = textureLoad(thickness_tex, iuv, 0).r;

    // Water absorption color (Beer's law)
    let water_color = vec3<f32>(0.1, 0.4, 0.8);
    let absorption_color = vec3<f32>(0.8, 0.95, 1.0); // What color light becomes after passing through
    let density = 3.0;
    let transmittance = exp(-density * thickness * (1.0 - absorption_color));

    // Refraction - sample environment behind the water (distorted by normal)
    let refract_dir = refract(ray_world, normal_world, 0.75); // water IOR ~1.33, air/water = 0.75
    var refraction_color: vec3<f32>;
    if (length(refract_dir) < 0.001) {
        // Total internal reflection
        refraction_color = reflection_color;
    } else {
        refraction_color = sample_environment(refract_dir) * transmittance;
    }

    // Fresnel (Schlick approximation)
    let F0 = water.fresnel_bias;
    let cos_theta = max(0.0, dot(normal_world, -ray_world));
    let fresnel = clamp(F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0), 0.0, 1.0);

    // Combine reflection and refraction based on Fresnel
    var color = mix(refraction_color, reflection_color, fresnel);

    // Add specular highlight
    color += vec3<f32>(1.0) * specular;

    // Subtle water tint based on thickness
    color = mix(color, color * water_color, clamp(thickness * 0.3, 0.0, 0.5));

    // Tone mapping and gamma correction for HDR
    color = hdr_to_sdr(color, 1.0);

    // ===== DEBUG VISUALIZATION =====
    // Uncomment ONE of these to diagnose rendering issues:
    // return vec4<f32>(vec3<f32>(depth * 0.1), 1.0);           // Raw Depth
    // return vec4<f32>(normal_world * 0.5 + 0.5, 1.0);         // Normals (RGB) - DEBUG ENABLED
    // return vec4<f32>(vec3<f32>(thickness), 1.0);             // Thickness
    // return vec4<f32>(vec3<f32>(fresnel), 1.0);               // Fresnel
    // return vec4<f32>(hdr_to_sdr(reflection_color, 1.0), 1.0);  // Reflection only
    // return vec4<f32>(hdr_to_sdr(refraction_color, 1.0), 1.0);  // Refraction only

    return vec4<f32>(color, 1.0);
}
