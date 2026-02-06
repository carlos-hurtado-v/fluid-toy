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
    specular_power: f32,
    ior: f32,  // Index of refraction (water = 1.333)
    refraction_strength: f32,
    ripple_scale: f32,
    ripple_strength: f32,
}

struct LightParams {
    sun_direction: vec3<f32>,
    sun_enabled: u32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    specular_power: f32,
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

// Hash function for procedural noise
fn hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

// Smooth noise
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash(i + vec2<f32>(0.0, 0.0));
    let b = hash(i + vec2<f32>(1.0, 0.0));
    let c = hash(i + vec2<f32>(0.0, 1.0));
    let d = hash(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Compute procedural normal perturbation for water surface detail
// Uses sum of directional sine waves with analytic gradients
fn surface_detail_normal(world_pos: vec3<f32>, base_normal: vec3<f32>, scale: f32, strength: f32) -> vec3<f32> {
    let p = world_pos.xz * scale;

    // Sum of directional sine waves (capillary ripples)
    // Analytic gradient: d/dp [A*sin(dot(p,k))] = A*cos(dot(p,k))*k
    var grad = vec2<f32>(0.0, 0.0);

    // Low-frequency swell
    let k0 = vec2<f32>(1.31, 0.97);
    grad += 0.040 * cos(dot(p, k0)) * k0;
    let k1 = vec2<f32>(-1.07, 1.43);
    grad += 0.035 * cos(dot(p, k1)) * k1;

    // Medium-frequency ripples
    let k2 = vec2<f32>(2.71, -1.83);
    grad += 0.025 * cos(dot(p, k2)) * k2;
    let k3 = vec2<f32>(-1.67, -2.31);
    grad += 0.020 * cos(dot(p, k3)) * k3;

    // High-frequency capillary detail
    let k4 = vec2<f32>(4.37, 1.51);
    grad += 0.015 * cos(dot(p, k4)) * k4;
    let k5 = vec2<f32>(-2.17, 3.93);
    grad += 0.012 * cos(dot(p, k5)) * k5;
    let k6 = vec2<f32>(3.91, -3.17);
    grad += 0.010 * cos(dot(p, k6)) * k6;
    let k7 = vec2<f32>(-4.53, -1.79);
    grad += 0.008 * cos(dot(p, k7)) * k7;

    // Modulate amplitude with noise to break repetition
    let n = noise(p * 0.5) * 0.4;
    grad *= (1.0 + n);

    // Apply strength
    grad *= strength;

    // Normal from height gradient
    let detail_normal = normalize(vec3<f32>(-grad.x, 1.0, -grad.y));

    // Blend detail normal with base normal using TBN-style perturbation
    let up = vec3<f32>(0.0, 1.0, 0.0);
    var tangent = cross(up, base_normal);
    if (length(tangent) < 0.001) {
        tangent = vec3<f32>(1.0, 0.0, 0.0);
    }
    tangent = normalize(tangent);
    let bitangent = normalize(cross(base_normal, tangent));

    // Transform detail from tangent space to world space
    let perturbed = tangent * detail_normal.x + base_normal * detail_normal.y + bitangent * detail_normal.z;
    return normalize(perturbed);
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

// Linearize depth from depth buffer (reverse-Z or standard)
fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let view_dir = normalize(camera.camera_pos - input.world_position);

    // Get the surface normal - ensure it faces toward the camera
    var normal = normalize(input.world_normal);
    // If normal points away from camera, flip it (ensures correct reflection)
    if (dot(normal, view_dir) < 0.0) {
        normal = -normal;
    }

    // Add procedural surface detail to break up the polygonal look
    if (water.ripple_strength > 0.001) {
        normal = surface_detail_normal(input.world_position, normal, water.ripple_scale, water.ripple_strength);
    }

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
    let absorption_coeffs = vec3<f32>(0.45, 0.09, 0.02);  // RGB absorption rates
    let density = 1.5;  // Water density factor
    let transmittance = exp(-absorption_coeffs * density * thickness);

    // Reflection - sample environment (sky, surroundings)
    let reflect_dir = reflect(-view_dir, normal);
    let reflection_color = sample_environment(reflect_dir);

    // === SCREEN-SPACE REFRACTION ===
    // Transform normal to view space for screen-space distortion
    let normal_view = (camera.view * vec4<f32>(normal, 0.0)).xyz;

    // Compute refraction UV offset based on normal and thickness
    // Stronger distortion for thicker water and more angled surfaces
    let refract_strength = water.refraction_strength * (1.0 + thickness * 0.5);
    let uv_offset = normal_view.xy * refract_strength;

    // Sample background with distorted UVs (clamp to avoid sampling outside)
    let refract_uv = clamp(screen_uv + uv_offset, vec2<f32>(0.001), vec2<f32>(0.999));
    var refracted_background = textureSample(background_tex, env_sampler, refract_uv).rgb;

    // Apply absorption to refracted light (Beer-Lambert)
    refracted_background = refracted_background * transmittance;

    // Deep water color (what you see when looking deep)
    let deep_color = vec3<f32>(0.01, 0.04, 0.1);

    // Blend between refracted background and deep water based on thickness
    let depth_blend = 1.0 - exp(-thickness * 0.6);
    let water_interior = mix(refracted_background, deep_color, depth_blend);

    // Add water's own color contribution (subsurface scattering approximation)
    let scatter_color = water.water_color * 0.1;
    let interior_with_scatter = water_interior + scatter_color * (1.0 - transmittance);

    // Fresnel (Schlick approximation) - controls reflection vs transmission
    // F0 = ((n1 - n2) / (n1 + n2))^2 where n1=1.0 (air), n2=IOR (water)
    let cos_theta = max(0.0, dot(normal, view_dir));
    let F0 = pow((water.ior - 1.0) / (water.ior + 1.0), 2.0);  // ~0.02 for water
    let fresnel = clamp(F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0), 0.0, 1.0);

    // === DIRECTIONAL LIGHT (SUN) ===
    var sun_specular = vec3<f32>(0.0);
    var sun_subsurface = vec3<f32>(0.0);
    if (light.sun_enabled == 1u) {
        let light_dir = normalize(light.sun_direction);
        let NdotL = max(0.0, dot(normal, light_dir));

        // Specular reflection (Blinn-Phong), modulated by Fresnel
        // At steep angles most light enters water; at grazing angles it reflects
        let half_vec = normalize(light_dir + view_dir);
        let NdotH = max(0.0, dot(normal, half_vec));
        let spec_intensity = pow(NdotH, light.specular_power);
        sun_specular = light.sun_color * light.sun_intensity * spec_intensity * fresnel;

        // Subsurface illumination — light enters water, scatters, exits toward viewer
        let light_entering = NdotL * (1.0 - fresnel);
        let interior_glow = water.water_color * transmittance;
        sun_subsurface = interior_glow * light_entering * light.sun_color * light.sun_intensity * 0.25;

        // Forward scattering — thin areas glow when backlit (translucency)
        let VdotL = max(0.0, dot(-view_dir, light_dir));
        let forward_scatter = pow(VdotL, 4.0) * exp(-thickness * 2.0);
        sun_subsurface += water.water_color * forward_scatter * light.sun_color * light.sun_intensity * 0.15;
    }

    // Add sun subsurface to interior (before Fresnel blend, it's inside the water)
    let lit_interior = interior_with_scatter + sun_subsurface;

    // Combine reflection and refraction based on Fresnel
    // At grazing angles (high fresnel): more reflection
    // Looking straight on (low fresnel): more refraction/transmission
    var color = mix(lit_interior, reflection_color, fresnel);

    // Add specular on top (pure surface reflection, independent of interior)
    color += sun_specular;

    // Tone mapping (Reinhard)
    color = color / (color + vec3<f32>(1.0));

    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
