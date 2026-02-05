// Marching Cubes - Mesh Rendering
// Renders the generated triangle mesh with water shading

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct WaterParams {
    water_color: vec3<f32>,
    specular_power: f32,
    fresnel_bias: f32,
    refraction_strength: f32,
    ripple_scale: f32,
    ripple_strength: f32,
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

// Fractal Brownian Motion for natural-looking detail
fn fbm(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var pos = p;

    for (var i = 0; i < 4; i++) {
        value += amplitude * noise(pos);
        amplitude *= 0.5;
        pos *= 2.0;
    }
    return value;
}

// Compute procedural normal perturbation for surface detail
fn surface_detail_normal(world_pos: vec3<f32>, base_normal: vec3<f32>, scale: f32, strength: f32) -> vec3<f32> {
    // Use world XZ position for ripple pattern
    let p = world_pos.xz * scale;

    // Multiple wave layers for organic look
    let wave1 = sin(p.x * 1.0) * sin(p.y * 1.2) * 0.5;
    let wave2 = sin(p.x * 2.3 + 0.5) * sin(p.y * 1.8 - 0.3) * 0.3;
    let wave3 = fbm(p * 0.3) * 2.0 - 1.0;

    // Compute gradient for normal perturbation
    let eps = 0.05;
    let h = (wave1 + wave2 + wave3 * 0.4) * strength;

    let p_dx = (world_pos.xz + vec2<f32>(eps, 0.0)) * scale;
    let wave1_dx = sin(p_dx.x * 1.0) * sin(p_dx.y * 1.2) * 0.5;
    let wave2_dx = sin(p_dx.x * 2.3 + 0.5) * sin(p_dx.y * 1.8 - 0.3) * 0.3;
    let wave3_dx = fbm(p_dx * 0.3) * 2.0 - 1.0;
    let h_dx = (wave1_dx + wave2_dx + wave3_dx * 0.4) * strength;

    let p_dy = (world_pos.xz + vec2<f32>(0.0, eps)) * scale;
    let wave1_dy = sin(p_dy.x * 1.0) * sin(p_dy.y * 1.2) * 0.5;
    let wave2_dy = sin(p_dy.x * 2.3 + 0.5) * sin(p_dy.y * 1.8 - 0.3) * 0.3;
    let wave3_dy = fbm(p_dy * 0.3) * 2.0 - 1.0;
    let h_dy = (wave1_dy + wave2_dy + wave3_dy * 0.4) * strength;

    // Normal from height gradient
    let dx = (h_dx - h) / eps;
    let dy = (h_dy - h) / eps;
    let detail_normal = normalize(vec3<f32>(-dx, 1.0, -dy));

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

    // Convert to linear depth (approximate with near=0.1, far=100)
    let near = 0.1;
    let far = 100.0;
    let front_linear = linearize_depth(front_depth_raw, near, far);
    let back_linear = linearize_depth(back_depth_raw, near, far);

    // Thickness in world units (clamped to reasonable range)
    var thickness = max(0.0, back_linear - front_linear);
    thickness = min(thickness, 5.0);  // Cap at 5 units

    // === ABSORPTION (Beer's Law) ===
    // Light attenuates exponentially through water
    // Different wavelengths absorb at different rates (red absorbs fastest)
    let absorption_coeffs = vec3<f32>(0.8, 0.3, 0.1);  // RGB absorption rates
    let density = 2.5;  // Water density factor
    let transmittance = exp(-absorption_coeffs * density * thickness);

    // Light direction (approximate sun)
    let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.3));

    // Reflection - sample environment (sky, surroundings)
    let reflect_dir = reflect(-view_dir, normal);
    let reflection_color = sample_environment(reflect_dir);

    // Refraction direction for environment sampling
    let refract_dir = refract(-view_dir, normal, 0.75);  // water IOR ~1.33
    var refraction_env = sample_environment(refract_dir);

    // Apply absorption to refracted/transmitted light
    let absorbed_color = refraction_env * transmittance;

    // Deep water color (what you see when looking deep)
    let deep_color = vec3<f32>(0.01, 0.04, 0.1);

    // Blend between absorbed environment and deep water based on thickness
    let depth_blend = 1.0 - exp(-thickness * 1.5);
    let water_interior = mix(absorbed_color, deep_color, depth_blend);

    // Add water's own color contribution
    let scatter_color = water.water_color * 0.3;
    let interior_with_scatter = water_interior + scatter_color * (1.0 - transmittance);

    // Fresnel (Schlick approximation) - controls reflection vs transmission
    let cos_theta = max(0.0, dot(normal, view_dir));
    let F0 = water.fresnel_bias;
    let fresnel = clamp(F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0), 0.0, 1.0);

    // Specular highlight (Blinn-Phong)
    let half_vec = normalize(light_dir + view_dir);
    let specular = pow(max(0.0, dot(half_vec, normal)), water.specular_power) * 1.5;

    // Combine reflection and interior based on Fresnel
    var color = mix(interior_with_scatter, reflection_color, fresnel);

    // Add specular highlight
    color += vec3<f32>(1.0, 1.0, 0.95) * specular;

    // Subtle overall tint
    color = color * mix(vec3<f32>(1.0), water.water_color, 0.15);

    // Tone mapping
    color = color / (color + vec3<f32>(1.0));
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
