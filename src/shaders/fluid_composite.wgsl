// Fluid Composite - Final water surface rendering with Fresnel, refraction, specular

struct FluidRenderParams {
    water_color: vec3<f32>,
    absorption: f32,
    specular_power: f32,
    fresnel_power: f32,
    fresnel_scale: f32,
    refraction_strength: f32,
    ambient: f32,
    screen_width: f32,
    screen_height: f32,
    _padding: f32,
}

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var depth_texture: texture_2d<f32>;
@group(0) @binding(1) var thickness_texture: texture_2d<f32>;
@group(0) @binding(2) var background_texture: texture_2d<f32>;
@group(0) @binding(3) var texture_sampler: sampler;
@group(0) @binding(4) var<uniform> params: FluidRenderParams;
@group(0) @binding(5) var<uniform> camera: CameraParams;

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

    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv = uvs[vertex_index];
    return output;
}

fn reconstruct_normal(uv: vec2<f32>, pixel_size: vec2<f32>) -> vec3<f32> {
    // Sample neighboring depths with larger offset for smoother normals
    let offset = pixel_size * 2.0;

    let depth_c = textureSample(depth_texture, texture_sampler, uv).r;
    let depth_l = textureSample(depth_texture, texture_sampler, uv - vec2<f32>(offset.x, 0.0)).r;
    let depth_r = textureSample(depth_texture, texture_sampler, uv + vec2<f32>(offset.x, 0.0)).r;
    let depth_u = textureSample(depth_texture, texture_sampler, uv - vec2<f32>(0.0, offset.y)).r;
    let depth_d = textureSample(depth_texture, texture_sampler, uv + vec2<f32>(0.0, offset.y)).r;

    // Handle edges where neighbors might be empty
    var dzdx = 0.0;
    var dzdy = 0.0;

    if (depth_l > 0.001 && depth_r > 0.001) {
        dzdx = (depth_r - depth_l) * 0.5;
    } else if (depth_l > 0.001) {
        dzdx = depth_c - depth_l;
    } else if (depth_r > 0.001) {
        dzdx = depth_r - depth_c;
    }

    if (depth_u > 0.001 && depth_d > 0.001) {
        dzdy = (depth_d - depth_u) * 0.5;
    } else if (depth_u > 0.001) {
        dzdy = depth_c - depth_u;
    } else if (depth_d > 0.001) {
        dzdy = depth_d - depth_c;
    }

    // Scale factor for normal strength - adjust based on depth
    let scale = 30.0 / max(depth_c, 0.1);

    // Construct normal in view space
    let normal = normalize(vec3<f32>(-dzdx * scale, -dzdy * scale, 1.0));
    return normal;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let pixel_size = vec2<f32>(1.0 / params.screen_width, 1.0 / params.screen_height);

    let depth = textureSample(depth_texture, texture_sampler, input.uv).r;
    let thickness = textureSample(thickness_texture, texture_sampler, input.uv).r;

    // If no fluid, return background
    if (depth <= 0.001 || depth > 100.0) {
        let bg = textureSample(background_texture, texture_sampler, input.uv).rgb;
        return vec4<f32>(bg, 1.0);
    }

    // Reconstruct normal from depth
    let normal = reconstruct_normal(input.uv, pixel_size);

    // View direction (in view space, looking down -Z)
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);

    // Multiple light sources for better coverage
    let light_dir1 = normalize(vec3<f32>(0.5, 0.8, 0.6));
    let light_dir2 = normalize(vec3<f32>(-0.3, 0.6, 0.4));

    // Fresnel effect - controls reflection vs refraction
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let fresnel = params.fresnel_scale + (1.0 - params.fresnel_scale) * pow(1.0 - n_dot_v, params.fresnel_power);

    // Specular highlights from multiple lights
    let half_vec1 = normalize(light_dir1 + view_dir);
    let half_vec2 = normalize(light_dir2 + view_dir);
    let specular1 = pow(max(dot(normal, half_vec1), 0.0), params.specular_power);
    let specular2 = pow(max(dot(normal, half_vec2), 0.0), params.specular_power) * 0.5;
    let specular = specular1 + specular2;

    // Refraction - distort UV based on normal for "seeing through" effect
    let refract_strength = params.refraction_strength * (1.0 + thickness * 0.5);
    let refract_offset = normal.xy * refract_strength;
    let refract_uv = clamp(input.uv + refract_offset, vec2<f32>(0.0), vec2<f32>(1.0));
    let background = textureSample(background_texture, texture_sampler, refract_uv).rgb;

    // Water color absorption - thicker = more colored, less transparent
    let absorption_factor = 1.0 - exp(-thickness * params.absorption);

    // Base water color with depth-based tint
    let deep_color = params.water_color * 0.6;
    let shallow_color = params.water_color * 1.2;
    let water_tint = mix(shallow_color, deep_color, clamp(thickness * 2.0, 0.0, 1.0));

    // Diffuse lighting
    let ndotl1 = max(dot(normal, light_dir1), 0.0);
    let ndotl2 = max(dot(normal, light_dir2), 0.0);
    let diffuse = (ndotl1 + ndotl2 * 0.5) * 0.4 + params.ambient;

    // Combine everything
    // Start with background seen through water
    var color = background;

    // Blend in water color based on thickness/absorption
    color = mix(color, water_tint, absorption_factor * 0.7);

    // Apply diffuse lighting
    color = color * diffuse;

    // Add reflection tint based on Fresnel
    let reflection_color = vec3<f32>(0.7, 0.85, 1.0); // Sky-ish color
    color = mix(color, reflection_color, fresnel * 0.4);

    // Add specular highlights
    color = color + vec3<f32>(1.0, 1.0, 1.0) * specular * 0.9;

    // Subtle edge darkening for depth
    let edge_factor = 1.0 - pow(n_dot_v, 0.5) * 0.15;
    color = color * edge_factor;

    return vec4<f32>(color, 1.0);
}
