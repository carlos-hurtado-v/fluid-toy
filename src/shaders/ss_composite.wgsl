// Screen-Space Fluid - Composite Pass
// Reconstructs normals from depth, applies water shading

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
}

@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var thickness_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> camera: CameraParams;
@group(0) @binding(3) var<uniform> water: WaterParams;

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

// Reconstruct view-space position from UV and depth
// Note: UV has (0,0) at top-left matching texture coordinates after flip
fn compute_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // NDC coordinates: UV (0,0) = top-left = NDC (-1, +1)
    // UV (1,1) = bottom-right = NDC (+1, -1)
    var ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, 0.0, 1.0);

    // Compute NDC z from view depth using projection matrix
    ndc.z = -camera.projection[2][2] + camera.projection[3][2] / depth;

    // Unproject to view space
    var view_pos = water.inv_projection * ndc;
    return view_pos.xyz / view_pos.w;
}

fn get_view_pos_at(uv: vec2<f32>, iuv: vec2<i32>) -> vec3<f32> {
    let depth = abs(textureLoad(depth_tex, iuv, 0).r);
    return compute_view_pos(uv, depth);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(depth_tex));
    // Flip Y for texture coordinate (UV 0,0 is bottom-left, texture 0,0 is top-left)
    let flipped_uv = vec2<f32>(input.uv.x, 1.0 - input.uv.y);
    let iuv = vec2<i32>(flipped_uv * tex_size);

    let depth = abs(textureLoad(depth_tex, iuv, 0).r);
    let bg_color = vec3<f32>(0.02, 0.02, 0.05);

    // Background - no fluid here
    if (depth == 0.0 || depth >= 1e4) {
        return vec4<f32>(bg_color, 1.0);
    }

    // Get view-space position (use flipped UV for consistent coordinate system)
    let view_pos = compute_view_pos(flipped_uv, depth);
    
    // === NORMAL RECONSTRUCTION WITH STRIDE ===
    // Widen the sampling kernel to smooth out faceted "grape" artifacts.
    // stride = 1.0 : Sharpest, but shows artifacts if blur isn't perfect
    // stride = 2.0 or 3.0 : Smoother normals, hides the sphere shapes
    let stride = 2.0; 
    let stride_i = i32(stride);

    // Compute gradients (Central differences with wider gap)
    // Note: We multiply texel_size by stride for UVs, and use stride_i for integer lookups
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

    // Use the gradient with smaller z change (edge preservation logic remains the same)
    var ddx = ddx1;
    if (abs(ddx2.z) < abs(ddx1.z)) {
        ddx = ddx2;
    }
    var ddy = ddy1;
    if (abs(ddy2.z) < abs(ddy1.z)) {
        ddy = ddy2;
    }

    // Normal from cross product (negate for correct orientation)
    var normal = -normalize(cross(ddx, ddy));

    // Ray direction (view space, pointing from camera into scene)
    let ray_dir = normalize(view_pos);

    // Light direction in view space
    let light_dir = normalize((camera.view * vec4<f32>(0.5, 1.0, 0.3, 0.0)).xyz);

    // Blinn-Phong specular
    let H = normalize(light_dir - ray_dir);
    let specular = pow(max(0.0, dot(H, normal)), water.specular_power);

    // Diffuse
    let diffuse = max(0.0, dot(light_dir, normal));

    // Thickness for absorption (use same flipped coordinates)
    let thickness = textureLoad(thickness_tex, iuv, 0).r;

    // Water color and absorption (Beer's law)
    let water_color = vec3<f32>(0.1, 0.5, 0.9);
    let density = 2.0;
    let transmittance = exp(-density * thickness * (1.0 - water_color));
    let refraction_color = bg_color * transmittance;

    // Fresnel (Schlick approximation)
    let F0 = water.fresnel_bias;
    let fresnel = clamp(F0 + (1.0 - F0) * pow(1.0 - dot(normal, -ray_dir), 5.0), 0.0, 1.0);

    // Reflection (simple sky gradient)
    let reflect_dir = reflect(ray_dir, normal);
    let reflect_world = (water.inv_view * vec4<f32>(reflect_dir, 0.0)).xyz;
    let sky_color = mix(
        vec3<f32>(0.4, 0.6, 0.9),
        vec3<f32>(0.7, 0.85, 1.0),
        max(0.0, reflect_world.y * 0.5 + 0.5)
    );

    // Final color
    var color = specular + mix(refraction_color, sky_color, fresnel);

    // Add subtle diffuse contribution
    color += water_color * diffuse * 0.2;

    // ===== DEBUG VISUALIZATION (uncomment ONE to diagnose) =====
    // return vec4<f32>(vec3<f32>(depth * 0.1), 1.0);           // Raw Depth
    // return vec4<f32>(vec3<f32>(depth * 0.3), 1.0);           // Depth (gray)
    // return vec4<f32>(normal * 0.5 + 0.5, 1.0);               // Normals (RGB)
    // return vec4<f32>(vec3<f32>(thickness), 1.0);             // Thickness (gray)
    // return vec4<f32>(vec3<f32>(fresnel), 1.0);               // Fresnel (gray)
    // return vec4<f32>(vec3<f32>(specular), 1.0);              // Specular (gray)
    // return vec4<f32>(vec3<f32>(diffuse), 1.0);               // Diffuse (gray)
    // return vec4<f32>(refraction_color, 1.0);                 // Refraction only
    // return vec4<f32>(sky_color * fresnel, 1.0);              // Reflection only

    return vec4<f32>(color, 1.0);
}
