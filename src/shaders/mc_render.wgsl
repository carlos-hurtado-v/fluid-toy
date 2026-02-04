// Marching Cubes - Surface Rendering
// Renders the extracted mesh with water-like appearance

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct WaterParams {
    water_color: vec3<f32>,
    ambient: f32,
    specular_power: f32,
    fresnel_power: f32,
    fresnel_bias: f32,
    _padding: f32,
}

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> water: WaterParams;
@group(0) @binding(2) var<storage, read> vertices: array<Vertex>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let vertex = vertices[vertex_index];

    var output: VertexOutput;
    output.world_pos = vertex.position;
    output.world_normal = vertex.normal;

    let view_pos = camera.view * vec4<f32>(vertex.position, 1.0);
    output.clip_position = camera.projection * view_pos;

    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let view_dir = normalize(camera.camera_pos - input.world_pos);

    // Get normal and flip if facing away from camera
    var normal = normalize(input.world_normal);
    let facing_away = dot(normal, view_dir) < 0.0;
    if (facing_away) {
        normal = -normal;
    }

    // DEBUG: Visualize normal direction (comment out for final render)
    // Red = +X, Green = +Y, Blue = +Z
    // return vec4<f32>(normal * 0.5 + 0.5, 1.0);

    // Light directions
    let light_dir1 = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let light_dir2 = normalize(vec3<f32>(-0.3, 0.8, -0.5));
    let light_color = vec3<f32>(1.0, 0.98, 0.95);

    // Fresnel - more reflection at grazing angles
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let fresnel = water.fresnel_bias + (1.0 - water.fresnel_bias) * pow(1.0 - n_dot_v, water.fresnel_power);

    // Diffuse lighting
    let n_dot_l1 = max(dot(normal, light_dir1), 0.0);
    let n_dot_l2 = max(dot(normal, light_dir2), 0.0);
    let diffuse = (n_dot_l1 + n_dot_l2 * 0.3) * 0.6;

    // Specular highlights (Blinn-Phong)
    let half_vec1 = normalize(light_dir1 + view_dir);
    let half_vec2 = normalize(light_dir2 + view_dir);
    let spec1 = pow(max(dot(normal, half_vec1), 0.0), water.specular_power);
    let spec2 = pow(max(dot(normal, half_vec2), 0.0), water.specular_power) * 0.5;
    let specular = (spec1 + spec2) * light_color;

    // Environment reflection approximation
    let reflect_dir = reflect(-view_dir, normal);
    let env_color = mix(
        vec3<f32>(0.3, 0.5, 0.7),  // Horizon color
        vec3<f32>(0.6, 0.8, 1.0),  // Sky color
        max(reflect_dir.y * 0.5 + 0.5, 0.0)
    );

    // Combine water color with lighting
    let base_color = water.water_color * (water.ambient + diffuse);

    // Mix between refracted color and reflected environment based on fresnel
    var color = mix(base_color, env_color, fresnel * 0.6);

    // Add specular on top
    color = color + specular * 0.8;

    // Subtle rim lighting for depth
    let rim = pow(1.0 - n_dot_v, 3.0) * 0.2;
    color = color + vec3<f32>(0.5, 0.7, 1.0) * rim;

    return vec4<f32>(color, 1.0);
}
