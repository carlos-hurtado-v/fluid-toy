// 3D Particle rendering shader - billboard spheres with perspective projection

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct RenderParams {
    particle_radius: f32,
    color_by_velocity: u32,
    _padding1: vec2<u32>,
    particle_color: vec4<f32>,
}

struct LightParams {
    sun_direction: vec3<f32>,
    sun_enabled: u32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    _pad2: f32,
    _padding: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) velocity: vec3<f32>,
    @location(2) world_center: vec3<f32>,
    @location(3) sphere_radius: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> params: RenderParams;
@group(0) @binding(2) var<uniform> light: LightParams;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) particle_pos: vec3<f32>,
    @location(1) particle_vel: vec3<f32>,
) -> VertexOutput {
    // Generate quad vertices (2 triangles, 6 vertices)
    var quad_verts = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );

    let local_pos = quad_verts[vertex_index];

    // Transform particle center to view space
    let view_center = camera.view * vec4<f32>(particle_pos, 1.0);

    // Create billboard quad in view space (facing camera)
    let view_pos = view_center.xyz + vec3<f32>(local_pos * params.particle_radius, 0.0);

    // Project to clip space
    let clip_pos = camera.projection * vec4<f32>(view_pos, 1.0);

    var output: VertexOutput;
    output.position = clip_pos;
    output.uv = local_pos * 0.5 + 0.5;
    output.velocity = particle_vel;
    output.world_center = particle_pos;
    output.sphere_radius = params.particle_radius;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Draw circle within quad
    let centered_uv = input.uv - vec2<f32>(0.5, 0.5);
    let dist_sq = dot(centered_uv, centered_uv);

    // Discard pixels outside the circle
    if (dist_sq > 0.25) {
        discard;
    }

    // Compute sphere normal for shading
    let dist = sqrt(dist_sq) * 2.0; // 0 to 1 range
    let z = sqrt(max(0.0, 1.0 - dist * dist));
    let normal = vec3<f32>(centered_uv.x * 2.0, centered_uv.y * 2.0, z);

    // Base color from params
    var color = params.particle_color.rgb;

    // Optionally modify by velocity
    if (params.color_by_velocity != 0u) {
        let speed = length(input.velocity);
        let t = clamp(speed * 2.0, 0.0, 1.0);
        // Blend toward brighter color based on speed
        color = mix(color, vec3<f32>(0.3, 0.8, 1.0), t);
    }

    // Simple lighting using sun direction from uniform
    let light_dir = normalize(light.sun_direction);
    let ndotl = max(dot(normal, light_dir), 0.0);
    let ambient = 0.3;
    let diffuse = 0.7;
    let lighting = ambient + diffuse * ndotl;

    color = color * lighting;

    // Soft edge
    let edge_dist = sqrt(dist_sq) * 2.0;
    let alpha = smoothstep(1.0, 0.9, edge_dist);

    return vec4<f32>(color, alpha);
}
