// Environment background rendering for marching cubes
// Renders a fullscreen quad with the environment map as background

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var env_tex: texture_2d<f32>;
@group(0) @binding(2) var env_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

const PI: f32 = 3.14159265359;

// Fullscreen triangle vertices
const POSITIONS: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    let pos = POSITIONS[vertex_index];
    output.position = vec4<f32>(pos, 0.9999, 1.0);  // Far plane
    output.uv = pos * 0.5 + 0.5;  // Convert to 0..1 range

    return output;
}

// Sample equirectangular environment map
fn sample_environment(dir: vec3<f32>) -> vec3<f32> {
    let phi = atan2(dir.z, dir.x);
    let theta = acos(clamp(dir.y, -1.0, 1.0));
    let u = (phi + PI) / (2.0 * PI);
    let v = 1.0 - theta / PI;  // Flip V to match screen-space shader
    return textureSample(env_tex, env_sampler, vec2<f32>(u, v)).rgb;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Compute world-space ray direction using inverse matrices (same as screen-space shader)
    let ndc = vec2<f32>(input.uv.x * 2.0 - 1.0, 1.0 - 2.0 * input.uv.y);
    let view_ray = normalize((camera.inv_projection * vec4<f32>(ndc, 1.0, 1.0)).xyz);
    let world_ray = normalize((camera.inv_view * vec4<f32>(view_ray, 0.0)).xyz);

    var color = sample_environment(world_ray);

    // Simple tone mapping (match screen-space HDR handling)
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

    return vec4<f32>(color, 1.0);
}
