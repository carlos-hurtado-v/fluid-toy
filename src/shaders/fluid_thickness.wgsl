// Fluid Thickness Pass - Additive rendering for thickness/absorption

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct FluidParams {
    particle_radius: f32,
    screen_width: f32,
    screen_height: f32,
    near: f32,
    far: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> params: FluidParams;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) particle_pos: vec3<f32>,
    @location(1) particle_vel: vec3<f32>,
) -> VertexOutput {
    var quad_verts = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );

    let local_pos = quad_verts[vertex_index];
    let view_center = (camera.view * vec4<f32>(particle_pos, 1.0)).xyz;
    let view_pos = view_center + vec3<f32>(local_pos * params.particle_radius, 0.0);
    let clip_pos = camera.projection * vec4<f32>(view_pos, 1.0);

    var output: VertexOutput;
    output.position = clip_pos;
    output.uv = local_pos;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) f32 {
    let dist_sq = dot(input.uv, input.uv);
    if (dist_sq > 1.0) {
        discard;
    }

    // Soft sphere falloff for smoother thickness accumulation
    let thickness = (1.0 - dist_sq) * 0.1;
    return thickness;
}
