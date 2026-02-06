// Marching Cubes - Back Face Depth Pass
// Renders only back faces to capture the "exit" depth for thickness calculation

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

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let vertex = vertices[vertex_index];

    var output: VertexOutput;
    let world_pos = vec4<f32>(vertex.position, 1.0);
    let view_pos = camera.view * world_pos;
    output.clip_position = camera.projection * view_pos;

    return output;
}

@fragment
fn fs_main() {
    // Just write depth - no color output needed
}
