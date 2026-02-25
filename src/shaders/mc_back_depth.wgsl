// Marching Cubes - Back/Front Face Depth Pass
// Renders faces to capture depth; clips to container bounds when enabled

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

struct ContainerClipParams {
    half_width: f32,
    half_depth: f32,
    half_height: f32,
    center_y: f32,
    sin_x: f32,
    cos_x: f32,
    sin_z: f32,
    cos_z: f32,
    clip_enabled: u32,
    clip_margin: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<uniform> clip: ContainerClipParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let vertex = vertices[vertex_index];

    var output: VertexOutput;
    let world_pos = vec4<f32>(vertex.position, 1.0);
    output.world_position = vertex.position;
    let view_pos = camera.view * world_pos;
    output.clip_position = camera.projection * view_pos;

    return output;
}

// Transform world position to container-local space (inverse tilt)
fn world_to_container(world_pos: vec3<f32>) -> vec3<f32> {
    var p = world_pos;
    p.y -= clip.center_y;

    // Inverse Z rotation (negate sin_z)
    let x1 = p.x * clip.cos_z + p.y * clip.sin_z;
    let y1 = -p.x * clip.sin_z + p.y * clip.cos_z;

    // Inverse X rotation (negate sin_x)
    let y2 = y1 * clip.cos_x + p.z * clip.sin_x;
    let z2 = -y1 * clip.sin_x + p.z * clip.cos_x;

    return vec3<f32>(x1, y2, z2);
}

struct FragmentInput {
    @location(0) world_position: vec3<f32>,
}

@fragment
fn fs_main(in: FragmentInput) {
    if (clip.clip_enabled != 0u) {
        let local = world_to_container(in.world_position);
        let m = clip.clip_margin;
        if (local.x < -(clip.half_width + m) || local.x > (clip.half_width + m) ||
            local.y < -(clip.half_height + m) || local.y > (clip.half_height + m) ||
            local.z < -(clip.half_depth + m) || local.z > (clip.half_depth + m)) {
            discard;
        }
    }
}

// === Front face pass with normal output (for SSR) ===

struct NormalVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}

@vertex
fn vs_normal(@builtin(vertex_index) vertex_index: u32) -> NormalVertexOutput {
    let vertex = vertices[vertex_index];

    var output: NormalVertexOutput;
    output.world_position = vertex.position;
    output.world_normal = vertex.normal;
    let view_pos = camera.view * vec4<f32>(vertex.position, 1.0);
    output.clip_position = camera.projection * view_pos;

    return output;
}

struct NormalFragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}

@fragment
fn fs_normal(in: NormalFragmentInput) -> @location(0) vec4<f32> {
    if (clip.clip_enabled != 0u) {
        let local = world_to_container(in.world_position);
        let m = clip.clip_margin;
        if (local.x < -(clip.half_width + m) || local.x > (clip.half_width + m) ||
            local.y < -(clip.half_height + m) || local.y > (clip.half_height + m) ||
            local.z < -(clip.half_depth + m) || local.z > (clip.half_depth + m)) {
            discard;
        }
    }

    let n = normalize(in.world_normal);
    return vec4<f32>(n, 1.0);
}
