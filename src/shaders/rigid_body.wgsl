// Rigid body cube rendering shader
// Generates a solid cube from vertex_index with per-face normals and simple diffuse lighting

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    near_plane: f32,
    far_plane: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct RigidBodyParams {
    position: vec3<f32>,
    half_extent: f32,
    color: vec4<f32>,
    light_dir: vec3<f32>,
    _pad: f32,
    rot_row0: vec4<f32>,
    rot_row1: vec4<f32>,
    rot_row2: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> body: RigidBodyParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
}

// 36 vertices for 6 faces (2 triangles each), unit cube [-1, 1]
const CUBE_POSITIONS: array<vec3<f32>, 36> = array<vec3<f32>, 36>(
    // -X face
    vec3(-1.0, -1.0, -1.0), vec3(-1.0, -1.0,  1.0), vec3(-1.0,  1.0,  1.0),
    vec3(-1.0, -1.0, -1.0), vec3(-1.0,  1.0,  1.0), vec3(-1.0,  1.0, -1.0),
    // +X face
    vec3( 1.0, -1.0,  1.0), vec3( 1.0, -1.0, -1.0), vec3( 1.0,  1.0, -1.0),
    vec3( 1.0, -1.0,  1.0), vec3( 1.0,  1.0, -1.0), vec3( 1.0,  1.0,  1.0),
    // -Y face
    vec3(-1.0, -1.0, -1.0), vec3( 1.0, -1.0, -1.0), vec3( 1.0, -1.0,  1.0),
    vec3(-1.0, -1.0, -1.0), vec3( 1.0, -1.0,  1.0), vec3(-1.0, -1.0,  1.0),
    // +Y face
    vec3(-1.0,  1.0,  1.0), vec3( 1.0,  1.0,  1.0), vec3( 1.0,  1.0, -1.0),
    vec3(-1.0,  1.0,  1.0), vec3( 1.0,  1.0, -1.0), vec3(-1.0,  1.0, -1.0),
    // -Z face
    vec3( 1.0, -1.0, -1.0), vec3(-1.0, -1.0, -1.0), vec3(-1.0,  1.0, -1.0),
    vec3( 1.0, -1.0, -1.0), vec3(-1.0,  1.0, -1.0), vec3( 1.0,  1.0, -1.0),
    // +Z face
    vec3(-1.0, -1.0,  1.0), vec3( 1.0, -1.0,  1.0), vec3( 1.0,  1.0,  1.0),
    vec3(-1.0, -1.0,  1.0), vec3( 1.0,  1.0,  1.0), vec3(-1.0,  1.0,  1.0),
);

const CUBE_NORMALS: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3(-1.0,  0.0,  0.0),  // -X
    vec3( 1.0,  0.0,  0.0),  // +X
    vec3( 0.0, -1.0,  0.0),  // -Y
    vec3( 0.0,  1.0,  0.0),  // +Y
    vec3( 0.0,  0.0, -1.0),  // -Z
    vec3( 0.0,  0.0,  1.0),  // +Z
);

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let local_pos = CUBE_POSITIONS[vi] * body.half_extent;

    // Rotate from local to world space (transpose of stored world→local rows)
    let world_offset = vec3<f32>(
        body.rot_row0.x * local_pos.x + body.rot_row1.x * local_pos.y + body.rot_row2.x * local_pos.z,
        body.rot_row0.y * local_pos.x + body.rot_row1.y * local_pos.y + body.rot_row2.y * local_pos.z,
        body.rot_row0.z * local_pos.x + body.rot_row1.z * local_pos.y + body.rot_row2.z * local_pos.z,
    );
    let world_pos = world_offset + body.position;

    // Rotate normal the same way
    let local_n = CUBE_NORMALS[vi / 6u];
    let world_n = vec3<f32>(
        body.rot_row0.x * local_n.x + body.rot_row1.x * local_n.y + body.rot_row2.x * local_n.z,
        body.rot_row0.y * local_n.x + body.rot_row1.y * local_n.y + body.rot_row2.y * local_n.z,
        body.rot_row0.z * local_n.x + body.rot_row1.z * local_n.y + body.rot_row2.z * local_n.z,
    );

    var out: VertexOutput;
    out.position = camera.projection * camera.view * vec4<f32>(world_pos, 1.0);
    out.normal = world_n;
    out.world_pos = world_pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(body.light_dir);

    // Ambient + diffuse
    let ambient = 0.15;
    let diffuse = max(dot(n, l), 0.0) * 0.7;

    // Simple specular
    let view_dir = normalize(camera.camera_pos - in.world_pos);
    let half_vec = normalize(l + view_dir);
    let spec = pow(max(dot(n, half_vec), 0.0), 32.0) * 0.3;

    let brightness = ambient + diffuse + spec;
    let color = body.color.rgb * brightness;
    return vec4<f32>(color, body.color.a);
}
