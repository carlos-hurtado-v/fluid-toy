// Mesh rigid body rendering shader — vertex buffer + textured
// Used for custom GLB models alongside the procedural shape shader

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
    shape: u32,
    rot_row0: vec4<f32>,
    rot_row1: vec4<f32>,
    rot_row2: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> body: RigidBodyParams;

@group(1) @binding(0) var base_texture: texture_2d<f32>;
@group(1) @binding(1) var base_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
}

fn rotate_local_to_world(local: vec3<f32>) -> vec3<f32> {
    return vec3(
        body.rot_row0.x * local.x + body.rot_row1.x * local.y + body.rot_row2.x * local.z,
        body.rot_row0.y * local.x + body.rot_row1.y * local.y + body.rot_row2.y * local.z,
        body.rot_row0.z * local.x + body.rot_row1.z * local.y + body.rot_row2.z * local.z,
    );
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let local_pos = in.position * body.half_extent;
    let world_pos = rotate_local_to_world(local_pos) + body.position;
    let world_n = normalize(rotate_local_to_world(in.normal));

    var out: VertexOutput;
    out.position = camera.projection * camera.view * vec4(world_pos, 1.0);
    out.normal = world_n;
    out.world_pos = world_pos;
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(body.light_dir);

    let tex_color = textureSample(base_texture, base_sampler, in.uv);

    let ambient = 0.15;
    let diffuse = max(dot(n, l), 0.0) * 0.7;

    let view_dir = normalize(camera.camera_pos - in.world_pos);
    let half_vec = normalize(l + view_dir);
    let spec = pow(max(dot(n, half_vec), 0.0), 32.0) * 0.3;

    let brightness = ambient + diffuse + spec;
    // in.color = per-vertex material color (white for textured primitives)
    // body.color = global tint (white = no tinting)
    let color = tex_color.rgb * in.color.rgb * body.color.rgb * brightness;
    return vec4(color, tex_color.a * in.color.a * body.color.a);
}
