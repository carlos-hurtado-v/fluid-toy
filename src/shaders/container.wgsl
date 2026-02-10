// Container (opaque pool) vertex + fragment shader
// Blinn-Phong lit pool material with inward-facing normals

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    eye_position: vec4<f32>,
};

struct ContainerRenderParams {
    wall_color: vec3<f32>,
    roughness: f32,
    floor_color: vec3<f32>,
    specular_strength: f32,
    light_dir: vec3<f32>,
    _pad0: f32,
    rotation_row0: vec4<f32>,
    rotation_row1: vec4<f32>,
    rotation_row2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> params: ContainerRenderParams;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) face_id: f32,
    @location(3) _pad: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) face_id: f32,
};

fn rotate_local_to_world(v: vec3<f32>) -> vec3<f32> {
    let r0 = params.rotation_row0.xyz;
    let r1 = params.rotation_row1.xyz;
    let r2 = params.rotation_row2.xyz;
    return vec3<f32>(
        dot(r0, v),
        dot(r1, v),
        dot(r2, v),
    );
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let world_pos = rotate_local_to_world(input.position);
    let world_normal = rotate_local_to_world(input.normal);

    var out: VertexOutput;
    out.clip_position = camera.projection * camera.view * vec4<f32>(world_pos, 1.0);
    out.world_position = world_pos;
    out.world_normal = world_normal;
    out.face_id = input.face_id;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let V = normalize(camera.eye_position.xyz - input.world_position);
    var N = normalize(input.world_normal);

    // Two-sided lighting: flip normal if facing away from camera
    if dot(N, V) < 0.0 {
        N = -N;
    }

    let L = normalize(params.light_dir);
    let H = normalize(L + V);

    // Pick color based on face_id (0 = floor, 1 = wall)
    let base_color = mix(params.floor_color, params.wall_color, input.face_id);

    // Ambient
    let ambient = 0.25 * base_color;

    // Diffuse
    let NdotL = max(dot(N, L), 0.0);
    let diffuse = 0.65 * NdotL * base_color;

    // Specular (Blinn-Phong)
    let NdotH = max(dot(N, H), 0.0);
    let spec = params.specular_strength * pow(NdotH, 16.0);

    let color = ambient + diffuse + vec3<f32>(spec);
    return vec4<f32>(color, 1.0);
}
