// Mesh rigid body rendering shader — vertex buffer + textured
// Used for custom GLB models alongside the procedural shape shader
// Concatenated with container_common.wgsl (rim_visibility for sun shadowing).

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
    // Unused since scene lighting moved to RbLightParams; kept for layout
    light_dir: vec3<f32>,
    shape: u32,
    rot_row0: vec4<f32>,
    rot_row1: vec4<f32>,
    rot_row2: vec4<f32>,
    prop_blades: u32,
    prop_pitch: f32,
    _pad0: f32,
    _pad1: f32,
}

// Scene lighting shared by all bodies — mirrors GpuRbLightParams (Rust)
struct RbLightParams {
    // Direction toward the sun (world space, normalized)
    sun_dir: vec3<f32>,
    // SH ambient scale (environment intensity)
    ibl_strength: f32,
    // Sun color x intensity, zeroed when the sun is disabled
    sun_rgb: vec3<f32>,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<storage, read> bodies: array<RigidBodyParams>;
@group(0) @binding(2) var<uniform> rb_light: RbLightParams;
@group(0) @binding(3) var<uniform> sh_coeffs: array<vec4<f32>, 9>;
@group(0) @binding(4) var<uniform> container: ContainerGeometry;

@group(1) @binding(0) var base_texture: texture_2d<f32>;
@group(1) @binding(1) var base_sampler: sampler;

const INV_PI: f32 = 0.31830988;
const BODY_SPEC_STRENGTH: f32 = 0.5;

// evaluate_sh_irradiance(): sh_common.wgsl, prepended at module creation

// Scene-coherent body shading — identical to rigid_body.wgsl's shade_body
fn shade_body(albedo: vec3<f32>, n: vec3<f32>, v: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let l = normalize(rb_light.sun_dir);
    let ndotl = max(dot(n, l), 0.0);

    let local = world_to_local(container, world_pos);
    let l_local = world_dir_to_local(container, l);
    let rim = rim_visibility(container, local, l_local);

    let ambient = evaluate_sh_irradiance(n) * rb_light.ibl_strength;
    let sun = rb_light.sun_rgb * (ndotl * rim);

    let h = normalize(l + v);
    let ndoth = max(dot(n, h), 0.0);
    let ndotv = max(dot(n, v), 0.0);
    let fresnel = 0.04 + 0.96 * pow(1.0 - ndotv, 5.0);
    let spec = rb_light.sun_rgb * (fresnel * pow(ndoth, 64.0) * BODY_SPEC_STRENGTH * rim);

    return albedo * (ambient + sun) * INV_PI + spec;
}

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
    @location(4) @interpolate(flat) body_idx: u32,
}

fn rotate_local_to_world(body: RigidBodyParams, local: vec3<f32>) -> vec3<f32> {
    return vec3(
        body.rot_row0.x * local.x + body.rot_row1.x * local.y + body.rot_row2.x * local.z,
        body.rot_row0.y * local.x + body.rot_row1.y * local.y + body.rot_row2.y * local.z,
        body.rot_row0.z * local.x + body.rot_row1.z * local.y + body.rot_row2.z * local.z,
    );
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) ii: u32) -> VertexOutput {
    let body = bodies[ii];
    let local_pos = in.position * body.half_extent;
    let world_pos = rotate_local_to_world(body, local_pos) + body.position;
    let world_n = normalize(rotate_local_to_world(body, in.normal));

    var out: VertexOutput;
    out.position = camera.projection * camera.view * vec4(world_pos, 1.0);
    out.normal = world_n;
    out.world_pos = world_pos;
    out.uv = in.uv;
    out.color = in.color;
    out.body_idx = ii;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let body = bodies[in.body_idx];
    let n = normalize(in.normal);
    let v = normalize(camera.camera_pos - in.world_pos);

    let tex_color = textureSample(base_texture, base_sampler, in.uv);

    // in.color = per-vertex material color (white for textured primitives)
    // body.color = global tint (white = no tinting)
    let albedo = tex_color.rgb * in.color.rgb * body.color.rgb;
    let color = shade_body(albedo, n, v, in.world_pos);
    return vec4(color, tex_color.a * in.color.a * body.color.a);
}
