// Screen-space fluid rendering — Normal Reconstruction
// Edge-aware central-difference method matching Splash implementation.
// After the narrow-range filter, edge depths are smooth (background was
// clamped, not rejected), so no boundary erosion is needed.

struct NormalParams {
    screen_width: u32,
    screen_height: u32,
    _pad0: u32,
    _pad1: u32,
}

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

@group(0) @binding(0) var<uniform> params: NormalParams;
@group(0) @binding(1) var<uniform> camera: CameraParams;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var output_normal: texture_storage_2d<rgba16float, write>;

fn uv_to_view(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = (1.0 - uv.y) * 2.0 - 1.0;
    let clip = vec4<f32>(ndc_x, ndc_y, 0.5, 1.0);
    let view_h = camera.inv_projection * clip;
    let view_dir = view_h.xyz / view_h.w;
    let t = -depth / view_dir.z;
    return view_dir * t;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= params.screen_width || y >= params.screen_height) {
        return;
    }

    let coord = vec2<i32>(i32(x), i32(y));
    let z_c = textureLoad(depth_tex, coord, 0).r;

    if (z_c <= 0.0) {
        textureStore(output_normal, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let xi = i32(x);
    let yi = i32(y);
    let w_max = i32(params.screen_width) - 1;
    let h_max = i32(params.screen_height) - 1;

    // Per-pixel derivatives (step=1, matching Splash)
    let step = 1i;
    var z_l = textureLoad(depth_tex, vec2<i32>(max(xi - step, 0), yi), 0).r;
    var z_r = textureLoad(depth_tex, vec2<i32>(min(xi + step, w_max), yi), 0).r;
    var z_b = textureLoad(depth_tex, vec2<i32>(xi, max(yi - step, 0)), 0).r;
    var z_t = textureLoad(depth_tex, vec2<i32>(xi, min(yi + step, h_max)), 0).r;

    // If a neighbor is empty (no coverage), substitute a very large depth
    // (matching Splash's 1e4 background convention). This creates an enormous
    // |z| derivative that the edge-aware selector rejects in favor of the
    // valid interior derivative. Using z_c here would be WRONG — it creates
    // near-zero |z| change that the selector prefers, producing lateral normals.
    if (z_l <= 0.0) { z_l = 1e8; }
    if (z_r <= 0.0) { z_r = 1e8; }
    if (z_b <= 0.0) { z_b = 1e8; }
    if (z_t <= 0.0) { z_t = 1e8; }

    let inv_w = 1.0 / f32(params.screen_width);
    let inv_h = 1.0 / f32(params.screen_height);
    let uv_c = vec2<f32>((f32(x) + 0.5) * inv_w, (f32(y) + 0.5) * inv_h);

    let pos_c = uv_to_view(uv_c, z_c);
    let step_f = f32(step);
    let pos_l = uv_to_view(vec2<f32>(uv_c.x - inv_w * step_f, uv_c.y), z_l);
    let pos_r = uv_to_view(vec2<f32>(uv_c.x + inv_w * step_f, uv_c.y), z_r);
    let pos_b = uv_to_view(vec2<f32>(uv_c.x, uv_c.y - inv_h * step_f), z_b);
    let pos_t = uv_to_view(vec2<f32>(uv_c.x, uv_c.y + inv_h * step_f), z_t);

    // Edge-aware: pick the derivative with smaller depth change (matching Splash)
    let ddx  = pos_r - pos_c;
    let ddx2 = pos_c - pos_l;
    let dx = select(ddx, ddx2, abs(ddx.z) > abs(ddx2.z));

    let ddy  = pos_t - pos_c;
    let ddy2 = pos_c - pos_b;
    let dy = select(ddy, ddy2, abs(ddy.z) > abs(ddy2.z));

    // Negative sign: screen y+ is view y-, so cross(right, down) points
    // into screen. Negating gives camera-facing normal (matching Splash).
    var normal = -normalize(cross(dx, dy));

    // Safety: ensure normal points toward camera (positive Z in view space)
    if (normal.z < 0.0) {
        normal = -normal;
    }

    textureStore(output_normal, coord, vec4<f32>(normal, 1.0));
}
