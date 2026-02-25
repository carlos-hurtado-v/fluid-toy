// Container (opaque pool) vertex + fragment shader
// Blue tile pattern + IBL diffuse lighting

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    eye_position: vec4<f32>,
};

struct PoolStyle {
    tile_color: vec3<f32>,
    tile_scale: f32,
    grout_color: vec3<f32>,
    specular_strength: f32,
    light_dir: vec3<f32>,
    grout_width: f32,
    ibl_strength: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> container: ContainerGeometry;
@group(0) @binding(2) var<uniform> sh_coeffs: array<vec4<f32>, 9>;
@group(0) @binding(3) var<uniform> pool: PoolStyle;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) face_id: f32,
    @location(3) is_inner: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) face_id: f32,
    @location(3) is_inner: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let world_pos = local_to_world(container, input.position);
    let world_normal = local_dir_to_world(container, input.normal);

    var out: VertexOutput;
    out.clip_position = camera.projection * camera.view * vec4<f32>(world_pos, 1.0);
    out.world_position = world_pos;
    out.world_normal = world_normal;
    out.face_id = input.face_id;
    out.is_inner = input.is_inner;
    return out;
}

// --- Utility functions ---

// Simple hash for per-tile variation
fn hash2(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

// Evaluate order-2 spherical harmonics irradiance
fn evaluate_sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    var irradiance = sh_coeffs[0].rgb * 0.282095;
    irradiance += sh_coeffs[1].rgb * 0.488603 * n.y;
    irradiance += sh_coeffs[2].rgb * 0.488603 * n.z;
    irradiance += sh_coeffs[3].rgb * 0.488603 * n.x;
    irradiance += sh_coeffs[4].rgb * 1.092548 * n.x * n.y;
    irradiance += sh_coeffs[5].rgb * 1.092548 * n.y * n.z;
    irradiance += sh_coeffs[6].rgb * 0.315392 * (3.0 * n.z * n.z - 1.0);
    irradiance += sh_coeffs[7].rgb * 1.092548 * n.x * n.z;
    irradiance += sh_coeffs[8].rgb * 0.546274 * (n.x * n.x - n.y * n.y);
    return max(irradiance, vec3<f32>(0.0));
}

// --- Tile pattern ---
// Returns (variation, grout_factor) where grout_factor=1 means grout, 0 means tile
fn tile_pattern(uv: vec2<f32>) -> vec2<f32> {
    let scaled = uv * pool.tile_scale;
    let cell = floor(scaled);
    let f = fract(scaled);

    // Per-tile brightness variation (+-8%)
    let variation = (hash2(cell) - 0.5) * 0.16;

    // Grout detection: distance from edges
    let gw = pool.grout_width;
    let edge_x = min(f.x, 1.0 - f.x);
    let edge_y = min(f.y, 1.0 - f.y);
    let edge_dist = min(edge_x, edge_y);
    let grout = 1.0 - smoothstep(0.0, gw, edge_dist);

    return vec2<f32>(variation, grout);
}

// Project world position onto 2D UV based on face normal
fn face_uv(world_pos: vec3<f32>, normal: vec3<f32>) -> vec2<f32> {
    let abs_n = abs(normal);
    if abs_n.y > abs_n.x && abs_n.y > abs_n.z {
        return vec2<f32>(world_pos.x, world_pos.z);
    } else if abs_n.x > abs_n.z {
        return vec2<f32>(world_pos.z, world_pos.y);
    } else {
        return vec2<f32>(world_pos.x, world_pos.y);
    }
}

// --- Fragment shader ---

// Exterior color for outer faces and rim (neutral gray)
const EXTERIOR_COLOR: vec3<f32> = vec3<f32>(0.55, 0.55, 0.55);

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let V = normalize(camera.eye_position.xyz - input.world_position);
    var N = normalize(input.world_normal);

    // Two-sided lighting: flip normal if facing away from camera
    if dot(N, V) < 0.0 {
        N = -N;
    }

    let L = normalize(pool.light_dir);
    let H = normalize(L + V);
    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.0);

    // --- Outer faces: plain exterior ---
    if input.is_inner < 0.5 {
        let ibl = evaluate_sh_irradiance(N) * 0.4;
        let color = EXTERIOR_COLOR * (ibl + NdotL * 0.5);
        return vec4<f32>(color, 1.0);
    }

    // --- Inner faces: pool tile material ---

    // Get face UV for tile pattern
    let uv = face_uv(input.world_position, N);

    // Tile pattern
    let tile_info = tile_pattern(uv);
    let variation = tile_info.x;
    let grout = tile_info.y;

    // Blend tile and grout colors with per-tile variation
    let tile_col = pool.tile_color * (1.0 + variation);
    let base_color = mix(tile_col, pool.grout_color, grout);

    // IBL diffuse lighting
    let ibl_diffuse = evaluate_sh_irradiance(N) * pool.ibl_strength;

    // Wet specular (Blinn-Phong with Schlick Fresnel, F0=0.04 for wet ceramic)
    let NdotH = max(dot(N, H), 0.0);
    let f0 = 0.04;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);
    let spec = fresnel * pool.specular_strength * pow(NdotH, 64.0);

    // Final composition
    let color = base_color * (ibl_diffuse + NdotL * 0.65) + vec3<f32>(spec);
    return vec4<f32>(color, 1.0);
}
