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
    // Sun color x intensity, zeroed when the sun is disabled
    sun_rgb: vec3<f32>,
    ibl_strength: f32,
    caustic_strength: f32,
    shadow_strength: f32,
    caustic_focus: f32,
    _pad0: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> container: ContainerGeometry;
@group(0) @binding(2) var<uniform> sh_coeffs: array<vec4<f32>, 9>;
@group(0) @binding(3) var<uniform> pool: PoolStyle;
// Caustic floor map: RGB = refracted sun irradiance, A = water shadow coverage.
// Spans the inner floor rect in container-local space.
@group(0) @binding(4) var caustic_map: texture_2d<f32>;
@group(0) @binding(5) var caustic_sampler: sampler;

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

// Sun term scales, chosen so the default sun (intensity 2.0, warm color,
// luminance ~0.84) reproduces the legacy white NdotL * 0.65 brightness.
const SUN_DIFFUSE_SCALE: f32 = 0.39;
const SUN_SPECULAR_SCALE: f32 = 0.6;
const SUN_EXTERIOR_SCALE: f32 = 0.3;

// Analytic container self-shadowing. The pool walls all top out at local
// y = 0 (the mesh caps them at 50% of container height), so an interior
// point receives direct sun iff its ray to the sun exits through the open
// top rectangle. Penumbra widens with ray length to the opening.
fn rim_visibility(local_pos: vec3<f32>, light_dir_local: vec3<f32>) -> f32 {
    if light_dir_local.y < 0.02 {
        return 0.0; // sun at or below the rim plane: no direct sun inside
    }
    let t = -local_pos.y / light_dir_local.y;
    if t <= 0.0 {
        return 1.0; // already above the rim
    }
    let exit = local_pos.xz + light_dir_local.xz * t;
    // Signed distance to the opening boundary (positive = inside)
    let d = min(container.half_width - abs(exit.x), container.half_depth - abs(exit.y));
    let penumbra = 0.02 + 0.08 * t;
    return smoothstep(-penumbra, penumbra, d);
}

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
        let color = EXTERIOR_COLOR * (ibl + NdotL * pool.sun_rgb * SUN_EXTERIOR_SCALE);
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

    // --- Container self-shadowing (analytic, walls block the sun) ---
    let local = world_to_local(container, input.world_position);
    let L_local = world_dir_to_local(container, L);
    let rim_vis = rim_visibility(local, L_local);

    // --- Caustics + water shadow (floor face only) ---
    // The caustic map's shadow channel removes direct sun where water covers
    // the floor; the RGB channels re-deposit it as refracted irradiance.
    // Both are in units of "fraction of full direct sun", so flat water leaves
    // the overall floor brightness nearly unchanged (energy conservation).
    // The light raster sees the container as an occluder, so rim-shadowed
    // water emits neither photons nor shadow splats - no double counting.
    var sun_direct = NdotL * rim_vis;
    var caustic = vec3<f32>(0.0);
    if pool.caustic_strength > 0.0 {
        let n_local = world_dir_to_local(container, N);
        if n_local.y > 0.9 {
            let uv = vec2<f32>(
                local.x / container.half_width,
                local.z / container.half_depth,
            ) * 0.5 + 0.5;
            // SampleLevel: inside non-uniform control flow, no derivatives
            let c = textureSampleLevel(caustic_map, caustic_sampler, uv, 0.0);
            sun_direct = max(sun_direct - c.a * pool.shadow_strength, 0.0);
            // Focus: contrast exponent around the flat-water irradiance level,
            // which is ~NdotL (not 1.0) under a slanted sun. Filaments amplify,
            // dark lanes deepen, and the mean stays anchored at the local
            // direct-sun level, so it punches through tile camouflage without
            // blowout at any sun elevation.
            let anchor = max(NdotL, 0.05);
            let focused = pow(max(c.rgb, vec3<f32>(0.0)) / anchor, vec3<f32>(pool.caustic_focus)) * anchor;
            caustic = focused * pool.caustic_strength;
        }
    }

    // Final composition: direct sun, caustics, and specular all carry the
    // sun's color and intensity; specular is blocked in rim shadow too
    let sun_diffuse = pool.sun_rgb * SUN_DIFFUSE_SCALE;
    let color = base_color * (ibl_diffuse + (vec3<f32>(sun_direct) + caustic) * sun_diffuse)
        + pool.sun_rgb * SUN_SPECULAR_SCALE * spec * rim_vis;
    return vec4<f32>(color, 1.0);
}
