// Environment background rendering for marching cubes
// Renders a fullscreen quad with the environment map as background

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

struct EnvParams {
    use_env_background: u32,
    background_r: f32,
    background_g: f32,
    background_b: f32,
    env_intensity: f32,
    // Ground-projected backdrop (GroundStaging in state/rendering.rs)
    ground_y: f32,
    ground_capture_height: f32,
    ground_enabled: u32,
    occluder_half_x: f32,
    occluder_half_z: f32,
    occluder_height: f32,
    ground_sky_share: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var env_tex: texture_2d<f32>;
@group(0) @binding(2) var env_sampler: sampler;
@group(0) @binding(3) var<uniform> env_params: EnvParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

const PI: f32 = 3.14159265359;

// Fullscreen triangle vertices
const POSITIONS: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    let pos = POSITIONS[vertex_index];
    output.position = vec4<f32>(pos, 0.9999, 1.0);  // Far plane
    output.uv = pos * 0.5 + 0.5;  // Convert to 0..1 range

    return output;
}

// Sample equirectangular environment map. Same convention as the CPU SH
// projection (compute_sh_irradiance in environment.rs): row 0 = +Y, and
// u = phi / 2pi for dir = (sin t cos phi, cos t, sin t sin phi). (The old
// mapping, v = 1 - t/pi with u offset by pi, sampled the antipode -dir.)
fn sample_environment(dir: vec3<f32>) -> vec3<f32> {
    let phi = atan2(dir.z, dir.x);
    let theta = acos(clamp(dir.y, -1.0, 1.0));
    let u = fract(phi / (2.0 * PI) + 1.0);
    let v = theta / PI;
    // Explicit LOD 0 (the map has a single mip): called from per-pixel branches
    return textureSampleLevel(env_tex, env_sampler, vec2<f32>(u, v), 0.0).rgb;
}

// Depth the backdrop has always drawn at (vs_main): "infinitely far"
const SKY_DEPTH: f32 = 0.9999;

struct Backdrop {
    color: vec3<f32>,
    depth: f32,
}

// Sky the pool box hides from the ground beside it. A wall of height H seen
// from a horizontal receiver at distance d blocks (1 - d/sqrt(d^2 + H^2)) / 2
// of the sky hemisphere (half at its foot); distance is to the footprint, so
// corners are slightly overestimated. Only the sky's share of the ground's
// light is removed — the sun part would need the box's shadow.
fn contact_occlusion(p: vec2<f32>) -> f32 {
    if (env_params.occluder_height <= 0.0) {
        return 1.0;
    }
    let outside = max(abs(p) - vec2<f32>(env_params.occluder_half_x, env_params.occluder_half_z), vec2<f32>(0.0));
    let d = length(outside);
    let h = env_params.occluder_height;
    let blocked = 0.5 * (1.0 - d / sqrt(d * d + h * h));
    return 1.0 - env_params.ground_sky_share * blocked;
}

// The HDR's nadir (straight down from where it was shot) is the worst part
// of any map: the equirect pole pinches it into radial seams, and it usually
// holds the tripod/photographer's shadow and stitching blur. The projected
// ground puts it right under the container, which refraction then shows
// through the water. Past NADIR_FADE_START the ground is re-read from a
// capture point shifted sideways (NADIR_SHIFT capture heights), which lands
// those points on ordinary textured ground ~35 deg below the horizon.
const NADIR_FADE_START: f32 = 0.866;  // sin(60 deg) below the horizon
const NADIR_FADE_END: f32 = 0.940;    // sin(70 deg)
const NADIR_SHIFT: f32 = 1.5;

fn ground_radiance(hit: vec3<f32>, capture: vec3<f32>) -> vec3<f32> {
    let dir = normalize(hit - capture);
    let color = sample_environment(dir);
    let fade = smoothstep(NADIR_FADE_START, NADIR_FADE_END, -dir.y);
    if (fade <= 0.0) {
        return color;
    }
    let shifted = capture + vec3<f32>(NADIR_SHIFT * env_params.ground_capture_height, 0.0, 0.0);
    return mix(color, sample_environment(normalize(hit - shifted)), fade);
}

fn backdrop(uv: vec2<f32>) -> Backdrop {
    var out: Backdrop;
    out.depth = SKY_DEPTH;
    // Solid color background
    if (env_params.use_env_background == 0u) {
        out.color = vec3<f32>(env_params.background_r, env_params.background_g, env_params.background_b);
        return out;
    }

    // Compute world-space ray direction using inverse matrices. uv comes from
    // the fullscreen triangle's clip xy, so it is already y-up: no flip.
    let ndc = uv * 2.0 - 1.0;
    let view_ray = normalize((camera.inv_projection * vec4<f32>(ndc, 1.0, 1.0)).xyz);
    let world_ray = normalize((camera.inv_view * vec4<f32>(view_ray, 0.0)).xyz);

    var radiance: vec3<f32>;
    var occlusion = 1.0;
    if (env_params.ground_enabled != 0u && world_ray.y < 0.0 && camera.camera_pos.y > env_params.ground_y) {
        // Ground-projected HDR ("HDRI backdrop"): the ray stops on a ground
        // plane, and the map is read from where it was shot (capture height
        // above the origin), so the photographed ground gets real parallax
        // and scale under the container. Far hits converge to the plain ray
        // direction, so the horizon stays seamless.
        let hit = camera.camera_pos + world_ray * ((env_params.ground_y - camera.camera_pos.y) / world_ray.y);
        let capture = vec3<f32>(0.0, env_params.ground_y + env_params.ground_capture_height, 0.0);
        radiance = ground_radiance(hit, capture);
        occlusion = contact_occlusion(hit.xz);
        let clip = camera.projection * camera.view * vec4<f32>(hit, 1.0);
        out.depth = clamp(clip.z / clip.w, 0.0, SKY_DEPTH);
    } else {
        radiance = sample_environment(world_ray);
    }

    // Unclipped HDR radiance: post_process tonemaps (the bright sky and the
    // sun disk keep their range for bloom and for refraction/SSR reads)
    out.color = max(radiance * env_params.env_intensity * occlusion, vec3<f32>(0.0));
    return out;
}

// Color only (Particles mode: its backdrop pipeline has no depth attachment)
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(backdrop(input.uv).color, 1.0);
}

struct GroundOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

// Color + depth: the projected ground is real geometry for the depth test,
// physical refraction and SSR (MC and SS backdrop passes)
@fragment
fn fs_ground(input: VertexOutput) -> GroundOutput {
    let b = backdrop(input.uv);
    var out: GroundOutput;
    out.color = vec4<f32>(b.color, 1.0);
    out.depth = b.depth;
    return out;
}
