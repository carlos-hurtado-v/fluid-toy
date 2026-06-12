// Whitewater rendering — kind-dependent billboard sprites.
//   spray  (0): velocity-stretched soft streaks (misty look)
//   foam   (1): round, dense, fake-sphere sunlit white caps; biased toward the
//               camera so surface foam survives the depth test against the
//               water mesh it sits on
//   bubble (2): small dim rings, hidden when bubbles_visible == 0 (they are
//               occluded by the water mesh in the main pass and show through
//               refraction via the background pass)
// Uses instance_index to read diffuse particles from the storage buffer.

struct Camera {
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
};

struct SprayParticle {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    lifetime: f32,
    vel_x: f32,
    vel_y: f32,
    vel_z: f32,
    max_lifetime: f32,
    kind: u32,
    age: f32,
    _pad1: f32,
    _pad2: f32,
};

struct RenderParams {
    particle_size: f32,
    max_particles: u32,
    bubbles_visible: u32,
    // 1 = foam is drawn by the screen-space density field instead (MC mode)
    foam_as_field: u32,
};

struct LightParams {
    sun_direction: vec3<f32>,
    sun_enabled: u32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    _pad2: f32,
    _padding: vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> spray_particles: array<SprayParticle>;
@group(0) @binding(2) var<uniform> render_params: RenderParams;
@group(0) @binding(3) var<uniform> light: LightParams;

const KIND_SPRAY: u32 = 0u;
const KIND_FOAM: u32 = 1u;
const KIND_BUBBLE: u32 = 2u;

// World-space offset toward the camera for foam, so billboards centered just
// under the marching-cubes skin still pass the depth test against it
const FOAM_DEPTH_BIAS: f32 = 0.02;

// Foam coalesces: caps grow from droplet size up to FOAM_GROWN_SCALE x over
// FOAM_GROW_TIME seconds of age, starting once the newborn spray grace ends
// (must match NEWBORN_SPRAY_TIME in spray_simulate.wgsl). Persistent settle
// foam reads as fat caps instead of faint dust, with no extra slider —
// particle_size stays the master scale.
const FOAM_GROW_START: f32 = 0.15;
const FOAM_GROW_TIME: f32 = 0.6;
const FOAM_GROWN_SCALE: f32 = 2.5;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
    @location(2) stretch_ratio: f32,  // how elongated this particle is (for fragment shaping)
    @location(3) @interpolate(flat) kind: u32,
    @location(4) @interpolate(flat) sun_view: vec4<f32>,  // xyz: view-space sun dir, w: enabled
};

fn hash(seed: u32) -> u32 {
    var x = seed;
    x = x ^ (x >> 16u);
    x = x * 0x45d9f3bu;
    x = x ^ (x >> 16u);
    x = x * 0x45d9f3bu;
    x = x ^ (x >> 16u);
    return x;
}

fn hash_float(seed: u32) -> f32 {
    return f32(hash(seed) & 0xFFFFu) / 65535.0;
}

// Billboard quad: 6 vertices for 2 triangles
const QUAD_POS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
);

fn discard_vertex() -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0);
    out.alpha = 0.0;
    out.stretch_ratio = 1.0;
    out.kind = 0u;
    out.sun_view = vec4<f32>(0.0);
    return out;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    if (instance_index >= render_params.max_particles) {
        return discard_vertex();
    }

    let p = spray_particles[instance_index];

    // Dead particles: clip them off-screen
    if (p.lifetime <= 0.0) {
        return discard_vertex();
    }
    if (p.kind == KIND_BUBBLE && render_params.bubbles_visible == 0u) {
        return discard_vertex();
    }
    if (p.kind == KIND_FOAM && render_params.foam_as_field != 0u) {
        return discard_vertex();
    }

    var out: VertexOutput;
    out.kind = p.kind;

    var world_pos = vec3<f32>(p.pos_x, p.pos_y, p.pos_z);
    let world_vel = vec3<f32>(p.vel_x, p.vel_y, p.vel_z);

    // Surface foam sits at the water mesh; nudge it toward the camera so it
    // wins the depth test instead of vanishing under the skin
    if (p.kind == KIND_FOAM) {
        world_pos = world_pos + normalize(camera.camera_pos - world_pos) * FOAM_DEPTH_BIAS;
    }

    // View-space sun direction for fake-sphere foam shading
    let sun_vs = (camera.view * vec4<f32>(light.sun_direction, 0.0)).xyz;
    out.sun_view = vec4<f32>(normalize(sun_vs), f32(light.sun_enabled));

    // Camera basis vectors
    let cam_right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let cam_up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);

    // Project velocity onto the camera plane (screen-space velocity direction)
    let vel_right = dot(world_vel, cam_right);
    let vel_up = dot(world_vel, cam_up);
    let screen_speed = sqrt(vel_right * vel_right + vel_up * vel_up);

    // Per-particle size variation, heavy-tailed: cubing the uniform random
    // gives mostly small droplets with occasional 2.5x outliers (0.3x-2.8x,
    // mean ~0.9x), which reads as natural variance instead of uniform grain
    let size_rand = hash_float(instance_index * 7u + 31u);
    var base_size = render_params.particle_size
        * (0.3 + 2.5 * size_rand * size_rand * size_rand);
    if (p.kind == KIND_BUBBLE) {
        base_size = base_size * 0.6;
    } else if (p.kind == KIND_FOAM) {
        let grow = smoothstep(FOAM_GROW_START, FOAM_GROW_START + FOAM_GROW_TIME, p.age);
        base_size = base_size * mix(1.0, FOAM_GROWN_SCALE, grow);
    }

    // Shrink as particle ages
    let life_frac = clamp(p.lifetime / max(p.max_lifetime, 0.001), 0.0, 1.0);
    let age_scale = 0.6 + 0.4 * life_frac;
    let size = base_size * age_scale;

    // Velocity stretching (spray only): elongate billboard along travel direction
    var stretch_factor = 1.0;
    if (p.kind == KIND_SPRAY) {
        stretch_factor = 1.0 + clamp(screen_speed * 0.15, 0.0, 4.0);
    }

    let quad = QUAD_POS[vertex_index % 6u];

    var offset: vec3<f32>;
    if (p.kind == KIND_SPRAY && screen_speed > 0.1) {
        // Build a billboard oriented along the velocity direction
        let vel_dir = vec2<f32>(vel_right, vel_up) / screen_speed;
        // "along" = velocity direction on screen, "across" = perpendicular
        let along = cam_right * vel_dir.x + cam_up * vel_dir.y;
        let across = cam_right * (-vel_dir.y) + cam_up * vel_dir.x;

        // Stretch along velocity, normal width across
        offset = along * quad.x * size * stretch_factor + across * quad.y * size;
    } else {
        // Round billboard (foam, bubbles, slow spray)
        offset = cam_right * quad.x * size + cam_up * quad.y * size;
    }

    let view_pos = camera.view * vec4<f32>(world_pos + offset, 1.0);
    out.position = camera.projection * view_pos;
    out.uv = quad;

    // Alpha fades with remaining lifetime, with per-particle brightness variation
    let alpha_rand = hash_float(instance_index * 13u + 97u);
    let brightness = 0.6 + alpha_rand * 0.4;
    out.alpha = life_frac * brightness;
    out.stretch_ratio = stretch_factor;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Elliptical shape: compress UV along stretch axis so the shape
    // remains roughly circular in the narrow dimension
    let uv = vec2<f32>(in.uv.x / max(in.stretch_ratio, 1.0), in.uv.y);
    let dist = length(uv);

    var color: vec3<f32>;
    var alpha: f32;

    if (in.kind == KIND_FOAM) {
        // Dense white cap with fake-sphere sun shading
        let n = vec3<f32>(uv.x, uv.y, sqrt(max(1.0 - dist * dist, 0.0)));
        let ndotl = max(dot(n, in.sun_view.xyz), 0.0);
        let shade = 0.45 + 0.7 * ndotl * in.sun_view.w;
        color = vec3<f32>(0.97, 0.99, 1.0) * shade;
        alpha = in.alpha * smoothstep(1.0, 0.6, dist) * 0.9;
    } else if (in.kind == KIND_BUBBLE) {
        // Dim ring: brighter toward the rim, soft outer cutoff
        let ring = 0.25 + 0.75 * smoothstep(0.35, 0.85, dist);
        color = vec3<f32>(0.85, 0.94, 1.0);
        alpha = in.alpha * ring * smoothstep(1.0, 0.85, dist) * 0.45;
    } else {
        // Spray: very soft gaussian falloff — wide and diffuse for a misty look
        let falloff = exp(-dist * dist * 1.8);
        // Subtle per-fragment color variation: warmer at center, cooler at edges
        let warm = vec3<f32>(1.0, 0.98, 0.95);   // slight warm white
        let cool = vec3<f32>(0.88, 0.93, 1.0);    // blue-white
        color = mix(warm, cool, dist);
        alpha = in.alpha * falloff * 0.35;
    }

    // Discard fully transparent fragments
    if (alpha < 0.005) {
        discard;
    }

    // Non-premultiplied output — alpha blending handles the math
    return vec4<f32>(color, alpha);
}
