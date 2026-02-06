// Spray particle rendering — velocity-stretched billboard sprites
// Uses instance_index to read spray particles from storage buffer.
// Each instance generates a 6-vertex quad (2 triangles), stretched along
// the screen-space velocity direction for a motion-streak look.

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
};

struct RenderParams {
    particle_size: f32,
    max_particles: u32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> spray_particles: array<SprayParticle>;
@group(0) @binding(2) var<uniform> render_params: RenderParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
    @location(2) stretch_ratio: f32,  // how elongated this particle is (for fragment shaping)
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

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    if (instance_index >= render_params.max_particles) {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        out.uv = vec2<f32>(0.0);
        out.alpha = 0.0;
        out.stretch_ratio = 1.0;
        return out;
    }

    let p = spray_particles[instance_index];

    // Dead particles: clip them off-screen
    if (p.lifetime <= 0.0) {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        out.uv = vec2<f32>(0.0);
        out.alpha = 0.0;
        out.stretch_ratio = 1.0;
        return out;
    }

    let world_pos = vec3<f32>(p.pos_x, p.pos_y, p.pos_z);
    let world_vel = vec3<f32>(p.vel_x, p.vel_y, p.vel_z);

    // Camera basis vectors
    let cam_right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let cam_up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
    let cam_fwd = vec3<f32>(camera.view[0][2], camera.view[1][2], camera.view[2][2]);

    // Project velocity onto the camera plane (screen-space velocity direction)
    let vel_right = dot(world_vel, cam_right);
    let vel_up = dot(world_vel, cam_up);
    let screen_speed = sqrt(vel_right * vel_right + vel_up * vel_up);

    // Per-particle size variation: 0.4x to 1.6x base size
    let size_rand = hash_float(instance_index * 7u + 31u);
    let base_size = render_params.particle_size * (0.4 + size_rand * 1.2);

    // Shrink as particle ages
    let life_frac = clamp(p.lifetime / max(p.max_lifetime, 0.001), 0.0, 1.0);
    let age_scale = 0.6 + 0.4 * life_frac;
    let size = base_size * age_scale;

    // Velocity stretching: elongate billboard along direction of travel
    // stretch_amount is how many extra radii to add along velocity axis
    let stretch_amount = clamp(screen_speed * 0.15, 0.0, 4.0);
    let stretch_factor = 1.0 + stretch_amount;

    let quad = QUAD_POS[vertex_index % 6u];

    var offset: vec3<f32>;
    if (screen_speed > 0.1) {
        // Build a billboard oriented along the velocity direction
        let vel_dir = vec2<f32>(vel_right, vel_up) / screen_speed;
        // "along" = velocity direction on screen, "across" = perpendicular
        let along = cam_right * vel_dir.x + cam_up * vel_dir.y;
        let across = cam_right * (-vel_dir.y) + cam_up * vel_dir.x;

        // Stretch along velocity, normal width across
        offset = along * quad.x * size * stretch_factor + across * quad.y * size;
    } else {
        // Slow particle: regular circular billboard
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

    // Very soft gaussian falloff — wide and diffuse for a misty look
    let falloff = exp(-dist * dist * 1.8);

    // Discard fully transparent fragments
    if (falloff < 0.01) {
        discard;
    }

    let alpha = in.alpha * falloff * 0.35;

    // Subtle per-fragment color variation: warmer at center, cooler at edges
    let warm = vec3<f32>(1.0, 0.98, 0.95);   // slight warm white
    let cool = vec3<f32>(0.88, 0.93, 1.0);    // blue-white
    let color = mix(warm, cool, dist);

    // Non-premultiplied output — alpha blending handles the math
    return vec4<f32>(color, alpha);
}
