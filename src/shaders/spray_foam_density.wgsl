// Whitewater field splatting — foam and bubble particles draw as soft
// additive weights into a half-resolution RG16Float buffer instead of as
// individual sprites. R = surface foam: depth-gated against the water front
// face, composited by the MC shader as surface whitening (connected patches,
// FLIP-style). G = aeration: bubbles plus the depth-discarded share of
// submerged foam, composited as milkiness INSIDE the water (entrained-air
// plumes, vortex cores). Splats stretch along screen-space velocity so
// advected whitewater forms streaks.

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
    foam_as_field: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> spray_particles: array<SprayParticle>;
@group(0) @binding(2) var<uniform> render_params: RenderParams;
// Water front depth (1 frame stale: the prepass runs inside the MC render,
// after this pass). Splats below the visible surface fade out — submerged
// whitewater is bubbles, not surface foam, and without this the screen-space
// field accumulates the whole churned VOLUME as one saturated slab.
@group(0) @binding(4) var water_depth_tex: texture_depth_2d;
@group(0) @binding(5) var water_depth_sampler: sampler;

const KIND_FOAM: u32 = 1u;
const KIND_BUBBLE: u32 = 2u;

// View-space falloff (meters) for SURFACE foam below the water front face;
// BIAS spares foam straddling the iso-surface itself (MC skin sits above
// particle centers)
const DEPTH_FADE: f32 = 0.2;
const DEPTH_BIAS: f32 = 0.045;
// Aeration falloff — submerged whitewater keeps contributing well below the
// surface (that IS the phenomenon); extinction through the water column caps
// it on a much longer scale
const AERATION_FADE: f32 = 0.5;
// Share of the depth-discarded surface-foam energy converted to aeration
const FOAM_AERATION_SHARE: f32 = 0.8;
// Bubble splats are individually weaker than foam (they represent diffuse
// volume scattering, not an opaque carpet) but stack deep in plumes
const BUBBLE_SPLAT_WEIGHT: f32 = 0.35;

// Splat footprint relative to the sprite base size — wide enough that nearby
// foam overlaps into patches; the gaussian falloff keeps isolated splats dim.
const SPLAT_SCALE: f32 = 4.5;
// Peak density contribution of one fully-grown splat at its center. The
// composite threshold (mc_render FOAM_DENSITY_LO) sits below a single grown
// splat but above a newborn one, so accumulations and mature foam read while
// fresh sparse emission stays subtle.
const SPLAT_WEIGHT: f32 = 0.7;
// Foam coalesces: contribution ramps in with age (replaces the old
// sprite-size growth). GROW_START mirrors NEWBORN_SPRAY_TIME in
// spray_simulate.wgsl — keep them in sync.
const GROW_START: f32 = 0.15;
const GROW_TIME: f32 = 0.6;
const NEWBORN_WEIGHT: f32 = 0.3;
// Screen-space velocity stretching (milder than spray streaks)
const STRETCH_PER_SPEED: f32 = 0.2;
const MAX_STRETCH: f32 = 2.0;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) weight: f32,
    @location(2) stretch_ratio: f32,
    // NDC xy interpolated window-linear = exact per-fragment NDC, giving the
    // full-res screen UV for the depth sample without knowing target dims
    @location(3) @interpolate(linear) ndc: vec2<f32>,
    @location(4) view_dist: f32,
    @location(5) @interpolate(flat) kind: u32,
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
    out.weight = 0.0;
    out.stretch_ratio = 1.0;
    out.ndc = vec2<f32>(0.0);
    out.view_dist = 0.0;
    out.kind = KIND_FOAM;
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
    if (p.lifetime <= 0.0 || (p.kind != KIND_FOAM && p.kind != KIND_BUBBLE)) {
        return discard_vertex();
    }

    var out: VertexOutput;

    let world_pos = vec3<f32>(p.pos_x, p.pos_y, p.pos_z);
    let world_vel = vec3<f32>(p.vel_x, p.vel_y, p.vel_z);

    // Camera basis + screen-space velocity for stretching
    let cam_right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let cam_up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
    let vel_right = dot(world_vel, cam_right);
    let vel_up = dot(world_vel, cam_up);
    let screen_speed = sqrt(vel_right * vel_right + vel_up * vel_up);

    // Same heavy-tailed per-particle size variance as the sprite path
    let size_rand = hash_float(instance_index * 7u + 31u);
    let size = render_params.particle_size
        * (0.3 + 2.5 * size_rand * size_rand * size_rand)
        * SPLAT_SCALE;

    // Density weight: lifetime fade x age-based coalescence. Same formula for
    // both kinds (continuity when a particle flickers across the foam/bubble
    // neighbor threshold), only the base weight differs.
    let life_frac = clamp(p.lifetime / max(p.max_lifetime, 0.001), 0.0, 1.0);
    let grow = smoothstep(GROW_START, GROW_START + GROW_TIME, p.age);
    var base_weight = SPLAT_WEIGHT;
    if (p.kind == KIND_BUBBLE) {
        base_weight = BUBBLE_SPLAT_WEIGHT;
    }
    out.weight = base_weight * life_frac * mix(NEWBORN_WEIGHT, 1.0, grow);
    out.kind = p.kind;

    let stretch = 1.0 + clamp(screen_speed * STRETCH_PER_SPEED, 0.0, MAX_STRETCH);
    out.stretch_ratio = stretch;

    let quad = QUAD_POS[vertex_index % 6u];
    var offset: vec3<f32>;
    if (screen_speed > 0.1) {
        let vel_dir = vec2<f32>(vel_right, vel_up) / screen_speed;
        let along = cam_right * vel_dir.x + cam_up * vel_dir.y;
        let across = cam_right * (-vel_dir.y) + cam_up * vel_dir.x;
        offset = along * quad.x * size * stretch + across * quad.y * size;
    } else {
        offset = cam_right * quad.x * size + cam_up * quad.y * size;
    }

    let view_pos = camera.view * vec4<f32>(world_pos + offset, 1.0);
    out.position = camera.projection * view_pos;
    out.ndc = out.position.xy / max(out.position.w, 1e-6);
    out.view_dist = -view_pos.z;
    out.uv = quad;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Compress UV along the stretch axis so the splat stays soft and elliptical
    let uv = vec2<f32>(in.uv.x / max(in.stretch_ratio, 1.0), in.uv.y);
    let r_sq = dot(uv, uv);
    let falloff = exp(-r_sq * 4.0);

    // Depth below the visible water surface. Linearize the sampled depth
    // from the projection matrix itself (convention-exact): with
    // clip.w = -view_z, view distance = m32 / (m22 + ndc_depth).
    let screen_uv = vec2<f32>(in.ndc.x * 0.5 + 0.5, 0.5 - in.ndc.y * 0.5);
    let water_ndc_depth = textureSampleLevel(water_depth_tex, water_depth_sampler, screen_uv, 0u);
    let m22 = camera.projection[2][2];
    let m32 = camera.projection[3][2];
    let water_dist = m32 / (m22 + water_ndc_depth);
    let depth_below = max(in.view_dist - water_dist - DEPTH_BIAS, 0.0);
    let surf_atten = exp(-depth_below / DEPTH_FADE);
    let aer_atten = exp(-depth_below / AERATION_FADE);

    let base = in.weight * falloff;
    var field = vec2<f32>(0.0);
    if (in.kind == KIND_FOAM) {
        // Surface share stays in R; the depth-discarded remainder becomes
        // aeration instead of vanishing (submerged foam = entrained air)
        field = vec2<f32>(
            base * surf_atten,
            base * (1.0 - surf_atten) * aer_atten * FOAM_AERATION_SHARE,
        );
    } else {
        field = vec2<f32>(0.0, base * aer_atten);
    }
    return vec4<f32>(field, 0.0, 1.0);
}
