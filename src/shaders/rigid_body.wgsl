// Rigid body rendering shader — procedural shape generation
// Generates Cube, Sphere, Cylinder, or Torus from vertex_index
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

const INV_PI: f32 = 0.31830988;
const BODY_SPEC_STRENGTH: f32 = 0.5;

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

// Scene-coherent body shading: SH ambient + rim-shadowed sun, Lambert with
// the proper 1/pi (evaluate_sh_irradiance returns irradiance, and the sun
// term ndotl * sun_rgb is one too), plus a Schlick-Fresnel Blinn-Phong lobe
// carrying the sun color (F0 = 0.04 dielectric).
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

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) @interpolate(flat) body_idx: u32,
}

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

const SHAPE_CUBE: u32 = 0u;
const SHAPE_SPHERE: u32 = 1u;
const SHAPE_CYLINDER: u32 = 2u;
const SHAPE_TORUS: u32 = 3u;
const SHAPE_PROPELLER: u32 = 5u;

// Propeller proportions in units of half_extent (spin axis = Y).
// Must match state/rigid_body.rs PROP_* and the SDF in sph_integrate_3d.wgsl.
const PROP_HUB_RADIUS: f32 = 0.25;
const PROP_HUB_HALF_HEIGHT: f32 = 0.30;
const PROP_BLADE_CENTER: f32 = 0.55;
const PROP_BLADE_HALF: vec3<f32> = vec3<f32>(0.44, 0.18, 0.045);

const SPHERE_SLICES: u32 = 32u;
const SPHERE_STACKS: u32 = 16u;
const CYL_SEGMENTS: u32 = 32u;
const TORUS_MAJOR: u32 = 32u;
const TORUS_MINOR: u32 = 16u;

struct ShapeVertex {
    pos: vec3<f32>,
    norm: vec3<f32>,
}

// === CUBE (36 vertices) ===
fn cube_vertex(vi: u32) -> ShapeVertex {
    let face = vi / 6u;
    let vert = vi % 6u;

    // Quad vertex pattern (CCW winding from outside)
    var u: f32; var v: f32;
    switch (vert) {
        case 0u: { u = -1.0; v = -1.0; }
        case 1u: { u =  1.0; v =  1.0; }
        case 2u: { u =  1.0; v = -1.0; }
        case 3u: { u = -1.0; v = -1.0; }
        case 4u: { u = -1.0; v =  1.0; }
        default: { u =  1.0; v =  1.0; }
    }

    var pos: vec3<f32>;
    var norm: vec3<f32>;
    switch (face) {
        case 0u: { pos = vec3(-1.0,  v,  -u); norm = vec3(-1.0, 0.0, 0.0); } // -X
        case 1u: { pos = vec3( 1.0,  v,   u); norm = vec3( 1.0, 0.0, 0.0); } // +X
        case 2u: { pos = vec3( u, -1.0,  -v); norm = vec3(0.0, -1.0, 0.0); } // -Y
        case 3u: { pos = vec3( u,  1.0,   v); norm = vec3(0.0,  1.0, 0.0); } // +Y
        case 4u: { pos = vec3( u,  v, -1.0); norm = vec3(0.0, 0.0, -1.0); }  // -Z
        default: { pos = vec3(-u,  v,  1.0); norm = vec3(0.0, 0.0,  1.0); }  // +Z
    }

    return ShapeVertex(pos, norm);
}

// === SPHERE (32×16 UV sphere = 3072 vertices) ===
fn sphere_point(stack: u32, slice: u32) -> vec3<f32> {
    let theta = f32(stack) * PI / f32(SPHERE_STACKS);
    let phi = f32(slice % SPHERE_SLICES) * TWO_PI / f32(SPHERE_SLICES);
    return vec3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
}

fn sphere_vertex(vi: u32) -> ShapeVertex {
    let vert_in_tri = vi % 3u;
    let tri_idx = vi / 3u;
    let tri_in_quad = tri_idx % 2u;
    let quad_idx = tri_idx / 2u;
    let stack = quad_idx / SPHERE_SLICES;
    let slice = quad_idx % SPHERE_SLICES;

    var s = stack;
    var sl = slice;
    if (tri_in_quad == 0u) {
        // Triangle 0: (s,sl), (s+1,sl+1), (s+1,sl) — CCW from outside
        if (vert_in_tri == 1u) { s += 1u; sl += 1u; }
        else if (vert_in_tri == 2u) { s += 1u; }
    } else {
        // Triangle 1: (s,sl), (s,sl+1), (s+1,sl+1) — CCW from outside
        if (vert_in_tri == 1u) { sl += 1u; }
        else if (vert_in_tri == 2u) { s += 1u; sl += 1u; }
    }

    let p = sphere_point(s, sl);
    return ShapeVertex(p, p); // normal = position for unit sphere
}

// === CYLINDER (32 segments, capped, height=2, radius=1) ===
// Layout: barrel (32×6=192 verts) + top cap (32×3=96) + bottom cap (32×3=96) = 384
fn cylinder_vertex(vi: u32) -> ShapeVertex {
    let barrel_verts = CYL_SEGMENTS * 6u;
    let cap_verts = CYL_SEGMENTS * 3u;

    if (vi < barrel_verts) {
        // Barrel
        let vert_in_tri = vi % 3u;
        let tri_idx = vi / 3u;
        let tri_in_quad = tri_idx % 2u;
        let seg = tri_idx / 2u;

        var s = seg;
        var top = false;
        if (tri_in_quad == 0u) {
            // (seg,bot), (seg+1,top), (seg+1,bot) — CCW from outside
            if (vert_in_tri == 1u) { s += 1u; top = true; }
            else if (vert_in_tri == 2u) { s += 1u; }
        } else {
            // (seg,bot), (seg,top), (seg+1,top) — CCW from outside
            if (vert_in_tri == 1u) { top = true; }
            else if (vert_in_tri == 2u) { s += 1u; top = true; }
        }

        let phi = f32(s % CYL_SEGMENTS) * TWO_PI / f32(CYL_SEGMENTS);
        let x = cos(phi);
        let z = sin(phi);
        var y = -1.0;
        if (top) { y = 1.0; }

        return ShapeVertex(vec3(x, y, z), vec3(x, 0.0, z));
    } else if (vi < barrel_verts + cap_verts) {
        // Top cap (y = +1)
        let local_vi = vi - barrel_verts;
        let tri = local_vi / 3u;
        let vert_in_tri = local_vi % 3u;

        if (vert_in_tri == 0u) {
            return ShapeVertex(vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0));
        }
        // CCW from above: center, seg+1, seg
        let seg = tri + 2u - vert_in_tri;
        let phi = f32(seg % CYL_SEGMENTS) * TWO_PI / f32(CYL_SEGMENTS);
        return ShapeVertex(vec3(cos(phi), 1.0, sin(phi)), vec3(0.0, 1.0, 0.0));
    } else {
        // Bottom cap (y = -1)
        let local_vi = vi - barrel_verts - cap_verts;
        let tri = local_vi / 3u;
        let vert_in_tri = local_vi % 3u;

        if (vert_in_tri == 0u) {
            return ShapeVertex(vec3(0.0, -1.0, 0.0), vec3(0.0, -1.0, 0.0));
        }
        // CCW from below: center, seg, seg+1
        let seg = tri + vert_in_tri - 1u;
        let phi = f32(seg % CYL_SEGMENTS) * TWO_PI / f32(CYL_SEGMENTS);
        return ShapeVertex(vec3(cos(phi), -1.0, sin(phi)), vec3(0.0, -1.0, 0.0));
    }
}

// === TORUS (32×16, major_radius=1, minor_radius=0.3) ===
const TORUS_MINOR_R: f32 = 0.3;

fn torus_point(major_idx: u32, minor_idx: u32) -> ShapeVertex {
    let u_angle = f32(major_idx % TORUS_MAJOR) * TWO_PI / f32(TORUS_MAJOR);
    let v_angle = f32(minor_idx % TORUS_MINOR) * TWO_PI / f32(TORUS_MINOR);

    let cos_u = cos(u_angle);
    let sin_u = sin(u_angle);
    let cos_v = cos(v_angle);
    let sin_v = sin(v_angle);

    let r = 1.0 + TORUS_MINOR_R * cos_v;
    let pos = vec3(r * cos_u, TORUS_MINOR_R * sin_v, r * sin_u);
    let norm = vec3(cos_v * cos_u, sin_v, cos_v * sin_u);

    return ShapeVertex(pos, norm);
}

fn torus_vertex(vi: u32) -> ShapeVertex {
    let vert_in_tri = vi % 3u;
    let tri_idx = vi / 3u;
    let tri_in_quad = tri_idx % 2u;
    let quad_idx = tri_idx / 2u;
    let major = quad_idx / TORUS_MINOR;
    let minor = quad_idx % TORUS_MINOR;

    var ma = major;
    var mi = minor;
    if (tri_in_quad == 0u) {
        // CCW from outside
        if (vert_in_tri == 1u) { ma += 1u; mi += 1u; }
        else if (vert_in_tri == 2u) { ma += 1u; }
    } else {
        // CCW from outside
        if (vert_in_tri == 1u) { mi += 1u; }
        else if (vert_in_tri == 2u) { ma += 1u; mi += 1u; }
    }

    return torus_point(ma, mi);
}

// === PROPELLER (hub cylinder 384 verts + 36 per blade) ===
// Forward transform per blade: scale → offset radially → pitch around X →
// rotate into sector around Y. The SDF applies the exact inverse.
fn propeller_vertex(vi: u32, blades: u32, pitch: f32) -> ShapeVertex {
    let hub_verts = CYL_SEGMENTS * 12u; // 384
    if (vi < hub_verts) {
        let cv = cylinder_vertex(vi);
        // Non-uniform scale keeps cylinder normals valid (barrel normals are
        // horizontal, cap normals vertical)
        return ShapeVertex(
            vec3(cv.pos.x * PROP_HUB_RADIUS, cv.pos.y * PROP_HUB_HALF_HEIGHT, cv.pos.z * PROP_HUB_RADIUS),
            cv.norm,
        );
    }

    let bi = (vi - hub_verts) / 36u;
    let bv = (vi - hub_verts) % 36u;
    let cv = cube_vertex(bv);

    var pos = cv.pos * PROP_BLADE_HALF;
    pos.x += PROP_BLADE_CENTER;
    var norm = cv.norm; // box face normals stay axis-aligned under scale

    // Pitch around the radial (X) axis
    let cp = cos(pitch);
    let sp = sin(pitch);
    pos = vec3(pos.x, cp * pos.y - sp * pos.z, sp * pos.y + cp * pos.z);
    norm = vec3(norm.x, cp * norm.y - sp * norm.z, sp * norm.y + cp * norm.z);

    // Place the blade in its sector around Y
    let theta = f32(bi) * TWO_PI / f32(blades);
    let ct = cos(theta);
    let st = sin(theta);
    pos = vec3(ct * pos.x + st * pos.z, pos.y, -st * pos.x + ct * pos.z);
    norm = vec3(ct * norm.x + st * norm.z, norm.y, -st * norm.x + ct * norm.z);

    return ShapeVertex(pos, norm);
}

// === Shared rotation helper ===
fn rotate_local_to_world(body: RigidBodyParams, local: vec3<f32>) -> vec3<f32> {
    return vec3(
        body.rot_row0.x * local.x + body.rot_row1.x * local.y + body.rot_row2.x * local.z,
        body.rot_row0.y * local.x + body.rot_row1.y * local.y + body.rot_row2.y * local.z,
        body.rot_row0.z * local.x + body.rot_row1.z * local.y + body.rot_row2.z * local.z,
    );
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VertexOutput {
    let body = bodies[ii];
    var sv: ShapeVertex;
    switch (body.shape) {
        case SHAPE_SPHERE:    { sv = sphere_vertex(vi); }
        case SHAPE_CYLINDER:  { sv = cylinder_vertex(vi); }
        case SHAPE_TORUS:     { sv = torus_vertex(vi); }
        case SHAPE_PROPELLER: { sv = propeller_vertex(vi, max(body.prop_blades, 1u), body.prop_pitch); }
        default:              { sv = cube_vertex(vi); }
    }

    let local_pos = sv.pos * body.half_extent;
    let world_pos = rotate_local_to_world(body, local_pos) + body.position;
    let world_n = rotate_local_to_world(body, sv.norm);

    var out: VertexOutput;
    out.position = camera.projection * camera.view * vec4(world_pos, 1.0);
    out.normal = world_n;
    out.world_pos = world_pos;
    out.body_idx = ii;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let body = bodies[in.body_idx];
    let n = normalize(in.normal);
    let v = normalize(camera.camera_pos - in.world_pos);
    let color = shade_body(body.color.rgb, n, v, in.world_pos);
    return vec4(color, body.color.a);
}
