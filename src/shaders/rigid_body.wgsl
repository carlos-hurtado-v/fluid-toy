// Rigid body rendering shader — procedural shape generation
// Generates Cube, Sphere, Cylinder, or Torus from vertex_index

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
    light_dir: vec3<f32>,
    shape: u32,
    rot_row0: vec4<f32>,
    rot_row1: vec4<f32>,
    rot_row2: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> body: RigidBodyParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
}

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

const SHAPE_CUBE: u32 = 0u;
const SHAPE_SPHERE: u32 = 1u;
const SHAPE_CYLINDER: u32 = 2u;
const SHAPE_TORUS: u32 = 3u;

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

// === Shared rotation helper ===
fn rotate_local_to_world(local: vec3<f32>) -> vec3<f32> {
    return vec3(
        body.rot_row0.x * local.x + body.rot_row1.x * local.y + body.rot_row2.x * local.z,
        body.rot_row0.y * local.x + body.rot_row1.y * local.y + body.rot_row2.y * local.z,
        body.rot_row0.z * local.x + body.rot_row1.z * local.y + body.rot_row2.z * local.z,
    );
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var sv: ShapeVertex;
    switch (body.shape) {
        case SHAPE_SPHERE:   { sv = sphere_vertex(vi); }
        case SHAPE_CYLINDER: { sv = cylinder_vertex(vi); }
        case SHAPE_TORUS:    { sv = torus_vertex(vi); }
        default:             { sv = cube_vertex(vi); }
    }

    let local_pos = sv.pos * body.half_extent;
    let world_pos = rotate_local_to_world(local_pos) + body.position;
    let world_n = rotate_local_to_world(sv.norm);

    var out: VertexOutput;
    out.position = camera.projection * camera.view * vec4(world_pos, 1.0);
    out.normal = world_n;
    out.world_pos = world_pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(body.light_dir);

    let ambient = 0.15;
    let diffuse = max(dot(n, l), 0.0) * 0.7;

    let view_dir = normalize(camera.camera_pos - in.world_pos);
    let half_vec = normalize(l + view_dir);
    let spec = pow(max(dot(n, half_vec), 0.0), 32.0) * 0.3;

    let brightness = ambient + diffuse + spec;
    let color = body.color.rgb * brightness;
    return vec4(color, body.color.a);
}
