// Caustics - Photon Splat Pass
// One photon per light-space G-buffer texel, drawn as an instanced quad into
// the caustic atlas (additive). Four kinds per photon:
//   kind 0..2: refracted ray per color channel (chromatic dispersion),
//              carrying Fresnel-transmitted, Beer-Lambert-attenuated flux
//   kind 3:    straight (unrefracted) ray, marking direct sunlight removed
//              from the receiving face (alpha channel = water shadow)
// Both use the same normalized gaussian kernel, so the energy removed by the
// shadow channel equals the energy re-deposited by the caustic channels
// (modulo Fresnel reflection and absorption) - conservation by construction.
//
// Each photon lands on the FIRST interior face its ray hits: the floor or
// one of the four walls (up-refracted rays off steep wave faces are valid -
// they make the shimmer bands above the waterline). Rays exiting the open
// top are culled. Atlas layout (width res, height 3*res, vertical strip):
//   rows [0, res)              face 0: floor  (fp = local.xz)
//   rows [res + k*res/2, +res/2) face 1+k: wall +X/-X/+Z/-Z, v = -local.y/hh
// Walls span local y in [-hh, 0] (the pool mesh caps them at the rim plane).
// Concatenated with container_common.wgsl (local-space transforms).

struct CausticsParams {
    light_view_proj: mat4x4<f32>,
    sun_dir: vec3<f32>,
    flux_area: f32,
    ior_rgb: vec3<f32>,
    sigma: f32,
    absorb_rgb: vec3<f32>,
    inv_two_sigma_sq: f32,
    splat_norm: f32,
    splat_radius: f32,
    optical_density: f32,
    light_res: u32,
    time: f32,
    ripple_strength: f32,
    // Photon kinds per texel: 4 = R/G/B/shadow (chromatic), 2 = white/shadow
    kinds: u32,
    // Enabled rigid bodies in the render array (photon occluders)
    body_count: u32,
}

// Mirrors GpuRigidBodyRender (rigid_body.wgsl RigidBodyParams, 112 bytes)
struct RigidBodyParams {
    position: vec3<f32>,
    half_extent: f32,
    color: vec4<f32>,
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

@group(0) @binding(0) var<uniform> params: CausticsParams;
@group(0) @binding(1) var<uniform> container: ContainerGeometry;
@group(0) @binding(2) var gbuffer_position: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> rigid_bodies: array<RigidBodyParams>;

// --- Rigid body photon occlusion ---
// Bodies absorb the light that hits them, so photons whose path crosses a
// body are culled (the shadow kind still splats: direct sun is removed from
// that corridor whether water refracted it away or a body swallowed it).

// World -> body-local (columns of the rotation are rot_row0/1/2.xyz, matching
// rotate_local_to_world in rigid_body.wgsl, so world->local dots the rows)
fn body_local_point(body: RigidBodyParams, p: vec3<f32>) -> vec3<f32> {
    let q = p - body.position;
    return vec3<f32>(
        dot(body.rot_row0.xyz, q),
        dot(body.rot_row1.xyz, q),
        dot(body.rot_row2.xyz, q),
    );
}

fn body_local_dir(body: RigidBodyParams, d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(body.rot_row0.xyz, d),
        dot(body.rot_row1.xyz, d),
        dot(body.rot_row2.xyz, d),
    );
}

// Conservative bounding-sphere radius per shape (world units)
fn body_bound_radius(body: RigidBodyParams) -> f32 {
    if (body.shape == 0u) { return body.half_extent * 1.7320508; } // cube corner
    if (body.shape == 2u) { return body.half_extent * 1.4142136; } // cylinder rim
    if (body.shape == 3u) { return body.half_extent * 1.3; }       // torus R+r
    // Sphere: exact. Propeller: blade-sweep disc — a solid occlusion disc is
    // deliberate (per-blade shadows would strobe against the temporal EMA).
    // Custom mesh: half_extent sphere approximation.
    return body.half_extent;
}

// Does the segment a->b pass through any enabled rigid body? Sphere and cube
// are exact; the other shapes use their bounding sphere.
fn segment_blocked(a: vec3<f32>, b: vec3<f32>) -> bool {
    let seg = b - a;
    let len = length(seg);
    if (len < 1e-5) {
        return false;
    }
    let dir = seg / len;
    let n = min(params.body_count, 8u);
    for (var i = 0u; i < n; i++) {
        let body = rigid_bodies[i];
        let r = body_bound_radius(body);
        let oc = a - body.position;
        let tca = -dot(oc, dir);
        let d2 = dot(oc, oc) - tca * tca;
        if (d2 > r * r) {
            continue;
        }
        let thc = sqrt(max(r * r - d2, 0.0));
        if (tca + thc < 0.0 || tca - thc > len) {
            continue;
        }
        if (body.shape == 0u) {
            // Exact rotated-box slab test in body-local space
            let ro = body_local_point(body, a);
            let rd = body_local_dir(body, dir);
            let he = vec3<f32>(body.half_extent);
            let safe_rd = select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));
            let inv = vec3<f32>(1.0) / safe_rd;
            let t1 = (-he - ro) * inv;
            let t2 = (he - ro) * inv;
            let tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
            let tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
            if (tmax >= max(tmin, 0.0) && tmin <= len) {
                return true;
            }
            continue;
        }
        return true; // sphere: exact hit; other shapes: bounding sphere decides
    }
    return false;
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Offset from splat center in the receiving face's plane (meters)
    @location(0) offset: vec2<f32>,
    // Per-channel flux carried by this quad (RGB caustic, A shadow)
    @location(1) flux: vec4<f32>,
    // Atlas row bounds [start, end) of the target tile in pixels; fragments
    // outside contribute zero (quads near tile edges must not spill into
    // the neighboring face's tile)
    @location(2) @interpolate(flat) row_bounds: vec2<f32>,
}

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    // Quad corners as two CCW triangles
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    return corners[vertex_index];
}

fn culled() -> VertexOutput {
    var out: VertexOutput;
    // Zero-area quad: every corner collapses to the same point
    out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.offset = vec2<f32>(0.0);
    out.flux = vec4<f32>(0.0);
    out.row_bounds = vec2<f32>(0.0);
    return out;
}

// --- First-hit face selection ---

struct FaceHit {
    // 0 floor, 1 +X wall, 2 -X wall, 3 +Z wall, 4 -Z wall, -1 none
    face: i32,
    // Ray parameter at the hit (meters — d must be normalized)
    t: f32,
    // Face-plane coordinates relative to the face center (meters)
    fp: vec2<f32>,
}

// Half-extents of a face in its own plane coordinates
fn face_half_extents(face: i32) -> vec2<f32> {
    if (face == 0) {
        return vec2<f32>(container.half_width, container.half_depth);
    }
    if (face <= 2) {
        return vec2<f32>(container.half_depth, container.half_height * 0.5);
    }
    return vec2<f32>(container.half_width, container.half_height * 0.5);
}

// First interior face hit by the local-space ray p + d*t. Walls only count
// below the rim plane (local y <= 0); a ray that clears every face exits the
// open top and is lost. `margin` accepts hits slightly off-face so gaussian
// tails whose center lands just outside still deposit their in-tile energy.
fn first_face_hit(p: vec3<f32>, d: vec3<f32>, margin: f32) -> FaceHit {
    let hw = container.half_width;
    let hh = container.half_height;
    let hd = container.half_depth;
    var best: FaceHit;
    best.face = -1;
    best.t = 1e9;

    // Floor (y = -hh)
    if (d.y < -1e-5) {
        let t = (-hh - p.y) / d.y;
        if (t > 0.0 && t < best.t) {
            let h = p + d * t;
            if (abs(h.x) <= hw + margin && abs(h.z) <= hd + margin) {
                best = FaceHit(0, t, vec2<f32>(h.x, h.z));
            }
        }
    }
    // Walls: fp.v = -y - hh/2 puts v = 0 at the rim plane, v = 1 at the floor
    let v_center = hh * 0.5;
    // +X (x = +hw)
    if (d.x > 1e-5) {
        let t = (hw - p.x) / d.x;
        if (t > 0.0 && t < best.t) {
            let h = p + d * t;
            if (abs(h.z) <= hd + margin && h.y >= -hh - margin && h.y <= margin) {
                best = FaceHit(1, t, vec2<f32>(h.z, -h.y - v_center));
            }
        }
    }
    // -X (x = -hw)
    if (d.x < -1e-5) {
        let t = (-hw - p.x) / d.x;
        if (t > 0.0 && t < best.t) {
            let h = p + d * t;
            if (abs(h.z) <= hd + margin && h.y >= -hh - margin && h.y <= margin) {
                best = FaceHit(2, t, vec2<f32>(h.z, -h.y - v_center));
            }
        }
    }
    // +Z (z = +hd)
    if (d.z > 1e-5) {
        let t = (hd - p.z) / d.z;
        if (t > 0.0 && t < best.t) {
            let h = p + d * t;
            if (abs(h.x) <= hw + margin && h.y >= -hh - margin && h.y <= margin) {
                best = FaceHit(3, t, vec2<f32>(h.x, -h.y - v_center));
            }
        }
    }
    // -Z (z = -hd)
    if (d.z < -1e-5) {
        let t = (-hd - p.z) / d.z;
        if (t > 0.0 && t < best.t) {
            let h = p + d * t;
            if (abs(h.x) <= hw + margin && h.y >= -hh - margin && h.y <= margin) {
                best = FaceHit(4, t, vec2<f32>(h.x, -h.y - v_center));
            }
        }
    }
    return best;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let photons = params.light_res * params.light_res;
    let kind = instance_index / photons;
    let photon = instance_index % photons;
    let texel = vec2<u32>(photon % params.light_res, photon / params.light_res);

    let pos4 = textureLoad(gbuffer_position, texel, 0);
    if (pos4.w < 0.5) {
        return culled(); // no water surface at this light texel
    }

    let local_pos = world_to_local(container, pos4.xyz);

    // Incident ray travels opposite the sun direction
    let incident = -params.sun_dir;
    let is_shadow = kind == params.kinds - 1u;

    if (!is_shadow) {
        let normal = textureLoad(gbuffer_normal, texel, 0).xyz;
        let cos_i = dot(normal, params.sun_dir);
        if (cos_i < 0.02) {
            return culled(); // grazing or back-facing: ~no transmitted light
        }
        // Chromatic mode (kinds=4): one quad per channel with its own IOR.
        // White mode (kinds=2): one quad carrying all channels (green IOR
        // geometry, per-channel Beer-Lambert) - identical result when
        // dispersion is zero, at half the total splat cost.
        var ior = params.ior_rgb.y;
        if (params.kinds == 4u) {
            ior = params.ior_rgb[kind];
        }
        let ray_world = refract(incident, normal, 1.0 / ior);
        if (dot(ray_world, ray_world) < 0.5) {
            return culled(); // degenerate refraction
        }

        // Fresnel transmittance (Schlick)
        let f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
        let fresnel_t = 1.0 - (f0 + (1.0 - f0) * pow(1.0 - cos_i, 5.0));

        // First interior face along the refracted ray (floor or wall; rays
        // out the open top are lost). Up-going rays are valid — steep wave
        // faces refract light onto the walls above the waterline.
        let ray_local = world_dir_to_local(container, normalize(ray_world));
        let hit = first_face_hit(local_pos, ray_local, params.splat_radius);
        if (hit.face < 0) {
            return culled();
        }

        // Rigid bodies absorb this photon if they block the sun's path to the
        // surface (dry part above the waterline) or the refracted path to the
        // receiving face (submerged part)
        if (params.body_count > 0u) {
            let world_hit = local_to_world(container, local_pos + ray_local * hit.t);
            let surface = pos4.xyz;
            if (segment_blocked(surface + params.sun_dir * 0.02, surface + params.sun_dir * 3.0)
                || segment_blocked(surface + normalize(ray_world) * 0.02, world_hit)) {
                return culled();
            }
        }

        // Beer-Lambert along the in-water path (mirrors mc_render.wgsl).
        // Up-refracted rays re-exit the surface before reaching the wall; the
        // second interface is not modeled, so the whole path is treated as
        // submerged — a slight over-absorption on centimeter scales.
        let base = params.flux_area * fresnel_t;
        var flux = vec4<f32>(0.0);
        if (params.kinds == 4u) {
            let energy = base * exp(-params.absorb_rgb[kind] * params.optical_density * hit.t);
            if (kind == 0u) {
                flux = vec4<f32>(energy, 0.0, 0.0, 0.0);
            } else if (kind == 1u) {
                flux = vec4<f32>(0.0, energy, 0.0, 0.0);
            } else {
                flux = vec4<f32>(0.0, 0.0, energy, 0.0);
            }
        } else {
            let transmittance = exp(-params.absorb_rgb * params.optical_density * hit.t);
            flux = vec4<f32>(base * transmittance, 0.0);
        }
        return emit_quad(vertex_index, hit.face, hit.fp, flux);
    }

    // Shadow photon - straight ray removes direct sun from whichever face
    // the unrefracted corridor lands on
    let ray_local = world_dir_to_local(container, incident);
    let hit = first_face_hit(local_pos, ray_local, params.splat_radius);
    if (hit.face < 0) {
        return culled();
    }
    return emit_quad(vertex_index, hit.face, hit.fp, vec4<f32>(0.0, 0.0, 0.0, params.flux_area));
}

fn emit_quad(vertex_index: u32, face: i32, center_fp: vec2<f32>, flux: vec4<f32>) -> VertexOutput {
    let corner = quad_corner(vertex_index);
    let offset = corner * params.splat_radius;
    let p = center_fp + offset;

    // Face-plane meters -> face uv -> atlas clip position (vertical strip:
    // floor tile res tall at the top, wall tiles res/2 below)
    let ext = face_half_extents(face);
    let uv = p / (2.0 * ext) + vec2<f32>(0.5);
    let res = f32(params.light_res);
    var row_start = 0.0;
    var row_h = res;
    if (face > 0) {
        row_start = res + f32(face - 1) * res * 0.5;
        row_h = res * 0.5;
    }
    let av = (row_start + uv.y * row_h) / (3.0 * res);

    var out: VertexOutput;
    // v flipped: NDC +y is texel row 0
    out.clip_position = vec4<f32>(uv.x * 2.0 - 1.0, -(av * 2.0 - 1.0), 0.5, 1.0);
    out.offset = offset;
    out.flux = flux;
    out.row_bounds = vec2<f32>(row_start, row_start + row_h);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Clip to the target tile row (additive blend: zero adds nothing)
    if (in.clip_position.y < in.row_bounds.x || in.clip_position.y >= in.row_bounds.y) {
        return vec4<f32>(0.0);
    }
    let r_sq = dot(in.offset, in.offset);
    let kernel = exp(-r_sq * params.inv_two_sigma_sq) * params.splat_norm;
    return in.flux * kernel;
}
