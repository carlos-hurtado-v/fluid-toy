// Caustics - Photon Splat Pass
// One photon per light-space G-buffer texel, drawn as an instanced quad into
// the floor caustic map (additive). Four kinds per photon:
//   kind 0..2: refracted ray per color channel (chromatic dispersion),
//              carrying Fresnel-transmitted, Beer-Lambert-attenuated flux
//   kind 3:    straight (unrefracted) ray, marking direct sunlight removed
//              from the floor (alpha channel = water shadow)
// Both use the same normalized gaussian kernel, so the energy removed by the
// shadow channel equals the energy re-deposited by the caustic channels
// (modulo Fresnel reflection and absorption) - conservation by construction.
//
// Map space: caustic map spans the pool floor rect in container-local space,
//   u = local.x / half_width * 0.5 + 0.5,  v = local.z / half_depth * 0.5 + 0.5
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
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> params: CausticsParams;
@group(0) @binding(1) var<uniform> container: ContainerGeometry;
@group(0) @binding(2) var gbuffer_position: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_normal: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Offset from splat center on the floor (container-local meters)
    @location(0) offset: vec2<f32>,
    // Per-channel flux carried by this quad (RGB caustic, A shadow)
    @location(1) flux: vec4<f32>,
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
    return out;
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
    let floor_y = -container.half_height;
    let hw = container.half_width;
    let hd = container.half_depth;

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

        let ray_local = world_dir_to_local(container, normalize(ray_world));
        if (ray_local.y > -0.05) {
            return culled(); // not heading toward the floor
        }
        let t = (floor_y - local_pos.y) / ray_local.y;
        let hit = local_pos + ray_local * t;
        if (abs(hit.x) > hw + params.splat_radius || abs(hit.z) > hd + params.splat_radius) {
            return culled(); // misses the floor rect (walls are v2)
        }

        // Beer-Lambert along the in-water path (mirrors mc_render.wgsl)
        let base = params.flux_area * fresnel_t;
        var flux = vec4<f32>(0.0);
        if (params.kinds == 4u) {
            let energy = base * exp(-params.absorb_rgb[kind] * params.optical_density * t);
            if (kind == 0u) {
                flux = vec4<f32>(energy, 0.0, 0.0, 0.0);
            } else if (kind == 1u) {
                flux = vec4<f32>(0.0, energy, 0.0, 0.0);
            } else {
                flux = vec4<f32>(0.0, 0.0, energy, 0.0);
            }
        } else {
            let transmittance = exp(-params.absorb_rgb * params.optical_density * t);
            flux = vec4<f32>(base * transmittance, 0.0);
        }
        return emit_quad(vertex_index, hit.xz, flux);
    }

    // Shadow photon - straight ray removes direct sun from the floor
    let ray_local = world_dir_to_local(container, incident);
    if (ray_local.y > -0.05) {
        return culled();
    }
    let t = (floor_y - local_pos.y) / ray_local.y;
    let hit = local_pos + ray_local * t;
    if (abs(hit.x) > hw + params.splat_radius || abs(hit.z) > hd + params.splat_radius) {
        return culled();
    }
    return emit_quad(vertex_index, hit.xz, vec4<f32>(0.0, 0.0, 0.0, params.flux_area));
}

fn emit_quad(vertex_index: u32, center: vec2<f32>, flux: vec4<f32>) -> VertexOutput {
    let corner = quad_corner(vertex_index);
    let offset = corner * params.splat_radius;
    let p = center + offset;

    var out: VertexOutput;
    // Floor local -> caustic map clip space (v flipped: NDC +y is texel row 0)
    out.clip_position = vec4<f32>(
        p.x / container.half_width,
        -(p.y / container.half_depth),
        0.5,
        1.0,
    );
    out.offset = offset;
    out.flux = flux;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let r_sq = dot(in.offset, in.offset);
    let kernel = exp(-r_sq * params.inv_two_sigma_sq) * params.splat_norm;
    return in.flux * kernel;
}
