// Scene-lit rigid body shading, shared by the body renderers (rigid_body.wgsl,
// rigid_body_mesh.wgsl) and the water shader, which shades the bodies its
// refracted and mirrored rays hit (mc_render/bodies.wgsl): the same surface
// has to look the same seen directly and through the water.
//
// Uses the including shader's `container` (ContainerGeometry) binding with
// container_common.wgsl's rim_visibility, and sh_common.wgsl's
// evaluate_sh_irradiance (both prepended before this file).

const BODY_INV_PI: f32 = 0.31830988;
const BODY_SPEC_STRENGTH: f32 = 0.5;

// SH ambient + rim-shadowed sun, Lambert with the proper 1/pi
// (evaluate_sh_irradiance returns irradiance, and the sun term ndotl * sun_rgb
// is one too), plus a Schlick-Fresnel Blinn-Phong lobe carrying the sun color
// (F0 = 0.04 dielectric). `v` points from the surface to the viewer;
// `sun_rgb` is color x intensity (zero when the sun is off); `ibl_strength`
// scales the SH ambient (environment intensity).
fn shade_body_lit(
    albedo: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    world_pos: vec3<f32>,
    sun_dir: vec3<f32>,
    sun_rgb: vec3<f32>,
    ibl_strength: f32,
) -> vec3<f32> {
    let l = normalize(sun_dir);
    let ndotl = max(dot(n, l), 0.0);

    let local = world_to_local(container, world_pos);
    let l_local = world_dir_to_local(container, l);
    let rim = rim_visibility(container, local, l_local);

    let ambient = evaluate_sh_irradiance(n) * ibl_strength;
    let sun = sun_rgb * (ndotl * rim);

    let h = normalize(l + v);
    let ndoth = max(dot(n, h), 0.0);
    let ndotv = max(dot(n, v), 0.0);
    let fresnel = 0.04 + 0.96 * pow(1.0 - ndotv, 5.0);
    let spec = sun_rgb * (fresnel * pow(ndoth, 64.0) * BODY_SPEC_STRENGTH * rim);

    return albedo * (ambient + sun) * BODY_INV_PI + spec;
}
