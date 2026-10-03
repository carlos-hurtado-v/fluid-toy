// Water shader (mc_render), part: filtered re-read of the final refraction lookup

// Longest axis of the footprint a filtered lookup may cover (level-0
// texels). Within one route neighbouring pixels can still land far apart
// (a mirrored tank corner, floor vs wall): unbounded, those pixels would
// average half the screen into a smear along the seam.
const LOOK_MAX_FOOTPRINT: f32 = 64.0;

// Screen-space gradients for a filtered read, or zero where they mean
// nothing: `same_x` / `same_y` say whether this pixel and its neighbour along
// that axis took the same route. One axis lost: assume a round footprint from
// the other. Then bound the footprint (g in uv, dims = texture size).
fn footprint_gradients(gx_in: vec2<f32>, gy_in: vec2<f32>, same_x: bool, same_y: bool, dims: vec2<f32>) -> array<vec2<f32>, 2> {
    var gx = gx_in;
    var gy = gy_in;
    if (!same_x) {
        gx = vec2<f32>(-gy.y, gy.x) * vec2<f32>(dims.y / dims.x, dims.x / dims.y);
    }
    if (!same_y) {
        gy = vec2<f32>(-gx.y, gx.x) * vec2<f32>(dims.y / dims.x, dims.x / dims.y);
    }
    let longest = max(length(gx * dims), length(gy * dims));
    let scale = min(1.0, LOOK_MAX_FOOTPRINT / max(longest, 1e-6));
    return array<vec2<f32>, 2>(gx * scale, gy * scale);
}

// Equirect uv of a direction (as sample_environment)
fn environment_uv(dir: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(fract(atan2(dir.z, dir.x) / (2.0 * PI) + 1.0), acos(clamp(dir.y, -1.0, 1.0)) / PI);
}

// Equirect uv change for a change dd of the unit direction d. Taken from the
// direction, not from uv: u wraps at +-pi, where a uv derivative is a jump
// across the whole map.
fn environment_uv_gradient(d: vec3<f32>, dd: vec3<f32>) -> vec2<f32> {
    let r2 = max(d.x * d.x + d.z * d.z, 1e-6);
    let dphi = (d.x * dd.z - d.z * dd.x) / r2;
    let dtheta = -dd.y / sqrt(max(1.0 - d.y * d.y, 1e-6));
    return vec2<f32>(dphi / (2.0 * PI), dtheta / PI);
}

// The refracted ray's colour, read again over this pixel's footprint: the
// background's mip chain (anisotropic: a grazing mirror squeezes the image
// along one axis only) or the environment map's. `unfiltered` is the single
// sample already taken; it stands wherever there is nothing to filter or no
// neighbour to measure a footprint against. Call in uniform control flow.
fn resolve_lookup(unfiltered: vec3<f32>) -> vec3<f32> {
    // Screen derivatives first: of the route (do the neighbours' lookups
    // belong to the same image?), the uv, and the environment direction
    let route = f32((((((dbg_path * 4u + look_kind) * 4u + dbg_bounces) * 4u + dbg_exit_kind) * 4u
        + dbg_mirror_kind) * 2u) + u32(dbg_end == DBG_END_BODY));
    let same_x = dpdx(route) == 0.0;
    let same_y = dpdy(route) == 0.0;
    let uv_dx = dpdx(look_uv);
    let uv_dy = dpdy(look_uv);
    // (the ground is read along the direction from its capture point, and
    // under the capture point from a shifted one: ground_radiance)
    var dir = look_vec;
    var dir_far = look_vec;
    if (look_kind == LOOK_GROUND) {
        let capture = vec3<f32>(0.0, water.ground_y + water.ground_capture_height, 0.0);
        dir = normalize(look_vec - capture);
        dir_far = normalize(look_vec - capture - vec3<f32>(NADIR_SHIFT * water.ground_capture_height, 0.0, 0.0));
    }
    let dir_dx = dpdx(dir);
    let dir_dy = dpdy(dir);
    let far_dx = dpdx(dir_far);
    let far_dy = dpdy(dir_far);

    if (look_kind == LOOK_NONE || (!same_x && !same_y)) {
        return unfiltered;
    }
    if (look_kind == LOOK_SCREEN) {
        let g = footprint_gradients(uv_dx, uv_dy, same_x, same_y, vec2<f32>(textureDimensions(background_tex)));
        return textureSampleGrad(background_tex, background_sampler, look_uv, g[0], g[1]).rgb;
    }
    let env_dims = vec2<f32>(textureDimensions(env_tex));
    let g = footprint_gradients(
        environment_uv_gradient(dir, dir_dx), environment_uv_gradient(dir, dir_dy), same_x, same_y, env_dims,
    );
    var color = textureSampleGrad(env_tex, env_sampler, environment_uv(dir), g[0], g[1]).rgb;
    if (look_kind == LOOK_GROUND) {
        // ground_radiance's nadir re-read, with its own footprint (near the
        // nadir the first read's longitude gradient is enormous)
        let fade = smoothstep(NADIR_FADE_START, NADIR_FADE_END, -dir.y);
        if (fade > 0.0) {
            let gf = footprint_gradients(
                environment_uv_gradient(dir_far, far_dx), environment_uv_gradient(dir_far, far_dy), same_x, same_y, env_dims,
            );
            let far = textureSampleGrad(env_tex, env_sampler, environment_uv(dir_far), gf[0], gf[1]).rgb;
            color = mix(color, far, fade);
        }
    }
    return max(color * water.env_intensity, vec3<f32>(0.0));
}
