// Water shader (mc_render), part: what a ray that left the water sees (background, ground, environment)

// Sample equirectangular environment map. Same convention as the CPU SH
// projection (compute_sh_irradiance in environment.rs): row 0 = +Y, and
// u = phi / 2pi for dir = (sin t cos phi, cos t, sin t sin phi). (The old
// mapping, v = 1 - t/pi with u offset by pi, sampled the antipode -dir.)
fn sample_environment(dir: vec3<f32>) -> vec3<f32> {
    let phi = atan2(dir.z, dir.x);
    let theta = acos(clamp(dir.y, -1.0, 1.0));
    let u = fract(phi / (2.0 * PI) + 1.0);
    let v = theta / PI;
    // Explicit LOD 0 (the map has a single mip) so refraction can call this
    // from per-pixel branches
    return textureSampleLevel(env_tex, env_sampler, vec2<f32>(u, v), 0.0).rgb;
}

// Background at a refracted uv. If what sits there is in front of the water
// (e.g. the pool's near wall), the refracted ray can't have reached it — keep
// the straight-through sample instead of leaking the occluder into the water.
fn background_at(uv: vec2<f32>, front_depth_raw: f32, straight: vec3<f32>) -> vec3<f32> {
    let depth = background_depth_at(uv);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || depth < front_depth_raw) {
        dbg_end = DBG_END_STRAIGHT;
        look_kind = LOOK_NONE;
        probe_event(PRB_LOOKUP, vec3<f32>(uv, depth), vec3<f32>(front_depth_raw, 0.0, 0.0), 0.0);
        return straight;
    }
    dbg_end = DBG_END_SURFACE;
    dbg_uv = uv;
    look_kind = LOOK_SCREEN;
    look_uv = uv;
    probe_event(PRB_LOOKUP, vec3<f32>(uv, depth), vec3<f32>(front_depth_raw, 1.0, 0.0), 0.0);
    return textureSampleLevel(background_tex, env_sampler, uv, 0.0).rgb;
}

// The projected ground's radiance at a point on it (keep in sync with
// ground_radiance in mc_environment.wgsl: the map read from its capture
// point, the nadir patch re-read from a shifted one)
const NADIR_FADE_START: f32 = 0.866;
const NADIR_FADE_END: f32 = 0.940;
const NADIR_SHIFT: f32 = 1.5;

fn ground_radiance(hit: vec3<f32>) -> vec3<f32> {
    let capture = vec3<f32>(0.0, water.ground_y + water.ground_capture_height, 0.0);
    let dir = normalize(hit - capture);
    let color = sample_environment(dir);
    let fade = smoothstep(NADIR_FADE_START, NADIR_FADE_END, -dir.y);
    if (fade <= 0.0) {
        return color;
    }
    let shifted = capture + vec3<f32>(NADIR_SHIFT * water.ground_capture_height, 0.0, 0.0);
    return mix(color, sample_environment(normalize(hit - shifted)), fade);
}

// Distance along a ray to the projected ground, or -1 if it does not reach it
// (the ground exists under the same condition as in the backdrop pass:
// Environment mode, projection on, camera above the plane)
fn ground_distance(origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    if (water.use_env_background != 0u && water.ground_enabled != 0u && dir.y < 0.0
        && origin.y > water.ground_y && camera.camera_pos.y > water.ground_y) {
        return (water.ground_y - origin.y) / dir.y;
    }
    return -1.0;
}

// Did the last march_to_background stop on something nearer than the ground
// along this ray? The ground itself is known exactly (backdrop_along), which
// beats reading it off the screen: a ray that lands on ground hidden behind a
// body, as the camera sees it, finds no ground pixel there. The march skipped
// the body's pixels and stopped on the first ground pixel past its outline,
// so a mirror showing that ground showed the body's edge colours instead, in
// stripes (hatched ball reflections in side-wall mirrors). Glass tank only:
// a pool's floor and walls are opaque and always come first.
fn march_hit_before_ground(hit: bool, t_ground: f32) -> bool {
    if (container.is_pool != 0u) {
        return hit;
    }
    return hit && (t_ground < 0.0 || march_dist < 0.98 * t_ground - 0.01);
}

// Radiance reaching `origin` from far along a world direction, past
// everything the marches can find. Heading down onto the projected ground,
// where the ray lands sets what it shows (the map is read from its capture
// point, with parallax): the map along the direction is the ground at
// infinity, i.e. the horizon's hills and trees where grass belongs. That
// seam ran along every lookup that left the screen (snap_004: a gray panel
// with ragged tabs inside a side-wall mirror). Otherwise the backdrop pass's
// own radiance: the map along the direction, or the solid colour.
// The pool's contact occlusion on the ground is not applied here.
fn backdrop_along(origin: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let t_ground = ground_distance(origin, dir);
    if (t_ground > 0.0) {
        let hit = origin + dir * t_ground;
        dbg_end = DBG_END_GROUND;
        look_kind = LOOK_GROUND;
        look_vec = hit;
        probe_event(PRB_BACKDROP, vec3<f32>(-1.0, -1.0, 0.0), dir, f32(dbg_end));
        return max(ground_radiance(hit) * water.env_intensity, vec3<f32>(0.0));
    }
    let vanishing = vec3<f32>(-1.0, -1.0, 0.0);
    if (water.use_env_background == 0u) {
        dbg_end = DBG_END_SOLID;
        look_kind = LOOK_NONE;
        probe_event(PRB_BACKDROP, vanishing, dir, f32(dbg_end));
        return vec3<f32>(water.background_r, water.background_g, water.background_b);
    }
    // Same radiance as the backdrop pass (mc_environment.wgsl)
    dbg_end = DBG_END_ENV;
    look_kind = LOOK_ENV;
    look_vec = dir;
    probe_event(PRB_BACKDROP, vanishing, dir, f32(dbg_end));
    return max(sample_environment(dir) * water.env_intensity, vec3<f32>(0.0));
}
