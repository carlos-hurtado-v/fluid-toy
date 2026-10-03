// Water shader (mc_render), part: refract_scene, the two-interface refraction entry point

// Two-interface image-space refraction (after Wyman 2005). Snell-refract the
// view ray at the front surface. If an opaque surface (floor, wall, body) sits
// inside the water behind this pixel, the ray ends on it — this is the
// apparent-depth shift that makes a pool look shallower than it is. Otherwise
// the ray crosses the body (front-to-back distance), refracts out through the
// back-face normal where it lands, and we look up what the exit ray reaches.
// Curved bodies (drops, crests) bend the exit ray: they act as lenses.
// Returns the radiance arriving from behind, before absorption.
fn refract_scene(
    p: vec3<f32>,
    n: vec3<f32>,
    view_dir: vec3<f32>,
    screen_uv: vec2<f32>,
    front_depth_raw: f32,
    back_depth_raw: f32,
) -> vec3<f32> {
    let straight = textureSampleLevel(background_tex, env_sampler, screen_uv, 0.0).rgb;
    // n faces the camera; entering the denser medium never totally reflects
    let t1 = refract(-view_dir, n, 1.0 / water.ior);
    let bg_depth = background_depth_at(screen_uv);
    probe_event(PRB_REFRACT_IN, t1, vec3<f32>(bg_depth, back_depth_raw, f32(container.is_pool)), 0.0);

    // Opaque surface inside the water: the ray ends on it
    if (bg_depth < BACKDROP_DEPTH && bg_depth <= back_depth_raw) {
        // No crossing: the ray left the screen, or rose toward the water
        // surface from below (side faces near the rounded top edge), where it
        // would reflect back down rather than reach anything far away. The
        // surface first seen is the safe answer; the furthest point reached
        // would paint the horizon into the water.
        dbg_path = DBG_PATH_INSIDE;
        let m = march_to_background(p, t1, screen_uv, bg_depth);
        if (container.is_pool != 0u) {
            return background_at(m.xy, front_depth_raw, straight);
        }
        // Glass tank: the refracted ray can also miss what the view ray sees
        // (it bends away past a body's edge). It then crosses the tank like
        // any other ray: carry on to the exit search below. Painting the
        // surface first seen there instead drew bodies a little too large.
        if (m.z > 0.5) {
            refracted_path = march_dist;
            if (march_hit_before_ground(true, ground_distance(p, t1))) {
                return background_at(m.xy, front_depth_raw, straight);
            }
            // It is the ground the ray ends on, through the tank's floor
            return backdrop_along(p, t1);
        }
        missed_inside = true;
    }
    // No back face behind this pixel (mesh clipped open): treat the body as
    // deep and let the refracted ray run out to the backdrop
    if (back_depth_raw >= 1.0) {
        dbg_path = DBG_PATH_NO_BACK;
        return backdrop_along(p, t1);
    }

    var p_exit: vec3<f32>;
    var p_inside: vec3<f32>;
    var n_exit: vec3<f32>;
    var exit_on_wall = false;
    if (container.is_pool == 0u) {
        // Wireframe tank: find the exit along the refracted ray itself. The
        // back face behind this pixel is where the VIEW ray leaves, and the
        // refracted ray can leave somewhere else entirely: near the sides the
        // view ray meets a side wall while the refracted ray, bent toward the
        // front wall's normal, runs on to the back wall (or vice versa)
        let ex = water_exit(p, t1);
        if (ex.blocked) {
            dbg_path = DBG_PATH_BLOCKED;
            return background_at(ex.blocked_uv, front_depth_raw, straight);
        }
        p_exit = ex.point;
        p_inside = ex.inside;
        n_exit = ex.normal;
        exit_on_wall = ex.on_wall;
    } else {
        // Cross the body to the back face, refract out through its normal there
        p_exit = p + t1 * distance(p, screen_to_world(screen_uv, back_depth_raw));
        p_inside = p_exit;
        let exit_uv = screen_point(p_exit).xy;
        var back_n = textureLoad(back_normal_tex, texel_at(exit_uv, textureDimensions(back_normal_tex)), 0);
        if (back_n.w < 0.5) {
            // Landed outside the body's silhouette: use the exit face behind this pixel
            back_n = textureLoad(back_normal_tex, texel_at(screen_uv, textureDimensions(back_normal_tex)), 0);
        }
        n_exit = normalize(back_n.xyz);
    }
    dbg_water_path = distance(p, p_exit);
    if (missed_inside) {
        refracted_path = dbg_water_path;
    }
    let t2 = refract(t1, -n_exit, water.ior);
    probe_event(PRB_SECOND, t2, n_exit, dbg_water_path);
    if (dot(t2, t2) < 0.5) {
        // Total internal reflection: in bulk water, follow the mirror bounce;
        // inside a thin drop or crest the next interface isn't knowable here
        if (distance(p, p_exit) < TIR_MIN_BODY) {
            dbg_path = DBG_PATH_THIN_TIR;
            dbg_end = DBG_END_STRAIGHT;
            look_kind = LOOK_NONE;
            return straight;
        }
        dbg_mirror_kind = dbg_exit_interface(p_exit, exit_on_wall);
        return follow_internal_reflection(p_inside, reflect(t1, n_exit), front_depth_raw, straight);
    }
    dbg_path = DBG_PATH_EXIT;
    dbg_exit_cos = dot(t2, n_exit);
    dbg_exit_kind = dbg_exit_interface(p_exit, exit_on_wall);
    return scene_from(p_exit, t2, front_depth_raw, straight);
}
