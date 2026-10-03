// Water shader (mc_render), part: where a ray leaves the water, and internal reflection

// The MC surface does not meet a wireframe tank wall at a corner: it rounds
// over into the particle bulge past the wall, unevenly along the wall. Real
// water meets glass with a meniscus of millimetres, flat right up to the
// wall. Mirrored in that rim at grazing angles, the uneven lean strung dark
// beads along wall/surface seams (the Mirror debug view shows rays whose last
// reflection was there). Within this distance of a side wall (m), the free
// surface's normal loses its lean into the wall (full at half the band,
// fading out by the band's edge). The rim's uneven HEIGHT remains: rays near
// the seam still flip between leaving through the surface and the wall.
const RIM_BAND: f32 = 0.05;

fn flatten_wall_rim(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    if (container.is_pool != 0u) {
        return n;
    }
    let l = world_to_local(container, p);
    let n_local = world_dir_to_local(container, n);
    // Only the free surface: the bulge's side faces (normals along a wall)
    // would be left pointing along the wall
    if (n_local.y < 0.5) {
        return n;
    }
    var out = n_local;
    // Each side wall in turn (a corner is near two)
    let to_x = container.half_width - abs(l.x);
    if (to_x < RIM_BAND) {
        let wn = vec3<f32>(sign(l.x), 0.0, 0.0);
        let w = 1.0 - smoothstep(0.5 * RIM_BAND, RIM_BAND, to_x);
        out = out - wn * max(dot(out, wn), 0.0) * w;
    }
    let to_z = container.half_depth - abs(l.z);
    if (to_z < RIM_BAND) {
        let wn = vec3<f32>(0.0, 0.0, sign(l.z));
        let w = 1.0 - smoothstep(0.5 * RIM_BAND, RIM_BAND, to_z);
        out = out - wn * max(dot(out, wn), 0.0) * w;
    }
    if (dot(out, out) < 0.25) {
        return n;
    }
    return local_dir_to_world(container, normalize(out));
}

struct WaterExit {
    // Where the ray leaves the water, and the interface's outward normal
    point: vec3<f32>,
    normal: vec3<f32>,
    // Last point found still in the water: where a mirrored ray continues
    // (the exit point itself lies just outside a back-face crossing)
    inside: vec3<f32>,
    // Left through a container wall or the floor (exact plane)
    on_wall: bool,
    // Something opaque came first: the ray ends there. A body met exactly
    // (body.t > 0: shade it, body_radiance), else whatever the depth buffer
    // shows at this screen uv
    blocked: bool,
    blocked_uv: vec2<f32>,
    body: BodyHit,
}

// How a ray inside the water leaves it, decided along the ray itself: through
// the free surface or a drop's far side (where it crosses a back face, which
// supplies the normal), through a container wall or the floor (where the box
// bounds it: exact plane, and the MC bulge past a wall counts as the wall),
// or not at all because something opaque is in the way.
fn water_exit(origin: vec3<f32>, dir: vec3<f32>) -> WaterExit {
    let wall = box_interior_exit(world_to_local(container, origin), world_dir_to_local(container, dir));
    let body_hit = ray_body_hit(origin, dir, wall.w);
    let t_body = body_hit.t;
    let trace_max = select(wall.w, t_body, t_body > 0.0);
    probe_event(PRB_EXIT_BEGIN, origin, dir, wall.w);
    probe_event(PRB_EXIT_BOX, local_dir_to_world(container, wall.xyz), vec3<f32>(t_body, trace_max, 0.0), 0.0);
    let ev = trace_in_water(origin, dir, trace_max);
    var out: WaterExit;
    out.body = NO_BODY_HIT;
    // Reached the body with nothing in between, or left the water through the
    // film just in front of it
    let opaque = ev.kind > 1.5 && ev.kind < 2.5;
    if (t_body > 0.0 && (ev.kind < 0.5
        || (!opaque && t_body - ev.dist < BODY_WET_GAP && body_has_film(body_hit.index)))) {
        out.blocked = true;
        out.body = body_hit;
        out.blocked_uv = screen_point(origin + dir * t_body).xy;
        probe_event(PRB_EXIT_END, origin + dir * t_body, vec3<f32>(0.0), 6.0);
        probe_event(PRB_EXIT_INSIDE, origin, vec3<f32>(out.blocked_uv, 0.0), 0.0);
        return out;
    }
    out.blocked = opaque;
    out.blocked_uv = ev.uv;
    var floor_contact = 0.0;
    var crossing = ev.kind > 0.5 && ev.dist < wall.w - (container.clip_margin + WALL_SNAP_TOLERANCE);
    if (crossing) {
        // The interface crossed: a back face, or (kind 3) a front face; with
        // the world-space test, the field's own surface at the crossing
        let front_layer = ev.kind > 2.5;
        var back_n: vec4<f32>;
        if (water.volume_trace != 0u) {
            back_n = water_normal(origin + dir * ev.dist);
            if ((back_n.w < 0.5 || dot(back_n.xyz, dir) <= 0.0) && ev.dist < VOLUME_ENTRY_SLACK) {
                // Never got under the surface: a sheet or drop thinner than
                // the field resolves. The ray carries straight on.
                back_n = vec4<f32>(dir, 1.0);
            }
        } else if (front_layer) {
            back_n = front_normal_smooth(ev.uv);
        } else {
            back_n = back_normal_smooth(ev.uv);
        }
        if (back_n.w > 0.5) {
            out.normal = flatten_wall_rim(origin + dir * ev.dist, back_n.xyz);
        } else {
            // Crossing on a silhouette edge with no normal written: the free
            // surface is the likely interface
            out.normal = local_dir_to_world(container, vec3<f32>(0.0, 1.0, 0.0));
        }
        // A ray can only leave through a face it is heading out of. A crossing
        // whose outward normal points back against the ray is the in-water
        // test misreading a wavy surface seen edge-on (a trough between the
        // camera and a point just under the surface reads as dry): ignore it
        // and let the box bound the ray
        crossing = dot(out.normal, dir) > 0.0;
        probe_event(PRB_EXIT_CROSS, back_n.xyz, out.normal, back_n.w + 2.0 * f32(crossing) + 4.0 * f32(front_layer));
    }
    if (crossing) {
        out.point = origin + dir * ev.dist;
        out.inside = origin + dir * ev.dist_in;
        out.on_wall = false;
    } else {
        out.point = origin + dir * wall.w;
        out.inside = out.point;
        out.normal = local_dir_to_world(container, wall.xyz);
        out.on_wall = true;
        // A tank standing on something (the projected ground): its floor is in
        // contact with whatever the depth buffer shows right there, not a
        // window onto air. Treated as air, grazing rays reflected off it and
        // zig-zagged between floor and surface until some leaked out to the
        // sky, painting sky into the side faces.
        if (!out.blocked && wall.y < -0.5) {
            let qf = screen_point(out.point);
            let bgf = depth_smooth(background_depth_tex, qf.xy);
            if (abs(qf.z - bgf) <= SURFACE_CONTACT_REL * (1.0 - bgf)) {
                out.blocked = true;
                out.blocked_uv = qf.xy;
                floor_contact = 1.0;
            }
        }
    }
    probe_event(PRB_EXIT_END, out.point, out.normal, f32(out.on_wall) + 2.0 * f32(out.blocked));
    probe_event(PRB_EXIT_INSIDE, out.inside, vec3<f32>(out.blocked_uv, floor_contact), 0.0);
    return out;
}

// What a ray leaving the water at `p_out` along `dir` reaches: a body, the
// surface behind that point, if any, else the backdrop far along it
fn scene_from(p_out: vec3<f32>, dir: vec3<f32>, front_depth_raw: f32, straight: vec3<f32>) -> vec3<f32> {
    let uv = screen_point(p_out).xy;
    let depth = background_depth_at(uv);
    probe_event(PRB_SCENE, vec3<f32>(uv, depth), dir, 0.0);
    // Only where the depth buffer holds something the ray could land on that
    // is not known exactly (pool walls and floor, a Custom body): outside the
    // water of a glass tank there are just the bodies and the backdrop
    if (water.depth_occluders != 0u && depth < BACKDROP_DEPTH) {
        // Nothing reached: the ray passes the surface behind the exit point
        // (e.g. leaves through the free surface toward the sky while a body
        // sits behind the exit point on screen) and escapes
        let m = march_to_background(p_out, dir, uv, depth);
        if (march_body.t > 0.0) {
            return body_radiance(p_out, dir, march_body);
        }
        if (march_hit_before_ground(m.z > 0.5, ground_distance(p_out, dir))) {
            return background_at(m.xy, front_depth_raw, straight);
        }
        return backdrop_along(p_out, dir);
    }
    let hit = ray_body_hit(p_out, dir, BODY_MAX_REACH);
    if (hit.t > 0.0) {
        return body_radiance(p_out, dir, hit);
    }
    return backdrop_along(p_out, dir);
}

// Total internal reflection: the interface is a mirror. Follow the reflected
// ray through the bulk to where it next leaves the water (a container wall or
// the floor, or the free surface from below) and either get out there or
// reflect again. This is the aquarium look: side walls and the underside of
// the surface mirror the tank interior.
fn follow_internal_reflection(
    start: vec3<f32>,
    dir: vec3<f32>,
    front_depth_raw: f32,
    straight: vec3<f32>,
) -> vec3<f32> {
    var o = start;
    var d = dir;
    for (var bounce = 0; bounce < TIR_MAX_BOUNCES; bounce++) {
        let ex = water_exit(o, d);
        dbg_bounces = u32(bounce + 1);
        if (ex.blocked) {
            dbg_path = DBG_PATH_TIR_BLOCKED;
            if (ex.body.t > 0.0) {
                return body_radiance(o, d, ex.body);
            }
            return background_at(ex.blocked_uv, front_depth_raw, straight);
        }
        if (container.is_pool != 0u && ex.on_wall) {
            // Opaque pool walls should have blocked the ray already
            dbg_path = DBG_PATH_TIR_POOL;
            dbg_end = DBG_END_STRAIGHT;
            look_kind = LOOK_NONE;
            return straight;
        }
        let out_dir = refract(d, -ex.normal, water.ior);
        probe_event(PRB_BOUNCE, out_dir, d, f32(bounce + 1));
        if (dot(out_dir, out_dir) > 0.5) {
            dbg_path = DBG_PATH_TIR_EXIT;
            dbg_exit_cos = dot(out_dir, ex.normal);
            dbg_exit_kind = dbg_exit_interface(ex.point, ex.on_wall);
            return scene_from(ex.point, out_dir, front_depth_raw, straight);
        }
        // Reflects again: continue inside the water
        dbg_mirror_kind = dbg_exit_interface(ex.point, ex.on_wall);
        o = ex.inside;
        d = reflect(d, ex.normal);
    }
    // Out of bounces: wherever the ray is heading beats the view straight
    // through (which would paint what lies behind the tank, often sky, into
    // the mirror)
    dbg_path = DBG_PATH_TIR_SPENT;
    return scene_from(o, d, front_depth_raw, straight);
}
