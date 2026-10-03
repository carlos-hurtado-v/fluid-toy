// Water shader (mc_render), part: debug-view, pixel-probe and lookup records + the debug view output

// === Refraction debug record (rendering.mc_debug_view) ===
// What the refraction code did for this pixel, written at each decision and
// shown instead of the shaded color when a debug view is on. The ids are
// decoded by scripts/debug_decode.py: keep the two tables in sync.
// Route (dbg_path)
const DBG_PATH_LEGACY: u32 = 1u;       // physical refraction off
const DBG_PATH_INSIDE: u32 = 2u;       // opaque surface inside the water: marched onto it
const DBG_PATH_NO_BACK: u32 = 3u;      // no back face behind the pixel: straight out to the backdrop
const DBG_PATH_BLOCKED: u32 = 4u;      // something opaque before the water ends
const DBG_PATH_THIN_TIR: u32 = 5u;     // total internal reflection in a thin body: straight through
const DBG_PATH_EXIT: u32 = 6u;         // refracted out of the water
const DBG_PATH_TIR_BLOCKED: u32 = 7u;  // mirrored, then something opaque
const DBG_PATH_TIR_EXIT: u32 = 8u;     // mirrored, then refracted out
const DBG_PATH_TIR_SPENT: u32 = 9u;    // mirrored until out of bounces
const DBG_PATH_TIR_POOL: u32 = 10u;    // mirrored onto an opaque pool wall: straight through
// Final lookup (dbg_end)
const DBG_END_STRAIGHT: u32 = 1u;      // the view straight through (occluded or fallback)
const DBG_END_SURFACE: u32 = 2u;       // background texture where the ray met a surface
const DBG_END_SCREEN_SKY: u32 = 3u;    // (until 2026-10: backdrop read on screen at the vanishing point; no longer produced)
const DBG_END_ENV: u32 = 4u;           // environment map along the direction
const DBG_END_SOLID: u32 = 5u;         // solid background color
const DBG_END_BODY: u32 = 6u;          // a body met exactly, shaded at the hit (body_radiance)
const DBG_END_GROUND: u32 = 7u;        // projected ground where the ray lands on it, off screen or past the march
var<private> dbg_path: u32 = 0u;
var<private> dbg_end: u32 = 0u;
var<private> dbg_bounces: u32 = 0u;
var<private> dbg_uv: vec2<f32> = vec2<f32>(-1.0);
var<private> dbg_exit_cos: f32 = 0.0;
var<private> dbg_water_path: f32 = 0.0;
var<private> dbg_exit_kind: u32 = 0u;
// Interface of the last mirror (total internal) reflection, same ids
var<private> dbg_mirror_kind: u32 = 0u;
// Interface a ray finally refracted out through (dbg_exit_kind)
const DBG_EXIT_WALL: u32 = 1u;          // container wall or floor (exact plane)
const DBG_EXIT_SURFACE: u32 = 2u;       // back face (free surface, drop) away from the walls
const DBG_EXIT_SURFACE_WALL: u32 = 3u;  // back face near a wall: the MC contact line / bulge
const DBG_NEAR_WALL: f32 = 0.06;

// === Pixel probe events (--probe; decoded by scripts/probe_decode.py) ===
// probe_event(tag, a, b, c) records one step of the refraction code at a
// probed pixel; normal rendering links no-op stubs (mc_probe_off.wgsl). Keep
// this table and the field meanings in sync with the script's EVENTS table.
const PRB_FRAG: u32 = 1u;          // a = world pos, b = mesh normal (camera-facing), c = front raw depth
const PRB_NORMAL: u32 = 2u;        // a = shading normal (wall snap + ripple), b = (on_wall, physical, -), c = back raw depth
const PRB_REFRACT_IN: u32 = 3u;    // a = t1, b = (background raw depth, back raw depth, is_pool), c = -
const PRB_SECOND: u32 = 4u;        // a = t2 (0 = TIR), b = exit normal, c = water path to the exit (m)
const PRB_EXIT_BEGIN: u32 = 10u;   // water_exit: a = origin, b = dir, c = box exit distance
const PRB_EXIT_BOX: u32 = 11u;     // a = box exit normal, b = (t_body, trace max distance, -), c = -
const PRB_TRACE: u32 = 12u;        // in-water sample: a = (uv, raw depth), b = (background depth, back depth, kind), c = distance, d.x = front depth
const PRB_TRACE_REFINE: u32 = 13u; // bisection sample, same fields
const PRB_TRACE_END: u32 = 14u;    // a = (kind, dist_in, dist), b = (uv, -), c = max distance
const PRB_EXIT_CROSS: u32 = 15u;   // a = layer normal (smooth), b = after rim flattening, c = n.w + 2 * accepted + 4 * front layer
const PRB_EXIT_END: u32 = 16u;     // a = exit point, b = exit normal, c = on_wall + 2 blocked + 4 body
const PRB_EXIT_INSIDE: u32 = 17u;  // a = last in-water point, b = (blocked uv, floor contact), c = -
const PRB_SILHOUETTE: u32 = 18u;   // out-of-water bracket whose back depth jumps: a = (back in, back out, mode), b = (lo, hi, action 0 exit 1 continue), c = -
const PRB_HIDDEN: u32 = 19u;       // ray going out of sight behind a body: a = (outline distance, gap to the back face there (m), closing rate), b = (estimated crossing distance, normal uv), c = 1 crossing taken, 0 crossing in sight (the samples decide), 2 beyond the trace's reach
const PRB_VTRACE: u32 = 21u;        // world-space in-water sample: a = world pos, b = (density / iso, background depth or -1 off screen, kind), c = distance
const PRB_VTRACE_REFINE: u32 = 22u; // bisection sample, same fields
const PRB_BODY: u32 = 23u;         // body_radiance: a = hit point, b = surface normal, c = body index
const PRB_BOUNCE: u32 = 20u;       // a = refracted dir (0 = reflects again), b = incoming dir, c = bounce index
const PRB_MARCH_BEGIN: u32 = 30u;  // march_to_background: a = origin, b = dir, c = reach
const PRB_MARCH: u32 = 31u;        // a = (uv, raw depth), b = (background depth, behind, -), c = distance
const PRB_MARCH_REFINE: u32 = 32u; // bisection sample, same fields
const PRB_MARCH_END: u32 = 33u;    // a = (uv, hit), b = (lo, hi, t_body), c = -
const PRB_SCENE: u32 = 40u;        // scene_from: a = (exit uv, background depth there), b = dir, c = -
const PRB_BACKDROP: u32 = 41u;     // a = (vanishing-point uv or -1, on screen backdrop), b = dir, c = end id
const PRB_LOOKUP: u32 = 42u;       // background_at: a = (uv, background depth), b = (front raw depth, accepted, -), c = -
const PRB_RESULT: u32 = 50u;       // a = (dbg path, end, bounces), b = (dbg uv, exit kind), c = mirror kind
const PRB_COLOR: u32 = 51u;        // a = refracted scene, b = shaded color, c = fresnel
// Background / front depth at the last trace_event / behind_background verdict (probe only)
var<private> prb_bg: f32 = -1.0;
var<private> prb_front: f32 = -1.0;

// === Final lookup record (rendering.mc_filtered_lookup) ===
// Where the refracted ray's colour was read, noted at the read itself, so
// fs_main can read it again over the pixel's footprint (resolve_lookup). A
// refraction that shrinks what it shows (a grazing mirror, strong lensing)
// moves many texels between neighbouring pixels; one sample each skips the
// texels in between: streaks, sparkle, and shimmer as soon as anything moves.
const LOOK_NONE: u32 = 0u;     // nothing to filter: the straight view, a solid colour
const LOOK_SCREEN: u32 = 1u;   // background texture at look_uv
const LOOK_ENV: u32 = 2u;      // environment map along look_vec
const LOOK_GROUND: u32 = 3u;   // projected ground at world point look_vec
var<private> look_kind: u32 = 0u;
var<private> look_uv: vec2<f32> = vec2<f32>(0.0);
var<private> look_vec: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);

fn dbg_exit_interface(p: vec3<f32>, on_wall: bool) -> u32 {
    if (on_wall) {
        return DBG_EXIT_WALL;
    }
    let l = world_to_local(container, p);
    let to_wall = min(
        min(container.half_width - abs(l.x), container.half_depth - abs(l.z)),
        l.y + container.half_height,
    );
    return select(DBG_EXIT_SURFACE, DBG_EXIT_SURFACE_WALL, to_wall < DBG_NEAR_WALL);
}

// Encoding per McDebugView (state/rendering.rs) - decoded by scripts/debug_decode.py
fn debug_view_output() -> vec3<f32> {
    let end = dbg_end;
    let bounces = f32(dbg_bounces + 1u) / 8.0;
    switch (water.debug_view) {
        case 1u: {
            return vec3<f32>(f32(dbg_path) / 16.0, f32(end) / 8.0, bounces);
        }
        case 2u: {
            return vec3<f32>(max(dbg_uv, vec2<f32>(0.0)), f32(end) / 8.0);
        }
        case 3u: {
            // Lookup jump between neighbouring pixels, in background texels
            let dims = vec2<f32>(textureDimensions(background_tex));
            let jump = max(length(dpdx(dbg_uv) * dims), length(dpdy(dbg_uv) * dims));
            return vec3<f32>(clamp(log2(1.0 + jump) / 8.0, 0.0, 1.0));
        }
        case 4u: {
            return vec3<f32>(clamp(dbg_exit_cos, 0.0, 1.0), clamp(dbg_water_path / 4.0, 0.0, 1.0), f32(dbg_exit_kind) / 8.0);
        }
        default: {
            return vec3<f32>(f32(dbg_mirror_kind) / 8.0, f32(dbg_exit_kind) / 8.0, bounces);
        }
    }
}
