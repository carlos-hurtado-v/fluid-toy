// Water shader (mc_render), part: world-space in-water test on the density field + the mesh normal

// === World-space in-water test (rendering.mc_volume_trace) ===
// The screen-space test above knows the water only by the nearest back and
// front faces at a pixel: it cannot tell what a ray does behind a body, behind
// a nearer fold of the surface (a crater wall, a crest), or off screen, and
// each of those blind spots drew its own stripes and stair steps into mirrors.
// The mesh is the iso-surface of a density field that is still on the GPU:
// a point is in the water iff the field there is at least the iso value, and
// the surface normal is the field's gradient. No view dependence.

// Within two cells of a wall the field is not the water's own: mc_wall_bound
// cuts it with a ramp that puts the mesh's sides on the wall plane. Points
// are read at least this many cells inside the walls and floor: the last
// stretch to a wall sees the field as it is that far in (the walls themselves
// are exact planes, box_interior_exit).
const VOLUME_WALL_INSET: f32 = 3.0;
// water_normal's spline reaches 2 voxels either way: its own inset
const NORMAL_WALL_INSET: f32 = 4.0;
// A ray starts on the mesh, which is not exactly the interpolated field's
// surface: samples that read as out of the water before any read as in it,
// this close to the start, are the ray still getting under the surface (m)
const VOLUME_ENTRY_SLACK: f32 = 0.03;

// Where the field is read for world point p: out of any body's film, and
// inside the container by `inset_cells` (sides, floor and top)
fn volume_point(p: vec3<f32>, inset_cells: f32) -> vec3<f32> {
    var q = p;
    if (water.body_count > 0u) {
        q = body_film_push(q);
    }
    if (container.clip_enabled != 0u) {
        let inset = inset_cells * mc_grid.cell_size;
        let h = max(vec3<f32>(container.half_width, container.half_height, container.half_depth) - inset, vec3<f32>(0.0));
        let l = world_to_local(container, q);
        q = local_to_world(container, vec3<f32>(clamp(l.x, -h.x, h.x), clamp(l.y, -h.y, h.y), clamp(l.z, -h.z, h.z)));
    }
    return q;
}

// The field at a point already placed by volume_point, in units of the iso
// value (>= 1: water)
fn field_at(q: vec3<f32>) -> f32 {
    let g = (q - mc_grid.grid_min) / mc_grid.cell_size;
    let uvw = (g + 0.5) / f32(mc_grid.grid_size);
    return textureSampleLevel(density_tex, density_sampler, uvw, 0.0).r / mc_grid.iso_value;
}

// The field at world point p in units of the iso value. Above the open top
// of the container there is no field: not water.
fn water_density(p: vec3<f32>) -> f32 {
    if (container.clip_enabled != 0u && world_to_local(container, p).y > container.half_height) {
        return 0.0;
    }
    return field_at(volume_point(p, VOLUME_WALL_INSET));
}

// mc_generate's normal at a voxel: both read the texture mc_voxel_normals.wgsl
// writes (the field's central-difference gradient, denoised on calm water)
fn voxel_normal(i: vec3<i32>) -> vec3<f32> {
    let top = vec3<i32>(i32(mc_grid.grid_size) - 1);
    return oct_decode(textureLoad(normal_tex, clamp(i, vec3<i32>(0), top), 0).r);
}

// Marching-cubes cell layout, as in mc_generate.wgsl (keep in sync): corner
// offsets, and the two corners each of the 12 edges joins
const MC_CORNER: array<vec3<i32>, 8> = array<vec3<i32>, 8>(
    vec3<i32>(0, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(0, 1, 1), vec3<i32>(0, 0, 1),
    vec3<i32>(1, 0, 0), vec3<i32>(1, 1, 0), vec3<i32>(1, 1, 1), vec3<i32>(1, 0, 1),
);
const MC_EDGE: array<vec2<u32>, 12> = array<vec2<u32>, 12>(
    vec2<u32>(0u, 1u), vec2<u32>(1u, 2u), vec2<u32>(2u, 3u), vec2<u32>(3u, 0u),
    vec2<u32>(4u, 5u), vec2<u32>(5u, 6u), vec2<u32>(6u, 7u), vec2<u32>(7u, 4u),
    vec2<u32>(0u, 4u), vec2<u32>(1u, 5u), vec2<u32>(2u, 6u), vec2<u32>(3u, 7u),
);

// Barycentric coordinates of the point of triangle abc closest to p
// (Ericson, Real-Time Collision Detection 5.1.5)
fn closest_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / max(d1 - d3, 1e-12);
        return vec3<f32>(1.0 - v, v, 0.0);
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / max(d2 - d6, 1e-12);
        return vec3<f32>(1.0 - w, 0.0, w);
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-12);
        return vec3<f32>(0.0, 1.0 - w, w);
    }
    let denom = 1.0 / max(va + vb + vc, 1e-12);
    let v = vb * denom;
    let w = vc * denom;
    return vec3<f32>(1.0 - v - w, v, w);
}

// Outward surface normal at world point p (w = 1): the MESH's normal there.
// The cell p lies in is triangulated exactly as mc_generate did it (same
// corner values, same table, same vertex normals: the voxel normals of the
// two voxels of each cut edge, blended along it), and the normal is
// interpolated across the triangle nearest p, as the rasteriser would. 16
// texel loads. Anything "smoother" or cheaper was tried and shows
// in refractions: differences of interpolated samples, a blend of the cell's
// eight voxel normals and a cubic B-spline gradient all draw contour bands
// two cells apart (they let in voxels a cell or more off the surface); an
// inverse-distance blend of the cell's edge vertices is right on average but
// jumps at every cell face (cell-sized blocks and beaded rings, seen from
// close by). Only the mesh's own interpolation is both right and continuous.
fn water_normal(p: vec3<f32>) -> vec4<f32> {
    let q = volume_point(p, NORMAL_WALL_INSET);
    let g = (q - mc_grid.grid_min) / mc_grid.cell_size;
    let i0 = vec3<i32>(floor(g));
    let f = g - floor(g);
    let top = vec3<i32>(i32(mc_grid.grid_size) - 1);
    var value: array<f32, 8>;
    var normal: array<vec3<f32>, 8>;
    var all = vec3<f32>(0.0);
    var case_index = 0u;
    for (var k = 0u; k < 8u; k++) {
        let o = MC_CORNER[k];
        value[k] = textureLoad(density_tex, clamp(i0 + o, vec3<i32>(0), top), 0).r;
        normal[k] = voxel_normal(i0 + o);
        if (value[k] >= mc_grid.iso_value) {
            case_index |= 1u << k;
        }
        let w3 = select(vec3<f32>(1.0) - f, f, o == vec3<i32>(1));
        all += normal[k] * (w3.x * w3.y * w3.z);
    }
    var n = vec3<f32>(0.0);
    if (case_index != 0u && case_index != 255u) {
        // The mesh's vertices on this cell's edges (cell coordinates) and
        // their normals
        var edge_pos: array<vec3<f32>, 12>;
        var edge_normal: array<vec3<f32>, 12>;
        for (var e = 0u; e < 12u; e++) {
            let c0 = MC_EDGE[e].x;
            let c1 = MC_EDGE[e].y;
            var t = 0.5;
            if (abs(value[c1] - value[c0]) > 0.00001) {
                t = (mc_grid.iso_value - value[c0]) / (value[c1] - value[c0]);
            }
            edge_pos[e] = mix(vec3<f32>(MC_CORNER[c0]), vec3<f32>(MC_CORNER[c1]), t);
            edge_normal[e] = normalize(mix(normal[c0], normal[c1], t));
        }
        var best = 1e9;
        let table = case_index * 16u;
        for (var i = 0u; i < 15u; i += 3u) {
            let e0 = mc_tri_table[table + i];
            if (e0 < 0) {
                break;
            }
            let e1 = mc_tri_table[table + i + 1u];
            let e2 = mc_tri_table[table + i + 2u];
            let bary = closest_on_triangle(f, edge_pos[e0], edge_pos[e1], edge_pos[e2]);
            let d = f - (edge_pos[e0] * bary.x + edge_pos[e1] * bary.y + edge_pos[e2] * bary.z);
            let dist = dot(d, d);
            if (dist < best) {
                best = dist;
                n = edge_normal[e0] * bary.x + edge_normal[e1] * bary.y + edge_normal[e2] * bary.z;
            }
        }
    }
    if (dot(n, n) < 1e-8) {
        // No surface in this cell (p a little off it): the cell's eight
        n = all;
    }
    if (dot(n, n) < 1e-8) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(normalize(n), 1.0);
}

// What a point on a ray inside the water has run into: (kind, field / iso).
// Kinds as trace_event: 0 in the water, 1 out of it, 2 an opaque surface that
// only the depth buffer knows (water.depth_occluders: pool walls, Custom
// bodies; unknown off screen). In a glass tank with procedural bodies there
// is none: ray_body_hit bounds the ray, and the test (five texel loads a
// sample) is skipped.
fn volume_event(p: vec3<f32>) -> vec2<f32> {
    prb_bg = -1.0;
    if (water.depth_occluders != 0u) {
        let q = screen_point(p);
        if (all(q.xy >= vec2<f32>(0.0)) && all(q.xy <= vec2<f32>(1.0))) {
            let bg = depth_smooth(background_depth_tex, q.xy);
            prb_bg = bg;
            if (q.z >= bg && (water.body_count == 0u || !on_analytic_body(screen_to_world(q.xy, bg)))) {
                return vec2<f32>(2.0, 0.0);
            }
        }
    }
    let d = water_density(p);
    return vec2<f32>(select(1.0, 0.0, d >= 1.0), d);
}
