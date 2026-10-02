// Marching Cubes - Container wall bound
//
// Last field pass before mesh generation: makes the water end ON the
// container walls. Until here, voxels outside the container hold a sentinel
// (-1, see mc_density.wgsl) that the smoothing filters skip, so a wall is not
// a water/air edge to them and the surface stays flat right up to it. This
// pass turns the sentinel into geometry:
//
//   final = min(field extended across the wall, wall ramp)
//
// The ramp is linear in the distance to the nearest wall plane and equals the
// iso value exactly on it, so marching cubes' linear interpolation puts the
// mesh's sides on the plane itself at any tilt (cutting at the sentinel
// instead would follow the voxel grid: stair-steps on a tilted wall). The
// extension gives voxels just outside the wall the field of the nearest point
// inside, so the free surface runs straight into the wall rather than
// rounding off toward it.
//
// container_common.wgsl is prepended (ContainerGeometry, world_to_local).

struct GridParams {
    grid_min: vec3<f32>,
    grid_size: u32,
    grid_max: vec3<f32>,
    cell_size: f32,
    kernel_radius: f32,
    iso_value: f32,
    num_particles: u32,
    max_vertices: u32,
}

struct BoundParams {
    // Field value of bulk water (the number density at rest spacing)
    interior: f32,
    // The bounding planes sit this far outside the walls (m). 0 for the glass
    // tank; the opaque pool pushes the mesh's sides just behind its wall
    // faces, where they cannot z-fight with them.
    wall_offset: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var src_field: texture_3d<f32>;
@group(0) @binding(1) var dst_field: texture_storage_3d<r32float, write>;
@group(0) @binding(2) var<uniform> params: GridParams;
@group(0) @binding(3) var<uniform> container: ContainerGeometry;
@group(0) @binding(4) var<uniform> bound: BoundParams;

// Voxels deeper inside than this keep their value untouched (cells)
const RAMP_REACH: f32 = 2.0;
// Voxels further outside than this hold no extension, just the ramp (cells)
const EXTEND_REACH: f32 = 2.0;

// The filtered field at the nearest point one cell inside the walls: trilinear
// over the in-container voxels around it.
fn extended_field(local: vec3<f32>) -> f32 {
    let half = vec3<f32>(container.half_width, container.half_height, container.half_depth);
    let lim = max(half - params.cell_size, vec3<f32>(0.0));
    let q = local_to_world(container, clamp(local, -lim, lim));
    let g = (q - params.grid_min) / params.cell_size;
    let g0 = vec3<i32>(floor(g));
    let f = g - floor(g);
    let grid_max = vec3<i32>(i32(params.grid_size) - 1);
    var sum = 0.0;
    var w_sum = 0.0;
    for (var k = 0; k < 8; k++) {
        let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let v = textureLoad(src_field, clamp(g0 + o, vec3<i32>(0), grid_max), 0).r;
        let w3 = select(1.0 - f, f, o == vec3<i32>(1));
        let w = w3.x * w3.y * w3.z;
        if (v >= 0.0) {
            sum += v * w;
            w_sum += w;
        }
    }
    return select(0.0, sum / max(w_sum, 1e-6), w_sum > 1e-6);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= vec3<u32>(params.grid_size))) {
        return;
    }
    let p = vec3<i32>(id);
    let cell = params.cell_size;
    let value = textureLoad(src_field, p, 0).r;

    // Voxel i sits at grid_min + i * cell_size (as in mc_density / mc_generate)
    let local = world_to_local(container, params.grid_min + vec3<f32>(id) * cell);
    let half = vec3<f32>(container.half_width, container.half_height, container.half_depth);
    // Distance inside the nearest bounding plane, negative outside
    let to_wall = half + bound.wall_offset - abs(local);
    let dist = min(to_wall.x, min(to_wall.y, to_wall.z));

    if (value >= 0.0 && dist > RAMP_REACH * cell) {
        textureStore(dst_field, p, vec4<f32>(value, 0.0, 0.0, 0.0));
        return;
    }

    var field = value;
    if (value < 0.0) {
        field = 0.0;
        if (dist > -EXTEND_REACH * cell) {
            field = extended_field(local);
        }
    }

    // iso on the plane, bulk water one cell in: only the voxels next to a wall
    // are limited by it, and those on either side of the plane lie on one
    // line, which is what places the mesh exactly. Bounded below so far
    // voxels stay finite (mesh edges span one cell).
    let ramp = params.iso_value + (bound.interior - params.iso_value) * max(dist / cell, -1.5);
    textureStore(dst_field, p, vec4<f32>(min(field, ramp), 0.0, 0.0, 0.0));
}
