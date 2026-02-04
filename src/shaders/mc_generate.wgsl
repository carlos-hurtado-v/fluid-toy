// Marching Cubes - Mesh Generation
// Extracts surface triangles from density grid

struct GridParams {
    grid_min: vec3<f32>,
    cell_size: f32,
    grid_dims: vec3<u32>,
    num_particles: u32,
    smoothing_radius: f32,
    surface_threshold: f32,
    _padding: vec2<f32>,
}

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
}

struct Counter {
    count: atomic<u32>,
}

@group(0) @binding(0) var<storage, read> density_grid: array<f32>;
@group(0) @binding(1) var<uniform> params: GridParams;
@group(0) @binding(2) var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(3) var<storage, read_write> counter: Counter;
@group(0) @binding(4) var<storage, read> edge_table: array<u32>;      // 256 entries
@group(0) @binding(5) var<storage, read> tri_table: array<i32>;       // 256 * 16 entries

fn grid_index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.grid_dims.x + z * params.grid_dims.x * params.grid_dims.y;
}

fn get_density(x: u32, y: u32, z: u32) -> f32 {
    if (x >= params.grid_dims.x || y >= params.grid_dims.y || z >= params.grid_dims.z) {
        return 0.0;
    }
    return density_grid[grid_index_3d(x, y, z)];
}

fn vertex_interp(p1: vec3<f32>, p2: vec3<f32>, v1: f32, v2: f32, threshold: f32) -> vec3<f32> {
    if (abs(threshold - v1) < 0.00001) { return p1; }
    if (abs(threshold - v2) < 0.00001) { return p2; }
    if (abs(v1 - v2) < 0.00001) { return p1; }

    let mu = (threshold - v1) / (v2 - v1);
    return p1 + mu * (p2 - p1);
}

fn compute_normal(pos: vec3<f32>) -> vec3<f32> {
    // Compute gradient of density field for normal
    let eps = params.cell_size * 0.5;

    // Sample density at offset positions (approximate gradient)
    let grid_pos = (pos - params.grid_min) / params.cell_size;
    let base = vec3<u32>(grid_pos);

    let dx = get_density(base.x + 1, base.y, base.z) - get_density(base.x, base.y, base.z);
    let dy = get_density(base.x, base.y + 1, base.z) - get_density(base.x, base.y, base.z);
    let dz = get_density(base.x, base.y, base.z + 1) - get_density(base.x, base.y, base.z);

    let n = -vec3<f32>(dx, dy, dz);
    let len = length(n);
    if (len > 0.0001) {
        return n / len;
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    // Need one less than grid dims (processing cells, not vertices)
    if (x >= params.grid_dims.x - 1 ||
        y >= params.grid_dims.y - 1 ||
        z >= params.grid_dims.z - 1) {
        return;
    }

    let threshold = params.surface_threshold;

    // Get density at 8 corners of this cell
    var corner_densities: array<f32, 8>;
    corner_densities[0] = get_density(x, y, z);
    corner_densities[1] = get_density(x + 1, y, z);
    corner_densities[2] = get_density(x + 1, y + 1, z);
    corner_densities[3] = get_density(x, y + 1, z);
    corner_densities[4] = get_density(x, y, z + 1);
    corner_densities[5] = get_density(x + 1, y, z + 1);
    corner_densities[6] = get_density(x + 1, y + 1, z + 1);
    corner_densities[7] = get_density(x, y + 1, z + 1);

    // Determine cube index (which corners are inside the surface)
    var cube_index = 0u;
    if (corner_densities[0] > threshold) { cube_index |= 1u; }
    if (corner_densities[1] > threshold) { cube_index |= 2u; }
    if (corner_densities[2] > threshold) { cube_index |= 4u; }
    if (corner_densities[3] > threshold) { cube_index |= 8u; }
    if (corner_densities[4] > threshold) { cube_index |= 16u; }
    if (corner_densities[5] > threshold) { cube_index |= 32u; }
    if (corner_densities[6] > threshold) { cube_index |= 64u; }
    if (corner_densities[7] > threshold) { cube_index |= 128u; }

    // Look up which edges are intersected
    let edge_mask = edge_table[cube_index];
    if (edge_mask == 0u) {
        return; // No triangles in this cell
    }

    // Corner positions in world space
    let base_pos = params.grid_min + vec3<f32>(f32(x), f32(y), f32(z)) * params.cell_size;
    let cs = params.cell_size;

    var corner_pos: array<vec3<f32>, 8>;
    corner_pos[0] = base_pos;
    corner_pos[1] = base_pos + vec3<f32>(cs, 0.0, 0.0);
    corner_pos[2] = base_pos + vec3<f32>(cs, cs, 0.0);
    corner_pos[3] = base_pos + vec3<f32>(0.0, cs, 0.0);
    corner_pos[4] = base_pos + vec3<f32>(0.0, 0.0, cs);
    corner_pos[5] = base_pos + vec3<f32>(cs, 0.0, cs);
    corner_pos[6] = base_pos + vec3<f32>(cs, cs, cs);
    corner_pos[7] = base_pos + vec3<f32>(0.0, cs, cs);

    // Compute edge vertices (interpolate where surface crosses)
    var edge_verts: array<vec3<f32>, 12>;
    if ((edge_mask & 1u) != 0u)    { edge_verts[0]  = vertex_interp(corner_pos[0], corner_pos[1], corner_densities[0], corner_densities[1], threshold); }
    if ((edge_mask & 2u) != 0u)    { edge_verts[1]  = vertex_interp(corner_pos[1], corner_pos[2], corner_densities[1], corner_densities[2], threshold); }
    if ((edge_mask & 4u) != 0u)    { edge_verts[2]  = vertex_interp(corner_pos[2], corner_pos[3], corner_densities[2], corner_densities[3], threshold); }
    if ((edge_mask & 8u) != 0u)    { edge_verts[3]  = vertex_interp(corner_pos[3], corner_pos[0], corner_densities[3], corner_densities[0], threshold); }
    if ((edge_mask & 16u) != 0u)   { edge_verts[4]  = vertex_interp(corner_pos[4], corner_pos[5], corner_densities[4], corner_densities[5], threshold); }
    if ((edge_mask & 32u) != 0u)   { edge_verts[5]  = vertex_interp(corner_pos[5], corner_pos[6], corner_densities[5], corner_densities[6], threshold); }
    if ((edge_mask & 64u) != 0u)   { edge_verts[6]  = vertex_interp(corner_pos[6], corner_pos[7], corner_densities[6], corner_densities[7], threshold); }
    if ((edge_mask & 128u) != 0u)  { edge_verts[7]  = vertex_interp(corner_pos[7], corner_pos[4], corner_densities[7], corner_densities[4], threshold); }
    if ((edge_mask & 256u) != 0u)  { edge_verts[8]  = vertex_interp(corner_pos[0], corner_pos[4], corner_densities[0], corner_densities[4], threshold); }
    if ((edge_mask & 512u) != 0u)  { edge_verts[9]  = vertex_interp(corner_pos[1], corner_pos[5], corner_densities[1], corner_densities[5], threshold); }
    if ((edge_mask & 1024u) != 0u) { edge_verts[10] = vertex_interp(corner_pos[2], corner_pos[6], corner_densities[2], corner_densities[6], threshold); }
    if ((edge_mask & 2048u) != 0u) { edge_verts[11] = vertex_interp(corner_pos[3], corner_pos[7], corner_densities[3], corner_densities[7], threshold); }

    // Generate triangles from the tri_table
    let tri_base = cube_index * 16u;
    var i = 0u;
    loop {
        let edge_idx = tri_table[tri_base + i];
        if (edge_idx < 0) {
            break;
        }

        // Get three vertices for this triangle
        let v0_idx = u32(tri_table[tri_base + i]);
        let v1_idx = u32(tri_table[tri_base + i + 1u]);
        let v2_idx = u32(tri_table[tri_base + i + 2u]);

        let p0 = edge_verts[v0_idx];
        let p1 = edge_verts[v1_idx];
        let p2 = edge_verts[v2_idx];

        // Allocate space for 3 vertices
        let base_idx = atomicAdd(&counter.count, 3u);

        // Use density gradient normals (point from high to low density = outward)
        vertices[base_idx].position = p0;
        vertices[base_idx].normal = compute_normal(p0);
        vertices[base_idx + 1u].position = p1;
        vertices[base_idx + 1u].normal = compute_normal(p1);
        vertices[base_idx + 2u].position = p2;
        vertices[base_idx + 2u].normal = compute_normal(p2);

        i += 3u;
        if (i >= 15u) {
            break;
        }
    }
}
