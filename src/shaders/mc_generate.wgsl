// Marching Cubes - Triangle Generation
// Processes each voxel and generates triangles using atomic counter

struct GridParams {
    grid_min: vec3<f32>,
    grid_size: u32,
    grid_max: vec3<f32>,
    cell_size: f32,
    kernel_radius: f32,
    iso_value: f32,
    num_particles: u32,
    _padding: f32,
}

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
}

struct Counter {
    vertex_count: atomic<u32>,
}

// Cube vertex offsets (same convention as reference)
//        3-------7
//       /|      /|
//      2-------6 |
//      | 1-----|-5
//      |/      |/
//      0-------4
const VERTEX_OFFSETS: array<vec3<f32>, 8> = array<vec3<f32>, 8>(
    vec3<f32>(0.0, 0.0, 0.0),  // 0
    vec3<f32>(0.0, 1.0, 0.0),  // 1
    vec3<f32>(0.0, 1.0, 1.0),  // 2
    vec3<f32>(0.0, 0.0, 1.0),  // 3
    vec3<f32>(1.0, 0.0, 0.0),  // 4
    vec3<f32>(1.0, 1.0, 0.0),  // 5
    vec3<f32>(1.0, 1.0, 1.0),  // 6
    vec3<f32>(1.0, 0.0, 1.0),  // 7
);

// Edge vertex pairs
const EDGE_VERTICES: array<vec2<u32>, 12> = array<vec2<u32>, 12>(
    vec2<u32>(0u, 1u), vec2<u32>(1u, 2u), vec2<u32>(2u, 3u), vec2<u32>(3u, 0u),  // bottom
    vec2<u32>(4u, 5u), vec2<u32>(5u, 6u), vec2<u32>(6u, 7u), vec2<u32>(7u, 4u),  // top
    vec2<u32>(0u, 4u), vec2<u32>(1u, 5u), vec2<u32>(2u, 6u), vec2<u32>(3u, 7u),  // vertical
);

@group(0) @binding(0) var density_field: texture_3d<f32>;
@group(0) @binding(1) var<uniform> params: GridParams;
@group(0) @binding(2) var<storage, read> edge_table: array<u32>;
@group(0) @binding(3) var<storage, read> tri_table: array<i32>;
@group(0) @binding(4) var<storage, read_write> counter: Counter;
@group(0) @binding(5) var<storage, read_write> vertices: array<Vertex>;

// Sample density at a grid point
fn sample_density(pos: vec3<i32>) -> f32 {
    let grid_size = i32(params.grid_size);
    // Clamp to grid bounds
    let clamped = clamp(pos, vec3<i32>(0), vec3<i32>(grid_size - 1));
    return textureLoad(density_field, clamped, 0).r;
}

// Interpolate vertex position along an edge
fn interpolate_vertex(p1: vec3<f32>, p2: vec3<f32>, v1: f32, v2: f32, iso: f32) -> vec3<f32> {
    if (abs(iso - v1) < 0.00001) {
        return p1;
    }
    if (abs(iso - v2) < 0.00001) {
        return p2;
    }
    if (abs(v1 - v2) < 0.00001) {
        return p1;
    }
    let t = (iso - v1) / (v2 - v1);
    return mix(p1, p2, t);
}

// Convert grid position to world position
fn grid_to_world(grid_pos: vec3<f32>) -> vec3<f32> {
    return params.grid_min + grid_pos * params.cell_size;
}

// Compute normal from density gradient (wider sampling for smoother normals)
fn compute_normal(pos: vec3<i32>) -> vec3<f32> {
    // Use step of 2 cells to average over a wider area, reducing noise from
    // individual particle contributions in the density field
    let dx = sample_density(pos + vec3<i32>(2, 0, 0)) - sample_density(pos - vec3<i32>(2, 0, 0));
    let dy = sample_density(pos + vec3<i32>(0, 2, 0)) - sample_density(pos - vec3<i32>(0, 2, 0));
    let dz = sample_density(pos + vec3<i32>(0, 0, 2)) - sample_density(pos - vec3<i32>(0, 0, 2));
    let grad = vec3<f32>(dx, dy, dz);
    let len = length(grad);
    if (len > 0.0001) {
        return -normalize(grad);  // Point outward from surface
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_size = params.grid_size;

    // Process voxels (grid_size - 1 in each dimension since we need 8 corners)
    if (global_id.x >= grid_size - 1u || global_id.y >= grid_size - 1u || global_id.z >= grid_size - 1u) {
        return;
    }

    let voxel_pos = vec3<i32>(global_id);
    let iso = params.iso_value;

    // Sample density at 8 cube corners
    var values: array<f32, 8>;
    values[0] = sample_density(voxel_pos + vec3<i32>(0, 0, 0));
    values[1] = sample_density(voxel_pos + vec3<i32>(0, 1, 0));
    values[2] = sample_density(voxel_pos + vec3<i32>(0, 1, 1));
    values[3] = sample_density(voxel_pos + vec3<i32>(0, 0, 1));
    values[4] = sample_density(voxel_pos + vec3<i32>(1, 0, 0));
    values[5] = sample_density(voxel_pos + vec3<i32>(1, 1, 0));
    values[6] = sample_density(voxel_pos + vec3<i32>(1, 1, 1));
    values[7] = sample_density(voxel_pos + vec3<i32>(1, 0, 1));

    // Compute case index (which corners are inside the surface)
    var case_index = 0u;
    for (var i = 0u; i < 8u; i++) {
        if (values[i] >= iso) {
            case_index |= 1u << i;
        }
    }

    // Skip empty or full voxels
    if (case_index == 0u || case_index == 255u) {
        return;
    }

    // Compute world positions of cube corners
    var corner_positions: array<vec3<f32>, 8>;
    for (var i = 0u; i < 8u; i++) {
        corner_positions[i] = grid_to_world(vec3<f32>(voxel_pos) + VERTEX_OFFSETS[i]);
    }

    // Compute normals at cube corners (from density gradient)
    var corner_normals: array<vec3<f32>, 8>;
    corner_normals[0] = compute_normal(voxel_pos + vec3<i32>(0, 0, 0));
    corner_normals[1] = compute_normal(voxel_pos + vec3<i32>(0, 1, 0));
    corner_normals[2] = compute_normal(voxel_pos + vec3<i32>(0, 1, 1));
    corner_normals[3] = compute_normal(voxel_pos + vec3<i32>(0, 0, 1));
    corner_normals[4] = compute_normal(voxel_pos + vec3<i32>(1, 0, 0));
    corner_normals[5] = compute_normal(voxel_pos + vec3<i32>(1, 1, 0));
    corner_normals[6] = compute_normal(voxel_pos + vec3<i32>(1, 1, 1));
    corner_normals[7] = compute_normal(voxel_pos + vec3<i32>(1, 0, 1));

    // Compute edge intersection points and interpolated normals
    var edge_verts: array<vec3<f32>, 12>;
    var edge_normals: array<vec3<f32>, 12>;
    let edge_mask = edge_table[case_index];

    for (var e = 0u; e < 12u; e++) {
        if ((edge_mask & (1u << e)) != 0u) {
            let v0 = EDGE_VERTICES[e].x;
            let v1 = EDGE_VERTICES[e].y;

            // Interpolation factor
            let val0 = values[v0];
            let val1 = values[v1];
            var t = 0.5;
            if (abs(val1 - val0) > 0.00001) {
                t = (iso - val0) / (val1 - val0);
            }

            // Interpolate position
            edge_verts[e] = mix(corner_positions[v0], corner_positions[v1], t);

            // Interpolate normal (and renormalize)
            edge_normals[e] = normalize(mix(corner_normals[v0], corner_normals[v1], t));
        }
    }

    // Generate triangles from lookup table
    // tri_table is 256 * 16 = 4096 entries
    let table_offset = case_index * 16u;

    var i = 0u;
    loop {
        let edge0 = tri_table[table_offset + i];
        if (edge0 < 0) {
            break;
        }
        let edge1 = tri_table[table_offset + i + 1u];
        let edge2 = tri_table[table_offset + i + 2u];

        // Claim space for 3 vertices
        let vertex_offset = atomicAdd(&counter.vertex_count, 3u);

        // Get vertex positions and smooth normals
        let p0 = edge_verts[edge0];
        let p1 = edge_verts[edge1];
        let p2 = edge_verts[edge2];

        let n0 = edge_normals[edge0];
        let n1 = edge_normals[edge1];
        let n2 = edge_normals[edge2];

        // Write vertices with smooth normals
        vertices[vertex_offset + 0u] = Vertex(p0, n0);
        vertices[vertex_offset + 1u] = Vertex(p1, n1);
        vertices[vertex_offset + 2u] = Vertex(p2, n2);

        i += 3u;
        if (i >= 15u) {
            break;
        }
    }
}
