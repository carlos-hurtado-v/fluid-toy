// Wireframe rendering shader for container visualization

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct ContainerParams {
    // Container bounds
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
    min_z: f32,
    max_z: f32,
    // Line color
    color_r: f32,
    color_g: f32,
    color_b: f32,
    color_a: f32,
    _padding: vec2<f32>,
    // Rotation matrix (local-to-world transform)
    rotation_row0: vec4<f32>,
    rotation_row1: vec4<f32>,
    rotation_row2: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> container: ContainerParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

// 24 vertices for 12 edges (2 vertices per edge)
// Edge indices: each edge connects two corners of the box
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // 8 corners of the box
    var corners = array<vec3<f32>, 8>(
        vec3<f32>(container.min_x, container.min_y, container.min_z), // 0: ---
        vec3<f32>(container.max_x, container.min_y, container.min_z), // 1: +--
        vec3<f32>(container.max_x, container.max_y, container.min_z), // 2: ++-
        vec3<f32>(container.min_x, container.max_y, container.min_z), // 3: -+-
        vec3<f32>(container.min_x, container.min_y, container.max_z), // 4: --+
        vec3<f32>(container.max_x, container.min_y, container.max_z), // 5: +-+
        vec3<f32>(container.max_x, container.max_y, container.max_z), // 6: +++
        vec3<f32>(container.min_x, container.max_y, container.max_z), // 7: -++
    );

    // 12 edges, 2 vertices each = 24 vertices
    // Bottom face edges (y = min)
    // Edge 0: 0-1, Edge 1: 1-5, Edge 2: 5-4, Edge 3: 4-0
    // Top face edges (y = max)
    // Edge 4: 3-2, Edge 5: 2-6, Edge 6: 6-7, Edge 7: 7-3
    // Vertical edges
    // Edge 8: 0-3, Edge 9: 1-2, Edge 10: 5-6, Edge 11: 4-7
    var edge_indices = array<u32, 24>(
        0u, 1u,  // bottom front
        1u, 5u,  // bottom right
        5u, 4u,  // bottom back
        4u, 0u,  // bottom left
        3u, 2u,  // top front
        2u, 6u,  // top right
        6u, 7u,  // top back
        7u, 3u,  // top left
        0u, 3u,  // front left vertical
        1u, 2u,  // front right vertical
        5u, 6u,  // back right vertical
        4u, 7u,  // back left vertical
    );

    let corner_index = edge_indices[vertex_index];
    let local_pos = corners[corner_index];

    // Apply rotation to transform from container-local space to world space
    let rot_row0 = container.rotation_row0.xyz;
    let rot_row1 = container.rotation_row1.xyz;
    let rot_row2 = container.rotation_row2.xyz;

    let world_pos = vec3<f32>(
        dot(rot_row0, local_pos),
        dot(rot_row1, local_pos),
        dot(rot_row2, local_pos)
    );

    let view_pos = camera.view * vec4<f32>(world_pos, 1.0);
    let clip_pos = camera.projection * view_pos;

    var output: VertexOutput;
    output.position = clip_pos;
    output.color = vec4<f32>(container.color_r, container.color_g, container.color_b, container.color_a);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
