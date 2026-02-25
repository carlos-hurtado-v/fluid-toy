// Wireframe rendering shader for container visualization

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct WireframeStyle {
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> container: ContainerGeometry;
@group(0) @binding(2) var<uniform> style: WireframeStyle;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

// 24 vertices for 12 edges (2 vertices per edge)
// Edge indices: each edge connects two corners of the box
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let hw = container.half_width;
    let hh = container.half_height;
    let hd = container.half_depth;

    // 8 corners of the box in local (container-centered) space
    var corners = array<vec3<f32>, 8>(
        vec3<f32>(-hw, -hh, -hd), // 0: ---
        vec3<f32>( hw, -hh, -hd), // 1: +--
        vec3<f32>( hw,  hh, -hd), // 2: ++-
        vec3<f32>(-hw,  hh, -hd), // 3: -+-
        vec3<f32>(-hw, -hh,  hd), // 4: --+
        vec3<f32>( hw, -hh,  hd), // 5: +-+
        vec3<f32>( hw,  hh,  hd), // 6: +++
        vec3<f32>(-hw,  hh,  hd), // 7: -++
    );

    // 12 edges, 2 vertices each = 24 vertices
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

    // Transform from container-local space to world space
    let world_pos = local_to_world(container, local_pos);

    let view_pos = camera.view * vec4<f32>(world_pos, 1.0);
    let clip_pos = camera.projection * view_pos;

    var output: VertexOutput;
    output.position = clip_pos;
    output.color = style.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
