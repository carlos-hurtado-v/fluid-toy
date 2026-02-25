// Shared container geometry — one struct, one buffer, used by every shader
// that needs the container bounds, rotation, or containment tests.
// Each consumer shader declares its own @group/@binding for ContainerGeometry.

struct ContainerGeometry {
    // Geometry (16 bytes)
    half_width: f32,
    half_height: f32,
    half_depth: f32,
    center_y: f32,
    // Forward rotation R = Rz * Rx: local -> world (48 bytes)
    forward_row0: vec4<f32>,
    forward_row1: vec4<f32>,
    forward_row2: vec4<f32>,
    // Inverse rotation R^T: world -> local (48 bytes)
    inverse_row0: vec4<f32>,
    inverse_row1: vec4<f32>,
    inverse_row2: vec4<f32>,
    // Physics + clip (16 bytes)
    wall_stiffness: f32,
    damping: f32,
    clip_enabled: u32,
    clip_margin: f32,
}

// Transform a world-space position to container-local space.
// Subtracts center_y, then applies the inverse rotation (R^T).
fn world_to_local(c: ContainerGeometry, pos: vec3<f32>) -> vec3<f32> {
    let p = vec3<f32>(pos.x, pos.y - c.center_y, pos.z);
    return vec3<f32>(
        dot(c.inverse_row0.xyz, p),
        dot(c.inverse_row1.xyz, p),
        dot(c.inverse_row2.xyz, p),
    );
}

// Transform a container-local position to world space.
// Applies the forward rotation (R), then adds center_y.
fn local_to_world(c: ContainerGeometry, local: vec3<f32>) -> vec3<f32> {
    let rotated = vec3<f32>(
        dot(c.forward_row0.xyz, local),
        dot(c.forward_row1.xyz, local),
        dot(c.forward_row2.xyz, local),
    );
    return vec3<f32>(rotated.x, rotated.y + c.center_y, rotated.z);
}

// Rotate a direction vector from world to local space (no translation).
fn world_dir_to_local(c: ContainerGeometry, dir: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c.inverse_row0.xyz, dir),
        dot(c.inverse_row1.xyz, dir),
        dot(c.inverse_row2.xyz, dir),
    );
}

// Rotate a direction vector from local to world space (no translation).
fn local_dir_to_world(c: ContainerGeometry, dir: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c.forward_row0.xyz, dir),
        dot(c.forward_row1.xyz, dir),
        dot(c.forward_row2.xyz, dir),
    );
}

// Signed distance from a local-space point to the box boundary.
// Negative inside, positive outside.
fn box_sdf(c: ContainerGeometry, local_pos: vec3<f32>) -> f32 {
    let d = abs(local_pos) - vec3<f32>(c.half_width, c.half_height, c.half_depth);
    return max(d.x, max(d.y, d.z));
}

// Returns true if local_pos is inside the box with the given margin.
fn is_inside_box(c: ContainerGeometry, local_pos: vec3<f32>, margin: f32) -> bool {
    let hw = c.half_width + margin;
    let hh = c.half_height + margin;
    let hd = c.half_depth + margin;
    return local_pos.x >= -hw && local_pos.x <= hw &&
           local_pos.y >= -hh && local_pos.y <= hh &&
           local_pos.z >= -hd && local_pos.z <= hd;
}
