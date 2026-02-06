// SPH 3D Integration Shader
// Soft wall penalty forces + hard boundary backstop

struct SphParticle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    force: vec3<f32>,
    density: f32,
    near_density: f32,
    // WGSL automatically pads struct to 64 bytes for arrays
}

struct SphParams {
    kernel_radius: f32,
    kernel_radius_sq: f32,
    kernel_radius_pow5: f32,
    kernel_radius_pow6: f32,
    kernel_radius_pow9: f32,
    mass: f32,
    rest_density: f32,
    stiffness: f32,
    near_stiffness: f32,
    viscosity: f32,
    dt: f32,
    num_particles: u32,
}

struct BoundsParams {
    bound_x: f32,        // Half-width (symmetric: -bound_x to +bound_x)
    bound_z: f32,        // Half-depth (symmetric: -bound_z to +bound_z)
    floor_y: f32,        // Floor Y position (asymmetric)
    ceiling_y: f32,      // Ceiling Y position (asymmetric)
    wall_stiffness: f32,
    damping: f32,
    _padding1: f32,
    _padding2: f32,
    // Rotation matrix rows (transforms world -> container local space)
    rotation_row0: vec4<f32>,
    rotation_row1: vec4<f32>,
    rotation_row2: vec4<f32>,
}

struct MouseForce {
    position: vec3<f32>,
    radius: f32,
    strength: f32,
    is_active: u32,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SphParticle3D>;
@group(0) @binding(2) var<uniform> bounds: BoundsParams;
@group(0) @binding(3) var<uniform> mouse_force: MouseForce;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_particles) {
        return;
    }

    var pos = particles[i].position;
    var vel = particles[i].velocity;

    // Force field contains acceleration directly (Monaghan symmetric formulation)
    var accel = particles[i].force;

    // Clamp SPH acceleration to prevent explosions from extreme pressure
    let max_accel = 200.0;
    let accel_mag = length(accel);
    if (accel_mag > max_accel) {
        accel = accel * (max_accel / accel_mag);
    }

    // === SOFT WALL PENALTY FORCES ===
    // Repulsive force within one kernel radius of each wall.
    // Quadratic ramp: zero at boundary_layer distance, wall_stiffness at wall surface.
    let rot_row0 = bounds.rotation_row0.xyz;
    let rot_row1 = bounds.rotation_row1.xyz;
    let rot_row2 = bounds.rotation_row2.xyz;

    // Transform position to container-local space
    let local_pos = vec3<f32>(
        dot(rot_row0, pos),
        dot(rot_row1, pos),
        dot(rot_row2, pos)
    );

    let boundary_layer = params.kernel_radius;
    var wall_accel = vec3<f32>(0.0, 0.0, 0.0);

    // X axis (symmetric: -bound_x to +bound_x)
    let dist_neg_x = local_pos.x - (-bounds.bound_x);
    if (dist_neg_x < boundary_layer) {
        let t = 1.0 - dist_neg_x / boundary_layer;
        wall_accel.x += bounds.wall_stiffness * t * t;
    }
    let dist_pos_x = bounds.bound_x - local_pos.x;
    if (dist_pos_x < boundary_layer) {
        let t = 1.0 - dist_pos_x / boundary_layer;
        wall_accel.x -= bounds.wall_stiffness * t * t;
    }

    // Y axis (asymmetric: floor_y to ceiling_y)
    let dist_floor = local_pos.y - bounds.floor_y;
    if (dist_floor < boundary_layer) {
        let t = 1.0 - dist_floor / boundary_layer;
        wall_accel.y += bounds.wall_stiffness * t * t;
    }
    let dist_ceiling = bounds.ceiling_y - local_pos.y;
    if (dist_ceiling < boundary_layer) {
        let t = 1.0 - dist_ceiling / boundary_layer;
        wall_accel.y -= bounds.wall_stiffness * t * t;
    }

    // Z axis (symmetric: -bound_z to +bound_z)
    let dist_neg_z = local_pos.z - (-bounds.bound_z);
    if (dist_neg_z < boundary_layer) {
        let t = 1.0 - dist_neg_z / boundary_layer;
        wall_accel.z += bounds.wall_stiffness * t * t;
    }
    let dist_pos_z = bounds.bound_z - local_pos.z;
    if (dist_pos_z < boundary_layer) {
        let t = 1.0 - dist_pos_z / boundary_layer;
        wall_accel.z -= bounds.wall_stiffness * t * t;
    }

    // Transform wall acceleration from local to world space (multiply by R^T)
    accel += vec3<f32>(
        rot_row0.x * wall_accel.x + rot_row1.x * wall_accel.y + rot_row2.x * wall_accel.z,
        rot_row0.y * wall_accel.x + rot_row1.y * wall_accel.y + rot_row2.y * wall_accel.z,
        rot_row0.z * wall_accel.x + rot_row1.z * wall_accel.y + rot_row2.z * wall_accel.z,
    );

    // Apply mouse force (if active)
    if (mouse_force.is_active == 1u) {
        let to_mouse = mouse_force.position - pos;
        let dist = length(to_mouse);
        if (dist < mouse_force.radius && dist > 0.001) {
            // Smooth falloff: stronger near center, weaker at edge
            let falloff = 1.0 - (dist / mouse_force.radius);
            let force_dir = normalize(to_mouse);
            // Negative strength = repel (push away), positive = attract
            accel += force_dir * mouse_force.strength * falloff * -1.0;
        }
    }

    // Semi-implicit Euler integration
    vel = vel + params.dt * accel;
    pos = pos + params.dt * vel;

    // === HARD BOUNDARY BACKSTOP ===
    // Safety clamp for particles that escape the soft penalty layer.
    // This should rarely activate — the soft forces above handle normal containment.
    var local_pos_new = vec3<f32>(
        dot(rot_row0, pos),
        dot(rot_row1, pos),
        dot(rot_row2, pos)
    );

    var local_vel_new = vec3<f32>(
        dot(rot_row0, vel),
        dot(rot_row1, vel),
        dot(rot_row2, vel)
    );

    let restitution = bounds.damping;

    // X axis (symmetric: -bound_x to +bound_x)
    if (local_pos_new.x < -bounds.bound_x) {
        local_pos_new.x = -bounds.bound_x;
        local_vel_new.x = abs(local_vel_new.x) * restitution;
    } else if (local_pos_new.x > bounds.bound_x) {
        local_pos_new.x = bounds.bound_x;
        local_vel_new.x = -abs(local_vel_new.x) * restitution;
    }

    // Y axis (asymmetric: floor_y to ceiling_y)
    if (local_pos_new.y < bounds.floor_y) {
        local_pos_new.y = bounds.floor_y;
        local_vel_new.y = abs(local_vel_new.y) * restitution;
    } else if (local_pos_new.y > bounds.ceiling_y) {
        local_pos_new.y = bounds.ceiling_y;
        local_vel_new.y = -abs(local_vel_new.y) * restitution;
    }

    // Z axis (symmetric: -bound_z to +bound_z)
    if (local_pos_new.z < -bounds.bound_z) {
        local_pos_new.z = -bounds.bound_z;
        local_vel_new.z = abs(local_vel_new.z) * restitution;
    } else if (local_pos_new.z > bounds.bound_z) {
        local_pos_new.z = bounds.bound_z;
        local_vel_new.z = -abs(local_vel_new.z) * restitution;
    }

    // Transform back to world space (multiply by transpose of rotation matrix)
    pos = vec3<f32>(
        rot_row0.x * local_pos_new.x + rot_row1.x * local_pos_new.y + rot_row2.x * local_pos_new.z,
        rot_row0.y * local_pos_new.x + rot_row1.y * local_pos_new.y + rot_row2.y * local_pos_new.z,
        rot_row0.z * local_pos_new.x + rot_row1.z * local_pos_new.y + rot_row2.z * local_pos_new.z
    );

    vel = vec3<f32>(
        rot_row0.x * local_vel_new.x + rot_row1.x * local_vel_new.y + rot_row2.x * local_vel_new.z,
        rot_row0.y * local_vel_new.x + rot_row1.y * local_vel_new.y + rot_row2.y * local_vel_new.z,
        rot_row0.z * local_vel_new.x + rot_row1.z * local_vel_new.y + rot_row2.z * local_vel_new.z
    );

    particles[i].velocity = vel;
    particles[i].position = pos;
}
