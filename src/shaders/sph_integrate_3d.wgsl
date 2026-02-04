// SPH 3D Integration Shader
// Soft boundary forces only (matching reference implementation)

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
    bound_x: f32,
    bound_y: f32,
    bound_z: f32,
    wall_stiffness: f32,
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

    // Use safety minimum for density to match force shader
    let density = max(particles[i].density, 1.0);
    var pos = particles[i].position;
    var vel = particles[i].velocity;
    let force = particles[i].force;

    // Compute acceleration from forces
    var accel = force / density;

    // Clamp acceleration to prevent explosions from extreme pressure
    let max_accel = 50.0;
    let accel_mag = length(accel);
    if (accel_mag > max_accel) {
        accel = accel * (max_accel / accel_mag);
    }

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

    // Transform particle position to container-local space
    // The rotation matrix transforms world coords -> container local coords
    let rot_row0 = bounds.rotation_row0.xyz;
    let rot_row1 = bounds.rotation_row1.xyz;
    let rot_row2 = bounds.rotation_row2.xyz;

    let local_pos = vec3<f32>(
        dot(rot_row0, pos),
        dot(rot_row1, pos),
        dot(rot_row2, pos)
    );

    // Soft boundary forces in container-local space
    let x_plus_dist = bounds.bound_x - local_pos.x;
    let x_minus_dist = bounds.bound_x + local_pos.x;
    let y_plus_dist = bounds.bound_y - local_pos.y;
    let y_minus_dist = bounds.bound_y + local_pos.y;
    let z_plus_dist = bounds.bound_z - local_pos.z;
    let z_minus_dist = bounds.bound_z + local_pos.z;

    // Compute forces in container-local space
    var local_force = vec3<f32>(0.0, 0.0, 0.0);
    local_force.x += bounds.wall_stiffness * min(x_plus_dist, 0.0);   // Right wall
    local_force.x -= bounds.wall_stiffness * min(x_minus_dist, 0.0);  // Left wall
    local_force.y += bounds.wall_stiffness * min(y_plus_dist, 0.0);   // Top wall
    local_force.y -= bounds.wall_stiffness * min(y_minus_dist, 0.0);  // Bottom wall
    local_force.z += bounds.wall_stiffness * min(z_plus_dist, 0.0);   // Front wall
    local_force.z -= bounds.wall_stiffness * min(z_minus_dist, 0.0);  // Back wall

    // Transform force back to world space (multiply by transpose = inverse rotation)
    // For rotation matrix R, transpose is [column0, column1, column2] as rows
    let world_force = vec3<f32>(
        rot_row0.x * local_force.x + rot_row1.x * local_force.y + rot_row2.x * local_force.z,
        rot_row0.y * local_force.x + rot_row1.y * local_force.y + rot_row2.y * local_force.z,
        rot_row0.z * local_force.x + rot_row1.z * local_force.y + rot_row2.z * local_force.z
    );

    accel += world_force;

    // Semi-implicit Euler integration
    vel = vel + params.dt * accel;

    // Apply velocity damping to dissipate energy (time-step aware)
    // Using pow(0.3, dt) gives ~70% velocity loss per second, which feels fluid-like
    let damping_per_second = 0.3;  // Retain 30% velocity per second
    let damping = pow(damping_per_second, params.dt);
    vel = vel * damping;

    pos = pos + params.dt * vel;

    particles[i].velocity = vel;
    particles[i].position = pos;
}
