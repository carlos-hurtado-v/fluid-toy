// SPH 3D Integration Shader
// Soft wall penalty forces + hard boundary backstop

struct SphParticle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    force: vec3<f32>,
    density: f32,
    near_density: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
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
    surface_tension: f32,
    _pad_st0: f32,
    _pad_st1: f32,
    _pad_st2: f32,
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
    mode: u32,
    _pad: f32,
    direction: vec3<f32>,
    _pad2: f32,
}

const FORCE_PUSH: u32 = 0u;
const FORCE_PULL: u32 = 1u;
const FORCE_VORTEX: u32 = 2u;
const FORCE_EXPLODE: u32 = 3u;
const FORCE_DRAIN: u32 = 4u;

const SHAPE_CUBE: u32 = 0u;
const SHAPE_SPHERE: u32 = 1u;
const SHAPE_CYLINDER: u32 = 2u;
const SHAPE_TORUS: u32 = 3u;
const SHAPE_CUSTOM: u32 = 4u;

struct RigidBody {
    position: vec3<f32>,
    half_extent: f32,
    velocity: vec3<f32>,
    is_active: u32,
    stiffness: f32,
    shape: u32,
    _pad1: f32,
    _pad2: f32,
    rot_row0: vec4<f32>,
    rot_row1: vec4<f32>,
    rot_row2: vec4<f32>,
}

struct RigidBodyAccum {
    force_x: atomic<i32>,
    force_y: atomic<i32>,
    force_z: atomic<i32>,
    contact_count: atomic<u32>,
    torque_x: atomic<i32>,
    torque_y: atomic<i32>,
    torque_z: atomic<i32>,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read_write> particles: array<SphParticle3D>;
@group(0) @binding(2) var<uniform> bounds: BoundsParams;
@group(0) @binding(3) var<uniform> mouse_force: MouseForce;
@group(0) @binding(4) var<uniform> rigid_body: RigidBody;
@group(0) @binding(5) var<storage, read_write> body_accum: RigidBodyAccum;
@group(0) @binding(6) var sdf_texture: texture_3d<f32>;
@group(0) @binding(7) var sdf_sampler: sampler;

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
            let falloff = 1.0 - (dist / mouse_force.radius);
            let force_dir = normalize(to_mouse);
            let s = mouse_force.strength * falloff;

            switch mouse_force.mode {
                case FORCE_PUSH, default: {
                    // Repel from cursor
                    accel -= force_dir * s;
                }
                case FORCE_PULL: {
                    // Attract toward cursor
                    accel += force_dir * s;
                }
                case FORCE_VORTEX: {
                    // Tangential swirl around cursor ray direction
                    let axis = normalize(mouse_force.direction);
                    let tangent = cross(force_dir, axis);
                    let tlen = length(tangent);
                    if (tlen > 0.001) {
                        accel += (tangent / tlen) * s;
                    }
                }
                case FORCE_EXPLODE: {
                    // One-shot burst outward (same direction as push, CPU handles one-shot)
                    accel -= force_dir * s * 3.0;
                }
                case FORCE_DRAIN: {
                    // Pull inward + downward (funnel)
                    let drain_dir = normalize(force_dir + vec3<f32>(0.0, -1.0, 0.0));
                    accel += drain_dir * s;
                }
            }
        }
    }

    // Rigid body penalty forces (per-shape SDF)
    if (rigid_body.is_active != 0u) {
        // Transform particle position to body-local frame
        let world_rel = pos - rigid_body.position;
        let rb_local = vec3<f32>(
            dot(rigid_body.rot_row0.xyz, world_rel),
            dot(rigid_body.rot_row1.xyz, world_rel),
            dot(rigid_body.rot_row2.xyz, world_rel),
        );

        let he = rigid_body.half_extent;
        var sdf: f32;
        var local_normal = vec3<f32>(0.0);

        switch (rigid_body.shape) {
            case SHAPE_SPHERE: {
                // Sphere SDF: distance from origin minus radius
                let dist = length(rb_local);
                sdf = dist - he;
                if (dist > 0.001) {
                    local_normal = rb_local / dist;
                } else {
                    local_normal = vec3<f32>(0.0, 1.0, 0.0);
                }
            }
            case SHAPE_CYLINDER: {
                // Capped cylinder: radius=he, height=2*he (y-axis)
                let radial_dist = length(rb_local.xz);
                let d_radial = radial_dist - he;
                let d_cap = abs(rb_local.y) - he;
                sdf = max(d_radial, d_cap);

                if (d_radial > d_cap) {
                    // Closest to barrel
                    if (radial_dist > 0.001) {
                        local_normal = vec3<f32>(rb_local.x / radial_dist, 0.0, rb_local.z / radial_dist);
                    } else {
                        local_normal = vec3<f32>(1.0, 0.0, 0.0);
                    }
                } else {
                    // Closest to cap
                    local_normal = vec3<f32>(0.0, sign(rb_local.y), 0.0);
                }
            }
            case SHAPE_TORUS: {
                // Torus: major_radius=he, minor_radius=0.3*he
                let minor_r = he * 0.3;
                let xz_len = length(rb_local.xz);
                let q = vec2<f32>(xz_len - he, rb_local.y);
                sdf = length(q) - minor_r;

                // Normal: direction from nearest point on ring to particle
                let q_len = length(q);
                if (q_len > 0.001 && xz_len > 0.001) {
                    let ring_dir = vec2<f32>(q.x, q.y) / q_len;
                    local_normal = vec3<f32>(
                        ring_dir.x * rb_local.x / xz_len,
                        ring_dir.y,
                        ring_dir.x * rb_local.z / xz_len,
                    );
                } else {
                    local_normal = vec3<f32>(0.0, 1.0, 0.0);
                }
            }
            case SHAPE_CUSTOM: {
                // Voxelized SDF from 3D texture
                let normalized = rb_local / he;
                let uvw = normalized * 0.5 + 0.5; // [-1,1] → [0,1]
                let raw_sdf = textureSampleLevel(sdf_texture, sdf_sampler, uvw, 0.0).r;
                sdf = raw_sdf * he; // Scale to world space

                // Normal via central differences
                let eps = 1.0 / 32.0; // One voxel step in texture space
                let dx = textureSampleLevel(sdf_texture, sdf_sampler, uvw + vec3<f32>(eps, 0.0, 0.0), 0.0).r
                       - textureSampleLevel(sdf_texture, sdf_sampler, uvw - vec3<f32>(eps, 0.0, 0.0), 0.0).r;
                let dy = textureSampleLevel(sdf_texture, sdf_sampler, uvw + vec3<f32>(0.0, eps, 0.0), 0.0).r
                       - textureSampleLevel(sdf_texture, sdf_sampler, uvw - vec3<f32>(0.0, eps, 0.0), 0.0).r;
                let dz = textureSampleLevel(sdf_texture, sdf_sampler, uvw + vec3<f32>(0.0, 0.0, eps), 0.0).r
                       - textureSampleLevel(sdf_texture, sdf_sampler, uvw - vec3<f32>(0.0, 0.0, eps), 0.0).r;
                let grad = vec3<f32>(dx, dy, dz);
                let grad_len = length(grad);
                if (grad_len > 1e-6) {
                    local_normal = grad / grad_len;
                } else {
                    local_normal = normalize(rb_local + vec3<f32>(0.0, 1e-6, 0.0));
                }
            }
            default: {
                // Cube SDF: axis-aligned box
                let d = abs(rb_local) - vec3<f32>(he);
                sdf = max(d.x, max(d.y, d.z));
                if (d.x >= d.y && d.x >= d.z) {
                    local_normal = vec3<f32>(sign(rb_local.x), 0.0, 0.0);
                } else if (d.y >= d.x && d.y >= d.z) {
                    local_normal = vec3<f32>(0.0, sign(rb_local.y), 0.0);
                } else {
                    local_normal = vec3<f32>(0.0, 0.0, sign(rb_local.z));
                }
            }
        }

        let interact_range = params.kernel_radius * 0.7;

        if (sdf < interact_range) {
            // Transform normal from local to world (transpose multiply)
            let normal = vec3<f32>(
                rigid_body.rot_row0.x * local_normal.x + rigid_body.rot_row1.x * local_normal.y + rigid_body.rot_row2.x * local_normal.z,
                rigid_body.rot_row0.y * local_normal.x + rigid_body.rot_row1.y * local_normal.y + rigid_body.rot_row2.y * local_normal.z,
                rigid_body.rot_row0.z * local_normal.x + rigid_body.rot_row1.z * local_normal.y + rigid_body.rot_row2.z * local_normal.z,
            );

            // Quadratic penalty ramp
            let penetration = interact_range - sdf;
            let t = penetration / interact_range;
            let penalty = normal * rigid_body.stiffness * t * t;
            accel += penalty;

            // Velocity-dependent damping
            var damping_accel = vec3<f32>(0.0);
            let rel_vel = vel - rigid_body.velocity;
            let vn = dot(rel_vel, normal);
            if (vn < 0.0) {
                damping_accel = -normal * vn * 5.0;
                accel += damping_accel;
            }

            // Accumulate reaction force and torque (Newton's 3rd law)
            let reaction = -(penalty + damping_accel) * params.mass;
            atomicAdd(&body_accum.force_x, i32(reaction.x * 1000.0));
            atomicAdd(&body_accum.force_y, i32(reaction.y * 1000.0));
            atomicAdd(&body_accum.force_z, i32(reaction.z * 1000.0));
            atomicAdd(&body_accum.contact_count, 1u);

            // Torque: cross(r, F) where r = particle_pos - body_center
            let torque = cross(world_rel, reaction);
            atomicAdd(&body_accum.torque_x, i32(torque.x * 1000.0));
            atomicAdd(&body_accum.torque_y, i32(torque.y * 1000.0));
            atomicAdd(&body_accum.torque_z, i32(torque.z * 1000.0));
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
