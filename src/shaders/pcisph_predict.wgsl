// PCISPH Prediction — compute initial predicted state from non-pressure forces

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
    pcisph_delta: f32,
    xsph_epsilon: f32,
    _pad_st2: f32,
}

struct PredictedState {
    pred_pos_x: f32,
    pred_pos_y: f32,
    pred_pos_z: f32,
    pressure: f32,
    pred_vel_x: f32,
    pred_vel_y: f32,
    pred_vel_z: f32,
    pred_density: f32,
}

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read> particles: array<SphParticle3D>;
@group(0) @binding(2) var<storage, read_write> sorted_predicted: array<PredictedState>;
@group(0) @binding(3) var<storage, read> sorted_index: array<u32>;
@group(0) @binding(4) var<storage, read> prev_pressure: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_particles) {
        return;
    }

    let pos = particles[i].position;
    let vel = particles[i].velocity;
    let a_np = particles[i].force; // non-pressure acceleration from force shader

    // Semi-implicit Euler prediction
    let v_star = vel + params.dt * a_np;
    let x_star = pos + params.dt * v_star;

    // Write to sorted position for neighbor access in solve shader
    let si = sorted_index[i];
    sorted_predicted[si] = PredictedState(
        x_star.x, x_star.y, x_star.z,
        0.0, // pressure initialized to 0 each frame
        v_star.x, v_star.y, v_star.z,
        0.0, // predicted density (computed in solve)
    );
}
