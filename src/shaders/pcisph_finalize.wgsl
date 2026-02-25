// PCISPH Finalize — write corrected velocity back to particle buffer

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
@group(0) @binding(1) var<storage, read_write> particles: array<SphParticle3D>;
@group(0) @binding(2) var<storage, read> sorted_predicted: array<PredictedState>;
@group(0) @binding(3) var<storage, read> sorted_index: array<u32>;
@group(0) @binding(4) var<storage, read_write> pressure_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_particles) {
        return;
    }

    let si = sorted_index[i];
    let final_pred = sorted_predicted[si];

    let corrected_vel = vec3<f32>(
        final_pred.pred_vel_x,
        final_pred.pred_vel_y,
        final_pred.pred_vel_z,
    );

    // Store velocity delta for spray emission to read.
    // NOT divided by dt — keeps the threshold dt-independent.
    let original_vel = particles[i].velocity;
    particles[i].force = corrected_vel - original_vel;

    // Write corrected velocity (includes both non-pressure and pressure acceleration)
    particles[i].velocity = corrected_vel;

}
