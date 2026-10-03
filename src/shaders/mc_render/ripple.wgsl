// Water shader (mc_render), part: procedural micro-ripple normal

// Multi-octave noise normal perturbation with analytic derivatives.
// Each octave doubles frequency and halves amplitude (fBm).
fn ripple_normal(world_pos: vec3<f32>, t: f32) -> vec3<f32> {
    var grad = vec2<f32>(0.0);
    var amp = 1.0;
    var freq = 10.0;

    // 4 octaves at different time offsets to avoid coherent drift
    for (var oct = 0u; oct < 4u; oct++) {
        let time_offset = t * (0.3 + f32(oct) * 0.15);
        // Rotate sample coords per octave to break axis alignment
        let angle = f32(oct) * 1.8;
        let cs = cos(angle);
        let sn = sin(angle);
        let p = vec2<f32>(
            world_pos.x * cs - world_pos.z * sn,
            world_pos.x * sn + world_pos.z * cs,
        );
        let n = value_noise_grad(p * freq + vec2<f32>(time_offset, -time_offset * 0.7));
        // Rotate gradient back to world XZ
        grad += amp * vec2<f32>(
            n.y * cs + n.z * sn,
            -n.y * sn + n.z * cs,
        );
        freq *= 2.0;
        amp *= 0.5;
    }

    return vec3<f32>(grad.x, 0.0, grad.y);
}
