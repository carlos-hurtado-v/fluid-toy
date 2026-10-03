// Value noise shared by the water shaders (ripples, foam texture) and the
// caustics light raster (its own ripple): prepended at module creation.
// Needs nothing from the including shader.

// GPU-friendly hash -> pseudo-random [0,1]
fn hash2(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smooth value noise with analytic gradient (returns: vec3(noise, dN/dx, dN/dz))
fn value_noise_grad(p: vec2<f32>) -> vec3<f32> {
    let i = floor(p);
    let f = fract(p);
    // Quintic Hermite interpolation (C2 continuous — no grid artifacts)
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    let a = hash2(i + vec2<f32>(0.0, 0.0));
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));

    let val = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    let dx = du.x * ((b - a) + (a - b - c + d) * u.y);
    let dy = du.y * ((c - a) + (a - b - c + d) * u.x);
    return vec3<f32>(val, dx, dy);
}
