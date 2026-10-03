// Shared by every shader with SH ambient lighting (the water shaders,
// container, rigid bodies, spray): prepended at module creation.
//
// Needs from the including shader: the uniform
// `sh_coeffs: array<vec4<f32>, 9>` (GpuShCoefficients).

// Evaluate order-2 spherical harmonics irradiance
// Coefficients are pre-convolved with cosine lobe on CPU
fn evaluate_sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    // Band 0 (constant)
    var irradiance = sh_coeffs[0].rgb * 0.282095;
    // Band 1 (linear)
    irradiance += sh_coeffs[1].rgb * 0.488603 * n.y;
    irradiance += sh_coeffs[2].rgb * 0.488603 * n.z;
    irradiance += sh_coeffs[3].rgb * 0.488603 * n.x;
    // Band 2 (quadratic)
    irradiance += sh_coeffs[4].rgb * 1.092548 * n.x * n.y;
    irradiance += sh_coeffs[5].rgb * 1.092548 * n.y * n.z;
    irradiance += sh_coeffs[6].rgb * 0.315392 * (3.0 * n.z * n.z - 1.0);
    irradiance += sh_coeffs[7].rgb * 1.092548 * n.x * n.z;
    irradiance += sh_coeffs[8].rgb * 0.546274 * (n.x * n.x - n.y * n.y);
    return max(irradiance, vec3<f32>(0.0));
}
