// Screen-Space Fluid - Curvature Flow Smoothing
// Based on "Screen Space Fluid Rendering with Curvature Flow" (van der Laan, Green, Sainz - NVIDIA)
// This smooths the depth surface by flowing it in the direction that reduces mean curvature

// Reuse same struct layout as blur params for compatibility
struct FlowParams {
    _unused_blur_dir: vec2<f32>,  // Not used by curvature flow
    dt: f32,                      // depth_threshold field repurposed as time step
    _unused1: f32,
    _unused2: f32,
    _padding: vec3<f32>,
}

@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: FlowParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-screen triangle
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );

    let pos = positions[vertex_index];

    var output: VertexOutput;
    output.position = vec4<f32>(pos, 0.0, 1.0);
    output.uv = pos * 0.5 + 0.5;
    return output;
}

// Sample depth at offset (handles boundaries)
fn sample_depth(iuv: vec2<i32>, offset: vec2<i32>, tex_size: vec2<i32>) -> f32 {
    let coord = clamp(iuv + offset, vec2<i32>(0), tex_size - 1);
    return abs(textureLoad(depth_tex, coord, 0).r);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<i32>(textureDimensions(depth_tex));
    let iuv = vec2<i32>(input.uv * vec2<f32>(tex_size));

    let z = abs(textureLoad(depth_tex, iuv, 0).r);

    // Background check - don't process background pixels
    if (z <= 0.0 || z >= 1e4) {
        return vec4<f32>(z, 0.0, 0.0, 1.0);
    }

    // Sample neighboring depths for finite differences
    let z_xp = sample_depth(iuv, vec2<i32>(1, 0), tex_size);   // z(x+1, y)
    let z_xm = sample_depth(iuv, vec2<i32>(-1, 0), tex_size);  // z(x-1, y)
    let z_yp = sample_depth(iuv, vec2<i32>(0, 1), tex_size);   // z(x, y+1)
    let z_ym = sample_depth(iuv, vec2<i32>(0, -1), tex_size);  // z(x, y-1)

    // Corner samples for mixed derivative
    let z_xpyp = sample_depth(iuv, vec2<i32>(1, 1), tex_size);   // z(x+1, y+1)
    let z_xmym = sample_depth(iuv, vec2<i32>(-1, -1), tex_size); // z(x-1, y-1)
    let z_xpym = sample_depth(iuv, vec2<i32>(1, -1), tex_size);  // z(x+1, y-1)
    let z_xmyp = sample_depth(iuv, vec2<i32>(-1, 1), tex_size);  // z(x-1, y+1)

    // Handle background neighbors - use one-sided differences or skip that direction
    let xp_valid = z_xp > 0.0 && z_xp < 1e4;
    let xm_valid = z_xm > 0.0 && z_xm < 1e4;
    let yp_valid = z_yp > 0.0 && z_yp < 1e4;
    let ym_valid = z_ym > 0.0 && z_ym < 1e4;

    // Use valid neighbors, or current depth if neighbor is background
    let z_xp_safe = select(z, z_xp, xp_valid);
    let z_xm_safe = select(z, z_xm, xm_valid);
    let z_yp_safe = select(z, z_yp, yp_valid);
    let z_ym_safe = select(z, z_ym, ym_valid);

    // First derivatives (using safe values)
    let z_x = (z_xp_safe - z_xm_safe) * 0.5;
    let z_y = (z_yp_safe - z_ym_safe) * 0.5;

    // Second derivatives
    let z_xx = z_xp_safe - 2.0 * z + z_xm_safe;
    let z_yy = z_yp_safe - 2.0 * z + z_ym_safe;

    // For mixed derivative, use 0 if corners are invalid
    let corners_valid = xp_valid && xm_valid && yp_valid && ym_valid;
    let z_xy = select(0.0, (z_xpyp - z_xmyp - z_xpym + z_xmym) * 0.25, corners_valid);

    // Mean curvature: H = (z_xx*(1+z_y²) - 2*z_x*z_y*z_xy + z_yy*(1+z_x²)) / (2*(1+z_x²+z_y²)^(3/2))
    let z_x2 = z_x * z_x;
    let z_y2 = z_y * z_y;
    let denom = pow(1.0 + z_x2 + z_y2, 1.5);

    // Avoid division by zero
    let H = (z_xx * (1.0 + z_y2) - 2.0 * z_x * z_y * z_xy + z_yy * (1.0 + z_x2)) / (2.0 * max(denom, 0.0001));

    // Curvature flow: z_new = z + dt * H * |grad(z)|
    // The gradient magnitude term makes flow faster on steep parts
    let grad_mag = sqrt(1.0 + z_x2 + z_y2);
    let z_new = z + params.dt * H * grad_mag;

    return vec4<f32>(z_new, 0.0, 0.0, 1.0);
}
