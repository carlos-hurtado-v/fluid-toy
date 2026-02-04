// Particle rendering shader - state-driven configuration

struct RenderParams {
    particle_radius: f32,
    color_by_velocity: u32,
    _padding1: vec2<u32>,
    particle_color: vec4<f32>,
    background_color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) velocity: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: RenderParams;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) particle_pos: vec2<f32>,
    @location(1) particle_vel: vec2<f32>,
) -> VertexOutput {
    // Generate quad vertices (2 triangles, 6 vertices)
    var quad_verts = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );

    let local_pos = quad_verts[vertex_index];
    let world_pos = particle_pos + local_pos * params.particle_radius;

    var output: VertexOutput;
    output.position = vec4<f32>(world_pos, 0.0, 1.0);
    output.uv = local_pos * 0.5 + 0.5;
    output.velocity = particle_vel;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Draw circle within quad
    let dist = length(input.uv - vec2<f32>(0.5, 0.5));
    if (dist > 0.5) {
        discard;
    }

    // Base color from params
    var color = params.particle_color.rgb;

    // Optionally modify by velocity
    if (params.color_by_velocity != 0u) {
        let speed = length(input.velocity);
        let t = clamp(speed * 2.0, 0.0, 1.0);
        // Blend toward brighter color based on speed
        color = mix(color, vec3<f32>(0.3, 0.8, 1.0), t);
    }

    // Soft edge
    let alpha = smoothstep(0.5, 0.4, dist);

    return vec4<f32>(color, alpha);
}
