// Screen-space fluid rendering — Composition / Water Shading
// Fullscreen pass that reads filtered depth, thickness, normals, and the
// pre-rendered opaque scene (environment + container + rigid body + spray,
// with depth), then applies the same PBR water shading as mc_render (mc_render/main.wgsl).
// Writes every pixel: water shading where the water surface is the nearest
// thing, the opaque scene elsewhere. Refraction samples the scene texture,
// so submerged objects are visible through the water.
// Outputs linear HDR — post-process pipeline handles tonemapping.

struct CameraParams {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    camera_pos: vec3<f32>,
    near: f32,
    far: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct WaterParams {
    water_color: vec3<f32>,
    roughness: f32,
    ior: f32,
    refraction_strength: f32,
    env_intensity: f32,
    use_env_background: u32,
    background_r: f32,
    background_g: f32,
    background_b: f32,
    time: f32,
    deep_color_r: f32,
    deep_color_g: f32,
    deep_color_b: f32,
    ripple_strength: f32,
    clarity: f32,
    debug_mode: f32,  // 0=off, 1=raw depth, 2=filtered depth, 3=normals, 4=thickness
    foam_coverage: f32,
    aeration_strength: f32,
    // 1 = physical water medium (absorption + single scattering), 0 = legacy
    physical_medium: f32,
    _pad_m0: f32,
    _pad_m1: f32,
    _pad_m2: f32,
}

struct LightParams {
    sun_direction: vec3<f32>,
    sun_enabled: u32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    _pad2: f32,
    _padding: vec3<f32>,
}

// Group 0: uniforms
@group(0) @binding(0) var<uniform> camera: CameraParams;
@group(0) @binding(1) var<uniform> water: WaterParams;
@group(0) @binding(2) var<uniform> light: LightParams;
@group(0) @binding(3) var<uniform> sh_coeffs: array<vec4<f32>, 9>;

// Group 1: screen-space textures
@group(1) @binding(0) var filtered_depth_tex: texture_2d<f32>;
@group(1) @binding(1) var filtered_thickness_tex: texture_2d<f32>;  // half resolution
@group(1) @binding(2) var normal_tex: texture_2d<f32>;
@group(1) @binding(3) var background_tex: texture_2d<f32>;          // opaque scene color
@group(1) @binding(4) var env_tex: texture_2d<f32>;
@group(1) @binding(5) var tex_sampler: sampler;
@group(1) @binding(6) var background_depth_tex: texture_depth_2d;   // opaque scene depth
// Half-res whitewater field splatted by the spray system (R = surface foam,
// G = aeration), same texture the MC composite reads
@group(1) @binding(7) var foam_density_tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

const PI: f32 = 3.14159265359;

// Prepended at module creation (screen_space_fluid.rs): water_common.wgsl
// (GGX BRDF, physical water medium, whitewater calibration: one copy for this
// shader and the MC water shader), noise_common.wgsl (hash2,
// value_noise_grad), sh_common.wgsl (evaluate_sh_irradiance).

// Fullscreen triangle (3 vertices cover entire screen)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
    // UV: [0,0] at top-left, [1,1] at bottom-right
    output.uv = vec2<f32>(
        (pos[vertex_index].x + 1.0) * 0.5,
        (1.0 - pos[vertex_index].y) * 0.5,
    );
    return output;
}

// Sample equirectangular environment map. Same convention as the CPU SH
// projection (compute_sh_irradiance in environment.rs): row 0 = +Y, and
// u = phi / 2pi for dir = (sin t cos phi, cos t, sin t sin phi). (The old
// mapping, v = 1 - t/pi with u offset by pi, sampled the antipode -dir.)
fn sample_environment(dir: vec3<f32>) -> vec3<f32> {
    let phi = atan2(dir.z, dir.x);
    let theta = acos(clamp(dir.y, -1.0, 1.0));
    let u = fract(phi / (2.0 * PI) + 1.0);
    let v = theta / PI;
    // Level 0 explicitly: the map carries mips for the MC renderer's filtered
    // lookups, and implicit derivatives of u jump at the +-pi seam
    return textureSampleLevel(env_tex, tex_sampler, vec2<f32>(u, v), 0.0).rgb;
}

// Reconstruct view-space position from UV and linear depth
fn uv_to_view(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = (1.0 - uv.y) * 2.0 - 1.0;
    let clip = vec4<f32>(ndc_x, ndc_y, 0.5, 1.0);
    let view_h = camera.inv_projection * clip;
    let view_dir = view_h.xyz / view_h.w;
    let t = -depth / view_dir.z;
    return view_dir * t;
}

struct FragOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(input: VertexOutput) -> FragOutput {
    let coord = vec2<i32>(input.position.xy);

    // Read screen-space buffers
    let normal_sample = textureLoad(normal_tex, coord, 0);
    let linear_depth = textureLoad(filtered_depth_tex, coord, 0).r;
    // Thickness buffer is half resolution — bilinear upsample
    let thickness = textureSampleLevel(filtered_thickness_tex, tex_sampler, input.uv, 0.0).r;
    // Opaque scene rendered before the water (env + container + rigid body + spray)
    let scene_color = textureSampleLevel(background_tex, tex_sampler, input.uv, 0.0).rgb;
    let scene_depth = textureLoad(background_depth_tex, coord, 0);

    // Water surface hardware depth (for occlusion against the opaque scene)
    let view_z = -linear_depth;
    let water_hw_depth = clamp(
        (camera.projection[2][2] * view_z + camera.projection[3][2]) / (-view_z),
        0.0, 1.0);

    let min_thickness = 0.001;
    // Water shades this pixel only if it exists here and no opaque object is in front
    let has_water = linear_depth > 0.0 && thickness >= min_thickness
        && water_hw_depth < scene_depth;

    // Debug visualization modes
    let debug = u32(water.debug_mode);
    if (debug > 0u) {
        var out: FragOutput;
        if (linear_depth <= 0.0) {
            out.color = vec4<f32>(scene_color, 1.0);
            out.depth = scene_depth;
            return out;
        }
        var dbg: vec3<f32>;
        if (debug == 1u) {
            // Filtered depth: grayscale normalized to [0,1] over typical range
            let d = clamp(linear_depth / 5.0, 0.0, 1.0);
            dbg = vec3<f32>(d, d, d);
        } else if (debug == 2u) {
            // Normals: view-space mapped to [0,1] RGB
            dbg = select(vec3<f32>(0.0), normal_sample.xyz * 0.5 + 0.5, normal_sample.w > 0.0);
        } else if (debug == 3u) {
            // Thickness: world units, red ramp (~1.5 units saturates)
            let t = clamp(thickness * 1.5, 0.0, 1.0);
            dbg = vec3<f32>(t, t * 0.3, 0.0);
        } else {
            // Coverage: where thickness (red) / normals (green) are valid
            let has_thick = select(0.0, 1.0, thickness > min_thickness);
            let has_normal = select(0.0, 1.0, normal_sample.w > 0.0);
            dbg = vec3<f32>(has_thick, has_normal, 0.0);
        }
        out.color = vec4<f32>(dbg, 1.0);
        out.depth = 0.5;
        return out;
    }

    // No water at this pixel, or the opaque scene is in front of the water
    // surface — output the scene as-is.
    if (!has_water) {
        var out: FragOutput;
        out.color = vec4<f32>(scene_color, 1.0);
        out.depth = scene_depth;
        return out;
    }

    // (Normals are valid wherever filtered depth > 0 — ss_normal.wgsl writes
    // w=1 for every covered pixel — so no separate boundary path is needed;
    // the thin-coverage fade at the end handles water-air edges.)

    // View-space normal (already normalized in ss_normal.wgsl)
    let view_normal = normal_sample.xyz;

    // Reconstruct world-space position from depth
    let view_pos = uv_to_view(input.uv, linear_depth);
    let world_pos = (camera.inv_view * vec4<f32>(view_pos, 1.0)).xyz;

    // World-space normal from view-space normal
    let normal = normalize((camera.inv_view * vec4<f32>(view_normal, 0.0)).xyz);

    let view_dir = normalize(camera.camera_pos - world_pos);

    // === ABSORPTION (Beer's Law) ===
    let absorption_coeffs = vec3<f32>(0.30, 0.08, 0.02);
    let optical_density = (1.0 - water.clarity) * 2.5 + 0.05;
    let clamped_thickness = min(thickness, 5.0);
    let transmittance = exp(-absorption_coeffs * optical_density * clamped_thickness);

    // Roughness floor: reconstructed normals carry residual per-pixel noise
    // that sharp GGX lobes turn into firefly glints; MC's mesh normals don't,
    // so the floor lives here rather than on the shared slider.
    let ss_roughness = max(water.roughness, 0.03);

    // === REFLECTION ===
    let reflect_dir = reflect(-view_dir, normal);
    let roughness_sq = ss_roughness * ss_roughness;
    var reflection_color: vec3<f32>;
    var below_horizon = 0.0;
    if (water.use_env_background == 0u) {
        reflection_color = vec3<f32>(water.background_r, water.background_g, water.background_b);
    } else {
        let sharp_env = sample_environment(reflect_dir) * water.env_intensity;
        let diffuse_env = evaluate_sh_irradiance(reflect_dir) * water.env_intensity;
        let env_reflection = mix(sharp_env, diffuse_env, roughness_sq);

        // Below-horizon reflections can't come from the sky map. Sphere-like SS
        // features (droplets, choppy bumps) reflect in every direction, and a hard
        // fade-to-black paints dark rims on all of them; the faded share is filled
        // with the water's own body color once it's known (below).
        let horizon_fade = smoothstep(-0.15, 0.1, reflect_dir.y);
        reflection_color = env_reflection * horizon_fade;
        below_horizon = 1.0 - horizon_fade;
    }

    // === SCREEN-SPACE REFRACTION ===
    // Classic SSF: offset by the view-space normal's screen projection. Unlike
    // the deviation-from-flat-up variant (kept in MC for calm pool surfaces),
    // this stays bounded on vertical faces and sphere-like droplets.
    let normal_view = normalize((camera.view * vec4<f32>(normal, 0.0)).xyz);

    // Thin water barely displaces what's behind it — ramp in over ~0.1 units
    // so droplets don't grab background samples from far across the screen.
    let thin_atten = clamp(clamped_thickness / 0.1, 0.0, 1.0);
    let refract_strength = water.refraction_strength * (1.0 + clamped_thickness * 0.5) * thin_atten;
    let uv_offset = normal_view.xy * refract_strength;

    let refract_uv = clamp(input.uv + uv_offset, vec2<f32>(0.001), vec2<f32>(0.999));
    var refracted_background = textureSampleLevel(background_tex, tex_sampler, refract_uv, 0.0).rgb;

    let refracted_scene = refracted_background;
    // Apply Beer-Lambert absorption
    refracted_background = refracted_background * transmittance;

    // Deep water color blend
    let deep_color = vec3<f32>(water.deep_color_r, water.deep_color_g, water.deep_color_b);
    let depth_blend = 1.0 - exp(-clamped_thickness * optical_density * 0.5);
    let water_interior = mix(refracted_background, deep_color, depth_blend);

    // Subsurface scattering approximation
    let scatter_strength = 0.12 * (1.0 - exp(-clamped_thickness * optical_density * 1.2));
    let scatter_color = water.water_color * scatter_strength;
    let interior_with_scatter = water_interior + scatter_color * (1.0 - transmittance);

    // === FRESNEL ===
    let cos_theta = max(0.0, dot(normal, view_dir));
    let F0 = pow((water.ior - 1.0) / (water.ior + 1.0), 2.0);
    var fresnel = clamp(F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0), 0.0, 1.0);

    // Total internal reflection
    let sin_theta_sq = 1.0 - cos_theta * cos_theta;
    let sin_refracted_sq = sin_theta_sq / (water.ior * water.ior);
    if (sin_refracted_sq > 1.0) {
        fresnel = 1.0;
    }

    // === DIRECTIONAL LIGHT (SUN) ===
    var sun_specular = vec3<f32>(0.0);
    var sun_subsurface = vec3<f32>(0.0);
    if (light.sun_enabled == 1u) {
        let light_dir = normalize(light.sun_direction);
        let NdotL = max(0.0, dot(normal, light_dir));
        let NdotV = max(dot(normal, view_dir), 0.001);

        let alpha = ss_roughness * ss_roughness;
        let half_vec = normalize(light_dir + view_dir);
        let NdotH = max(dot(normal, half_vec), 0.0);
        let HdotV = max(dot(half_vec, view_dir), 0.0);

        let D = D_GGX(NdotH, alpha);
        let G = G_Smith(NdotV, max(NdotL, 0.001), ss_roughness);
        let F_spec = F0 + (1.0 - F0) * pow(1.0 - HdotV, 5.0);

        let denom = 4.0 * NdotV * max(NdotL, 0.001);
        let specular_brdf = (D * G * F_spec) / max(denom, 0.001);

        sun_specular = light.sun_color * light.sun_intensity * specular_brdf * NdotL;
        // Firefly clamp: noisy reconstructed normals + sharp GGX produce
        // pinpoint glints that bloom into white sparkle noise.
        sun_specular = min(sun_specular, vec3<f32>(3.0));

        // Body light sees the mean (flat) surface, not the facet — see mc_render/main.wgsl
        let light_entering = max(light_dir.y, 0.0) * (1.0 - F_spec);
        let interior_glow = water.water_color * transmittance;
        sun_subsurface = interior_glow * light_entering * light.sun_color * light.sun_intensity * 0.18;

        let VdotL = max(0.0, dot(-view_dir, light_dir));
        let forward_scatter = pow(VdotL, 4.0) * exp(-clamped_thickness * optical_density * 1.5);
        sun_subsurface += water.water_color * forward_scatter * light.sun_color * light.sun_intensity * 0.10;
    }

    // IBL diffuse irradiance
    let ambient_irradiance = evaluate_sh_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * water.env_intensity;
    let ambient_subsurface = ambient_irradiance * water.water_color * transmittance * scatter_strength * 0.6;

    var lit_interior = interior_with_scatter
        + sun_subsurface * (1.0 - fresnel)
        + ambient_subsurface * (1.0 - fresnel);
    if (water.physical_medium != 0.0) {
        // Splatted thickness is already a true path length in metres
        var sun_rgb = vec3<f32>(0.0);
        if (light.sun_enabled == 1u) {
            sun_rgb = light.sun_color * light.sun_intensity;
        }
        let medium = water_medium(
            clamped_thickness,
            refract(-view_dir, normal, 1.0 / water.ior),
            normalize(light.sun_direction),
            sun_rgb,
            evaluate_sh_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * water.env_intensity,
        );
        lit_interior = refracted_scene * medium.transmittance + medium.inscatter;
    }

    // === AERATION (submerged whitewater) ===
    // G channel of the whitewater field: milkiness INSIDE the water, mixed
    // before the Fresnel combine so reflections survive on top. Same mapping
    // as mc_render/main.wgsl.
    let whitewater_field = textureSampleLevel(foam_density_tex, tex_sampler, input.uv, 0.0).rg;
    let aeration = 1.0 - exp(-AERATION_K * water.aeration_strength * whitewater_field.g);
    if (aeration > 0.002) {
        let sun_dir = normalize(light.sun_direction);
        var aeration_light = evaluate_sh_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * water.env_intensity;
        if (light.sun_enabled == 1u) {
            aeration_light += light.sun_color * light.sun_intensity
                * max(sun_dir.y, 0.0) * 0.6;
        }
        lit_interior = mix(lit_interior, AERATION_ALBEDO * aeration_light, aeration);
    }

    // Below-horizon reflection rays mostly hit more water (see mc_render/main.wgsl)
    reflection_color += lit_interior * below_horizon;

    var color = mix(lit_interior, reflection_color, fresnel);
    color += sun_specular;

    // === FOAM OVERLAY ===
    // R channel: surface whitening after the Fresnel combine (foam is rough
    // and diffuse — it replaces the specular water response). Same mapping as
    // mc_render/main.wgsl.
    let foam_density = whitewater_field.r;
    if (foam_density > 0.01) {
        let n_coarse = value_noise_grad(world_pos.xz * FOAM_NOISE_SCALE).x;
        let n_fine = value_noise_grad(
            world_pos.xz * FOAM_NOISE_SCALE_FINE + vec2<f32>(37.42, 11.18),
        ).x;
        let breakup = (n_coarse - 0.5) * FOAM_NOISE_BREAKUP;
        let d_eff = max(foam_density * (1.0 + breakup) - FOAM_DENSITY_LO, 0.0);
        let coverage = 1.0 - exp(-FOAM_COVERAGE_K * water.foam_coverage * d_eff);
        if (coverage > 0.002) {
            var foam_light = evaluate_sh_irradiance(normal) * water.env_intensity;
            if (light.sun_enabled == 1u) {
                foam_light += light.sun_color * light.sun_intensity
                    * max(dot(normal, normalize(light.sun_direction)), 0.0);
            }
            let thick = smoothstep(FOAM_THICK_LO, FOAM_THICK_HI, coverage);
            let albedo = mix(FOAM_VEIL_ALBEDO, FOAM_ALBEDO, thick);
            let tex = 1.0 + (n_coarse - 0.5) * FOAM_TEX_CONTRAST
                + (n_fine - 0.5) * FOAM_TEX_CONTRAST_FINE;
            let foam_color = albedo * tex * foam_light;
            color = mix(color, foam_color, coverage);
        }
    }

    // Thin-coverage fade: the splat fringe (sub-particle thickness) carries
    // bead-shaped normals that read as scalloped silhouettes and milky halos.
    // Fading toward the scene over the thin fringe softens contact lines and
    // turns isolated droplets into translucent beads instead of opaque spheres.
    let edge_fade = smoothstep(min_thickness, 0.02, thickness);
    color = mix(scene_color, color, edge_fade);

    // Output linear HDR — post-process handles tonemapping + gamma
    var output: FragOutput;
    output.color = vec4<f32>(color, 1.0);
    output.depth = water_hw_depth;
    return output;
}
