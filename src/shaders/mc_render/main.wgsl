// Water shader (mc_render), part: vertex + fragment entry points

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let vertex = vertices[vertex_index];

    var output: VertexOutput;
    output.world_position = vertex.position;
    output.world_normal = vertex.normal;

    let world_pos = vec4<f32>(vertex.position, 1.0);
    let view_pos = camera.view * world_pos;
    output.clip_position = camera.projection * view_pos;

    return output;
}

// What a reflected ray (origin on the water surface, unit dir) meets before
// the environment: rgb + confidence.
// A rigid body is met exactly and shaded at the hit, as in the refraction
// (bodies.wgsl): confidence 1. The screen-space march knows a body by its
// camera-facing skin only: it showed that side where the reflected ray meets
// an underside, and its coarse steps hatched the image (the water inside a
// torus's hole).
// Everything else is the screen-space reflection (the ground, pool walls;
// also its false hits on rays that pass behind a body's outline), weaker on
// rough water: a sharp reflection looks wrong there.
fn near_reflection(screen_uv: vec2<f32>, origin: vec3<f32>, dir: vec3<f32>, roughness_sq: f32) -> vec4<f32> {
    if (water.body_count > 0u) {
        let body = ray_body_hit(origin, dir, BODY_MAX_REACH);
        if (body.t > 0.0) {
            return vec4<f32>(body_shade(origin, dir, body), 1.0);
        }
    }
    let ssr_dims = vec2<f32>(textureDimensions(ssr_tex));
    let ssr_sample = textureLoad(ssr_tex, vec2<i32>(screen_uv * ssr_dims), 0);
    return vec4<f32>(ssr_sample.rgb, ssr_sample.a * (1.0 - roughness_sq));
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Clip to container bounds with margin (MC interpolation can place vertices
    // slightly outside the container; clip_margin ≈ 1.5× MC cell_size).
    if (container.clip_enabled != 0u) {
        let local = world_to_local(container, input.world_position);
        if (!is_inside_box(container, local, container.clip_margin)) {
            discard;
        }
    }

    probe_begin(input.clip_position.xy, input.clip_position.z);
    let view_dir = normalize(camera.camera_pos - input.world_position);

    // Get the surface normal - ensure it faces toward the camera
    var normal = normalize(input.world_normal);
    // If normal points away from camera, flip it (ensures correct reflection)
    if (dot(normal, view_dir) < 0.0) {
        normal = -normal;
    }
    probe_event(PRB_FRAG, input.world_position, normal, input.clip_position.z);

    let local_pos = world_to_local(container, input.world_position);
    // Pixel footprint on the surface (m), for the foam lace's antialiasing.
    // Taken here, in uniform control flow, where derivatives are valid.
    let foam_px = max(length(dpdx(local_pos.xz)), length(dpdy(local_pos.xz)));

    // Water against a wireframe container wall is flat like water against
    // glass: use the wall plane, not the MC bulge (physical refraction only,
    // so the legacy path stays as it was)
    var on_wall = false;
    if (water.physical_refraction != 0.0) {
        let wall = wall_plane(local_pos, world_dir_to_local(container, normal));
        if (wall.w > 0.5) {
            normal = local_dir_to_world(container, wall.xyz);
            on_wall = true;
        }
    }

    // Mesh normal (camera-facing) before ripples: the foam map's top test
    let surface_up = world_dir_to_local(container, normal).y;

    // Micro-ripple perturbation: adds small-scale surface detail the MC mesh
    // can't capture (a free-surface effect: not on water held flat by a wall)
    if (!on_wall) {
        let ripple_grad = ripple_normal(input.world_position, water.time);
        normal = normalize(normal + ripple_grad * water.ripple_strength);
    }

    // === THICKNESS CALCULATION ===
    // Sample back face depth at this screen position
    let screen_size = vec2<f32>(textureDimensions(back_depth_tex));
    let screen_uv = input.clip_position.xy / screen_size;
    let back_depth_raw = textureSample(back_depth_tex, depth_sampler, screen_uv);
    let front_depth_raw = input.clip_position.z;
    probe_event(PRB_NORMAL, normal, vec3<f32>(f32(on_wall), water.physical_refraction, 0.0), back_depth_raw);

    // Convert to linear depth using actual camera near/far planes
    let front_linear = linearize_depth(front_depth_raw, camera.near, camera.far);
    let back_linear = linearize_depth(back_depth_raw, camera.near, camera.far);

    // Thickness in world units (clamped to reasonable range). Legacy: this
    // D3D-style linearization runs on a GL-style projection, so it reads
    // ~0.5x the true distance; the legacy look was tuned on it, so it stays.
    var thickness = max(0.0, back_linear - front_linear);
    thickness = min(thickness, 5.0);  // Cap at 5 units

    // True in-water path for the physical medium: to the back face, or to an
    // opaque surface inside the water (floor, wall, body) if that is nearer
    let path_end = min(back_depth_raw, background_depth_at(screen_uv));
    var path_length = 10.0;  // nothing behind: treat as deep
    if (path_end < BACKDROP_DEPTH) {
        path_length = min(distance(input.world_position, screen_to_world(screen_uv, path_end)), 10.0);
    }

    // === ABSORPTION (Beer's Law) ===
    // Light attenuates exponentially through water
    // Different wavelengths absorb at different rates (red absorbs fastest)
    // Clarity controls optical density: 0 = murky (dense), 1 = crystal clear (sparse)
    let absorption_coeffs = vec3<f32>(0.30, 0.08, 0.02);  // RGB absorption rates
    let optical_density = (1.0 - water.clarity) * 2.5 + 0.05;
    let transmittance = exp(-absorption_coeffs * optical_density * thickness);

    // Reflection — solid color or HDR environment
    let reflect_dir = reflect(-view_dir, normal);
    let roughness_sq = water.roughness * water.roughness;
    var reflection_color: vec3<f32>;
    var below_horizon = 0.0;
    if (water.use_env_background == 0u) {
        reflection_color = vec3<f32>(water.background_r, water.background_g, water.background_b);
    } else {
        // Roughness-blurred environment reflection:
        // Sharp env sample at roughness=0 (mirror), SH irradiance at roughness=1 (fully diffuse).
        // Squared roughness maps perceptual roughness to GGX lobe width more naturally.
        let sharp_env = sample_environment(reflect_dir) * water.env_intensity;
        let diffuse_env = evaluate_sh_irradiance(reflect_dir) * water.env_intensity;
        var env_reflection = mix(sharp_env, diffuse_env, roughness_sq);

        // Fade env reflection when reflect direction points below horizon.
        // The env map is a distant panorama — it can't represent nearby scene
        // geometry (walls, floor, other water). Downward reflections would show
        // the far-off ground of the HDRI instead.
        // SSR handles these directions; without SSR, the faded share is filled
        // with the water's own body color once it's known (below).
        let horizon_fade = smoothstep(-0.15, 0.1, reflect_dir.y);
        reflection_color = env_reflection * horizon_fade;
        below_horizon = 1.0 - horizon_fade;
    }

    // What the reflected ray meets nearby (a rigid body, the screen-space
    // reflection) over the environment, by its confidence
    let near = near_reflection(screen_uv, input.world_position, reflect_dir, roughness_sq);
    let ssr_confidence = near.w;
    reflection_color = mix(reflection_color, near.rgb, near.w);

    // === SCREEN-SPACE REFRACTION ===
    var refracted_background: vec3<f32>;
    if (water.physical_refraction != 0.0) {
        refracted_background = refract_scene(
            input.world_position, normal, view_dir, screen_uv, front_depth_raw, back_depth_raw,
        );
    } else {
        // Legacy: offset by the normal's deviation from flat up (a flat surface
        // shows no shift at all), scaled by the Refraction slider
        let flat_normal_view = normalize((camera.view * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz);
        let normal_view = normalize((camera.view * vec4<f32>(normal, 0.0)).xyz);
        let normal_deviation = normal_view - flat_normal_view;

        let refract_strength = water.refraction_strength * (1.0 + thickness * 0.5);
        let uv_offset = normal_deviation.xy * refract_strength;

        // Sample background with distorted UVs (clamp to avoid sampling outside)
        let refract_uv = clamp(screen_uv + uv_offset, vec2<f32>(0.001), vec2<f32>(0.999));
        refracted_background = textureSampleLevel(background_tex, env_sampler, refract_uv, 0.0).rgb;
        dbg_path = DBG_PATH_LEGACY;
        dbg_end = DBG_END_SURFACE;
        dbg_uv = refract_uv;
    }
    if (water.filtered_lookup != 0u) {
        refracted_background = resolve_lookup(refracted_background);
    }
    if (refracted_path >= 0.0) {
        path_length = min(refracted_path, 10.0);
    }

    let refracted_scene = refracted_background;
    // Apply absorption to refracted light (Beer-Lambert)
    refracted_background = refracted_background * transmittance;

    // Deep water color (what you see when looking deep)
    let deep_color = vec3<f32>(water.deep_color_r, water.deep_color_g, water.deep_color_b);

    // Blend between refracted background and deep water based on thickness
    // Clarity scales the depth blend rate — clearer water shows background longer
    let depth_blend = 1.0 - exp(-thickness * optical_density * 0.5);
    let water_interior = mix(refracted_background, deep_color, depth_blend);

    // Add water's own color contribution (subsurface scattering approximation)
    let scatter_strength = 0.12 * (1.0 - exp(-thickness * optical_density * 1.2));
    let scatter_color = water.water_color * scatter_strength;
    let interior_with_scatter = water_interior + scatter_color * (1.0 - transmittance);

    // Fresnel (Schlick approximation) - controls reflection vs transmission
    // F0 = ((n1 - n2) / (n1 + n2))^2 where n1=1.0 (air), n2=IOR (water)
    let cos_theta = max(0.0, dot(normal, view_dir));
    let F0 = pow((water.ior - 1.0) / (water.ior + 1.0), 2.0);  // ~0.02 for water
    var fresnel = clamp(F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0), 0.0, 1.0);

    // Total internal reflection — at extreme grazing angles, all light reflects
    let sin_theta_sq = 1.0 - cos_theta * cos_theta;
    let sin_refracted_sq = sin_theta_sq / (water.ior * water.ior);
    if (sin_refracted_sq > 1.0) {
        fresnel = 1.0;
    }

    // === DIRECTIONAL LIGHT (SUN) ===
    // Analytic rim shadow: pool walls block direct sun on the water surface,
    // matching the floor's rim shadowing and the caustics light raster (which
    // draws the container as an occluder). 1.0 outside pool mode.
    let sun_dir_ws = normalize(light.sun_direction);
    let rim_vis = rim_visibility(
        container,
        world_to_local(container, input.world_position),
        world_dir_to_local(container, sun_dir_ws),
    );

    var sun_specular = vec3<f32>(0.0);
    var sun_subsurface = vec3<f32>(0.0);
    if (light.sun_enabled == 1u) {
        let light_dir = sun_dir_ws;
        let NdotL = max(0.0, dot(normal, light_dir));
        let NdotV = max(dot(normal, view_dir), 0.001);

        // Cook-Torrance specular BRDF (GGX distribution)
        let alpha = water.roughness * water.roughness;
        let half_vec = normalize(light_dir + view_dir);
        let NdotH = max(dot(normal, half_vec), 0.0);
        let HdotV = max(dot(half_vec, view_dir), 0.0);

        let D = D_GGX(NdotH, alpha);
        let G = G_Smith(NdotV, max(NdotL, 0.001), water.roughness);
        // Fresnel at half-vector angle (physically correct for microfacet model)
        let F_spec = F0 + (1.0 - F0) * pow(1.0 - HdotV, 5.0);

        let denom = 4.0 * NdotV * max(NdotL, 0.001);
        let specular_brdf = (D * G * F_spec) / max(denom, 0.001);

        sun_specular = light.sun_color * light.sun_intensity * specular_brdf * NdotL * rim_vis;

        // Subsurface illumination — light enters water, scatters, exits toward viewer.
        // Driven by the mean (flat) surface, not the facet: refraction squeezes all
        // transmitted light into the ~49 deg Snell cone and it travels far past the
        // wave scale before scattering back, so body radiance is volumetric. A facet
        // NdotL here is a Lambert lobe — Lambert + sharp GGX is the CG plastic look.
        let light_entering = max(sun_dir_ws.y, 0.0) * (1.0 - F_spec);
        let interior_glow = water.water_color * transmittance;
        sun_subsurface = interior_glow * light_entering * light.sun_color * light.sun_intensity * 0.18;

        // Forward scattering — thin areas glow when backlit (translucency)
        let VdotL = max(0.0, dot(-view_dir, light_dir));
        let forward_scatter = pow(VdotL, 4.0) * exp(-thickness * optical_density * 1.5);
        sun_subsurface += water.water_color * forward_scatter * light.sun_color * light.sun_intensity * 0.10;

        // Both subsurface paths are fed by direct sun at this surface point
        sun_subsurface *= rim_vis;
    }

    // IBL diffuse irradiance from spherical harmonics
    // Light enters the water (1-F), travels through the volume (transmittance),
    // and scatters back (scatter_strength) — same physics as subsurface scattering,
    // so it also sees the mean surface (sky irradiance onto a flat water plane)
    let ambient_irradiance = evaluate_sh_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * water.env_intensity;
    let ambient_subsurface = ambient_irradiance * water.water_color * transmittance * scatter_strength * 0.6;

    // Add sun subsurface (weighted by 1-fresnel for energy conservation) and ambient irradiance
    var lit_interior = interior_with_scatter
        + sun_subsurface * (1.0 - fresnel)
        + ambient_subsurface * (1.0 - fresnel);
    if (water.physical_medium != 0.0) {
        var sun_rgb = vec3<f32>(0.0);
        if (light.sun_enabled == 1u) {
            sun_rgb = light.sun_color * light.sun_intensity * rim_vis;
        }
        let medium = water_medium(
            path_length,
            refract(-view_dir, normal, 1.0 / water.ior),
            sun_dir_ws,
            sun_rgb,
            evaluate_sh_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * water.env_intensity,
        );
        lit_interior = refracted_scene * medium.transmittance + medium.inscatter;
    }

    // === AERATION (submerged whitewater) ===
    // Entrained-air density along this ray (G channel of the whitewater
    // field): mix the interior toward a lit milky tone. White IN the water,
    // as opposed to the surface foam composited after the Fresnel combine.
    let whitewater_field = textureSampleLevel(foam_density_tex, env_sampler, screen_uv, 0.0).rg;
    let aeration = 1.0 - exp(-AERATION_K * water.aeration_strength * whitewater_field.g);
    if (aeration > 0.002) {
        // Bubble clouds sit in the volume: lit through the mean surface like
        // the body light above, not by the facet they're seen through
        var aeration_light = evaluate_sh_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * water.env_intensity;
        if (light.sun_enabled == 1u) {
            aeration_light += light.sun_color * light.sun_intensity
                * max(sun_dir_ws.y, 0.0) * 0.6 * rim_vis;
        }
        lit_interior = mix(lit_interior, AERATION_ALBEDO * aeration_light, aeration);
    }

    // Below-horizon reflection rays mostly hit more water, so the faded env
    // share takes the water's own color. (Black painted dark creases on every
    // wave back at grazing Fresnel; a blurred-env fallback overshoots into white
    // creases since the sky's lower hemisphere is bright haze.)
    reflection_color += lit_interior * below_horizon * (1.0 - ssr_confidence);

    // Combine reflection and refraction based on Fresnel
    // At grazing angles (high fresnel): more reflection
    // Looking straight on (low fresnel): more refraction/transmission
    var color = mix(lit_interior, reflection_color, fresnel);

    // Add specular on top (pure surface reflection, independent of interior)
    color += sun_specular;

    // === FOAM OVERLAY ===
    // Screen-space foam density (splatted half-res by the spray system):
    // whiten the surface where foam accumulates. Foam is rough and diffuse,
    // so it replaces the specular water response rather than adding to it.
    let foam_density = whitewater_field.r;
    if (foam_density > 0.01) {
        let n_coarse = value_noise_grad(input.world_position.xz * FOAM_NOISE_SCALE).x;
        let n_fine = value_noise_grad(
            input.world_position.xz * FOAM_NOISE_SCALE_FINE + vec2<f32>(37.42, 11.18),
        ).x;
        // Coarse noise raggedizes patch edges into stringy breakup
        let breakup = (n_coarse - 0.5) * FOAM_NOISE_BREAKUP;
        let d_eff = max(foam_density * (1.0 + breakup) - FOAM_DENSITY_LO, 0.0);
        let coverage = 1.0 - exp(-FOAM_COVERAGE_K * water.foam_coverage * d_eff);
        if (coverage > 0.002) {
            var foam_light = evaluate_sh_irradiance(normal) * water.env_intensity;
            if (light.sun_enabled == 1u) {
                foam_light += light.sun_color * light.sun_intensity
                    * max(dot(normal, sun_dir_ws), 0.0) * rim_vis;
            }
            // Thin veil -> dry white crest, plus saturation-proof brightness
            // texture so thick carpets keep internal structure
            let thick = smoothstep(FOAM_THICK_LO, FOAM_THICK_HI, coverage);
            let albedo = mix(FOAM_VEIL_ALBEDO, FOAM_ALBEDO, thick);
            let tex = 1.0 + (n_coarse - 0.5) * FOAM_TEX_CONTRAST
                + (n_fine - 0.5) * FOAM_TEX_CONTRAST_FINE;
            let foam_color = albedo * tex * foam_light;
            color = mix(color, foam_color, coverage);
        }
    }

    // Surface foam from the map: a bubble raft whose lace pattern rides the
    // flow (two flow-map phases crossfaded so each restart is invisible)
    let map = sample_map_foam(local_pos, surface_up);
    if (map.density > 0.01 && map.on_top > 0.0) {
        let raft_cover = 1.0 - exp(-RAFT_COVERAGE_K * water.foam_coverage * max(map.density - RAFT_DENSITY_LO, 0.0));
        let w_a = 1.0 - abs(2.0 * foam_map.flow_phase - 1.0);
        let mask = w_a * raft_mask(map.coords.xy, raft_cover, foam_px)
            + (1.0 - w_a) * raft_mask(map.coords.zw, raft_cover, foam_px);
        let grain = w_a * bubble_grain(map.coords.xy, foam_px)
            + (1.0 - w_a) * bubble_grain(map.coords.zw, foam_px);
        let raft_alpha = mix(RAFT_ALPHA_THIN, RAFT_ALPHA_THICK, 1.0 - exp(-RAFT_ALPHA_K * map.density));
        let alpha = mask * raft_alpha * map.on_top;
        if (alpha > 0.002) {
            var raft_light = evaluate_sh_irradiance(normal) * water.env_intensity;
            if (light.sun_enabled == 1u) {
                raft_light += light.sun_color * light.sun_intensity
                    * max(dot(normal, sun_dir_ws), 0.0) * rim_vis;
            }
            let albedo = mix(FOAM_VEIL_ALBEDO, FOAM_ALBEDO, smoothstep(0.3, 2.0, map.density));
            color = mix(color, albedo * grain * raft_light, alpha);
        }
    }

    // What the debug views would show, plus the shaded result (probe only)
    probe_event(
        PRB_RESULT,
        vec3<f32>(f32(dbg_path), f32(dbg_end), f32(dbg_bounces)),
        vec3<f32>(dbg_uv, f32(dbg_exit_kind)),
        f32(dbg_mirror_kind),
    );
    probe_event(PRB_COLOR, refracted_scene, color, fresnel);

    // Refraction debug view: data instead of color (the app bypasses post
    // processing, so these values reach the screen as written; uniform branch,
    // so the derivatives are legal)
    if (water.debug_view != 0u) {
        return vec4<f32>(debug_view_output(), 1.0);
    }

    // Output linear HDR — post-process pipeline handles tone mapping + gamma.
    // Bounded: a grazing sun glint off a near-mirror surface can exceed the
    // f16 scene buffer's range
    return vec4<f32>(min(color, vec3<f32>(HDR_OUTPUT_MAX)), 1.0);
}
