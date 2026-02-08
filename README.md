# fluid-toy

Real-time 3D fluid simulation using Smoothed Particle Hydrodynamics (SPH) on the GPU. Built with Rust, wgpu, and egui.

All physics run as compute shaders on the GPU. The CPU handles windowing, input, and configuration. Rendering uses either marching cubes surface extraction (default) or direct particle billboards.

## Building

Requires Rust toolchain (rustup in windows).

```
cargo run --release
```

Debug builds work but run significantly slower due to CPU-side overhead.

## Controls

- **Left mouse drag** -- orbit camera
- **Scroll wheel** -- zoom
- **Right mouse drag** -- apply force to fluid (mode selected in GUI)
- **Middle mouse hold** -- spawn particles at cursor

The GUI panel (egui) on the left exposes all simulation and rendering parameters. Changes apply immediately.

## Simulation

Grid-accelerated SPH with double density relaxation. The simulation runs as a 7-stage compute pipeline per step:

1. Clear spatial hash grid
2. Build grid (hash particles into cells)
3. Prefix sum (compute cell start offsets)
4. Reorder particles by cell
5. Density computation (density + near density)
6. Force computation (pressure, viscosity, surface tension)
7. Integration (velocity + position update, boundary enforcement)

Neighbor search uses a uniform spatial hash grid, giving O(n) scaling rather than O(n^2) brute force.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| kernel_radius | 0.08 | SPH smoothing length |
| rest_density | 8000.0 | Target fluid density |
| stiffness | 35.0 | Pressure response |
| near_stiffness | 0.45 | Near-field repulsion (prevents collapse) |
| viscosity | 15.0 | Velocity diffusion |
| surface_tension | 6.5 | Cohesive force between particles |
| wall_stiffness | 250.0 | Soft boundary penalty force |
| damping | 0.55 | Energy retained on wall bounce |
| delta_time | 0.008 | Fixed timestep per frame |
| max_particles | 50,000 | Buffer capacity |

### Rigid bodies

Primitive shapes (cube, sphere, cylinder, torus) and custom glTF meshes (duck.glb included as a test mesh). Rigid bodies interact with the fluid via penalty forces and accumulate buoyancy/torque. Bodies can be held in place or dropped to simulate freely.

Custom meshes use a signed distance field (SDF) generated at load time for fluid-body collision.

### Mouse force modes

Five interaction modes applied via right-click drag: Push, Pull, Vortex, Explode (one-shot), Drain.

## Rendering

### Marching cubes (default)

Particles are splatted onto a density field, optionally blurred (separable 3D Gaussian, configurable radius 0-5), then meshed via marching cubes. The water fragment shader implements:

- Fresnel reflectance (Schlick approximation)
- Screen-space refraction from the background texture
- Beer-Lambert absorption (wavelength-dependent)
- Total internal reflection at grazing angles
- Subsurface scattering (thickness-dependent)
- GGX specular from a directional sun light
- Configurable deep water color and roughness

All shaders output linear HDR. Tonemapping (ACES) and gamma correction happen in a single post-processing pass.

### Particle mode

Billboard spheres with optional velocity-based coloring. Useful for debugging.

### Post-processing

- ACES tonemapping + exposure control
- FXAA
- Bloom (threshold + intensity)
- Vignette
- Chromatic aberration
- Anamorphic streaks
- Color grading (saturation, contrast, brightness, temperature)

### Environment

Two bundled HDR environment maps (Farmland, Pure Sky) used for reflections and background. Switchable at runtime. Alternative: solid color background.

### Anti-aliasing

MSAA (Off/2x/4x/8x, requires restart) and FXAA (post-process, togglable at runtime).

## Spray particles

GPU-driven spray particle system. High-velocity fluid particles emit spray that simulates ballistically with gravity and air drag. Configurable emission threshold, lifetime, jitter, and particle size.

## Container

The fluid container has adjustable dimensions and can be tilted on two axes with smooth interpolation. "Flip Upside Down" inverts the container via rotation rather than negating gravity, so the fluid pours out naturally.

## Dependencies

| Crate | Purpose |
|-------|---------|
| wgpu 27 | GPU compute and rendering |
| winit 0.30 | Windowing and input |
| egui 0.33 | Immediate-mode GUI |
| bytemuck | Safe transmute for GPU structs |
| image | HDR environment map loading |
| gltf | Mesh loading for custom rigid bodies |
| half | f16 support for HDR textures |
| pollster | Minimal async runtime for wgpu init |
