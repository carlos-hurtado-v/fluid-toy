// Pixel probe off (normal rendering): no-op stubs that compile away, so the
// water shader keeps no storage writes. The --probe pipeline links
// mc_probe_on.wgsl instead.

fn probe_begin(frag_xy: vec2<f32>, depth: f32) {}

fn probe_event(tag: u32, a: vec3<f32>, b: vec3<f32>, c: f32) {}

fn probe_event4(tag: u32, a: vec3<f32>, b: vec3<f32>, c: f32, d: vec4<f32>) {}
