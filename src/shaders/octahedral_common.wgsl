// Octahedral unit vectors in 16 + 16 bits (0.003 deg): how the voxel normal
// texture stores a normal. Prepended to the writer (mc_voxel_normals.wgsl)
// and to both readers (mc_generate.wgsl, the water shader's water_normal),
// which must decode exactly what was encoded.
// Needs nothing from the including shader.

fn oct_encode(n: vec3<f32>) -> u32 {
    var o = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) {
        o = (1.0 - abs(o.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), o >= vec2<f32>(0.0));
    }
    let q = vec2<u32>(round(clamp(o, vec2<f32>(-1.0), vec2<f32>(1.0)) * 32767.0 + 32767.0));
    return q.x | (q.y << 16u);
}

fn oct_decode(v: u32) -> vec3<f32> {
    let o = (vec2<f32>(f32(v & 0xffffu), f32(v >> 16u)) - 32767.0) / 32767.0;
    var n = vec3<f32>(o.x, o.y, 1.0 - abs(o.x) - abs(o.y));
    let t = max(-n.z, 0.0);
    n.x += select(t, -t, n.x >= 0.0);
    n.y += select(t, -t, n.y >= 0.0);
    return normalize(n);
}
