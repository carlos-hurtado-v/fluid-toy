// Pixel probe (--probe): records, event by event, what the water shader's
// refraction code did at chosen pixels, for scripts/probe_decode.py. Linked
// into a separate MC render pipeline only while probing; normal rendering
// links mc_probe_off.wgsl (no-op stubs) instead, because any storage write in
// a fragment shader can cost the pass its early depth test.
//
// Every fragment that shades a probed pixel takes a slot (occluded ones and
// both triangles at an MSAA edge included): the decoder keeps the nearest.

const PROBE_MAX_PIXELS: u32 = 64u;

struct ProbeBuffer {
    pixel_count: u32,
    slots_used: atomic<u32>,
    events_per_slot: u32,
    _pad: u32,
    // Probed pixels (.xy, PNG pixel coordinates: x right, y down)
    pixels: array<vec4<u32>, PROBE_MAX_PIXELS>,
    // Slots of 2 + 2 * events_per_slot entries: [pixel index, frag x, frag y,
    // raw depth], [events recorded, overflowed, -, -], then per event
    // (a.xyz, tag) and (b.xyz, c)
    data: array<vec4<f32>>,
}
@group(0) @binding(20) var<storage, read_write> probe: ProbeBuffer;

var<private> probe_base: i32 = -1;
var<private> probe_n: u32 = 0u;

fn probe_begin(frag_xy: vec2<f32>, depth: f32) {
    let px = vec2<u32>(frag_xy);
    let n = min(probe.pixel_count, PROBE_MAX_PIXELS);
    for (var i = 0u; i < n; i++) {
        if (all(probe.pixels[i].xy == px)) {
            let stride = 2u + 2u * probe.events_per_slot;
            let slot = atomicAdd(&probe.slots_used, 1u);
            if ((slot + 1u) * stride > arrayLength(&probe.data)) {
                return;
            }
            let base = slot * stride;
            probe.data[base] = vec4<f32>(f32(i), frag_xy.x, frag_xy.y, depth);
            probe.data[base + 1u] = vec4<f32>(0.0);
            probe_base = i32(base);
            return;
        }
    }
}

fn probe_event(tag: u32, a: vec3<f32>, b: vec3<f32>, c: f32) {
    if (probe_base < 0) {
        return;
    }
    let base = u32(probe_base);
    if (probe_n >= probe.events_per_slot) {
        probe.data[base + 1u].y = 1.0;
        return;
    }
    let at = base + 2u + 2u * probe_n;
    probe.data[at] = vec4<f32>(a, f32(tag));
    probe.data[at + 1u] = vec4<f32>(b, c);
    probe_n += 1u;
    probe.data[base + 1u].x = f32(probe_n);
}
