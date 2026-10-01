// Fluid measurement pass: global fluid extents + per-column surface heights.
//
// One thread per SPH particle, atomicMax over order-preserving quantized
// coordinates. Results feed the stats CSV (--stats) and the GUI Measurements
// readout — e.g. dam-break front position is max_x, wave gauges are probes.

struct ProbeParams {
    num_particles: u32,
    num_probes: u32,
    _pad0: u32,
    _pad1: u32,
    // xyzw = probe x, probe z, radius squared, unused
    probes: array<vec4<f32>, 8>,
}

struct ProbeResults {
    // [0]=max_x [1]=-min_x [2]=max_y [3]=max_z [4]=-min_z [5..7]=reserved
    // [8..15] = per-probe max height
    slots: array<atomic<u32>, 16>,
}

@group(0) @binding(0) var<uniform> params: ProbeParams;
// SPH particle is 64 bytes = 4 vec4s; [i*4].xyz = position
@group(0) @binding(1) var<storage, read> particles: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> results: ProbeResults;

// Order-preserving quantization; 0 is reserved for "no sample".
// World range ±16 m at 1 micrometer resolution.
fn quantize(v: f32) -> u32 {
    return u32(clamp((v + 16.0) * 1000000.0, 1.0, 4200000000.0));
}

@compute @workgroup_size(256)
fn probe_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) {
        return;
    }
    let pos = particles[i * 4u].xyz;

    atomicMax(&results.slots[0], quantize(pos.x));
    atomicMax(&results.slots[1], quantize(-pos.x));
    atomicMax(&results.slots[2], quantize(pos.y));
    atomicMax(&results.slots[3], quantize(pos.z));
    atomicMax(&results.slots[4], quantize(-pos.z));

    for (var k = 0u; k < params.num_probes; k = k + 1u) {
        let p = params.probes[k];
        let dx = pos.x - p.x;
        let dz = pos.z - p.y;
        if (dx * dx + dz * dz <= p.z) {
            atomicMax(&results.slots[8u + k], quantize(pos.y));
        }
    }
}
