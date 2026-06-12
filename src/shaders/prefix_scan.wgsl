// Hierarchical inclusive prefix scan over per-cell particle counts.
// Replaces ceil(log2(total_cells)) Hillis-Steele full-array passes with 3
// dispatches:
//   1. scan_blocks:       each 256-element block scanned in shared memory,
//                          in place; block totals written to block_sums
//   2. scan_block_sums:   one workgroup scans block_sums in place, walking
//                          the array in 256-wide tiles with a running carry
//   3. add_block_offsets: every element of block b adds block_sums[b - 1]
// Result: data[] holds the inclusive prefix sum (same convention the grid
// consumers expect: cell range end = cell_starts[c], start = end - count).

struct GridParams {
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,
    total_cells: u32,
    cell_size: f32,
    inv_cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    num_particles: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(2) var<uniform> grid: GridParams;

// Must match SCAN_BLOCK_SIZE in sph_3d_grid.rs
const BLOCK_SIZE: u32 = 256u;

var<workgroup> scratch: array<u32, 256>;
var<workgroup> carry: u32;

// Inclusive Hillis-Steele scan of scratch[]. All barriers are in uniform
// control flow (callers must invoke from every thread in the workgroup).
fn scan_scratch(li: u32) {
    for (var offset = 1u; offset < BLOCK_SIZE; offset = offset << 1u) {
        var add = 0u;
        if (li >= offset) {
            add = scratch[li - offset];
        }
        workgroupBarrier();
        scratch[li] = scratch[li] + add;
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn scan_blocks(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    let li = lid.x;

    // Out-of-range lanes scan zeros so barriers stay uniform
    var v = 0u;
    if (i < grid.total_cells) {
        v = data[i];
    }
    scratch[li] = v;
    workgroupBarrier();

    scan_scratch(li);

    if (i < grid.total_cells) {
        data[i] = scratch[li];
    }
    if (li == BLOCK_SIZE - 1u) {
        block_sums[wid.x] = scratch[li];
    }
}

@compute @workgroup_size(256)
fn scan_block_sums(@builtin(local_invocation_id) lid: vec3<u32>) {
    let li = lid.x;
    let num_blocks = (grid.total_cells + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    if (li == 0u) {
        carry = 0u;
    }
    workgroupBarrier();

    for (var tile = 0u; tile < num_blocks; tile += BLOCK_SIZE) {
        let idx = tile + li;
        var v = 0u;
        if (idx < num_blocks) {
            v = block_sums[idx];
        }
        scratch[li] = v;
        workgroupBarrier();

        scan_scratch(li);

        // carry was last written behind a barrier; safe to read uniformly
        let tile_offset = carry;
        if (idx < num_blocks) {
            block_sums[idx] = scratch[li] + tile_offset;
        }
        workgroupBarrier();
        if (li == 0u) {
            // scratch[255] is the tile total (out-of-range lanes scanned zeros)
            carry = tile_offset + scratch[BLOCK_SIZE - 1u];
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn add_block_offsets(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    if (i >= grid.total_cells || wid.x == 0u) {
        return;
    }
    data[i] = data[i] + block_sums[wid.x - 1u];
}
