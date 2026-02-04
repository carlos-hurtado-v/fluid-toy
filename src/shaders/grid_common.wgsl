// Common grid structures and functions for spatial hashing

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

fn position_to_cell(pos: vec3<f32>, grid: GridParams) -> vec3<i32> {
    let local_pos = pos - vec3<f32>(grid.grid_origin_x, grid.grid_origin_y, grid.grid_origin_z);
    return vec3<i32>(floor(local_pos * grid.inv_cell_size));
}

fn cell_to_index(cell: vec3<i32>, grid: GridParams) -> u32 {
    let cx = clamp(cell.x, 0i, i32(grid.grid_size_x) - 1i);
    let cy = clamp(cell.y, 0i, i32(grid.grid_size_y) - 1i);
    let cz = clamp(cell.z, 0i, i32(grid.grid_size_z) - 1i);
    return u32(cx) + u32(cy) * grid.grid_size_x + u32(cz) * grid.grid_size_x * grid.grid_size_y;
}

fn is_valid_cell(cell: vec3<i32>, grid: GridParams) -> bool {
    return cell.x >= 0i && cell.x < i32(grid.grid_size_x) &&
           cell.y >= 0i && cell.y < i32(grid.grid_size_y) &&
           cell.z >= 0i && cell.z < i32(grid.grid_size_z);
}
