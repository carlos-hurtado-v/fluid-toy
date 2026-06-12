# Validate all WGSL shaders with naga-cli, mirroring how the app assembles them
# (some shaders get container_common.wgsl prepended before create_shader_module).
#
# Usage: .\scripts\validate-shaders.ps1
# Requires: cargo install naga-cli --version "^27"  (major must match wgpu in Cargo.lock)
#
# Keep PREFIXED in sync with the format!("{}\n{}", container_common_wgsl, ...) call
# sites: marching_cubes.rs, sph_3d_grid.rs, spray.rs, wireframe.rs, container_renderer.rs.

$ErrorActionPreference = "Stop"

$shaderDir = Join-Path $PSScriptRoot "..\src\shaders"
$common = Join-Path $shaderDir "container_common.wgsl"

$prefixed = @(
    "mc_density.wgsl",
    "mc_render.wgsl",
    "mc_back_depth.wgsl",
    "sph_density_3d_grid.wgsl",
    "sph_integrate_3d.wgsl",
    "spray_simulate.wgsl",
    "wireframe.wgsl",
    "container.wgsl"
)

$failures = 0
$tempFile = Join-Path ([System.IO.Path]::GetTempPath()) "naga_validate_concat.wgsl"

Get-ChildItem $shaderDir -Filter *.wgsl | Where-Object { $_.Name -ne "container_common.wgsl" } | ForEach-Object {
    $target = $_.FullName
    if ($prefixed -contains $_.Name) {
        (Get-Content $common -Raw) + "`n" + (Get-Content $target -Raw) | Set-Content $tempFile -NoNewline
        $target = $tempFile
    }
    $output = & naga $target 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PASS  $($_.Name)"
    } else {
        Write-Host "FAIL  $($_.Name)" -ForegroundColor Red
        Write-Host ($output | Out-String)
        $script:failures++
    }
}

Remove-Item $tempFile -ErrorAction SilentlyContinue

if ($failures -gt 0) {
    Write-Host "`n$failures shader(s) failed validation" -ForegroundColor Red
    exit 1
}
Write-Host "`nAll shaders valid"
