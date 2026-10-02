# Validate all WGSL shaders with naga-cli, mirroring how the app assembles them
# (some shaders get container_common.wgsl prepended before create_shader_module).
#
# Usage: .\scripts\validate-shaders.ps1
# Requires: cargo install naga-cli --version "^27"  (major must match wgpu in Cargo.lock)
#
# Keep PREFIXED in sync with the format!("{}\n{}", container_common_wgsl, ...) call
# sites: marching_cubes.rs, wall_bound.rs, sph_3d_grid.rs, spray.rs, wireframe.rs, container_renderer.rs,
# caustics.rs, rigid_body_renderer.rs. RIGIDPREFIXED shaders get
# container_common + rigid_body_common (sph_3d_grid.rs integrate/predict/solve).
# mc_render.wgsl links a pixel-probe snippet after container_common
# (marching_cubes.rs): it is validated with both mc_probe_off.wgsl (normal
# rendering) and mc_probe_on.wgsl (--probe pipeline).

$ErrorActionPreference = "Stop"

$shaderDir = Join-Path $PSScriptRoot "..\src\shaders"
$common = Join-Path $shaderDir "container_common.wgsl"
$rigidCommon = Join-Path $shaderDir "rigid_body_common.wgsl"

$prefixed = @(
    "mc_density.wgsl",
    "mc_anisotropy.wgsl",
    "mc_wall_bound.wgsl",
    "foam_map.wgsl",
    "mc_back_depth.wgsl",
    "mc_caustics_gbuffer.wgsl",
    "mc_caustics_splat.wgsl",
    "sph_density_3d_grid.wgsl",
    "spray_simulate.wgsl",
    "spray_render.wgsl",
    "wireframe.wgsl",
    "container.wgsl",
    "rigid_body.wgsl",
    "rigid_body_mesh.wgsl"
)

$rigidPrefixed = @(
    "sph_integrate_3d.wgsl",
    "pcisph_predict.wgsl",
    "pcisph_solve.wgsl"
)

$failures = 0
$tempFile = Join-Path ([System.IO.Path]::GetTempPath()) "naga_validate_concat.wgsl"

foreach ($variant in @("mc_probe_off.wgsl", "mc_probe_on.wgsl")) {
    (Get-Content $common -Raw) + "`n" + (Get-Content (Join-Path $shaderDir $variant) -Raw) + "`n" + (Get-Content (Join-Path $shaderDir "mc_render.wgsl") -Raw) | Set-Content $tempFile -NoNewline
    $output = & naga $tempFile 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PASS  mc_render.wgsl + $variant"
    } else {
        Write-Host "FAIL  mc_render.wgsl + $variant" -ForegroundColor Red
        Write-Host ($output | Out-String)
        $script:failures++
    }
}

Get-ChildItem $shaderDir -Filter *.wgsl | Where-Object { $_.Name -notin @("container_common.wgsl", "rigid_body_common.wgsl", "mc_render.wgsl") } | ForEach-Object {
    $target = $_.FullName
    if ($prefixed -contains $_.Name) {
        (Get-Content $common -Raw) + "`n" + (Get-Content $target -Raw) | Set-Content $tempFile -NoNewline
        $target = $tempFile
    }
    elseif ($rigidPrefixed -contains $_.Name) {
        (Get-Content $common -Raw) + "`n" + (Get-Content $rigidCommon -Raw) + "`n" + (Get-Content $target -Raw) | Set-Content $tempFile -NoNewline
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
