# Validate all WGSL shaders with naga-cli, mirroring how the app assembles them
# (WGSL has no includes: shared snippets, the *_common.wgsl files, are
# prepended to a shader before create_shader_module).
#
# Usage: .\scripts\validate-shaders.ps1
# Requires: cargo install naga-cli --version "^27"  (major must match wgpu in Cargo.lock)
#
# Keep $prefixes in sync with the format!("{}\n{}", <snippet>, <shader>) call
# sites in src\render and src\simulation: one entry per shader that gets
# snippets, listing them in order. A shader the table does not name is
# validated on its own.
#
# The water shader ("mc_render") is one module assembled from shared snippets,
# a pixel-probe snippet and the part files under src\shaders\mc_render\. Both
# lists and their order are read from WATER_SHADER_COMMON / WATER_SHADER_PARTS
# in src\render\mc_water.rs, so there is nothing to keep in sync here; a .wgsl
# file in that directory that the list does not name fails. The module is
# validated with both mc_probe_off.wgsl (normal rendering) and mc_probe_on.wgsl
# (--probe pipeline).

$ErrorActionPreference = "Stop"

$shaderDir = Join-Path $PSScriptRoot "..\src\shaders"

$container = "container_common.wgsl"
$prefixes = @{
    # MC field + G-buffers (mc_field.rs, mc_anisotropy.rs, wall_bound.rs, mc_faces.rs)
    "mc_density.wgsl"          = @($container)
    "mc_anisotropy.wgsl"       = @($container)
    "mc_wall_bound.wgsl"       = @($container)
    "mc_back_depth.wgsl"       = @($container)
    # Voxel normal texture: writer and mesh reader (voxel_normals.rs, mc_mesh.rs)
    "mc_voxel_normals.wgsl"    = @("octahedral_common.wgsl")
    "mc_generate.wgsl"         = @("octahedral_common.wgsl")
    # Caustics (caustics.rs)
    "mc_caustics_gbuffer.wgsl" = @($container, "noise_common.wgsl")
    "mc_caustics_splat.wgsl"   = @($container)
    # Screen-space water composite (screen_space_fluid.rs)
    "ss_composite.wgsl"        = @("water_common.wgsl", "noise_common.wgsl", "sh_common.wgsl")
    # Scene objects (container_renderer.rs, wireframe.rs, rigid_body_renderer.rs, spray_renderer.rs)
    "container.wgsl"           = @($container, "sh_common.wgsl")
    "wireframe.wgsl"           = @($container)
    "rigid_body.wgsl"          = @($container, "sh_common.wgsl")
    "rigid_body_mesh.wgsl"     = @($container, "sh_common.wgsl")
    "spray_render.wgsl"        = @($container, "sh_common.wgsl")
    # Simulation (sph_3d_grid.rs, spray.rs, foam_map.rs)
    "sph_density_3d_grid.wgsl" = @($container)
    "sph_integrate_3d.wgsl"    = @($container, "rigid_body_common.wgsl")
    "pcisph_predict.wgsl"      = @($container, "rigid_body_common.wgsl")
    "pcisph_solve.wgsl"        = @($container, "rigid_body_common.wgsl")
    "spray_simulate.wgsl"      = @($container)
    "foam_map.wgsl"            = @($container)
}

$failures = 0
$tempFile = Join-Path ([System.IO.Path]::GetTempPath()) "naga_validate_concat.wgsl"

function Read-Shader([string]$relative) {
    Get-Content (Join-Path $shaderDir $relative) -Raw
}

# --- The water shader: shared snippets + probe snippet + mc_render parts ---
$waterRs = Join-Path $PSScriptRoot "..\src\render\mc_water.rs"
$waterIncludes = @([regex]::Matches((Get-Content $waterRs -Raw), 'include_str!\("\.\./shaders/([a-z0-9_/]+\.wgsl)"\)') |
    ForEach-Object { $_.Groups[1].Value } | Where-Object { $_ -notlike "mc_probe_*" })
$waterCommon = @($waterIncludes | Where-Object { $_ -notlike "mc_render/*" })
$waterParts = @($waterIncludes | Where-Object { $_ -like "mc_render/*" })
if ($waterParts.Count -eq 0) {
    Write-Host "FAIL  no mc_render parts found in src\render\mc_water.rs (WATER_SHADER_PARTS)" -ForegroundColor Red
    $failures++
}
Get-ChildItem (Join-Path $shaderDir "mc_render") -Filter *.wgsl | Where-Object { "mc_render/$($_.Name)" -notin $waterParts } | ForEach-Object {
    Write-Host "FAIL  mc_render\$($_.Name) is not in WATER_SHADER_PARTS (src\render\mc_water.rs): the app would not link it" -ForegroundColor Red
    $script:failures++
}

foreach ($variant in @("mc_probe_off.wgsl", "mc_probe_on.wgsl")) {
    $source = ""
    $offsets = @()
    foreach ($file in ($waterCommon + @($variant) + $waterParts)) {
        $offsets += "{0,6}  {1}" -f (($source -split "`n").Count), $file
        $source += (Read-Shader $file) + "`n"
    }
    $source | Set-Content $tempFile -NoNewline
    $output = & naga $tempFile 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PASS  mc_render ($($waterCommon.Count) snippets + $($waterParts.Count) parts) + $variant"
    } else {
        Write-Host "FAIL  mc_render + $variant" -ForegroundColor Red
        Write-Host ($output | Out-String)
        Write-Host "Line numbers above count through the assembled module. Each file starts at line:"
        $offsets | ForEach-Object { Write-Host $_ }
        $script:failures++
    }
}

# --- Every other shader, on its own or behind its snippets ---
# (snippets are not valid on their own: they use the including shader's bindings)
Get-ChildItem $shaderDir -Filter *.wgsl | Where-Object { $_.Name -notlike "*_common.wgsl" } | ForEach-Object {
    $target = $_.FullName
    if ($prefixes.ContainsKey($_.Name)) {
        $source = ""
        foreach ($snippet in $prefixes[$_.Name]) {
            $source += (Read-Shader $snippet) + "`n"
        }
        $source + (Get-Content $target -Raw) | Set-Content $tempFile -NoNewline
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
