#!/usr/bin/env python3
"""Surface-noise lab for the marching-cubes density field (--dump-field dumps).

    # what the app meshes: any dump
    fluid-toy.exe --config snap.json --load-state snap.state --hold --size 1280x720 \
        --capture 2 --dump-field --out dir
    python scripts/field_lab.py dir/frame_00002_field [--png residual.png]

    # prototype field stages offline: a RAW dump, rebuilt here in numpy
    fluid-toy.exe ... --set rendering.mc_calm_smoothing=0 --set rendering.mc_blur_radius=0 \
        --capture 2 --dump-field --out raw
    python scripts/field_lab.py raw/frame_00002_field --replica [--png prefix]

Measures how noisy the top surface's normals are: the field's gradient at the
highest iso crossing of every column (what mc_voxel_normals.wgsl starts from),
minus its trend over ~8 cm, in degrees. Reported on calm columns (gentle, and
smooth in the widest variant) well inside the walls: median / rms / p90. The
PNG maps the residual (white = 2 deg): contour-line STRIPES are a field stage
that depends on where the surface sits in the voxel grid, a LABYRINTH is
particle noise that survived the smoothing.

--replica rebuilds the chain from a raw dump (base blur, calm S and G as in
calm_smoothing.rs / mc_calm_smooth.wgsl; exact to 1e-6 of the interior density
away from the walls - the dump is wall-bounded, so stay 25 cm clear) and
prints the variants side by side: base, the per-voxel-gated blend (how it was
before the gate moved onto the surface), S alone, and S with G's normals.
Edit `variants()` to try a filter, a gate or a normal source before writing a
shader for it. It does not model surface_gate or the normal denoise: measure
those on a normal dump and with scripts/normal_wobble.py.
"""
import argparse
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

# calm_smoothing.rs
REST_SPACING_FACTOR = 0.6
SMOOTH_HALF_WIDTH_H, GATE_HALF_WIDTH_H = 2.75, 4.1


def load(stem):
    meta = json.load(open(stem + ".json"))
    n = meta["grid_size"]
    return meta, np.fromfile(stem + ".bin", dtype=np.float32).reshape(n, n, n).astype(np.float64)  # [z, y, x]


def triangle(f, radius):
    """mc_blur.wgsl: separable triangle filter, weights r + 1 - |i|."""
    k = np.array([radius + 1 - abs(i) for i in range(-radius, radius + 1)], dtype=np.float64)
    for axis in range(3):
        f = ndimage.convolve1d(f, k / k.sum(), axis=axis, mode="nearest")
    return f


def downsample(f):
    n = f.shape[0] // 2
    return f[: 2 * n, : 2 * n, : 2 * n].reshape(n, 2, n, 2, n, 2).mean(axis=(1, 3, 5))


def upsample(half, n):
    """sample_half: full voxel p reads half-res coordinate (p + 0.5) / 2 - 0.5."""
    q = (np.arange(n) + 0.5) * 0.5 - 0.5
    return ndimage.map_coordinates(half, np.meshgrid(q, q, q, indexing="ij"), order=1, mode="nearest")


def half_radius(half_width_h, meta):
    return max(int(round(half_width_h * meta["kernel_radius"] / (2 * meta["cell_size"]))) - 1, 1)


def smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def top_surface(meta, field):
    """Highest iso crossing per (z, x) column: height, voxel below it, fraction to the next."""
    n, iso = meta["grid_size"], meta["iso_value"]
    inside = field >= iso
    top = n - 1 - np.argmax(inside[:, ::-1, :], axis=1)
    zi, xi = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v_in, v_out = field[zi, top, xi], field[zi, np.minimum(top + 1, n - 1), xi]
    t = np.clip((v_in - iso) / np.maximum(v_in - v_out, 1e-9), 0.0, 1.0)
    height = meta["grid_min"][1] + (top + t) * meta["cell_size"]
    return np.where(inside.any(axis=1), height, np.nan), top, t


def at_surface(field, top, t):
    n = field.shape[0]
    zi, xi = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    a, b = field[zi, top, xi], field[zi, np.minimum(top + 1, n - 1), xi]
    return a + (b - a) * t


def surface_normals(meta, shape_field, normal_field=None):
    """Unit normals of normal_field's gradient on shape_field's top surface."""
    height, top, t = top_surface(meta, shape_field)
    gz, gy, gx = np.gradient(shape_field if normal_field is None else normal_field)
    normal = -np.stack([at_surface(g, top, t) for g in (gx, gy, gz)], -1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
    normal[np.isnan(height)] = np.nan
    return height, normal


def residual(meta, normal, sigma_cm=8.0):
    """Angle (deg) between each normal and the gaussian trend of its neighbourhood."""
    ok = ~np.isnan(normal[..., 0])
    sigma = sigma_cm / 100 / meta["cell_size"]
    weight = ndimage.gaussian_filter(ok.astype(float), sigma)
    trend = np.stack([ndimage.gaussian_filter(np.where(ok, normal[..., k], 0.0), sigma) for k in range(3)], -1)
    trend /= np.maximum(np.linalg.norm(trend, axis=-1, keepdims=True), 1e-12)
    angle = np.degrees(np.arccos(np.clip((np.nan_to_num(normal) * trend).sum(-1), -1, 1)))
    return np.where(ok & (weight > 0.98), angle, np.nan), trend


def calm_columns(meta, height, res, trend, wall_margin=0.25, max_tilt_deg=30.0):
    n, cell, c = meta["grid_size"], meta["cell_size"], meta["container"]
    x = meta["grid_min"][0] + np.arange(n) * cell
    z = meta["grid_min"][2] + np.arange(n) * cell
    inner = np.outer(np.abs(z) < c["depth"] / 2 - wall_margin, np.abs(x) < c["width"] / 2 - wall_margin)
    window = max(3, int(round(0.11 / cell)))
    quiet = ndimage.uniform_filter(np.where(np.isnan(res), 9.0, res), window) < 1.5
    return inner & ~np.isnan(res) & (trend[..., 1] > np.cos(np.radians(max_tilt_deg))) & quiet


def save_map(path, res, calm):
    v = np.clip(np.nan_to_num(res) / 2.0, 0, 1)
    rgb = (np.stack([v] * 3, -1) * 255).astype(np.uint8)
    rgb[~calm] = (rgb[~calm] * 0.3 + np.array([30, 0, 50])).astype(np.uint8)
    Image.fromarray(rgb[::-1]).resize((800, 800), Image.NEAREST).save(path)


def variants(meta, raw):
    """name -> (field whose top surface is measured, field the normals come from or None)."""
    n = meta["grid_size"]
    interior = 1.0 / (REST_SPACING_FACTOR * meta["kernel_radius"]) ** 3
    base = triangle(raw, 1)
    s_half = triangle(downsample(raw), half_radius(SMOOTH_HALF_WIDTH_H, meta))
    g_half = triangle(s_half, half_radius(GATE_HALF_WIDTH_H, meta))
    S, G = upsample(s_half, n), upsample(g_half, n)
    per_voxel = base + (S - base) * smoothstep(0.22 * interior, 0.40 * interior, G)
    return {
        "S alone (gate = 1)": (S, None),               # first: defines the calm columns
        "base blur only": (base, None),
        "per-voxel gate (old)": (per_voxel, None),
        "S, normals from G": (S, G),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", help="dump path without extension (.bin + .json)")
    ap.add_argument("--replica", action="store_true", help="the dump is raw: rebuild the chain and compare variants")
    ap.add_argument("--png", help="residual map (with --replica: prefix, one map per variant)")
    args = ap.parse_args()

    meta, field = load(args.stem)
    todo = variants(meta, field) if args.replica else {"field": (field, None)}
    calm = None
    for i, (name, (shape, source)) in enumerate(todo.items()):
        height, normal = surface_normals(meta, shape, source)
        res, trend = residual(meta, normal)
        if calm is None:
            calm = calm_columns(meta, height, res, trend)
            print(f"{args.stem}: grid {meta['grid_size']}, cell {meta['cell_size'] * 100:.2f} cm, {calm.sum()} calm columns")
        v = res[calm & ~np.isnan(res)]
        if len(v) == 0:
            print(f"  {name:22s} no calm surface")
            continue
        print(f"  {name:22s} normal noise  median {np.median(v):.3f}  rms {np.sqrt((v ** 2).mean()):.3f}  p90 {np.percentile(v, 90):.3f} deg")
        if args.png:
            save_map(args.png if not args.replica else f"{args.png}_{i}.png", res, calm)


if __name__ == "__main__":
    sys.exit(main())
