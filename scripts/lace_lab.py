#!/usr/bin/env python3
"""Offline replica of mc_render.wgsl `raft_mask` (the foam-map bubble raft's
lace pattern), for trying lace geometry variants and calibrating their mean
coverage before writing WGSL.

    python scripts/lace_lab.py                      # sheet + calibration table
    python scripts/lace_lab.py --px 0.0006          # close-up footprint (camera at 1 m)
    python scripts/lace_lab.py --out lace.png
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw

# mc_render.wgsl constants
LACE_CELL = 0.045
LACE_CELL_FINE = 0.017
LACE_PATCH_FREQ = 8.0
LACE_SNAP_FREQ = 33.0
LACE_WARP_CELLS = 3.0
LACE_WARP_AMP = 0.3
LACE_FEATURE_WEIGHT = 0.3
LACE_ROUND_DENSE = 1.9
LACE_ROUND_COV = (0.45, 0.9)
LACE_METRIC_OFFSET = 0.3
LACE_WIDTH_NOISE = 0.5

W_M, H_M = 0.60, 0.40

# (label, raft_mask keyword overrides); first row = legacy F2 - F1 lace
VARIANTS = [
    ("legacy F2-F1", dict(warp=0.0, weight=0.0, round_dense=1.0, width_noise=0.0)),
    ("shipped", dict()),
    ("warp 0.4 (folds)", dict(warp=0.4)),
    ("rounder (c 2.5)", dict(round_dense=2.5)),
    ("no snap gate", dict(snap=False)),
]


def fract(x):
    return x - np.floor(x)


def hash2(px, py):
    p3x = fract(px * 0.1031)
    p3y = fract(py * 0.1031)
    p3z = fract(px * 0.1031)
    d = p3x * (p3y + 33.33) + p3y * (p3z + 33.33) + p3z * (p3x + 33.33)
    p3x = p3x + d
    p3y = p3y + d
    p3z = p3z + d
    return fract((p3x + p3y) * p3z)


def hash22(px, py):
    return hash2(px, py), hash2(px + 19.19, py + 73.31)


def value_noise_grad(px, py):
    ix, iy = np.floor(px), np.floor(py)
    fx, fy = px - ix, py - iy
    ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)
    dux = 30.0 * fx * fx * (fx * (fx - 2.0) + 1.0)
    duy = 30.0 * fy * fy * (fy * (fy - 2.0) + 1.0)
    a = hash2(ix, iy)
    b = hash2(ix + 1, iy)
    c = hash2(ix, iy + 1)
    d = hash2(ix + 1, iy + 1)
    val = a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy
    dx = dux * ((b - a) + (a - b - c + d) * uy)
    dy = duy * ((c - a) + (a - b - c + d) * ux)
    return val, dx, dy


def worley_weighted(px, py, weight):
    """(F1, F2) in cell units; weight > 0 = additively weighted features."""
    cx, cy = np.floor(px), np.floor(py)
    frx, fry = px - cx, py - cy
    f1 = np.full_like(px, 8.0)
    f2 = np.full_like(px, 8.0)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            hx, hy = hash22(cx + ox, cy + oy)
            dx = ox + hx - frx
            dy = oy + hy - fry
            d = np.sqrt(dx * dx + dy * dy) - weight * fract(hx * 7.0 + hy * 13.0)
            m1 = d < f1
            f2 = np.where(m1, f1, np.where(d < f2, d, f2))
            f1 = np.where(m1, d, f1)
    return f1, f2


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lace_warp(px, py, amp_cells, warp_cells):
    """Divergence-free displacement: rotated value-noise gradients, 2 octaves."""
    f = 1.0 / (warp_cells * LACE_CELL)
    _, g0x, g0y = value_noise_grad(px * f, py * f)
    _, g1x, g1y = value_noise_grad(px * 2 * f + 7.1, py * 2 * f + 3.3)
    k = amp_cells * LACE_CELL / 0.68
    return px + (g0y + 0.5 * g1y) * k, py + (-g0x - 0.5 * g1x) * k


def raft_mask(px, py, coverage, px_m, *, warp=LACE_WARP_AMP, warp_cells=LACE_WARP_CELLS,
              weight=LACE_FEATURE_WEIGHT, round_dense=LACE_ROUND_DENSE, round_cov=LACE_ROUND_COV,
              offset=LACE_METRIC_OFFSET, width_noise=LACE_WIDTH_NOISE, snap=True):
    patch_noise, _, _ = value_noise_grad(px * LACE_PATCH_FREQ, py * LACE_PATCH_FREQ)
    cov = np.clip(coverage * (0.45 + 1.1 * patch_noise), 0.0, 1.0)
    qx, qy = (px, py) if warp <= 0.0 else lace_warp(px, py, warp, warp_cells)
    c = 1.0 + (round_dense - 1.0) * smoothstep(round_cov[0], round_cov[1], cov)
    off = (c - 1.0) * offset
    f1c, f2c = worley_weighted(qx / LACE_CELL, qy / LACE_CELL, weight)
    f1f, f2f = worley_weighted(qx / LACE_CELL_FINE + 5.3, qy / LACE_CELL_FINE + 1.7, weight)
    coarse = f2c - c * f1c + off
    fine = f2f - c * f1f + off
    fine_weight = 3.5 + (1.4 - 3.5) * smoothstep(0.3, 0.8, cov)
    wall = np.minimum(coarse, fine * fine_weight)
    sn, _, _ = value_noise_grad(qx * LACE_SNAP_FREQ + 3.1, qy * LACE_SNAP_FREQ + 7.9)
    threshold = -np.log(np.maximum(1.0 - cov * 0.985, 1e-3)) * 0.22 \
        * (1.0 - width_noise + 2.0 * width_noise * sn)
    soft = max(px_m / LACE_CELL_FINE * 1.5, 0.03) * (0.5 * (1.0 + c))
    mask = 1.0 - smoothstep(threshold - soft, threshold + soft, wall)
    if snap:
        keep = np.clip(cov * 1.8, 0.0, 1.0)
        mask = mask * smoothstep(1.0 - keep - 0.12, 1.0 - keep + 0.12, sn)
    fade = smoothstep(0.25, 0.8, px_m / LACE_CELL_FINE)
    return mask * (1.0 - fade) + cov * fade


def render(mask):
    water = np.array([0.08, 0.20, 0.36])
    foam = np.array([0.93, 0.95, 0.97])
    rgb = water[None, None, :] * (1 - mask[..., None]) + foam[None, None, :] * mask[..., None]
    return (np.clip(rgb, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--px", type=float, default=0.0012, help="pixel footprint on the surface (m)")
    ap.add_argument("--coverages", default="0.25,0.5,0.8,0.95")
    ap.add_argument("--scale", type=int, default=2, help="magnification of the sheet")
    ap.add_argument("--out", default="captures/lace_lab.png")
    args = ap.parse_args()

    coverages = [float(c) for c in args.coverages.split(",")]
    nx, ny = int(W_M / args.px), int(H_M / args.px)
    xs = (np.arange(nx) + 0.5) * args.px + 0.37
    ys = (np.arange(ny) + 0.5) * args.px + 1.13
    px, py = np.meshgrid(xs, ys)

    s, pad, label_w = args.scale, 6, 220
    sheet = Image.new("RGB", (label_w + len(coverages) * (nx * s + pad), len(VARIANTS) * (ny * s + pad)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    cal_covs = (0.25, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0)
    print("mean mask per coverage  " + " ".join(f"{c:>5.2f}" for c in cal_covs))
    legacy = None
    for r, (name, kw) in enumerate(VARIANTS):
        y0 = r * (ny * s + pad)
        draw.text((4, y0 + ny * s // 2), name, fill=(230, 230, 230))
        for ci, cov in enumerate(coverages):
            img = Image.fromarray(render(raft_mask(px, py, cov, args.px, **kw)))
            sheet.paste(img.resize((nx * s, ny * s), Image.BICUBIC), (label_w + ci * (nx * s + pad), y0))
        means = np.array([raft_mask(px, py, cov, args.px, **kw).mean() for cov in cal_covs])
        if legacy is None:
            legacy = means
            tail = ""
        else:
            tail = f"   rms vs legacy {np.sqrt(np.mean((means - legacy) ** 2)):.3f}"
        print(f"{name:22s}  " + " ".join(f"{m:5.3f}" for m in means) + tail)
    sheet.save(args.out)
    print(args.out, sheet.size)


if __name__ == "__main__":
    main()
