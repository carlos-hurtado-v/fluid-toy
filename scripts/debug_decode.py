#!/usr/bin/env python3
"""Decode a refraction debug-view capture (rendering.mc_debug_view).

Capture with the view on (post-processing is bypassed automatically), e.g.

    fluid-toy --config my.json --set rendering.mc_debug_view=Paths \\
        --capture 600 --out captures/dbg

then

    python scripts/debug_decode.py captures/dbg/frame_00600.png --mode paths
    python scripts/debug_decode.py frame.png --mode paths --region 700,540,900,720
    python scripts/debug_decode.py frame.png --mode jump --map jump_map.png

Prints a histogram for the region (whole image by default) and writes a
false-color map (default: <capture>_<mode>.png) with the region outlined.

Encoding (written by debug_view_output() in src/shaders/mc_render.wgsl; the id
tables below must match its DBG_PATH_* / DBG_END_* constants):
  paths : R = path id / 16, G = lookup id / 8, B = (mirror bounces + 1) / 8
  lookup: R, G = screen uv of the final background lookup, B = lookup id / 8
  jump  : log2(1 + lookup jump between neighbouring pixels, texels) / 8
  exit  : R = cos(exit angle), G = water path to first exit / 4 m,
          B = exit interface id / 8 (see EXITS)
  mirror: R = interface of the last mirror reflection / 8, G = exit
          interface / 8 (both EXITS ids), B = (mirror bounces + 1) / 8
The swapchain stores sRGB: values are linearised before decoding. Paths mode
only counts pixels whose three channels all carry the exact 8-bit code of a
valid id; everything else (non-water pixels, edges blended by MSAA) is
reported as "not water / mixed" (a scene pixel passing by chance is rare,
~1 in 2000).
Lookup, jump and exit hold continuous values and cannot tell water from the
rest of the image: give them a --region over the water.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

PATHS = {
    1: "legacy refraction (physical off)",
    2: "inside: marched onto a surface in the water",
    3: "no back face: straight to the backdrop",
    4: "blocked in the water",
    5: "thin-body TIR: straight through",
    6: "refracted out",
    7: "mirrored, then blocked",
    8: "mirrored, then refracted out",
    9: "mirrored until out of bounces",
    10: "mirrored onto a pool wall: straight through",
}
ENDS = {
    1: "straight-through view",
    2: "surface reached (background texture)",
    3: "backdrop on screen (vanishing point)",
    4: "environment map",
    5: "solid background color",
    6: "exact body hit",
    7: "projected ground (where the ray lands, off screen)",
}
EXITS = {
    0: "no refracted exit (blocked / straight / legacy)",
    1: "container wall or floor (exact plane)",
    2: "back face away from the walls (free surface, drop)",
    3: "back face within 6 cm of a wall (MC contact line / bulge)",
}
# Distinct, readable colors per id (sRGB)
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (170, 110, 40),
]


def srgb_to_linear(a):
    a = a / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def heat(v):
    """0..1 -> dark blue, cyan, yellow, red"""
    stops = np.array([[0, 0, 60], [0, 160, 220], [250, 230, 40], [230, 30, 30]], float)
    x = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def parse_region(text, w, h):
    if not text:
        return 0, 0, w, h
    x0, y0, x1, y1 = (int(v) for v in text.split(","))
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def linear_to_code(v):
    """The 8-bit sRGB code the swapchain stores for linear value v"""
    s = np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)
    return np.rint(s * 255).astype(int)


def match_ids(codes, scale, ids):
    """Per pixel: the id whose encoded value (id / scale) produces this exact
    8-bit code (+-1), else -1. Exact codes rather than a tolerance on decoded
    values: ordinary scene pixels then almost never pass for data."""
    out = np.full(codes.shape, -1, int)
    for k in ids:
        out[np.abs(codes - linear_to_code(k / scale)) <= 1] = k
    return out


def decode_paths(rgb, region, args):
    codes = rgb.astype(int)
    path = match_ids(codes[..., 0], 16, PATHS.keys())
    end = match_ids(codes[..., 1], 8, list(ENDS.keys()) + [0])
    b = match_ids(codes[..., 2], 8, range(1, 5))
    water = (path > 0) & (end >= 0) & (b > 0)
    x0, y0, x1, y1 = region
    sub = water[y0:y1, x0:x1]
    total = sub.size
    n_water = int(sub.sum())
    print(f"region {x0},{y0} - {x1},{y1}: {total} px, {n_water} water ({100.0 * n_water / max(total, 1):.1f}%), "
          f"{total - n_water} not water / mixed")
    if n_water:
        p = path[y0:y1, x0:x1][sub]
        e = end[y0:y1, x0:x1][sub]
        bounces = b[y0:y1, x0:x1][sub] - 1
        combos = {}
        for pi, ei in zip(p.tolist(), e.tolist()):
            combos[(pi, ei)] = combos.get((pi, ei), 0) + 1
        print("\npath -> lookup                                                         px      %")
        for (pi, ei), n in sorted(combos.items(), key=lambda kv: -kv[1]):
            label = f"{pi:2d} {PATHS.get(pi, '?')}  ->  {ei} {ENDS.get(ei, '?')}"
            print(f"{label:<70} {n:7d} {100.0 * n / n_water:6.1f}")
        print("\nmirror bounces: " + ", ".join(
            f"{k}: {int((bounces == k).sum())}" for k in range(4) if (bounces == k).any()))
    return water, path, end


def write_map(base_rgb, colored, mask, region, out, legend):
    # Non-data pixels: the capture in dim gray, for orientation
    gray = (base_rgb.mean(axis=2, keepdims=True) * 0.35).astype(np.uint8).repeat(3, axis=2)
    img = np.where(mask[..., None], colored, gray)
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    x0, y0, x1, y1 = region
    if (x0, y0, x1, y1) != (0, 0, im.width, im.height):
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 255, 255))
    y = 6
    for color, text in legend:
        draw.rectangle([6, y, 20, y + 12], fill=color)
        draw.text((26, y), text, fill=(255, 255, 255))
        y += 16
    im.save(out)
    print(f"\nmap: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--mode", required=True, choices=["paths", "lookup", "jump", "exit", "mirror"])
    ap.add_argument("--region", help="x0,y0,x1,y1 in pixels (default: whole image)")
    ap.add_argument("--by", choices=["path", "end", "exit"], default="path",
                    help="paths mode: color the map by; exit mode: 'exit' colors by interface")
    ap.add_argument("--map", help="false-color map output path")
    args = ap.parse_args()

    rgb = np.asarray(Image.open(args.capture).convert("RGB"))
    lin = srgb_to_linear(rgb.astype(float))
    h, w = rgb.shape[:2]
    region = parse_region(args.region, w, h)
    x0, y0, x1, y1 = region
    out = args.map or os.path.splitext(args.capture)[0] + f"_{args.mode}.png"

    if args.mode == "paths":
        water, path, end = decode_paths(rgb, region, args)
        ids, table = (path, PATHS) if args.by == "path" else (end, ENDS)
        colored = np.zeros_like(rgb)
        legend = []
        for k, name in table.items():
            sel = water & (ids == k)
            if sel.any():
                colored[sel] = PALETTE[(k - 1) % len(PALETTE)]
                legend.append((PALETTE[(k - 1) % len(PALETTE)], f"{k} {name}"))
        write_map(rgb, colored, water, region, out, legend)
        return

    if args.mode == "mirror":
        codes = rgb.astype(int)
        mirror = match_ids(codes[..., 0], 8, EXITS.keys())
        exit_id = match_ids(codes[..., 1], 8, EXITS.keys())
        b = match_ids(codes[..., 2], 8, range(1, 5))
        data = (mirror >= 0) & (exit_id >= 0) & (b > 0)
        sub = data[y0:y1, x0:x1]
        n = int(sub.sum())
        print(f"region {x0},{y0} - {x1},{y1}: {sub.size} px, {n} decoded")
        combos = {}
        for m, e in zip(mirror[y0:y1, x0:x1][sub].tolist(), exit_id[y0:y1, x0:x1][sub].tolist()):
            combos[(m, e)] = combos.get((m, e), 0) + 1
        print("\nlast mirror reflection -> exit                                              px      %")
        for (m, e), c in sorted(combos.items(), key=lambda kv: -kv[1]):
            label = f"{m} {EXITS[m][:34]} -> {e} {EXITS[e][:34]}"
            print(f"{label:<74} {c:7d} {100.0 * c / max(n, 1):6.1f}")
        ids = mirror if args.by != "exit" else exit_id
        colored = np.zeros_like(rgb)
        legend = []
        for k, name in EXITS.items():
            sel = data & (ids == k)
            if sel.any():
                colored[sel] = PALETTE[k % len(PALETTE)]
                legend.append((PALETTE[k % len(PALETTE)], f"{k} {name}"))
        write_map(rgb, colored, data, region, out, legend)
        return

    sub = lin[y0:y1, x0:x1]
    if args.mode == "lookup":
        uv = sub[..., :2]
        # Staircase detector: neighbouring rows/columns that land on the very
        # same lookup (to 8-bit precision) - repeated rows are banding
        same_v = np.all(np.abs(np.diff(uv, axis=0)) < 1e-6, axis=2).mean()
        same_h = np.all(np.abs(np.diff(uv, axis=1)) < 1e-6, axis=2).mean()
        print(f"region {x0},{y0} - {x1},{y1}: u {uv[..., 0].min():.3f}-{uv[..., 0].max():.3f}, "
              f"v {uv[..., 1].min():.3f}-{uv[..., 1].max():.3f}")
        print(f"identical lookup in the next row: {100 * same_v:.1f}%   next column: {100 * same_h:.1f}%"
              "   (high = staircase / repeated rows)")
        colored = np.zeros_like(rgb)
        colored[..., 0] = (np.mod(lin[..., 0] * 16, 1) * 255).astype(np.uint8)
        colored[..., 1] = (np.mod(lin[..., 1] * 16, 1) * 255).astype(np.uint8)
        colored[..., 2] = (lin[..., 2] * 8 / 6 * 255).clip(0, 255).astype(np.uint8)
        mask = np.zeros((h, w), bool)
        mask[y0:y1, x0:x1] = True
        write_map(rgb, colored, mask, region, out, [((255, 0, 0), "u x16 (fract)"), ((0, 255, 0), "v x16 (fract)")])
        return

    if args.mode == "jump":
        jump = 2.0 ** (sub[..., 0] * 8) - 1
        pct = np.percentile(jump, [50, 90, 99])
        print(f"region {x0},{y0} - {x1},{y1}: lookup jump (texels per pixel) median {pct[0]:.1f}, "
              f"p90 {pct[1]:.1f}, p99 {pct[2]:.1f}; >4 texels: {100 * (jump > 4).mean():.1f}%, "
              f">16: {100 * (jump > 16).mean():.1f}%")
        colored = heat(lin[..., 0])
        mask = np.zeros((h, w), bool)
        mask[y0:y1, x0:x1] = True
        write_map(rgb, colored, mask, region, out,
                  [(tuple(heat(np.array(v))), f"{2 ** (v * 8) - 1:.0f} texels") for v in (0.0, 1 / 3, 2 / 3, 1.0)])
        return

    # exit
    cos_exit = sub[..., 0]
    path_m = sub[..., 1] * 4
    exit_id = match_ids(rgb[y0:y1, x0:x1, 2].astype(int), 8, EXITS.keys())
    print(f"region {x0},{y0} - {x1},{y1}: exit cos median {np.median(cos_exit):.2f} "
          f"(< 0.2 = grazing / near critical: {100 * (cos_exit < 0.2).mean():.1f}%), "
          f"water path median {np.median(path_m):.2f} m")
    for k, name in EXITS.items():
        sel = exit_id == k
        if sel.any():
            print(f"  exit {k} {name:<58} {int(sel.sum()):7d} px, grazing (<0.2) "
                  f"{100 * (cos_exit[sel] < 0.2).mean():5.1f}%")
    mask = np.zeros((h, w), bool)
    mask[y0:y1, x0:x1] = True
    if args.by == "exit":
        ids = match_ids(rgb[..., 2].astype(int), 8, EXITS.keys())
        colored = np.zeros_like(rgb)
        legend = []
        for k, name in EXITS.items():
            sel = ids == k
            if sel.any():
                colored[sel] = PALETTE[k % len(PALETTE)]
                legend.append((PALETTE[k % len(PALETTE)], f"{k} {name}"))
        write_map(rgb, colored, mask, region, out, legend)
        return
    colored = heat(1 - lin[..., 0])
    write_map(rgb, colored, mask, region, out, [(tuple(heat(np.array(0.0))), "straight out"),
                                                (tuple(heat(np.array(1.0))), "grazing exit")])


if __name__ == "__main__":
    sys.exit(main())
