#!/usr/bin/env python3
"""Measure the marching-cubes surface in a --dump-field dump.

    fluid-toy.exe ... --capture 2 --dump-field --out dir
    python scripts/field_profile.py dir/frame_00002_field [more stems...] [--png out.png]

For a calm, UNTILTED tank: finds the top surface (highest iso crossing of each
voxel column, linearly interpolated like mc_generate places vertices: voxel i
sits at grid_min + i * cell_size) and prints its height relative to the
interior against the distance from the walls - the "valley" where the water
meets the glass. Also prints how far the sides and bottom of the mesh stop
short of the walls and floor. Several stems print side by side (one column
per dump).
"""
import argparse
import json
import sys

import numpy as np

DIST_CM = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
WALLS = [("-X", 0, -1), ("+X", 0, 1), ("-Z", 2, -1), ("+Z", 2, 1)]
# Profiles and interior statistics stay this far from the other walls
CLEAR = 0.35


def load(stem):
    meta = json.load(open(stem + ".json"))
    n = meta["grid_size"]
    field = np.fromfile(stem + ".bin", dtype=np.float32).reshape(n, n, n)  # [z, y, x]
    return meta, field


def axis_coords(meta, axis):
    return meta["grid_min"][axis] + np.arange(meta["grid_size"]) * meta["cell_size"]


def half_extent(meta, axis):
    c = meta["container"]
    return (c["width"] if axis == 0 else c["depth"]) / 2


def top_surface(meta, field):
    """Height of the highest iso crossing per (z, x) column; NaN where none."""
    n = meta["grid_size"]
    iso = meta["iso_value"]
    inside = field >= iso                      # [z, y, x]
    any_in = inside.any(axis=1)
    top = n - 1 - np.argmax(inside[:, ::-1, :], axis=1)   # highest in-water voxel
    zi, xi = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v_in = field[zi, top, xi]
    v_out = field[zi, np.minimum(top + 1, n - 1), xi]
    t = np.clip((v_in - iso) / np.maximum(v_in - v_out, 1e-9), 0.0, 1.0)
    h = meta["grid_min"][1] + (top + t) * meta["cell_size"]
    return np.where(any_in, h, np.nan)


def interior(meta, h):
    x = axis_coords(meta, 0)
    z = axis_coords(meta, 2)
    return h[np.ix_(np.abs(z) < half_extent(meta, 2) - CLEAR, np.abs(x) < half_extent(meta, 0) - CLEAR)]


def profile(meta, h, axis, sign):
    """Mean surface height vs distance from the wall at sign * half along axis."""
    other = 2 if axis == 0 else 0
    keep = np.abs(axis_coords(meta, other)) < half_extent(meta, other) - CLEAR
    hm = h if axis == 0 else h.T                  # [other, along]
    with np.errstate(all="ignore"):
        line = np.nanmean(hm[keep], axis=0)
    d = half_extent(meta, axis) - sign * axis_coords(meta, axis)
    order = np.argsort(d)
    d, line = d[order], line[order]
    ok = ~np.isnan(line)
    return d[ok], line[ok]


def side_gap(meta, field, axis, sign, y_world):
    """Mean distance from a wall to the outermost iso crossing at height y."""
    n = meta["grid_size"]
    iso = meta["iso_value"]
    cell = meta["cell_size"]
    yi = int(round((y_world - meta["grid_min"][1]) / cell))
    plane = field[:, yi, :]                    # [z, x]
    if axis == 2:
        plane = plane.T                        # [x, z]: scan along z
    other = 2 if axis == 0 else 0
    rows = np.nonzero(np.abs(axis_coords(meta, other)) < half_extent(meta, other) - CLEAR)[0]
    gaps = []
    for r in rows:
        line = plane[r]
        idx = np.nonzero(line >= iso)[0]
        if idx.size == 0:
            continue
        i_in = idx[-1] if sign > 0 else idx[0]
        i_out = i_in + sign
        v_out = line[i_out] if 0 <= i_out < n else 0.0
        t = min((line[i_in] - iso) / max(line[i_in] - v_out, 1e-9), 1.0)
        pos = meta["grid_min"][axis] + (i_in + sign * t) * cell
        gaps.append(half_extent(meta, axis) - sign * pos)
    return float(np.mean(gaps)) if gaps else float("nan")


def floor_gap(meta, field):
    """Mean height of the lowest iso crossing above the floor, interior columns."""
    n = meta["grid_size"]
    iso = meta["iso_value"]
    x = axis_coords(meta, 0)
    z = axis_coords(meta, 2)
    sub = field[np.ix_(np.abs(z) < half_extent(meta, 2) - CLEAR, np.arange(n), np.abs(x) < half_extent(meta, 0) - CLEAR)]
    bot = np.argmax(sub >= iso, axis=1)
    zi, xi = np.meshgrid(np.arange(sub.shape[0]), np.arange(sub.shape[2]), indexing="ij")
    v_in = sub[zi, bot, xi]
    v_out = sub[zi, np.maximum(bot - 1, 0), xi]
    t = np.clip((v_in - iso) / np.maximum(v_in - v_out, 1e-9), 0, 1)
    return float(np.mean(meta["grid_min"][1] + (bot - t) * meta["cell_size"] - meta["container"]["floor_y"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--png", help="write a height map of the first dump (mm vs interior, +-30)")
    ap.add_argument("--labels", help="comma-separated column labels (default: each dump's directory)")
    args = ap.parse_args()

    dumps = [load(s) for s in args.stems]
    labels = args.labels.split(",") if args.labels else [s.replace("\\", "/").split("/")[-2] for s in args.stems]
    surfaces = [top_surface(m, f) for m, f in dumps]
    interiors = [float(np.nanmean(interior(m, h))) for (m, _), h in zip(dumps, surfaces)]

    m0 = dumps[0][0]
    print(f"grid {m0['grid_size']}^3, cell {m0['cell_size'] * 100:.2f} cm, kernel {m0['kernel_radius'] * 100:.2f} cm")
    width = max(max(len(l) for l in labels) + 2, 9)
    pad = 34
    head = "".join(f"{l:>{width}}" for l in labels)

    def row(name, values, fmt=".1f"):
        print(f"  {name:<{pad - 2}}" + "".join(f"{v:>{width}{fmt}}" for v in values))

    print("\n" + " " * pad + head)
    row("interior surface height (m)", interiors, ".4f")
    row("interior roughness rms (mm)", [float(np.nanstd(interior(m, h))) * 1000 for (m, _), h in zip(dumps, surfaces)], ".2f")

    dist = np.array(DIST_CM) / 100
    per_wall = []
    for k, ((m, _), h) in enumerate(zip(dumps, surfaces)):
        cols = {}
        for name, axis, sign in WALLS:
            d, line = profile(m, h, axis, sign)
            vals = np.interp(dist, d, line) - interiors[k]
            # no surface at that distance: the mesh stops short of the wall
            cols[name] = np.where(dist < d.min() - 1e-6, np.nan, vals) * 1000
        per_wall.append(cols)

    print("\nsurface height vs interior (mm) by distance from the wall (cm), mean of the four walls")
    for i, d in enumerate(DIST_CM):
        with np.errstate(all="ignore"):
            row(f"{d:5.1f}", [float(np.nanmean([c[n][i] for n, _, _ in WALLS])) for c in per_wall])

    print(f"\nper wall, first dump ({labels[0]}), mm")
    print("  dist(cm)" + "".join(f"{n:>9}" for n, _, _ in WALLS))
    for i, d in enumerate(DIST_CM):
        print(f"  {d:6.1f}  " + "".join(f"{per_wall[0][n][i]:>9.1f}" for n, _, _ in WALLS))

    print("\nmesh stops short of the wall by (mm): low = 10 cm above the floor, high = 10 cm below the surface")
    for name, axis, sign in WALLS:
        row(f"{name} low", [side_gap(m, f, axis, sign, m["container"]["floor_y"] + 0.10) * 1000 for m, f in dumps])
        row(f"{name} high", [side_gap(m, f, axis, sign, s - 0.10) * 1000 for (m, f), s in zip(dumps, interiors)])
    row("floor", [floor_gap(m, f) * 1000 for m, f in dumps])

    if args.png:
        from PIL import Image
        h = surfaces[0] - interiors[0]
        img = np.clip((h * 1000 + 30) / 60, 0, 1)
        rgb = np.stack([img, img, img], axis=-1)
        rgb[np.isnan(h)] = [0.3, 0.0, 0.0]
        Image.fromarray((rgb * 255).astype(np.uint8)).resize((800, 800), Image.NEAREST).save(args.png)
        print(f"\nwrote {args.png} (black -30 mm .. white +30 mm vs interior, dark red = no surface)")


if __name__ == "__main__":
    sys.exit(main())
