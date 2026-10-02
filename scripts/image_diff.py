#!/usr/bin/env python3
"""Pixel diff of two captures of the same frame (render A/B on a held state).

Typical drill: snapshot the moment (F12, or --snapshot in automation), then
render it frozen with each variant and diff:

    fluid-toy --config snap.json --load-state snap.state --hold \\
        --size WxH --capture 30 --out captures/ab/a
    (patched build, or a different --set) ... --out captures/ab/b
    python scripts/image_diff.py captures/ab/a/frame_00030.png \\
        captures/ab/b/frame_00030.png --region 700,540,900,720

Prints how many pixels differ (any change / over --threshold), the max and
mean difference and where the changes are (bounding box), and writes a
heatmap (default: <b>_diff.png): the per-pixel max channel difference x
--gain, black -> red -> yellow -> white, over the A capture in dim gray, with
the region outlined. --crop also writes an A | B | heat strip of the region.

Two holds of the same state with the same build are NOT bit-identical: the MC
mesh is appended through atomics, so triangle order varies and a few edge
pixels flip. Run the same variant twice for the noise floor before reading a
small diff as a change.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True  # no scripts/__pycache__ from the import below
from debug_decode import parse_region  # noqa: E402  (same --region semantics)


def heat_colors(v):
    """v in [0, 1] -> black, red, yellow, white"""
    stops = np.array([[0, 0, 0], [255, 0, 0], [255, 255, 0], [255, 255, 255]], float)
    x = np.clip(v, 0.0, 1.0) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--region", help="x0,y0,x1,y1 in pixels (default: whole image)")
    ap.add_argument("--threshold", type=int, default=8,
                    help="8-bit difference that counts as a visible change (default 8)")
    ap.add_argument("--gain", type=float, default=4.0,
                    help="heatmap gain: difference x gain / 255 (default 4)")
    ap.add_argument("--out", help="heatmap path (default: <b>_diff.png)")
    ap.add_argument("--crop", action="store_true", help="also write an A | B | heat strip of the region")
    args = ap.parse_args()

    a = np.asarray(Image.open(args.a).convert("RGB")).astype(int)
    b = np.asarray(Image.open(args.b).convert("RGB")).astype(int)
    if a.shape != b.shape:
        sys.exit(f"size mismatch: {args.a} is {a.shape[1]}x{a.shape[0]}, {args.b} is {b.shape[1]}x{b.shape[0]}")
    h, w = a.shape[:2]
    x0, y0, x1, y1 = parse_region(args.region, w, h)

    diff = np.abs(a - b).max(axis=2)
    d = diff[y0:y1, x0:x1]
    total = d.size
    changed = int((d > 0).sum())
    over = d > args.threshold
    n_over = int(over.sum())
    print(f"region {x0},{y0} - {x1},{y1}: {total} px")
    print(f"  changed (any):       {changed:8d} px  {100.0 * changed / max(total, 1):7.3f}%")
    print(f"  changed (>{args.threshold:3d}):       {n_over:8d} px  {100.0 * n_over / max(total, 1):7.3f}%")
    if changed:
        my, mx = np.unravel_index(int(d.argmax()), d.shape)
        print(f"  max diff:            {int(d.max()):8d}     at ({x0 + mx}, {y0 + my})")
        print(f"  mean diff (all px):  {d.mean():11.3f}")
        print(f"  mean diff (changed): {d[d > 0].mean():11.3f}")
        for q in (50, 90, 99):
            print(f"  p{q} of changed:      {np.percentile(d[d > 0], q):11.1f}")
    if n_over:
        ys, xs = np.nonzero(over)
        print(f"  bbox of >{args.threshold}:        {x0 + xs.min()},{y0 + ys.min()} - {x0 + xs.max() + 1},{y0 + ys.max() + 1}")
    if not changed:
        print("  identical")

    gray = (a.mean(axis=2, keepdims=True) * 0.3).astype(np.uint8).repeat(3, axis=2)
    heat = heat_colors(diff * args.gain / 255.0)
    img = np.where((diff > 0)[..., None], heat, gray)
    im = Image.fromarray(img)
    if (x0, y0, x1, y1) != (0, 0, w, h):
        ImageDraw.Draw(im).rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 160, 255))
    out = args.out or os.path.splitext(args.b)[0] + "_diff.png"
    im.save(out)
    print(f"\nheatmap: {out}")

    if args.crop:
        crop = lambda arr: arr[y0:y1, x0:x1].astype(np.uint8)
        strip = np.concatenate([crop(a), crop(b), np.asarray(im)[y0:y1, x0:x1]], axis=1)
        crop_out = os.path.splitext(out)[0] + "_crop.png"
        Image.fromarray(strip).save(crop_out)
        print(f"crop strip (A | B | heat): {crop_out}")


if __name__ == "__main__":
    main()
