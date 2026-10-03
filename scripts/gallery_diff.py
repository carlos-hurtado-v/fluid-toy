#!/usr/bin/env python3
"""Compare two refraction_gallery.py runs made with different builds.

A shader change with no toggle has nothing to A/B against inside one build:
keep a copy of the old executable, run the gallery once per build into its
own --out, then

    python scripts/gallery_diff.py gallery captures/gallery_base captures/gallery_new
    python scripts/gallery_diff.py pair a.png b.png 1560,700,1740,860 5 zoom.png

gallery  per view x variant: pixels differing by > 8/255 between the two
         runs' beauty frames, with their bounding box
pair     the same region of two images side by side, magnified (nearest)
"""
import sys
import os
import numpy as np
from PIL import Image


def gallery(base, new):
    views = sorted(d for d in os.listdir(new) if os.path.isdir(os.path.join(new, d)))
    for view in views:
        # Every variant both runs rendered (the gallery's variant names are
        # its own business: --variant replaces the defaults)
        for var in sorted(os.listdir(os.path.join(new, view))):
            pa = f"{base}/{view}/{var}/frame_00030.png"
            pb = f"{new}/{view}/{var}/frame_00030.png"
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            a = np.asarray(Image.open(pa).convert("RGB")).astype(int)
            b = np.asarray(Image.open(pb).convert("RGB")).astype(int)
            d = np.abs(a - b).max(axis=2) > 8
            ys, xs = np.nonzero(d)
            if len(xs) == 0:
                print(f"{view:22s} {var:10s}      0 px")
            else:
                print(f"{view:22s} {var:10s} {d.sum():6d} px {100 * d.mean():6.3f}%  bbox {xs.min()},{ys.min()} - {xs.max()},{ys.max()}")


def pair(a, b, box, scale, out):
    box = tuple(int(v) for v in box.split(","))
    scale = int(scale)
    A = Image.open(a).convert("RGB").crop(box)
    B = Image.open(b).convert("RGB").crop(box)
    w, h = A.size
    o = Image.new("RGB", (w * scale * 2 + 10, h * scale), (255, 0, 0))
    o.paste(A.resize((w * scale, h * scale), Image.NEAREST), (0, 0))
    o.paste(B.resize((w * scale, h * scale), Image.NEAREST), (w * scale + 10, 0))
    o.save(out)
    print(out, o.size)


if __name__ == "__main__":
    if sys.argv[1] == "gallery":
        gallery(sys.argv[2], sys.argv[3])
    else:
        pair(*sys.argv[2:7])
