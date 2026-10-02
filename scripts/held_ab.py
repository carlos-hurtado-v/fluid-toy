#!/usr/bin/env python3
"""A/B a saved moment (F12 / --snapshot state) held, under two settings or builds.

    python scripts/held_ab.py captures/snapshots/snap_004_2560x1351 \
        --b rendering.mc_calm_smoothing=0 --crop 1560,690,2180,880 --scale 3

    python scripts/held_ab.py <stem> --exe-a old.exe --exe-b target/release/fluid-toy.exe
    python scripts/held_ab.py <stem> --time 1200          # ms/frame instead of images

Renders the snapshot `--hold` with variant A (`--a path=value`, repeatable;
default none) and variant B, prints the pixels differing by > 8/255 with their
bounding box, and writes A over B (optionally cropped / magnified) to
captures/held_ab/<name>.png. `--set` applies to both. Held renders are
bit-identical for the same build and settings, so any difference is real.

`--time N` skips the images and times N held frames per variant instead
(difference of a 100-frame and a (100+N)-frame run, alternated over
`--rounds`), printing ms/frame: use >= 1200 frames, shorter runs scatter by
+-1 ms.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image


def launch(exe, stem, size, sets, extra):
    cmd = [os.path.abspath(exe), "--config", stem + ".json", "--load-state", stem + ".state", "--hold", "--size", size]
    for s in sets:
        cmd += ["--set", s]
    return subprocess.run(cmd + extra, capture_output=True, text=True, timeout=600)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", help="snapshot path without extension (.json + .state)")
    ap.add_argument("--name", help="output name (default: the snapshot's)")
    ap.add_argument("--size", help="WxH (default: from the snapshot's file name, else 1600x900)")
    ap.add_argument("--set", action="append", default=[], help="path=value for both variants")
    ap.add_argument("--a", action="append", default=[], help="path=value for variant A only")
    ap.add_argument("--b", action="append", default=[], help="path=value for variant B only")
    ap.add_argument("--exe-a", default="target/release/fluid-toy.exe")
    ap.add_argument("--exe-b", default=None, help="default: same as --exe-a")
    ap.add_argument("--frame", type=int, default=30, help="held frame to capture (default 30: temporal history settled)")
    ap.add_argument("--crop", help="x0,y0,x1,y1 of the saved comparison")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--time", type=int, default=0, metavar="FRAMES")
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()

    base = os.path.basename(args.stem)
    size = args.size or (base.rsplit("_", 1)[-1] if "x" in base.rsplit("_", 1)[-1] else "1600x900")
    name = args.name or base
    variants = [("a", args.exe_a, args.set + args.a), ("b", args.exe_b or args.exe_a, args.set + args.b)]

    if args.time:
        result = {tag: [] for tag, _, _ in variants}
        for _ in range(args.rounds):
            for tag, exe, sets in variants:
                spans = []
                for frames in (100, 100 + args.time):
                    t0 = time.time()
                    r = launch(exe, args.stem, size, sets, ["--exit-after", str(frames)])
                    if r.returncode != 0:
                        sys.exit(f"variant {tag} failed: {(r.stderr or r.stdout)[-400:]}")
                    spans.append(time.time() - t0)
                result[tag].append((spans[1] - spans[0]) / args.time * 1000.0)
        for tag, runs in result.items():
            print(f"{name} {tag}: " + " ".join(f"{v:.2f}" for v in runs) + f"  ms/frame  median {sorted(runs)[len(runs) // 2]:.2f}")
        return

    images = []
    for tag, exe, sets in variants:
        out = os.path.join("captures", "held_ab", f"{name}_{tag}")
        r = launch(exe, args.stem, size, sets, ["--capture", str(args.frame), "--out", out])
        png = os.path.join(out, f"frame_{args.frame:05d}.png")
        if r.returncode != 0 or not os.path.exists(png):
            sys.exit(f"variant {tag} failed: {(r.stderr or r.stdout)[-400:]}")
        images.append(Image.open(png).convert("RGB"))
    a, b = (np.asarray(i).astype(int) for i in images)
    changed = np.abs(a - b).max(axis=2) > 8
    ys, xs = np.nonzero(changed)
    bbox = f"  bbox {xs.min()},{ys.min()} - {xs.max()},{ys.max()}" if len(xs) else ""
    print(f"{name}: {int(changed.sum())} px changed ({100.0 * changed.mean():.2f}%){bbox}")
    if args.crop:
        box = tuple(int(v) for v in args.crop.split(","))
        images = [i.crop(box) for i in images]
    w, h = images[0].size
    sheet = Image.new("RGB", (w, h * 2 + 6), (255, 0, 0))
    sheet.paste(images[0], (0, 0))
    sheet.paste(images[1], (0, h + 6))
    if args.scale != 1.0:
        sheet = sheet.resize((int(sheet.width * args.scale), int(sheet.height * args.scale)),
                             Image.NEAREST if args.scale > 1 else Image.LANCZOS)
    path = os.path.join("captures", "held_ab", f"{name}.png")
    sheet.save(path)
    print(f"A over B: {path} {sheet.size}")


if __name__ == "__main__":
    main()
