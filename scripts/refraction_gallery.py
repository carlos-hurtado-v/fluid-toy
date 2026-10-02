#!/usr/bin/env python3
"""Refraction regression gallery: render saved moments (F12 / --snapshot
states) held, under several setting variants, and compare.

    python scripts/refraction_gallery.py
    python scripts/refraction_gallery.py --variant cont=rendering.mc_silhouette_exit=Continue
    python scripts/refraction_gallery.py --views my_views.json --size 2560x1351

For every view x variant it captures the beauty (held frame 30, temporal
history settled) and the Paths debug view (frame 2), then prints:
  flips   route changes between neighbouring water pixels per 100 water px
          (Paths view: stripes / speckle / jaggies in refraction routes;
          blind to stripes within one route, so also look at the sheets)
  changed % of pixels differing by > 8/255 from the first variant
and writes one contact sheet per view (variants side by side) into --out.

Default views: every captures/snapshots/snap_*.state at its own camera, plus
four extra cameras on snap_001 (its crater exercises the back-face
silhouette case that the other states never hit). Default variants:
  original  the pre-2026-10-01 behaviour (both refraction fixes off)
  default   the current defaults
  continue  current defaults + mc_silhouette_exit=Continue
A views file is JSON: [{"name": ..., "snapshot": "<path without extension>",
"set": ["camera.yaw=0.8", ...]}, ...].

Why: two fixes in a row were recommended after checking only the snapshot
they targeted, and each regressed another view. Run this before calling a
refraction change done.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True  # no scripts/__pycache__ from the import below
from debug_decode import PATHS, match_ids  # noqa: E402

DEFAULT_VARIANTS = [
    ("original", ["rendering.mc_silhouette_exit=Exit", "rendering.mc_front_face_exit=false"]),
    ("default", []),
    ("continue", ["rendering.mc_silhouette_exit=Continue"]),
]
SNAP_001_CAMERAS = ["camera.yaw=0.8", "camera.yaw=-0.9", "camera.pitch=0.5", "camera.pitch=-0.25"]


def default_views():
    views = []
    for state in sorted(glob.glob("captures/snapshots/snap_*.state")):
        stem = state[: -len(".state")]
        name = os.path.basename(stem).split("_")[0] + "_" + os.path.basename(stem).split("_")[1]
        views.append({"name": name, "snapshot": stem, "set": []})
        if name == "snap_001":
            for cam in SNAP_001_CAMERAS:
                views.append({"name": f"{name}[{cam.split('.', 1)[1]}]", "snapshot": stem, "set": [cam]})
    return views


def route_flips(png):
    img = np.asarray(Image.open(png).convert("RGB")).astype(int)
    path = match_ids(img[..., 0], 16, PATHS.keys())
    end = match_ids(img[..., 1], 8, range(1, 7))
    route = np.where((path >= 0) & (end >= 0), path * 10 + end, -1)
    v = (route[1:, :] >= 0) & (route[:-1, :] >= 0)
    h = (route[:, 1:] >= 0) & (route[:, :-1] >= 0)
    n = ((route[1:, :] != route[:-1, :]) & v).sum() + ((route[:, 1:] != route[:, :-1]) & h).sum()
    water = int((route >= 0).sum())
    return 100.0 * n / max(water, 1), water


def run(exe, view, sets, size, frame, out, timeout):
    cmd = [exe, "--config", view["snapshot"] + ".json", "--load-state", view["snapshot"] + ".state",
           "--hold", "--size", size, "--capture", str(frame), "--out", out]
    for s in view.get("set", []) + sets:
        cmd += ["--set", s]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    png = os.path.join(out, f"frame_{frame:05d}.png")
    if r.returncode != 0 or not os.path.exists(png):
        return None, (r.stderr or r.stdout).strip().splitlines()[-1:] or ["failed"]
    return png, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", default="target/release/fluid-toy.exe")
    ap.add_argument("--out", default="captures/gallery")
    ap.add_argument("--size", default="1600x900", help="window size for every view (default 1600x900)")
    ap.add_argument("--views", help="JSON views file (default: every saved snapshot + snap_001 cameras)")
    ap.add_argument("--variant", action="append", metavar="NAME=path=value[;path=value]",
                    help="replace the default variants (repeatable; the first is the baseline; "
                         "'NAME=' alone = current defaults)")
    ap.add_argument("--timeout", type=int, default=120, help="seconds per launch")
    args = ap.parse_args()

    args.exe = os.path.abspath(args.exe)
    if not os.path.exists(args.exe):
        sys.exit(f"no executable at {args.exe} (cargo build --release)")
    views = json.load(open(args.views)) if args.views else default_views()
    if not views:
        sys.exit("no views: save snapshots with F12 first, or pass --views")
    variants = DEFAULT_VARIANTS
    if args.variant:
        variants = []
        for v in args.variant:
            name, _, rest = v.partition("=")
            variants.append((name, [s for s in rest.split(";") if s]))

    t0 = time.time()
    total = len(views) * len(variants)
    rows = []
    done = 0
    for view in views:
        shots = []
        for name, sets in variants:
            done += 1
            base = os.path.join(args.out, view["name"].replace("[", "_").replace("]", "").replace("=", ""), name)
            beauty, err = run(args.exe, view, sets, args.size, 30, base, args.timeout)
            paths, err2 = run(args.exe, view, sets + ["rendering.mc_debug_view=Paths"], args.size, 2,
                              base + "_paths", args.timeout)
            print(f"[{done}/{total} {time.time() - t0:5.0f}s] {view['name']} / {name}"
                  + (f"  ERROR {err or err2}" if (err or err2) else ""), flush=True)
            shots.append((name, beauty, paths))
        base_img = None
        sheet = []
        for name, beauty, paths in shots:
            if beauty is None or paths is None:
                rows.append((view["name"], name, None, None))
                continue
            img = np.asarray(Image.open(beauty).convert("RGB")).astype(int)
            if base_img is None:
                base_img = img
            changed = 100.0 * (np.abs(img - base_img).max(axis=2) > 8).mean()
            flips, _ = route_flips(paths)
            rows.append((view["name"], name, flips, changed))
            sheet.append((name, Image.open(beauty).convert("RGB")))
        if sheet:
            w, h = sheet[0][1].size
            scale = min(1.0, 900 / w)
            tw, th = int(w * scale), int(h * scale)
            out = Image.new("RGB", (tw * len(sheet) + 6 * (len(sheet) - 1), th + 22), (25, 25, 25))
            d = ImageDraw.Draw(out)
            for i, (name, im) in enumerate(sheet):
                out.paste(im.resize((tw, th), Image.LANCZOS), (i * (tw + 6), 22))
                d.text((i * (tw + 6) + 4, 5), name, fill=(255, 255, 0))
            sheet_path = os.path.join(args.out, f"sheet_{view['name'].replace('[', '_').replace(']', '').replace('=', '')}.png")
            out.save(sheet_path)

    print(f"\n{'view':28s} {'variant':10s} {'flips':>7s} {'changed':>8s}")
    for view, name, flips, changed in rows:
        if flips is None:
            print(f"{view:28s} {name:10s}   (failed)")
        else:
            print(f"{view:28s} {name:10s} {flips:7.1f} {changed:7.2f}%")
    print(f"\nsheets in {args.out}/sheet_*.png ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
