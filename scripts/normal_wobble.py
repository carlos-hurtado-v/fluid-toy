#!/usr/bin/env python3
"""How bumpy is the water surface that refracted rays leave through?

Reads a pixel-probe dump (--probe-line across a mirror or refraction of the
free surface, see scripts/probe_decode.py) and, for the first water exit of
each probed pixel, measures how the exit normal wobbles from pixel to pixel
after removing its smooth trend along the line:

    fluid-toy --config snap.json --load-state snap.state --hold --size WxH \
        --probe-line 1800,772,1800,829 --capture 2 --out captures/wobble
    python scripts/normal_wobble.py captures/wobble/frame_00002_probe.json

  wobble   rms / peak angle between each exit normal and the trend (a
           quadratic fit per component along the line), degrees
  period   dominant period of that wobble, in probed pixels
  rough    rms second difference of the normal's tilt (degrees): pixel-to-
           pixel jitter, blind to slow waves
  jump     median distance between neighbouring pixels' final screen lookups
           (px): what the wobble does to the image (a grazing mirror turns
           0.7 deg into ~25 px)

A grazing mirror of the under-surface needs well under ~0.2 deg of wobble to
look like a mirror. Several dumps can be given to compare settings.
"""
import argparse
import json
import sys

import numpy as np


def first_exits(path):
    dump = json.load(open(path))
    best = {}
    for frag in dump["fragments"]:
        key = tuple(frag["pixel"])
        if key not in best or frag["depth"] < best[key]["depth"]:
            best[key] = frag
    order = [tuple(p) for p in dump["pixels"]]
    normals, lookups = [], []
    for key in order:
        if key not in best:
            continue
        events = best[key]["events"]
        # First EXIT_END (16) through the water surface (not a wall, body or blocked)
        end = next((e for e in events if round(e[0]) == 16), None)
        if end is None or int(round(end[7])) != 0:
            continue
        normals.append(end[4:7])
        result = next((e for e in events if round(e[0]) == 50), None)
        lookups.append(result[4:6] if result is not None else [-1.0, -1.0])
    return np.array(normals), np.array(lookups), dump["size"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+", help="frame_NNNNN_probe.json files")
    args = ap.parse_args()
    for path in args.dumps:
        n, uv, size = first_exits(path)
        if len(n) < 8:
            print(f"{path}: only {len(n)} surface exits on the probe line (need 8+)")
            continue
        x = np.arange(len(n))
        trend = np.stack([np.polyval(np.polyfit(x, n[:, c], 2), x) for c in range(3)], axis=1)
        trend /= np.linalg.norm(trend, axis=1, keepdims=True)
        dev = np.degrees(np.arccos(np.clip((n * trend).sum(axis=1), -1.0, 1.0)))
        # Signed wobble along the direction the normals vary most, for the period
        resid = n - trend
        axis = np.linalg.svd(resid - resid.mean(axis=0), full_matrices=False)[2][0]
        signed = resid @ axis
        spectrum = np.abs(np.fft.rfft(signed - signed.mean()))
        k = int(np.argmax(spectrum[1:])) + 1
        tilt = np.degrees(np.arcsin(np.clip(n @ axis, -1.0, 1.0)))
        rough = np.diff(tilt, 2).std()
        on_screen = (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
        jump = "n/a"
        if on_screen.sum() > 2:
            px = uv[on_screen] * np.array(size)
            jump = f"{np.median(np.linalg.norm(np.diff(px, axis=0), axis=1)):.0f} px"
        print(f"{path}\n  {len(n)} exits  wobble rms {np.sqrt((dev ** 2).mean()):.2f} deg  peak {dev.max():.2f} deg  "
              f"period ~{len(n) / k:.0f} px  rough {rough:.3f} deg  jump {jump}")


if __name__ == "__main__":
    sys.exit(main())
