#!/usr/bin/env python3
"""Track the Dynamic rigid bodies of a live run and summarize how they move.

    python scripts/rb_track.py captures/snapshots/snap_007_2560x1351 --frames 900
    python scripts/rb_track.py <stem> --set rigid_bodies.1.relative_density=0.1 \\
        --set "rigid_bodies.1.position=[-0.65,0.4,-0.28]" --csv drop.csv

Resumes the snapshot (not held), runs `--frames` simulation frames with
RB_DEBUG=1 and parses the per-frame lines the app prints for each Dynamic
body (`rb sub=.. count=.. expected=.. pos=x y z`, in body order) and for
frames with body-body contacts (`rb contacts=N deepest=D`). Per body it
prints where it started and ended, the range of its height, and over the last
two thirds of the run: mean height, its standard deviation, the rms vertical
speed, the submerged fraction and the horizontal drift.

Use it for anything that changes how bodies are driven (forces, contact,
frame order): a body at rest should stay within a millimetre (std 0.2-0.4 mm,
1.5-2 mm/s on snap_007), and LIGHT bodies are the sensitive case:
`relative_density=0.1` rests at ~0.5 mm / 3 mm/s, and goes to tens of mm and
hundreds of mm/s when its forces arrive a frame late. Runs are chaotic:
compare statistics, not trajectories.
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", help="snapshot path without extension (.json + .state)")
    ap.add_argument("--exe", default="target/release/fluid-toy.exe")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--set", action="append", default=[], help="path=value (repeatable)")
    ap.add_argument("--bodies", type=int, help="Dynamic bodies (default: counted from the config + --set)")
    ap.add_argument("--size", default="640x360", help="window size (the render is not what is measured)")
    ap.add_argument("--csv", help="write the tracks: one row per frame, (submerged, x, y, z) per body")
    args = ap.parse_args()

    cmd = [os.path.abspath(args.exe), "--config", args.stem + ".json", "--load-state", args.stem + ".state",
           "--size", args.size, "--exit-after", str(args.frames)]
    for s in args.set:
        cmd += ["--set", s]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, RB_DEBUG="1"), timeout=1800)
    if r.returncode != 0:
        sys.exit((r.stderr or r.stdout)[-800:])

    rows, contacts = [], []
    for line in r.stderr.splitlines():
        m = re.match(r"rb sub=([\d.]+) count=\d+ expected=\d+ pos=(\S+) (\S+) (\S+)", line)
        if m:
            rows.append([float(v) for v in m.groups()])
        m = re.match(r"rb contacts=\d+ deepest=(\S+)", line)
        if m:
            contacts.append(float(m.group(1)))
    if not rows:
        sys.exit("no `rb sub=...` lines: no Dynamic body in this run?")
    bodies = args.bodies or max(1, round(len(rows) / args.frames))
    n = len(rows) // bodies
    track = np.array(rows[: n * bodies]).reshape(n, bodies, 4)
    print(f"{n} frames, {bodies} dynamic bodies, {len(contacts)} contact frames"
          + (f" (deepest overlap {max(contacts) * 1000:.1f} mm)" if contacts else ""))
    for b in range(bodies):
        t = track[:, b]
        tail = t[n // 3:]
        vy = np.diff(t[:, 2]) * 60.0
        print(f"  body {b}: y {t[0, 2]:+.4f} -> {t[-1, 2]:+.4f} (min {t[:, 2].min():+.4f}, max {t[:, 2].max():+.4f}) | "
              f"last 2/3: mean {tail[:, 2].mean():+.4f}, std {tail[:, 2].std() * 1000:.2f} mm, "
              f"|vy| rms {np.sqrt((vy[n // 3:] ** 2).mean()) * 1000:.1f} mm/s, submerged {tail[:, 0].mean():.3f}, "
              f"drift x {t[-1, 1] - t[0, 1]:+.3f} z {t[-1, 3] - t[0, 3]:+.3f}")
    if args.csv:
        np.savetxt(args.csv, track.reshape(n, -1), delimiter=",", fmt="%.5f")


if __name__ == "__main__":
    main()
