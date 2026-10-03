#!/usr/bin/env python3
"""Decode a pixel-probe dump: what the water shader's refraction did, step by
step, at probed pixels.

Record (on a held state, so the frame is exactly reproducible):

    fluid-toy --config snap.json --load-state snap.state --hold --size WxH \\
        --probe-line 688,760,688,823 --capture 2 --out captures/probe

then

    python scripts/probe_decode.py captures/probe/frame_00002_probe.json
    python scripts/probe_decode.py dump.json --pixel 688,775
    python scripts/probe_decode.py dump.json --diff 688,775 688,776
    python scripts/probe_decode.py dump.json --check-paths paths_capture.png

Default output, one row per probed pixel (in probe order):
  route + final lookup (the Paths view ids), then one token per water_exit
  call (the initial exit and each mirror bounce):
      k05o>S*   first in-water sample that found an event = 5 of 16
                (a ^ before it per back-face silhouette the trace carried
                on past, rendering.mc_silhouette_exit=Continue);
                o = left the water (back face crossed / no water behind),
                h = left the water out of sight behind a body (crossing
                estimated from the gap closing at the body's outline),
                u = left through a front face (rendering.mc_front_face_exit),
                X = opaque surface,
                - = nothing up to the box; then the exit: S = through the
                back face (free surface / drop), W = container wall/floor
                plane, R = back-face crossing rejected (its normal faced the
                ray) so the box bounded it, B = blocked, F = floor contact
                (counts as blocked), Y = body, U = through a front face;
                * = reflected again (TIR)
followed by the FIRST DIVERGENCE between each pair of neighbouring pixels
whose route differs: which water_exit call, which sample, which test, and the
depth margins either side. --diff prints both traces around that point.

A pixel may be shaded by several fragments: triangles meeting inside it
under MSAA (each covers some samples, and the resolve blends them) or
triangles hidden behind it. The nearest (smallest depth) is summarized;
--pixel --all-fragments lists the rest. Event ids mirror the PRB_* constants in
src/shaders/mc_render/debug_records.wgsl: keep EVENTS below in sync with them.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True  # no scripts/__pycache__ from the import below
from debug_decode import ENDS, EXITS, PATHS  # noqa: E402  (shared id tables)

# tag -> name (fields documented in mc_render/debug_records.wgsl next to PRB_*)
EVENTS = {
    1: "FRAG", 2: "NORMAL", 3: "REFRACT_IN", 4: "SECOND",
    10: "EXIT_BEGIN", 11: "EXIT_BOX", 12: "TRACE", 13: "TRACE_REFINE", 14: "TRACE_END",
    15: "EXIT_CROSS", 16: "EXIT_END", 17: "EXIT_INSIDE", 18: "SILHOUETTE", 19: "HIDDEN",
    20: "BOUNCE", 21: "VTRACE", 22: "VTRACE_REFINE", 23: "BODY",
    30: "MARCH_BEGIN", 31: "MARCH", 32: "MARCH_REFINE", 33: "MARCH_END",
    40: "SCENE", 41: "BACKDROP", 42: "LOOKUP",
    50: "RESULT", 51: "COLOR",
}
KIND = {-1: "offscreen", 0: "in water", 1: "out of water", 2: "opaque",
        3: "out through a front face"}


def v3(x):
    return "(" + ", ".join(f"{c:+.4f}" for c in x) + ")"


def fmt_event(e):
    tag = int(round(e[0]))
    a, b, c = e[1:4], e[4:7], e[7]
    name = EVENTS.get(tag, f"?{tag}")
    if tag == 1:
        return f"{name:12s} pos={v3(a)} mesh_n={v3(b)} depth={c:.7f}"
    if tag == 2:
        return f"{name:12s} n={v3(a)} on_wall={int(b[0])} back_depth={c:.7f}"
    if tag == 3:
        return f"{name:12s} t1={v3(a)} bg_depth={b[0]:.7f} back_depth={b[1]:.7f} pool={int(b[2])}"
    if tag == 4:
        tir = " TIR" if abs(a[0]) + abs(a[1]) + abs(a[2]) < 1e-6 else ""
        return f"{name:12s} t2={v3(a)}{tir} n_exit={v3(b)} water_path={c:.4f}"
    if tag == 10:
        return f"{name:12s} origin={v3(a)} dir={v3(b)} box_dist={c:.4f}"
    if tag == 11:
        return f"{name:12s} box_n={v3(a)} t_body={b[0]:.4f} trace_max={b[1]:.4f}"
    if tag in (12, 13):
        if b[2] < -0.5:
            return f"{name:12s} s={c:.4f} uv=({a[0]:.5f}, {a[1]:.5f}) OFFSCREEN"
        kind = KIND.get(int(round(b[2])), "?")
        back = f"back={b[1]:.7f} z-back={a[2] - b[1]:+.2e}" if b[1] >= 0 else "back=  (not read)"
        front = ""
        if len(e) > 8 and e[8] > 0:
            # front margin in units of (1 - front): FRONT_EXIT_REL in mc_render/trace.wgsl
            front = f" front={e[8]:.7f} (front-z)/(1-front)={(e[8] - a[2]) / max(1 - e[8], 1e-6):+.4f}"
        return (f"{name:12s} s={c:.4f} uv=({a[0]:.5f}, {a[1]:.5f}) z={a[2]:.7f} "
                f"bg={b[0]:.7f} z-bg={a[2] - b[0]:+.2e} {back}{front} -> {kind}")
    if tag in (21, 22):
        kind = KIND.get(int(round(b[2])), "?")
        # (no depth test: off screen, or nothing in the depth buffer to test against)
        bg = f"bg={b[1]:.7f}" if b[1] >= 0 else "bg untested"
        return f"{name:12s} s={c:.4f} p={v3(a)} field/iso={b[0]:.4f} {bg} -> {kind}"
    if tag == 23:
        return f"{name:12s} hit={v3(a)} normal={v3(b)} body #{int(round(c))}: shaded at the hit"
    if tag == 14:
        return (f"{name:12s} kind={KIND.get(int(round(a[0])), '?')} dist_in={a[1]:.4f} "
                f"dist={a[2]:.4f} uv=({b[0]:.5f}, {b[1]:.5f}) max={c:.4f}")
    if tag == 18:
        action = {0: "taken as an exit (old)", 1: "CONTINUE behind the nearer layer"}.get(int(round(b[2])), "?")
        return (f"{name:12s} back in={a[0]:.7f} back out={a[1]:.7f} (jump {a[0] - a[1]:.2e}) "
                f"bracket=[{b[0]:.4f}, {b[1]:.4f}] -> {action}")
    if tag == 19:
        if b[0] < 0:
            why = "gap not closing" if a[2] <= 0 else "no rate"
            return f"{name:12s} outline s={a[0]:.4f} gap={a[1]:+.4f} m rate={a[2]:+.4f} -> {why}: carried on in the water"
        verdict = {1: "CROSSING behind the body", 2: "the wall or body comes first: carried on in the water"}.get(
            int(round(c)), "crossing back in sight: what is seen decides")
        return (f"{name:12s} outline s={a[0]:.4f} gap={a[1]:+.4f} m rate={a[2]:+.4f} "
                f"cross s={b[0]:.4f} normal uv=({b[1]:.5f}, {b[2]:.5f}) -> {verdict}")
    if tag == 15:
        has_n = int(round(c)) % 2
        accepted = int(round(c)) % 4 >= 2
        layer = "front" if int(round(c)) >= 4 else "back"
        return (f"{name:12s} {layer}_n={v3(a)} (written={has_n}) flattened={v3(b)} "
                f"{'ACCEPTED' if accepted else 'REJECTED (faces the ray)'}")
    if tag == 16:
        f = int(round(c))
        flags = [n for bit, n in ((1, "wall"), (2, "blocked"), (4, "body")) if f & bit]
        return f"{name:12s} point={v3(a)} normal={v3(b)} {'+'.join(flags) or 'back face'}"
    if tag == 17:
        return f"{name:12s} inside={v3(a)} blocked_uv=({b[0]:.5f}, {b[1]:.5f}) floor_contact={int(b[2])}"
    if tag == 20:
        tir = abs(a[0]) + abs(a[1]) + abs(a[2]) < 1e-6
        return f"{name:12s} #{int(c)} in={v3(b)} " + ("TIR: reflects again" if tir else f"out={v3(a)}")
    if tag == 30:
        return f"{name:12s} origin={v3(a)} dir={v3(b)} reach={c:.4f}"
    if tag in (31, 32):
        if b[0] < -0.5:
            return f"{name:12s} s={c:.4f} OFFSCREEN"
        return (f"{name:12s} s={c:.4f} uv=({a[0]:.5f}, {a[1]:.5f}) z={a[2]:.7f} bg={b[0]:.7f} "
                f"z-bg={a[2] - b[0]:+.2e} -> {'BEHIND' if b[1] > 0.5 else 'in front'}")
    if tag == 33:
        return f"{name:12s} uv=({a[0]:.5f}, {a[1]:.5f}) hit={int(a[2])} lo={b[0]:.4f} hi={b[1]:.4f} t_body={b[2]:.4f}"
    if tag == 40:
        return f"{name:12s} exit_uv=({a[0]:.5f}, {a[1]:.5f}) bg_depth={a[2]:.7f} dir={v3(b)}"
    if tag == 41:
        end = ENDS.get(int(round(c)), "?")
        return f"{name:12s} vanishing_uv=({a[0]:.5f}, {a[1]:.5f}) on_screen={int(a[2])} dir={v3(b)} -> {end}"
    if tag == 42:
        return (f"{name:12s} uv=({a[0]:.5f}, {a[1]:.5f}) bg_depth={a[2]:.7f} front={b[0]:.7f} "
                f"{'accepted' if b[1] > 0.5 else 'REJECTED: in front of the water -> straight'}")
    if tag == 50:
        return (f"{name:12s} path={int(a[0])} ({PATHS.get(int(a[0]), '?')}) end={int(a[1])} "
                f"({ENDS.get(int(a[1]), '?')}) bounces={int(a[2])} uv=({b[0]:.5f}, {b[1]:.5f}) "
                f"exit={int(b[2])} mirror={int(c)}")
    if tag == 51:
        return f"{name:12s} refracted={v3(a)} color={v3(b)} fresnel={c:.4f}"
    return f"{name:12s} a={v3(a)} b={v3(b)} c={c:.5f}"


def decision(e):
    """The branch an event took, for divergence finding (None = data only)"""
    tag = int(round(e[0]))
    if tag in (12, 13, 21, 22):
        return int(round(e[6]))            # trace kind (-1 offscreen)
    if tag == 15:
        return int(round(e[7])) % 4 >= 2     # crossing accepted
    if tag == 18:
        return int(round(e[6]))              # silhouette action
    if tag == 19:
        return int(round(e[7]))              # crossing estimated behind a body
    if tag == 16:
        return int(round(e[7]))              # exit flags
    if tag == 17:
        return int(round(e[6]))              # floor contact
    if tag == 20:
        return abs(e[1]) + abs(e[2]) + abs(e[3]) < 1e-6   # TIR again
    if tag in (31, 32):
        return round(e[5])                   # behind (or -1 offscreen)
    if tag == 33:
        return int(round(e[3]))              # march hit
    if tag == 41:
        return (int(round(e[3])), int(round(e[7])))
    if tag == 42:
        return e[5] > 0.5
    if tag == 4:
        return abs(e[1]) + abs(e[2]) + abs(e[3]) < 1e-6   # TIR at the first exit
    if tag == 2:
        return int(round(e[4]))              # on_wall snap
    return None


def summarize(events):
    """Route + one token per water_exit call"""
    s = {"path": 0, "end": 0, "bounces": 0, "exits": []}
    cur = None
    k = 0
    for e in events:
        tag = int(round(e[0]))
        if tag == 10:
            cur = {"k": None, "kind": "-", "exit": "?", "tir": False, "skips": 0}
            s["exits"].append(cur)
            k = 0
        elif tag in (12, 21) and cur is not None:
            k += 1
            if cur["k"] is None and e[6] > 0.5:
                cur["k"] = k
                cur["kind"] = {1: "o", 2: "X", 3: "u"}.get(int(round(e[6])), "?")
        elif tag == 18 and cur is not None:
            action = int(round(e[6]))
            if action == 1:
                cur["skips"] += 1
                cur["k"], cur["kind"] = None, "-"   # the next event counts
        elif tag == 19 and cur is not None and int(round(e[7])) == 1:
            cur["k"], cur["kind"] = k, "h"
        elif tag == 15 and cur is not None:
            cur["crossed"] = int(round(e[7])) % 4 >= 2
            cur["front"] = int(round(e[7])) >= 4
        elif tag == 16 and cur is not None:
            f = int(round(e[7]))
            if f & 4:
                cur["exit"] = "Y"
            elif f & 2:
                cur["exit"] = "B"
            elif f & 1:
                cur["exit"] = "R" if cur.get("crossed") is False else "W"
            else:
                cur["exit"] = "U" if cur.get("front") else "S"
        elif tag == 17 and cur is not None and e[6] > 0.5:
            cur["exit"] = "F"
        elif tag in (20, 4) and cur is not None:
            cur["tir"] = abs(e[1]) + abs(e[2]) + abs(e[3]) < 1e-6
        elif tag == 50:
            s["path"], s["end"], s["bounces"] = int(e[1]), int(e[2]), int(e[3])
    return s


def exit_token(x):
    k = f"k{x['k']:02d}" if x["k"] is not None else "k--"
    return f"{'^' * x['skips']}{k}{x['kind']}>{x['exit']}{'*' if x['tir'] else ' '}"


def locate(events, j, exact=True):
    """Where event j sits: water_exit call + sample (exact=False drops the
    sample number, grouping one cause across neighbouring sample positions)"""
    call, sample, refine = 0, 0, 0
    for e in events[: j + 1]:
        tag = int(round(e[0]))
        if tag == 10:
            call += 1
            sample, refine = 0, 0
        elif tag in (12, 21):
            sample += 1
        elif tag in (13, 22):
            refine += 1
    tag = int(round(events[j][0]))
    where = f"water_exit #{call}" if call else "before any water_exit"
    if not exact:
        return f"{where}, {EVENTS.get(tag, tag)}"
    if tag in (12, 21):
        return f"{where}, coarse sample {sample}"
    if tag in (13, 22):
        return f"{where}, bisection step {refine}"
    return f"{where}, {EVENTS.get(tag, tag)}"


def first_divergence(ea, eb):
    for j in range(min(len(ea), len(eb))):
        ta, tb = int(round(ea[j][0])), int(round(eb[j][0]))
        if ta != tb:
            return j, "next step differs"
        if decision(ea[j]) != decision(eb[j]):
            return j, "decision differs"
    if len(ea) != len(eb):
        return min(len(ea), len(eb)), "one trace ends"
    return None, None


def visible(frags):
    return min(frags, key=lambda f: f["depth"])


def parse_xy(text):
    x, y = (int(v) for v in text.split(","))
    return x, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--pixel", help="x,y: full event trace of this pixel")
    ap.add_argument("--diff", nargs=2, metavar="X,Y", help="side-by-side traces of two pixels")
    ap.add_argument("--all-fragments", action="store_true",
                    help="--pixel: also show the other fragments that shaded it")
    ap.add_argument("--context", type=int, default=6, help="--diff: events shown before the divergence")
    ap.add_argument("--check-paths", metavar="PNG",
                    help="Paths debug-view capture of the same frame: check the probe agrees per pixel")
    args = ap.parse_args()

    d = json.load(open(args.dump))
    by_pixel = {}
    for f in d["fragments"]:
        by_pixel.setdefault(tuple(f["pixel"]), []).append(f)
    order = [tuple(p) for p in d["pixels"]]
    print(f"{args.dump}: frame {d['frame']}, {d['size'][0]}x{d['size'][1]}, {d['render_mode']}, "
          f"{len(d['fragments'])} fragments on {len(by_pixel)}/{len(order)} pixels")
    if d.get("slots_dropped"):
        print(f"WARNING: {d['slots_dropped']} fragments dropped (slot buffer full)")
    if any(f["overflow"] for f in d["fragments"]):
        print("WARNING: some fragments ran out of event slots (traces truncated)")

    if args.pixel:
        p = parse_xy(args.pixel)
        frags = sorted(by_pixel.get(p, []), key=lambda f: f["depth"])
        if not frags:
            sys.exit(f"no fragment shaded pixel {p} (not water, or not probed)")
        for i, f in enumerate(frags if args.all_fragments else frags[:1]):
            label = "nearest" if i == 0 else "also shaded: MSAA edge share or hidden"
            print(f"\npixel {p} fragment {i} ({label}, depth {f['depth']:.7f}, {len(f['events'])} events)")
            for j, e in enumerate(f["events"]):
                print(f"  {j:3d} {fmt_event(e)}")
        return

    if args.diff:
        pa, pb = parse_xy(args.diff[0]), parse_xy(args.diff[1])
        for p in (pa, pb):
            if p not in by_pixel:
                sys.exit(f"no fragment shaded pixel {p}")
        ea, eb = visible(by_pixel[pa])["events"], visible(by_pixel[pb])["events"]
        j, why = first_divergence(ea, eb)
        if j is None:
            print(f"\n{pa} and {pb} take identical decisions ({len(ea)} events)")
            j = len(ea)
        else:
            print(f"\nfirst divergence: event {j} ({why}) at {locate(ea, j)}")
        lo = max(0, j - args.context)
        for k in range(lo, max(len(ea), len(eb))):
            mark = ">>" if k == j else "  "
            for p, ev in ((pa, ea), (pb, eb)):
                text = fmt_event(ev[k]) if k < len(ev) else "(end)"
                print(f"{mark}{k:3d} {p[0]},{p[1]}  {text}")
            print()
        return

    # Line / multi-pixel summary
    print(f"\n{'pixel':>10} frags  route -> lookup                       water_exit calls (see --help)")
    summaries = {}
    for p in order:
        frags = by_pixel.get(p)
        if not frags:
            print(f"{p[0]:>5},{p[1]:<4}    0  (no water fragment)")
            continue
        s = summarize(visible(frags)["events"])
        summaries[p] = s
        route = f"{s['path']:2d} {PATHS.get(s['path'], '?')[:22]:22s} -> {s['end']} {ENDS.get(s['end'], '?')[:12]:12s}"
        tokens = " ".join(exit_token(x) for x in s["exits"])
        print(f"{p[0]:>5},{p[1]:<4} {len(frags):4d}  {route}  {tokens}")

    # Where do neighbouring pixels with different routes part ways?
    flips = Counter()
    examples = {}
    for pa, pb in zip(order[:-1], order[1:]):
        if pa not in summaries or pb not in summaries:
            continue
        if (summaries[pa]["path"], summaries[pa]["end"]) == (summaries[pb]["path"], summaries[pb]["end"]):
            continue
        ea, eb = visible(by_pixel[pa])["events"], visible(by_pixel[pb])["events"]
        j, why = first_divergence(ea, eb)
        if j is None:
            continue
        tag = EVENTS.get(int(round(ea[j][0])), "?") if j < len(ea) else "end"
        key = f"{locate(ea, j, exact=False)}: {why}"
        flips[key] += 1
        examples.setdefault(key, (pa, pb, j))
    if flips:
        print(f"\nroute flips between neighbours: {sum(flips.values())}; first divergence:")
        for key, n in flips.most_common():
            pa, pb, j = examples[key]
            ea, eb = visible(by_pixel[pa])["events"], visible(by_pixel[pb])["events"]
            print(f"  {n:3d} x  {key}   e.g. --diff {pa[0]},{pa[1]} {pb[0]},{pb[1]} ({locate(ea, j)})")
            if j < len(ea):
                print(f"         {pa[0]},{pa[1]}: {fmt_event(ea[j])}")
            if j < len(eb):
                print(f"         {pb[0]},{pb[1]}: {fmt_event(eb[j])}")

    if args.check_paths:
        from PIL import Image
        import numpy as np
        from debug_decode import match_ids
        img = np.asarray(Image.open(args.check_paths).convert("RGB")).astype(int)
        path = match_ids(img[..., 0], 16, PATHS.keys())
        end = match_ids(img[..., 1], 8, ENDS.keys())
        agree, total, mismatches = 0, 0, []
        for p, s in summaries.items():
            pv, ev = path[p[1], p[0]], end[p[1], p[0]]
            if pv < 0 or ev < 0:
                continue  # mixed / not decodable in the capture
            total += 1
            if (pv, ev) == (s["path"], s["end"]):
                agree += 1
            else:
                mismatches.append((p, (pv, ev), (s["path"], s["end"])))
        print(f"\nPaths view cross-check: {agree}/{total} decodable pixels agree")
        for p, view, probe in mismatches[:10]:
            print(f"  {p}: view path/end {view}, probe {probe}")


if __name__ == "__main__":
    main()
