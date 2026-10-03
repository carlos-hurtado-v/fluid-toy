#!/usr/bin/env python3
"""Size budgets for source files and functions, with a ratchet.

    python scripts/check_size.py                # check src/ and scripts/
    python scripts/check_size.py src/app.rs ... # check only these files
    python scripts/check_size.py --list         # the 25 largest files and functions
    python scripts/check_size.py --update       # re-record the baseline

Budgets (raw lines, comments and blanks included):

    Rust file      {rs_file}      Rust function   {rs_fn}
    WGSL file      {wgsl_file}      WGSL function   {wgsl_fn}
    Python file    {py_file}

Anything over budget fails, except what scripts/size_baseline.json records:
files and functions that were already over when the budget was introduced.
Those are frozen at their recorded size: they may shrink, never grow. So a
large file cannot keep growing, and a new one cannot appear unnoticed.

When the check fails, move code out instead of raising the number: a new GPU
pass gets its own module (see render/wall_bound.rs, render/mc_*.rs), a new
shader topic its own file (src/shaders/mc_render/), a new frame phase its own
method (src/app/). `--update` is for the rare case where the growth is the
right call; say why in the commit.

Exit code: 0 clean, 1 violations (2 with --hook, so Claude Code shows the
report to the model after an edit).
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "scripts", "size_baseline.json")
SCAN_DIRS = ("src", "scripts")

FILE_BUDGET = {".rs": 1000, ".wgsl": 500, ".py": 600}
FN_BUDGET = {".rs": 200, ".wgsl": 200}

FN_START = {
    ".rs": re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)"),
    ".wgsl": re.compile(r"^\s*fn\s+(\w+)"),
}
CONTAINER = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:impl|mod|trait)\b")


def strip_code(line, state):
    """`line` without comments, string and char literals (so braces in them
    do not count). `state` carries an open block comment or raw string across lines."""
    out = []
    i, n = 0, len(line)
    while i < n:
        if state.get("block"):
            j = line.find("*/", i)
            if j < 0:
                return "".join(out)
            state["block"] = False
            i = j + 2
            continue
        if state.get("raw") is not None:
            end = '"' + "#" * state["raw"]
            j = line.find(end, i)
            if j < 0:
                return "".join(out)
            state["raw"] = None
            i = j + len(end)
            continue
        if state.get("string"):
            while i < n:
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == '"':
                    state["string"] = False
                    i += 1
                    break
                i += 1
            continue
        c = line[i]
        if line.startswith("//", i):
            break
        if line.startswith("/*", i):
            state["block"] = True
            i += 2
            continue
        m = re.match(r'b?r(#*)"', line[i:])
        if m and (i == 0 or not (line[i - 1].isalnum() or line[i - 1] == "_")):
            state["raw"] = len(m.group(1))
            i += m.end()
            continue
        if c == '"':
            state["string"] = True
            i += 1
            continue
        m = re.match(r"'(?:\\.[^']*|[^'\\])'", line[i:])
        if c == "'" and m:
            i += m.end()
            continue
        out.append(c)
        i += 1
    return "".join(out)


def functions(lines, ext):
    """(name, first line number, line count) of every function, by brace matching.
    Names are qualified by position in the file only: two functions with the
    same name in one file are told apart by their order (`new`, `new#2`)."""
    start_re = FN_START.get(ext)
    if start_re is None:
        return []
    found, seen = [], {}
    open_fns = []  # (name, start line, depth at which its body closes)
    depth, state = 0, {}
    pending = None
    for number, raw in enumerate(lines, 1):
        code = strip_code(raw, state)
        m = start_re.match(code)
        if m and not CONTAINER.match(code):
            pending = (m.group(1), number)
        for ch in code:
            if ch == "{":
                if pending:
                    open_fns.append((pending[0], pending[1], depth))
                    pending = None
                depth += 1
            elif ch == "}":
                depth -= 1
                if open_fns and open_fns[-1][2] == depth:
                    name, first, _ = open_fns.pop()
                    seen[name] = seen.get(name, 0) + 1
                    label = name if seen[name] == 1 else f"{name}#{seen[name]}"
                    found.append((label, first, number - first + 1))
            elif ch == ";" and pending and depth == (open_fns[-1][2] + 1 if open_fns else depth):
                pending = None  # a declaration without a body (trait method)
    return sorted(found, key=lambda f: f[1])


def measure(path):
    ext = os.path.splitext(path)[1]
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return len(lines), functions(lines, ext)


def source_files():
    for top in SCAN_DIRS:
        for folder, _, names in os.walk(os.path.join(ROOT, top)):
            for name in sorted(names):
                if os.path.splitext(name)[1] in FILE_BUDGET:
                    yield os.path.join(folder, name)


def rel(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


def over_budget(paths):
    """{key: size} for every file and function over its budget."""
    over = {}
    for path in paths:
        ext = os.path.splitext(path)[1]
        size, fns = measure(path)
        if size > FILE_BUDGET[ext]:
            over[rel(path)] = size
        for name, _, length in fns:
            if ext in FN_BUDGET and length > FN_BUDGET[ext]:
                over[f"{rel(path)}::{name}"] = length
    return over


def budget_of(key):
    path, _, fn = key.partition("::")
    ext = os.path.splitext(path)[1]
    return FN_BUDGET[ext] if fn else FILE_BUDGET[ext]


def load_baseline():
    if not os.path.exists(BASELINE):
        return {}
    with open(BASELINE, encoding="utf-8") as f:
        return json.load(f)["frozen"]


def main():
    doc = __doc__.format(rs_file=FILE_BUDGET[".rs"], rs_fn=FN_BUDGET[".rs"], wgsl_file=FILE_BUDGET[".wgsl"],
                         wgsl_fn=FN_BUDGET[".wgsl"], py_file=FILE_BUDGET[".py"])
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="check only these files (default: everything)")
    ap.add_argument("--update", action="store_true", help="re-record the baseline from the current tree")
    ap.add_argument("--list", action="store_true", help="print the largest files and functions")
    ap.add_argument("--hook", action="store_true",
                    help="Claude Code PostToolUse hook: read the edited file from the hook JSON on stdin, "
                         "report to stderr, exit 2 on violations")
    args = ap.parse_args()

    if args.list:
        sizes, fn_sizes = [], []
        for path in source_files():
            size, fns = measure(path)
            sizes.append((size, rel(path)))
            fn_sizes += [(length, f"{rel(path)}::{name}") for name, _, length in fns]
        for title, rows in (("files", sizes), ("functions", fn_sizes)):
            print(f"largest {title}:")
            for size, name in sorted(rows, reverse=True)[:25]:
                print(f"  {size:5d}  {name}")
        return 0

    if args.update:
        frozen = over_budget(source_files())
        with open(BASELINE, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"_comment": "Files and functions over the budgets in scripts/check_size.py, frozen at "
                                   "these sizes: they may shrink, never grow. Regenerate with "
                                   "`python scripts/check_size.py --update`.",
                       "frozen": dict(sorted(frozen.items()))}, f, indent=2)
            f.write("\n")
        print(f"baseline: {len(frozen)} frozen entries -> {rel(BASELINE)}")
        return 0

    paths = [os.path.abspath(p) for p in args.files]
    if args.hook:
        try:
            event = json.load(sys.stdin)
            edited = (event.get("tool_input") or {}).get("file_path") or ""
        except (ValueError, AttributeError):
            edited = ""
        paths = [os.path.abspath(edited)] if edited else []
        if not paths:
            return 0
    checking_all = not paths
    if checking_all:
        paths = list(source_files())
    else:
        inside = os.path.normcase(os.path.join(ROOT, ""))
        paths = [p for p in paths if os.path.splitext(p)[1] in FILE_BUDGET and os.path.exists(p)
                 and os.path.normcase(p).startswith(inside) and rel(p).split("/")[0] in SCAN_DIRS]

    frozen = load_baseline()
    over = over_budget(paths)
    problems, notes = [], []
    for key, size in sorted(over.items()):
        if key not in frozen:
            problems.append(f"{key}: {size} lines, budget {budget_of(key)}")
        elif size > frozen[key]:
            problems.append(f"{key}: {size} lines, frozen at {frozen[key]} (already over the budget of "
                            f"{budget_of(key)}: it may shrink, not grow)")
        elif size < frozen[key]:
            notes.append(f"{key}: now {size} (was {frozen[key]})")
    if checking_all:
        checked = {rel(p) for p in paths}
        for key in sorted(frozen):
            if key not in over and key.partition("::")[0] in checked:
                notes.append(f"{key}: no longer over budget")

    out = sys.stderr if args.hook else sys.stdout
    if problems:
        print("Size budget exceeded:", file=out)
        for p in problems:
            print("  " + p, file=out)
        print("Move code out rather than growing the file or function: a new pass, shader topic or frame phase "
              "gets its own module, file or method (see scripts/check_size.py). If the growth is right, "
              "`python scripts/check_size.py --update` re-records the baseline.", file=out)
        return 2 if args.hook else 1
    if notes and not args.hook:
        print("Shrunk since the baseline (run --update to lock it in):")
        for n in notes:
            print("  " + n)
    if not args.hook:
        print(f"size budgets ok ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
