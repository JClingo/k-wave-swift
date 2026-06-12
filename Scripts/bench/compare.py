#!/usr/bin/env python3
"""Join Swift and Python benchmark JSON outputs and print a comparison table.

    compare.py swift.json python.json [--metric best|median] [--strict [TOLERANCE]]

Speedup = python_time / swift_time (>1 means Swift is faster). With --strict, exit 1
if any scenario's speedup falls below TOLERANCE (default 1.0, i.e. Swift must be at
least as fast); use e.g. --strict 0.9 to allow 10% slack for run-to-run noise.
"""

import argparse
import json
import sys


def load(path):
    with open(path) as f:
        data = json.load(f)
    return {r["name"]: r for r in data["results"]}, data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("swift_json")
    ap.add_argument("python_json")
    ap.add_argument("--metric", default="median", choices=["best", "median"])
    ap.add_argument("--strict", nargs="?", const=1.0, type=float, default=None,
                    metavar="TOLERANCE")
    args = ap.parse_args()

    swift, swift_meta = load(args.swift_json)
    python, python_meta = load(args.python_json)

    names = [n for n in swift if n in python]
    missing = sorted(set(swift) ^ set(python))
    if missing:
        print(f"note: scenarios present on one side only, skipped: {', '.join(missing)}\n")

    header = (f"swift device={swift_meta.get('device', '?')} vs "
              f"python backend={python_meta.get('backend', '?')}  (metric: {args.metric})")
    print(header)
    rows = [("scenario", "grid", "steps", "swift (s)", "python (s)", "speedup")]
    failures = []
    for n in names:
        s, p = swift[n][args.metric], python[n][args.metric]
        speedup = p / s if s > 0 else float("inf")
        meta = swift[n].get("meta", {})
        rows.append((n, meta.get("grid", ""), meta.get("steps", ""),
                     f"{s:.4f}", f"{p:.4f}", f"{speedup:.2f}x"))
        if args.strict is not None and speedup < args.strict:
            failures.append((n, speedup))

    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for i, row in enumerate(rows):
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
        if i == 0:
            print("  ".join("-" * w for w in widths))

    if failures:
        print(f"\nFAIL: {len(failures)} scenario(s) below {args.strict:.2f}x:")
        for n, sp in failures:
            print(f"  {n}: {sp:.2f}x")
        sys.exit(1)
    if args.strict is not None:
        print(f"\nPASS: all {len(names)} scenarios at or above {args.strict:.2f}x")


if __name__ == "__main__":
    main()
