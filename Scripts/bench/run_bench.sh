#!/usr/bin/env bash
# Build and run the Swift and Python k-Wave benchmarks, then print the comparison.
#
#   Scripts/bench/run_bench.sh [--repeats N] [--filter substr] [--strict [TOL]] [--skip-build]
#
# The Swift bench must be built with xcodebuild (SwiftPM CLI cannot bundle MLX's metallib).
# The Python bench runs in the sibling ../k-wave-python checkout via uv.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYREPO="../k-wave-python"
OUTDIR=".build/bench"
DD=".build/bench-dd"
REPEATS=3
FILTER=""
STRICT=()
BUILD=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeats) REPEATS="$2"; shift 2 ;;
    --filter) FILTER="$2"; shift 2 ;;
    --strict)
      if [[ "${2:-}" =~ ^[0-9.]+$ ]]; then STRICT=(--strict "$2"); shift 2
      else STRICT=(--strict); shift; fi ;;
    --skip-build) BUILD=0; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

[[ -d "$PYREPO" ]] || { echo "error: $PYREPO not found (expected k-wave-python checkout)" >&2; exit 1; }
mkdir -p "$OUTDIR"

if [[ "$BUILD" == 1 ]]; then
  echo "== building kwave-bench (xcodebuild Release) =="
  xcodebuild build -scheme kwave-bench -destination 'platform=OS X' \
    -configuration Release -derivedDataPath "$DD" -quiet
fi
BIN="$DD/Build/Products/Release/kwave-bench"

FILTER_ARGS=()
[[ -n "$FILTER" ]] && FILTER_ARGS=(--filter "$FILTER")

echo "== swift bench =="
"$BIN" --repeats "$REPEATS" ${FILTER_ARGS[@]+"${FILTER_ARGS[@]}"} --out "$OUTDIR/swift.json"

echo "== python bench (numpy backend) =="
(cd "$PYREPO" && uv run --no-sync python "$OLDPWD/Scripts/bench/bench_python.py" \
  --repeats "$REPEATS" ${FILTER_ARGS[@]+"${FILTER_ARGS[@]}"} --out "$OLDPWD/$OUTDIR/python.json")

echo
python3 Scripts/bench/compare.py "$OUTDIR/swift.json" "$OUTDIR/python.json" ${STRICT[@]+"${STRICT[@]}"}
