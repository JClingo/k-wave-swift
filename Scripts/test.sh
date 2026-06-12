#!/usr/bin/env bash
# Run the test suite. MLX's Metal shaders cannot be compiled by `swift test` (SwiftPM CLI),
# only by xcodebuild, so tests must run through xcodebuild. Requires the Metal Toolchain
# (install once: `xcodebuild -downloadComponent MetalToolchain`).
set -euo pipefail
cd "$(dirname "$0")/.."
# KWave-Package is the auto-generated scheme carrying the test action now that the
# package also vends the kwave-bench executable product.
exec xcodebuild test -scheme KWave-Package -destination 'platform=OS X' "$@"
