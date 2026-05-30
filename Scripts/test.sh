#!/usr/bin/env bash
# Run the test suite. MLX's Metal shaders cannot be compiled by `swift test` (SwiftPM CLI),
# only by xcodebuild, so tests must run through xcodebuild. Requires the Metal Toolchain
# (install once: `xcodebuild -downloadComponent MetalToolchain`).
set -euo pipefail
cd "$(dirname "$0")/.."
exec xcodebuild test -scheme KWave -destination 'platform=OS X' "$@"
