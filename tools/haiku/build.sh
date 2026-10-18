#!/usr/bin/env bash
set -euo pipefail
LC_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$LC_ROOT"
. tools/haiku/env.sh
export CRAFT_FONTS_DIR=${CRAFT_FONTS_DIR:-"$LC_ROOT/../craft-fonts"}
export CRAFT_FONTS_REQUIRED=1
LC_SHA=$(git rev-parse HEAD 2>/dev/null || echo dev)
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n $(git ls-files --others --exclude-standard) ]]; then
    LC_SHA="$LC_SHA-dirty"
fi
export LIGHTCRAFT_BUILD_SHA=${LIGHTCRAFT_BUILD_SHA:-$LC_SHA}
cargo build --locked --release --target x86_64-unknown-haiku -Zbuild-std=std,panic_abort -j "${JOBS:-8}" -p lightcraft-haiku "$@"
