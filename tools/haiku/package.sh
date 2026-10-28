#!/usr/bin/env bash
# Cross-build and package entirely on Linux with Haiku's host tools.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
. tools/haiku/env.sh
TOOLS=${HAIKU_HOST_TOOLS:-/mnt/HaikuWork/build/fw13-x86_64/objects/linux/x86_64/release/tools}
RES="$ROOT/apps/lightcraft-haiku/resources"
OUT="$ROOT/artifacts/haiku"
VERSION=$(awk '$1 == "version" { print $2; exit }' "$RES/LightCraft.PackageInfo")
if [[ ${1:-} != --no-build ]]; then bash tools/haiku/build.sh; fi
mkdir -p "$OUT"
mkdir -p "$ROOT/target/tmp"
STAGE=$(mktemp -d "$ROOT/target/tmp/lightcraft-pkg-XXXXXX")
trap 'rm -rf -- "$STAGE"' EXIT
DOCS="$STAGE/documentation/packages/lightcraft"
mkdir -p "$STAGE/bin" "$STAGE/apps" "$STAGE/data/deskbar/menu/Applications" "$STAGE/boot/post-install" "$DOCS"
"$HAIKU_CROSS-strip" -o "$STAGE/apps/LightCraft" "$CARGO_TARGET_DIR/x86_64-unknown-haiku/release/LightCraft"
(cd "$RES" && "$TOOLS/rc/rc" -o "$STAGE/LightCraft.rsrc" LightCraft.rdef)
"$TOOLS/xres" -o "$STAGE/apps/LightCraft" "$STAGE/LightCraft.rsrc"
"$TOOLS/resattr/resattr" -O -o "$STAGE/apps/LightCraft" "$STAGE/LightCraft.rsrc"
rm "$STAGE/LightCraft.rsrc"
cp "$RES/LightCraft.PackageInfo" "$STAGE/.PackageInfo"
cp "$RES/lightcraft-post-install.sh" "$STAGE/boot/post-install/lightcraft.sh"
chmod 755 "$STAGE/boot/post-install/lightcraft.sh" "$STAGE/apps/LightCraft"
cp README.md LICENSE-MIT LICENSE-APACHE NOTICE docs/haiku.md docs/haiku-validation.json "$DOCS/"
# Redistribute the license texts for the assets embedded by the upstream editor.
while IFS= read -r license; do
    mkdir -p "$DOCS/licenses/$(dirname -- "$license")"
    cp "$license" "$DOCS/licenses/$license"
done < <(rg --files assets crates vendor apps/lightcraft-haiku apps/lightcraft-haiku-host tools/haiku | rg '(^|/)(LICENSE[^/]*|OFL[^/]*|COPYING[^/]*)$')
cp assets/ATTRIBUTION.md "$DOCS/ASSET-ATTRIBUTION.md"
cp apps/lightcraft-haiku/ATTRIBUTION.md "$DOCS/NATIVE-ATTRIBUTION.md"
CRAFT_FONTS_DIR=${CRAFT_FONTS_DIR:-"$ROOT/../craft-fonts"}
while IFS= read -r license; do
    relative=${license#"$CRAFT_FONTS_DIR/"}
    mkdir -p "$DOCS/licenses/craft-fonts/$(dirname -- "$relative")"
    cp "$license" "$DOCS/licenses/craft-fonts/$relative"
done < <(rg --files "$CRAFT_FONTS_DIR" | rg '(^|/)(LICENSE[^/]*|OFL[^/]*|COPYING[^/]*)$')
mkdir -p "$DOCS/licenses/epaint_default_fonts"
# egui's default fonts come from the storytold/egui commit patched in Cargo.toml (a git checkout).
EGUI_REV=$(sed -n 's/^epaint_default_fonts = .*rev = "\([0-9a-f]*\)".*/\1/p' Cargo.toml)
cp "$CARGO_HOME"/git/checkouts/egui-*/"${EGUI_REV:0:7}"/crates/epaint_default_fonts/fonts/*.txt "$DOCS/licenses/epaint_default_fonts/"
ln -s ../../../../apps/LightCraft "$STAGE/data/deskbar/menu/Applications/LightCraft"
"$TOOLS/package/package" create -C "$STAGE" "$OUT/lightcraft-$VERSION-x86_64.hpkg"
