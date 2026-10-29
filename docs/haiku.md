# LightCraft for air/OS

This is the independently maintained `airos-works/lightcraft` fork of LightCraft
0.15.0. The upstream project does not maintain this port.

The native application retains the full library, non-destructive developer,
import/export, presets, masks, history and external-editor workflows. The shared
Rust engine and egui UI run inside a Be API host adapted from PhotoCraft and
PdfCraft. This small isolated host provides the window, input, clipboard, Tracker
file references, native file panels and desktop integration.

## Build and package

The scripts use a Haiku x86_64 cross toolchain, a matching sysroot and Haiku's
Linux host tools. Their default paths are in `tools/haiku/env.sh`; override
`HAIKU_CROSS`, `HAIKU_SYSROOT`, `HAIKU_HOST_TOOLS`, `CARGO_HOME` and `RUSTUP_HOME`
for another installation. Rust needs `rust-src` because Haiku is a tier-3 target.
All build output stays under this checkout.

Clone `https://github.com/storytold/craft-fonts` beside this repository and check
out `b076417b5880bd4154bd22f00695b1d28c6bea83`, the upstream release's pinned
font input. Set `CRAFT_FONTS_DIR` if it lives elsewhere. These fonts are required
by the native release build and their OFL licenses are bundled in the package.

```sh
bash tools/haiku/build.sh
bash tools/haiku/package.sh --no-build
```

The executable is `target/x86_64-unknown-haiku/release/LightCraft`; the installable
package is under `artifacts/haiku/`. Install it with `pkgman install <package>`.
The package registers its application signature, icon and supported image types,
and adds a Deskbar entry. It does not replace existing preferred image editors.

## Rendering and settings

The photo pipeline uses upstream Vulkan compute shaders. The window renderer
uses an independent Vulkan device to draw egui, in BGRA (the screen's and a
`B_RGB32` bitmap's byte order), into the presenter's frame texture
(`apps/lightcraft-haiku/src/present.rs`, shared with PhotoCraft's host), which
gets it onto the window; Haiku does not provide a wgpu window surface. Identical
completed frames are not drawn again; geometry, clipping, scale or texture
changes redraw. `LIGHTCRAFT_PRESENT` picks how a frame reaches the window:

- `readback` (the default): a compute pass compares the frame with the one
  app_server already has, 64 px tile by tile; only the changed tiles, joined
  into a few rectangles (`damage.rs`), are copied back and drawn.
- `full`: the whole frame is read back and drawn every time (the first host).
- `direct` (experimental): the window is a `BDirectWindow` with air/OS's
  `B_DIRECT_DEVICE_PIXELS`, the screen's frame buffer is imported into Vulkan
  with the drivers' private `VkImportScanoutMemoryHAIKU`
  (`apps/lightcraft-haiku-host/src/scanout.rs`, with the host crate's other
  `unsafe`), and the GPU copies each frame into the visible parts of the window;
  app_server's own copy catches up through a readback a quarter of a second
  later, and the direct daemon thread waits for the GPU's copies before the
  window is moved or covered. `PRESENT_VERIFY=1` checks each direct frame
  against the frame buffer. Not the default: on the Framework 13 (RADV,
  amdgpu_drm) the copies were pixel-exact (0 of 2,746,548 visible pixels
  differed) but the Vulkan device is lost a few seconds after the frame buffer
  is imported, even with no copies at all, and the window renderer then falls
  back to the CPU. It has not been tried on NVK.

Measured on the Framework 13 (Radeon 760M, a 2076×1323-pixel window,
`tools/haiku/present-bench.py --samples 40`, against the 0.15.0~airos1-1
build): app_server's processor time 0.215 → 0.126 s, a median of 12,288 pixels
copied back per frame instead of 2,746,548; the frame thread's time and the
request-to-frame latency are about the same (hover 21.0 → 20.7 ms), being mostly
the GPU drawing the frame (3.9 ms of the 4.1 ms present). Per-frame costs are
logged with `RUST_LOG=warn,lightcraft_haiku=debug,present=debug`.
Both devices reject software Vulkan adapters. The native
window falls back to its CPU renderer if hardware initialization or rendering
fails; photo processing has the shared engine's separate CPU fallback.

`LIGHTCRAFT_RENDERER=auto` is the default. `cpu` selects CPU rendering and photo
processing; `vulkan` requires hardware at startup and reports an error if unavailable;
a later window-device fault still falls back safely.
`LIGHTCRAFT_GPU_BACKEND=off` disables photo compute independently. The
`ui.inspect` control response reports `perf.hostRenderer`, `hostFrameMs` and
`hostRenderMs`, alongside the engine's GPU adapter and fallback information. A failed native
GPU startup leaves a recovery marker; the next automatic launch starts on the
CPU. Memory-only runs leave recovery files untouched.

Settings live in `~/config/settings/LightCraft` and the library defaults to
`~/Pictures/LightCraft Library`. `LIGHTCRAFT_CONFIG_DIR`, `LIGHTCRAFT_LIBRARY`
and `--library` can isolate a run. Portable mode and `--memory` are retained.
The optional loopback JSON control channel is enabled with `--control PORT`;
see [the control protocol](control-protocol.md).

PhotoCraft handoff detects `/boot/system/apps/PhotoCraft` and the user's
non-packaged applications folder. AI features retain their upstream optional
model requirements; the package does not bundle downloaded models. HEIF remains
an optional build feature (`--features heif`).

For a machine requiring an experimental Vulkan driver, an optional app-private
manifest at `~/config/non-packaged/data/lightcraft/vulkan/icd.json` selects that
driver for LightCraft only, including launches from Tracker and Deskbar. Set
`LIGHTCRAFT_VULKAN_ICD` to another manifest or an empty string to ignore it.
Existing `VK_DRIVER_FILES` or `VK_ICD_FILENAMES` overrides take precedence. The
package neither bundles a driver nor changes system Vulkan configuration.

## Validation

Native validation used a Radeon 760M laptop (RADV, Mesa 25.3.6,
Haiku hrev60206+1566) and an X399 with a GeForce GTX 1070 (NVK,
Mesa 25.3.0-devel, Haiku hrev60206+1547). Both the photo pipeline and native
window used hardware Vulkan. The NVIDIA test used the app-private driver
described above; its exact hash is in [the validation record](haiku-validation.json).

The upstream `render_bench` rendered a procedural 24 MP source with eight Rayon
workers. These are the best wall times of three runs, excluding file decode,
window drawing and JPEG encoding. A cold preview has a new stage cache; driver
shaders were warmed.

| Operation | AMD CPU / GPU | NVIDIA CPU / GPU |
|---|---:|---:|
| 1920×1280 preview, cold | 454.9 / 64.7 ms | 1211.5 / 110.8 ms |
| Exposure change, warm preview | 40.5 / 9.0 ms | 66.3 / 14.6 ms |
| Highlights change, warm 1152×768 draft | 16.3 / 4.0 ms | 26.1 / 6.8 ms |
| 24 MP export render | 4454.6 / 1761.9 ms | 10949.4 / 1762.4 ms |

Unchanged window frames took a median 0.027 ms on AMD and 0.056 ms on NVIDIA
over 30 samples. These measure frame reuse, not the cost of drawing a changed
frame. Real edited 4032×3024 DNG exports differed from CPU output by at most one
8-bit channel value on both GPUs. This also held for the synthetic preview,
zoom and export comparisons.

Device workflows cover RAW import; exposure, highlights, shadows and radial
masks; undo/redo; persistent edits after reopening; JPEG/PNG export; PSD handoff
to PhotoCraft; single-writer library locking; native Tracker file references,
file-panel cancellation and clipboard round trips. CPU rendering, missing-driver
fallback and recovery from a previous failed GPU startup were also exercised.
Screenshots were captured from completed native window frames.

Native suites cover the catalog, GPU equivalence, limits, fallback, memory,
shutdown and adapter probes, plus the native host, frame reuse and window-device
loss. A separate process destroys the photo GPU device and verifies a CPU
fallback. A synthetic regression checks that a driver panic during scope cleanup
cannot escape the render boundary; it fails with the former scope placement.
The full `cargo xtask ci` gate also checks the original desktop and WASM builds.
Its optional corpus/model/hardware skips are reported separately: passing CI
does not establish coverage of every camera format or optional AI model.

The experimental NVK driver reset during both attempts when LightCraft and PhotoCraft were
using the GPU concurrently. This exposed a cleanup panic outside the shared
photo-render fallback guard. The port includes a fix covering error-scope
completion and buffer-pool cleanup, plus the regression tests above. Repeating
the handoff after the fix reproduced the driver reset and verified automatic CPU
recovery with a completed photo preview and no failed render. The driver timeout
briefly stalls the window before recovery. The application cannot
prevent a driver reset; after a device failure it disables that device and uses
the CPU. Users encountering repeated resets should use `LIGHTCRAFT_RENDERER=cpu`
or a stable driver. No system driver configuration is changed by the package.

## Native application icon (2026-10-26)

The air/OS icon is an original photo collection and exposure dial, licensed under
MIT OR Apache-2.0. Its 64-unit vector drawing follows the
[Haiku icon guidelines](https://www.haiku-os.org/development/icon-guidelines):
vertical edges stay upright in perspective, the main light comes from the top
left, the right faces are darker, a floor shadow extends to the right, and the
silhouette has a two-unit black outline (one pixel at 32px). Broad details are
intended to remain legible at 16px and 32px. No fonts or third-party artwork are used.

Rebuild both the native HVIF and its SVG preview from this checkout:

```sh
python3 tools/haiku/make-icon.py
```

`apps/lightcraft-haiku/resources/LightCraft.hvif` is the packaged icon;
`apps/lightcraft-haiku/resources/LightCraft.svg` exports the same geometry for inspection.
The existing `.rdef` imports the HVIF, and `tools/haiku/package.sh` embeds it in
the executable's resources. Tracker and Deskbar therefore use the vector icon.
The upstream icons for other platforms remain separate. Native previews can be
rendered on Haiku with `BIconUtils::GetVectorIcon`; review at 16, 32, 64 and 128px
on light and dark backgrounds before changing the drawing.

Package version `0.9.1~airos1-3` includes this icon update.

## Upstream 0.15.0 merge (2026-10-29)

The port follows upstream LightCraft 0.15.0 (egui 0.37 from storytold's egui fork,
still wgpu 30.0.1, so the vendored Haiku cfg patches apply unchanged). The native
host now uses the desktop app's own `prefs.rs` and `gpu_crash.rs` modules; the
start-up helpers that need eframe or rfd there are mirrored in
`apps/lightcraft-haiku/src/desktop.rs`. The library chooser at launch (Settings ▸
General ▸ At launch: Ask me, or Alt held), print templates and the model-folder
button are wired to native file panels and Tracker. The in-app update checker is
not started; air/OS updates come from the package. The release build uses
craft-fonts `b076417b5880bd4154bd22f00695b1d28c6bea83`, upstream's release pin.
Package version `0.15.0~airos1-1` is the first with this merge.

## Updates

LightCraft is a package on air/OS, updated by pkgman and SoftwareUpdater;
craft-update has no Haiku platform. Help ▸ Check for Updates is left out of the
menus, and the command (still reachable from the control channel) only says
that SoftwareUpdater installs newer versions, as the web build says it can't
check. The start-up check stays off. The community links (Help ▸ Join the
ArtCraft Discord, the About dialog's button) stay: they open the web browser,
and upstream has no switch a host could set.
