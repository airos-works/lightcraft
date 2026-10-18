# LightCraft for air/OS

This is the independently maintained `jmgasper/lightcraft` fork of LightCraft
0.9.1. The upstream project does not maintain this port.

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
out `3d9ab3a132ed0eff457e424226a435a901da30d3`, the upstream release's pinned
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
uses an independent Vulkan device to draw egui into a reusable offscreen target,
then copies the finished bitmap to the native view. Haiku does not provide a
wgpu window surface. Identical completed frames reuse the bitmap, avoiding GPU
submission and readback; geometry, clipping, scale or texture changes redraw it.
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
