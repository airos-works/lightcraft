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

## Touchpad scrolling

air/OS's mouse add-on marks a touchpad's scroll gesture (since 2026-10-27,
`docs/airos/TOUCHPAD-SCROLLING.md` in the haiku tree): `be:wheel_precise_delta_x/y`
(unrounded wheel steps) and `be:wheel_phase` (began, changed, ended). The host
turns those into egui's `MouseWheelSource::Trackpad` events in points (a step
is egui's line, `InputOptions::line_scroll_speed`, which LightCraft sets to
100 points), with `TouchPhase::Start` at the first message and `End` when the
fingers lift, so the grid, filmstrip and panels follow the fingers instead of
jumping a third of the grid per step. egui drops a gesture's unapplied deltas
at its end, so an end that arrives in the same frame as the last movement is
held for the next frame. A mouse wheel (no precise deltas) still sends whole
lines from a wheel, which the grid turns into its Lightroom-style page steps.
air/OS has no pinch gesture: Command+wheel and the zoom controls zoom.

## One running copy

LightCraft is `B_SINGLE_LAUNCH` (`LightCraft.rdef`): one copy runs from each
executable file, as the library has one writer. Photos and folders opened in
Tracker while it runs (double-click, Open With, drops on its icon) arrive in
the running copy as `B_REFS_RECEIVED`, and a launch from Terminal
(`LightCraft ~/Pictures/trip`) hands its arguments to it (`B_ARGV_RECEIVED`,
relative paths resolved against the launch's folder; options such as
`--library`, `--memory` and `--control` are ignored) and ends before it touches
the log file, the preferences, the GPU markers or the library (`BApplication`
reports `B_ALREADY_RUNNING`). The running copy imports them as the desktop app
imports its launch arguments (`launch_imports`, then the import into the open
library; while the library chooser is open they wait for the chosen library),
and its window comes to the front. A launch without files only brings the
window up. A test copy that must run beside the installed one runs from a file
of its own.

## Tracker, GPU loss, HDR and cameras

- Show in Folder, Help ▸ Open Log Folder and the SAM 3 dialog's Show in File
  Manager open Tracker through `lc_host_reveal`: a folder opens in its own
  window, a file is selected in its folder's window (Tracker's
  `nodeRefToSelect`). The model folder opens on a worker, as on Linux.
- When the window's Vulkan device is lost, the window goes on with the CPU
  renderer and `window-init.marker` makes the next launch start on the CPU, as
  before. If the photo pipeline has also stopped after the GPU failure (a
  driver reset takes both devices), the desktop app's `gpu-lost.marker`
  (`gpu_crash::record_gpu_lost`) is written too, at that moment or at quit, and
  the next launch turns GPU photo rendering off and says why; it removes the
  marker, a `--memory` run leaves it.
- HDR: the log says `display: SDR`. app_server composes the screen in 8-bit
  sRGB (`B_RGB32` frame buffers; `B_RGB30` is 10-bit SDR), and neither
  `BScreen`/`display_mode` nor the graphics drivers have an HDR transfer
  function, colour primaries or HDR metadata (the EDID parser reads a
  display's HDR block, nothing uses it), so there is no HDR output to report.
- Cameras and memory cards: File ▸ Import from Device lists the mounted
  volumes with a `DCIM` folder. air/OS mounts volumes at the root, beside
  network shares and virtual file systems, so the host lists the local,
  persistent volumes other than the boot volume (`lc_host_volumes`,
  `BVolumeRoster`) for the engine's scan (`devices::set_volume_lister`);
  `LIGHTCRAFT_DEVICE_ROOTS` still replaces it. Settings ▸ Import ▸ Cameras and
  Cards works the same way.
- Cameras that don't mount as a disk (PTP/MTP) and tethered capture: upstream
  LightCraft has neither on any platform (`docs/hardware-parity.md`), so there
  is nothing to port. The way upstream offers for tethering is Settings ▸
  Import ▸ Auto Import's watched folder, which works here as on the other
  desktops. HaikuPorts' gphoto2 2.5.27 (libgphoto2 with its PTP driver, over
  libusb and air/OS's `usb_raw`) runs on the X399 and lists the USB devices,
  so `gphoto2 --capture-tethered --filename "$HOME/Pictures/Tethered/%f.%C"`
  into the watched folder is the expected route; it hasn't been tried with a
  camera.

## Monitor profile

Neither Haiku nor air/OS keeps a colour profile for a screen (app_server has no
colour management, the Screen preferences none either), so a monitor's profile
is a file the user puts in `~/config/settings/ColorProfiles/`, the folder
PhotoCraft reads too (made at the first read). The first that exists is used,
letter case and `.icc`/`.icm` aside:

1. `<monitor name> <serial number>.icc`, for one of two identical monitors;
2. `<monitor name>.icc`, the name in the Screen preferences (`DELL P2415Q.icc`);
3. `<connector>.icc` (`DP-2.icc`), for a monitor without EDID;
4. `default.icc`.

While Settings ▸ Display has no profile chosen, previews go through the profile
of the monitor showing most of the window (`Services::screen_profile`, read
about once a second, so moving the window to another monitor is followed); the
settings then offer Use Screen Profile. The monitors come from air/OS's display
layout (`lc_host_displays`: connector, EDID name, vendor, serial number, frame),
or `BScreen::GetMonitorInfo` on plain Haiku. Choose Profile… starts in
`ColorProfiles` (`display_profile_dir`, which also knows
`~/config/non-packaged/data/color/icc` and `/boot/system/data/color/icc`).

## AI masks and AI Denoise on the GPU (issue #5, 2026-10-29)

candle, which upstream's SAM 3 port (`crates/segment`) runs on, has a GPU path only
through Metal, so AI masks ran entirely on the processor here. With the
`lightcraft-segment/wgpu` feature (engine feature `sam-gpu`, on in the air/OS
build) the heavy parts run on the same wgpu/Vulkan device type the photo pipeline
uses, as WGSL compute kernels in `crates/gpu/src/sam/` (`wgsl/sam_gemm.wgsl`,
`wgsl/sam_ops.wgsl`):

- the 32-layer image encoder (ViT with windowed and global attention, rotary
  position encoding; about 5.4 TFLOP per photo, nearly all of an AI mask's work);
- the feature pyramid (neck) of the click decoder and of the text detector;
- the text detector's DETR encoder (5184 image tokens attending to themselves and
  to the prompt) and its pixel decoder (3 × 3 convolutions, group norms, the
  200 per-query masks).

The click prompt encoder and two-way mask decoder, the CLIP text encoder, the DETR
decoder (200 queries with box refinement and box-relative attention bias) and the
scoring stay on candle's CPU path: together about 1 s of a click and 5 s of a
Describe prompt, spread over many small operations. candle's model stays the
reference and the fallback. A part whose kernels or buffers cannot be set up (no
Vulkan, a software adapter, too little memory, `LIGHTCRAFT_GPU=0`,
`LIGHTCRAFT_GPU_BACKEND=off`, the GPU preference off) runs on the CPU; a run that
fails (device error or loss, values that are not finite) is redone on the CPU and
the AI mask device is not used again until LightCraft restarts. Each kernel is
checked against plain loops on small inputs when the device is first set up.
`LIGHTCRAFT_GPU_ADAPTER` picks the adapter as for denoise. The device holds about
4 GB while SAM 3 is loaded (2.4 GB encoder weights and work buffers, 0.3 GB per
neck, 0.9 GB for the detector parts) and frees it when the model is unloaded.
The matrix-product kernel writes its accumulators out as named variables: NVK's
compiler kept an indexed array of them in memory, which made it 20× slower.

candle's CPU kernels also got slower with every thread past a few on air/OS, so the
AI mask worker runs in a rayon pool of 4 threads there
(`crates/engine/src/segment/worker.rs`, `POOL_THREADS`).

AI Denoise already had a GPU path upstream (`crates/gpu/src/nn.rs`, the U-Net as
WGSL convolutions, chosen per photo after a check tile agrees with the CPU and is
timed against it); it runs on NVK unchanged.

Measured on the X399 (Threadripper 1950X, GeForce GTX 1070 on NVK, hrev60207),
SAM 3 `facebook/sam3` checkpoint, `crates/segment/examples/sam3_gpu_bench.rs`
(a 1200 × 800 image) for the GPU columns and upstream's `sam3_bench` for the CPU:

| AI masks | CPU (before) | GPU (after) |
|---|---:|---:|
| Model load (GPU: set-up and 2.4 GB upload) | 2.4 s | 6.2 s |
| Image encoder, per photo | 310 s | 6.6 s |
| Click features (neck), per photo | 8.2 s | 1.2 s |
| A click | 1.3 s | 0.6 s |
| A Describe prompt on a new photo (detector neck and prediction) | 86 s | 6.8 s |

In the app, the first click on a 24 MP Sony A7 III raw (encode at 1008 × 672 and the
click features) took 8.4 s and each later click about 1.3 s; a Describe prompt took
6.3 s. The encoder's output differs from candle's CPU encoder by at most 1.3e-4 of
its largest value; the one- and two-click masks cut from either were identical and a
"sky" Describe mask differed in 0.001 % of its pixels (`tests/it/gpu.rs`, on the X399).

| AI Denoise (RawNIND Bayer, a7 III raw, 35 tiles) | CPU | GPU |
|---|---:|---:|
| One 512-cell tile | 2472 ms | 158 ms |
| Whole photo, model stage (16 CPU / 4 GPU workers) | 39.2 s | 7.6 s |

Tests: `LIGHTCRAFT_SAM_GPU_SOFTWARE=1 cargo test -p lightcraft-gpu --features sam`
runs the kernel, encoder, neck, DETR encoder, pixel decoder and device-loss tests on
a software adapter (lavapipe) against plain-loop references; on air/OS they run on
the GPU. With the model, `LIGHTCRAFT_SAM3_DIR=<dir> [LC_SAM3_GPU_TEXT=1] cargo
test -p lightcraft-segment --features wgpu --release --test it gpu` compares the GPU
and CPU paths end to end. The denoise numbers come from upstream's
`denoise_cpu::denoise_stage_compares_cpu_and_gpu` engine test.

## Printing

File ▸ Print… prints on air/OS: the Print dialog's button is Print…, and Save as
PDF… beside it still writes the print-ready PDF. Printing
opens the system print setup (the print server's window with its printer menu,
then that printer's paper and job setup, which also take copies and the page
range); the print server remembers the printer and its settings for LightCraft.
The pages are composed as for the PDF, at the Print Job panel's resolution, and
drawn into a `BPrintJob` (`apps/lightcraft-haiku/src/print.rs`, the Be print kit
calls in `apps/lightcraft-haiku-host/native/print.cpp`, both from WordCraft's
host): each page scaled to the printer's resolution (at most 300 dpi), placed
on the driver's paper at its real size (centred, shrunk only when bigger than
the paper, turned when a landscape page meets portrait paper), and spooled as
the strips of rows with ink. A progress window, and the activity row, cancel
the job.

The print kit has no colour management: its bitmaps are device RGB, which the
drivers (PostScript, PCL, Gutenprint) take as sRGB. Managed by Printer sends
sRGB (also when the panel says Adobe RGB, which only the PDF can carry); an
RGB printer profile converts the pages to the printer's values, which the
driver passes on; a CMYK profile can't be printed this way and asks for the PDF.
libprint's drivers start in monochrome: choose Color in the job setup.

Tested on the X399 with a PostScript printer on a JetDirect transport to a local
`nc -l -p 9100`, the output decoded to PNG: a 7.8 × 5.2 in photo on A4 landed
within a printer dot of its place and size, portrait and landscape pages on
portrait paper were turned with their tops to the left, and the printed colours
matched the app's own sRGB render of the photo to half a level on average.
