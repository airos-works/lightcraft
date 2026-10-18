//! Shared desktop library opening, preferences and crash recovery.
#![allow(dead_code)]
use lightcraft_engine::Session;
use lightcraft_ui_egui::panels::library_problem::LibraryProblem;
use lightcraft_ui_egui::{LightcraftApp, UiState};

/// Written while the GPU compute device is created, removed once that returned
/// (`lightcraft_gpu::backend::set_init_marker`). Not with `LIGHTCRAFT_NO_PREFS` (tests, scripts).
pub(crate) fn gpu_marker_path() -> Option<std::path::PathBuf> {
    if std::env::var_os("LIGHTCRAFT_NO_PREFS").is_some() {
        return None;
    }
    config_dir().map(|d| d.join("gpu-init.marker"))
}

/// Arm the GPU crash sentinel; if the previous launch left it behind (it died inside the GPU
/// driver while creating the device), the notice to show — GPU rendering then starts off.
///
/// A `--memory` session writes nothing (issue #169): it neither arms the sentinel — which would
/// create the settings folder — nor removes a marker it finds. It still honours one, so the GPU
/// starts off there too, and the next ordinary launch reports and clears it.
pub(crate) fn gpu_crash_check(in_memory: bool) -> Option<String> {
    gpu_crash_check_at(gpu_marker_path(), in_memory)
}

pub(crate) fn gpu_crash_check_at(marker: Option<std::path::PathBuf>, in_memory: bool) -> Option<String> {
    use lightcraft_engine::gpu::backend;
    let left = if in_memory {
        marker.as_deref().and_then(backend::read_init_marker)
    } else {
        let left = marker.as_deref().and_then(backend::take_init_marker);
        backend::set_init_marker(marker);
        left
    };
    left.map(|what| gpu_crash_notice(&what))
}

pub(crate) fn gpu_crash_notice(what: &str) -> String {
    format!(
        "LightCraft closed unexpectedly while starting the GPU last time ({what}), so GPU rendering is now off and photos render on the CPU. \
         To try the GPU again, turn on Settings ▸ Performance ▸ Use the GPU for rendering; to try another graphics backend, start LightCraft with \
         LIGHTCRAFT_GPU_BACKEND=dx12, vulkan or off (see docs/gpu-pipeline.md → Troubleshooting)."
    )
}

/// Written when the window stopped drawing after the GPU device was lost (issue #1337): the next
/// launch renders on the CPU and says why.
pub(crate) fn gpu_lost_marker_path() -> Option<std::path::PathBuf> {
    gpu_marker_path().map(|p| p.with_file_name("gpu-lost.marker"))
}

/// Why the GPU was lost, when the GPU pipeline stopped after a device failure in this process (a
/// driver reset, a hang, the Windows watchdog): `None` for other reasons it is off, or when it
/// never failed.
pub(crate) fn gpu_lost_reason() -> Option<String> {
    lightcraft_engine::gpu::unavailable_reason().and_then(|r| r.strip_prefix("stopped after a GPU failure: ").map(str::to_string))
}

/// Record that LightCraft stopped after the GPU was lost, for the next launch.
pub(crate) fn record_gpu_lost(marker: Option<&std::path::Path>, reason: &str) {
    let Some(m) = marker else { return };
    if let Some(d) = m.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    if let Err(e) = std::fs::write(m, format!("{reason}\n")) {
        log::warn!("could not write {}: {e}", m.display());
    }
}

/// A GPU loss the last launch stopped on: its notice, once (the marker is removed, except by a
/// `--memory` session, which leaves the disk as it found it).
pub(crate) fn gpu_lost_check_at(marker: Option<&std::path::Path>, in_memory: bool) -> Option<String> {
    let m = marker?;
    let reason = std::fs::read_to_string(m).ok()?;
    if !in_memory {
        let _ = std::fs::remove_file(m);
    }
    Some(format!(
        "LightCraft stopped last time after the graphics driver reset the GPU ({}) while photos rendered on it, so GPU rendering \
         is now off and photos render on the CPU. To try the GPU again, turn on Settings ▸ Performance ▸ Use the GPU for rendering; \
         an up-to-date graphics driver may help.",
        reason.trim()
    ))
}

pub(crate) fn config_dir() -> Option<std::path::PathBuf> {
    lightcraft_engine::config::config_dir()
}

/// `<config>/ui.json`, the saved UI state and app settings — `None` when nothing is read or
/// written: `LIGHTCRAFT_NO_PREFS` (tests, scripts) or no home folder.
pub(crate) fn prefs_path() -> Option<std::path::PathBuf> {
    if std::env::var_os("LIGHTCRAFT_NO_PREFS").is_some() {
        return None;
    }
    config_dir().map(|d| d.join("ui.json"))
}

/// The saved UI state and app settings (`<config>/ui.json`), if any, and a warning for the user
/// when the file exists but can't be used (issue #103): a damaged file is kept as
/// `ui.json.corrupt-<unix time>` first, so the next save can't lose the library location in it;
/// one that can't be read at all is not written this session (`keep_file`).
///
/// A `--memory` session reads the settings (so it looks like the user's app) but writes nothing
/// (issue #164), so a damaged file is left as it is there, not set aside.
pub(crate) fn load_prefs(in_memory: bool) -> (Option<UiState>, Option<String>, bool) {
    let Some(path) = prefs_path() else { return (None, None, false) };
    load_prefs_at(&path, !in_memory)
}

/// Compatibility recovery is limited to persisted docking, not runtime UiState commands.
pub(crate) fn decode_saved_ui(bytes: &[u8]) -> Result<(UiState, bool), String> {
    let mut saved: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    let fields = saved.as_object_mut().ok_or("settings must be an object")?;
    let recovered =
        fields.get("docking").is_some_and(|value| match serde_json::from_value::<Option<lightcraft_ui_egui::docking::Workspace>>(value.clone()) {
            Ok(Some(workspace)) => workspace.validate().is_err(),
            Ok(None) => false,
            Err(_) => true,
        });
    if recovered {
        fields.remove("docking");
    }
    let ui = serde_json::from_value(saved).map_err(|error| error.to_string())?;
    Ok((ui, recovered))
}

pub(crate) fn load_prefs_at(path: &std::path::Path, set_aside_damaged: bool) -> (Option<UiState>, Option<String>, bool) {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (None, None, false),
        Err(e) => {
            let msg = format!("The app settings ({}) couldn't be read: {e}. Defaults are used, and the file isn't overwritten.", path.display());
            return (None, Some(msg), true);
        }
    };
    match decode_saved_ui(&bytes) {
        Ok((ui, false)) => (Some(ui.sanitized()), None, false),
        Ok((ui, true)) => {
            let msg = format!(
                "The saved panel layout in {} couldn't be restored. Other settings were loaded; the original file is preserved and settings won't be saved this session.",
                path.display()
            );
            (Some(ui.sanitized()), Some(msg), true)
        }
        Err(e) if !set_aside_damaged => {
            let msg = format!("The app settings ({}) are damaged ({e}). Defaults are used; nothing is saved in this session.", path.display());
            (None, Some(msg), true)
        }
        Err(e) => {
            let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            let keep = path.with_file_name(format!("ui.json.corrupt-{secs}"));
            let kept = std::fs::rename(path, &keep);
            let what = match &kept {
                Ok(()) => format!("it was kept as {}", keep.display()),
                Err(r) => format!("it couldn't be set aside ({r})"),
            };
            let msg = format!(
                "The app settings ({}) are damaged ({e}); {what}. Defaults are used — reopen your library with Settings → Open Library… if it isn't shown.",
                path.display()
            );
            (None, Some(msg), kept.is_err())
        }
    }
}

/// Write `bytes` to `path` atomically: a temp file, fsynced, renamed over it (a crash or a full
/// disk leaves the old file or the new one, never a truncated one).
pub(crate) fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = path.with_extension("json.tmp");
    let written = (|| {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, path)?;
    if let Some(dir) = path.parent()
        && let Ok(d) = std::fs::File::open(dir)
    {
        let _ = d.sync_all();
    }
    Ok(())
}

/// Saves `ui.json` (app settings, incl. the library to open at launch): as soon as the library
/// changes, every few seconds when anything else changed, and at exit.
///
/// Nothing is written while the session is one that promises to save nothing: `--memory`
/// (`path` is `None`; issue #164), or the one running because the library couldn't be opened — the
/// question window and Continue Without Saving (issue #100; `app.openLibrary` ends it, and the
/// library it opened is saved then).
#[derive(Default)]
pub(crate) struct PrefsWriter {
    /// Where to write; `None` writes nothing (`--memory`, `LIGHTCRAFT_NO_PREFS`, no home folder).
    pub(crate) path: Option<std::path::PathBuf>,
    pub(crate) written: Vec<u8>,
    pub(crate) library: String,
    pub(crate) checked: f64,
    /// The last write failed (reported once until a write works).
    pub(crate) failing: bool,
    /// `ui.json` couldn't be read at launch: never overwrite it this session.
    pub(crate) keep_file: bool,
    /// Follows the window, to remember the size it rests at (`UiState::window`).
    pub(crate) window: lightcraft_ui_egui::window_place::WindowWatch,
    /// When the window was first looked at: its watch goes by the clock from then on, not by
    /// the interface's time, which stands still while the window is hidden.
    pub(crate) window_since: Option<std::time::Instant>,
}

impl PrefsWriter {
    pub(crate) fn save(&mut self, app: &LightcraftApp) -> Result<(), String> {
        self.save_ui(&app.ui, app.library_problem.is_some())
    }

    /// Write `ui` if it changed since the last write; `temporary` (the library couldn't be
    /// opened) saves nothing.
    pub(crate) fn save_ui(&mut self, ui: &UiState, temporary: bool) -> Result<(), String> {
        if self.keep_file || temporary {
            return Ok(());
        }
        let Some(path) = self.path.clone() else { return Ok(()) };
        if let Some(workspace) = &ui.docking {
            workspace.validate()?;
        }
        let bytes = serde_json::to_vec_pretty(ui).map_err(|e| e.to_string())?;
        if bytes == self.written {
            return Ok(());
        }
        path.parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| write_atomic(&path, &bytes))
            .map_err(|e| format!("saving the app settings failed: {e}"))?;
        self.written = bytes;
        self.library = ui.settings.library_path.clone();
        Ok(())
    }

    /// What to remember of the window as the app quits ([`WindowWatch::at_exit`]).
    ///
    /// [`WindowWatch::at_exit`]: lightcraft_ui_egui::window_place::WindowWatch::at_exit
    pub(crate) fn window_at_exit(
        &mut self,
        remembered: Option<lightcraft_ui_egui::window_place::WindowPlace>,
    ) -> Option<lightcraft_ui_egui::window_place::WindowPlace> {
        match self.window_since {
            Some(since) => self.window.at_exit(remembered, since.elapsed().as_secs_f64()),
            None => remembered,
        }
    }

    pub(crate) fn tick(&mut self, app: &mut LightcraftApp, ctx: &egui::Context) {
        // (asked for before the input is read: the context is not asked from inside its own lock)
        let zoom = ctx.zoom_factor();
        let (now, seen) = ctx.input(|i| (i.time, lightcraft_ui_egui::window_place::Seen::of(i.viewport(), i.content_rect(), zoom)));
        // how the window is, to open like that next time (written with the settings below). What
        // the settings hold is checked first: a script may have set it (`ui.set`)
        let remembered = app.ui.window.and_then(lightcraft_ui_egui::window_place::WindowPlace::checked);
        let at = self.window_since.get_or_insert_with(std::time::Instant::now).elapsed().as_secs_f64();
        app.ui.window = self.window.note(remembered, &seen, at);
        if let Some(wait) = self.window.waits(at) {
            // (a size is only remembered once the window has rested at it: look again then, though
            // nothing else asks for a frame)
            ctx.request_repaint_after(wait);
        }
        let moved = app.ui.settings.library_path != self.library;
        if !moved && now - self.checked < 3.0 {
            return;
        }
        self.checked = now;
        match self.save(app) {
            Ok(()) => self.failing = false,
            Err(e) => {
                if !self.failing {
                    log::warn!("{e}");
                    app.notices.push(format!("{e}. LightCraft keeps trying."));
                }
                self.failing = true;
                // don't retry every frame
                self.library = app.ui.settings.library_path.clone();
            }
        }
    }
}

/// The library session with the folders face models (`<config>/models`) and slideshow templates
/// (`<config>/slideshow-templates`) are kept in.
pub(crate) fn open_session(in_memory: bool, dir: Option<std::path::PathBuf>, seed_demo: bool) -> (Session, Option<LibraryProblem>) {
    let (mut s, problem) = open_library_session(in_memory, dir, seed_demo);
    s.face_models_dir = lightcraft_engine::config::default_face_models_dir();
    s.slide_templates_dir = lightcraft_engine::config::config_dir().map(|d| d.join(lightcraft_engine::slide_template::DIR_NAME));
    // Control requests are answered on the UI thread: Add to Apple Photos must not wait there
    s.apple_photos_background = true;
    (s, problem)
}

/// The persistent library session, or with `--memory` an in-memory one (demo photos).
///
/// If the library can't be opened the session is empty and in memory — never seeded with the
/// demo photos, never written anywhere — and the problem is returned: the window then says so
/// and offers Try Again / Choose Another Library… / Continue Without Saving / Quit (issue #100).
pub(crate) fn open_library_session(in_memory: bool, dir: Option<std::path::PathBuf>, seed_demo: bool) -> (Session, Option<LibraryProblem>) {
    if in_memory {
        return (if seed_demo { Session::with_demo() } else { Session::new() }.with_fs().with_default_denoise_models().with_system_clock(), None);
    }
    let unopened = || Session::new().with_fs().with_default_denoise_models().with_system_clock();
    let Some(dir) = dir else {
        log::warn!("no library location (set --library or LIGHTCRAFT_LIBRARY)");
        return (unopened(), Some(LibraryProblem::new("", "There is no home folder to keep the library in. Choose a folder for it.")));
    };
    let t0 = std::time::Instant::now();
    let mut s = Session::new().with_fs().with_default_denoise_models().with_system_clock();
    match s.open_library(&dir, seed_demo) {
        Ok(r) => {
            let (replayed, torn) = (r.replayed, r.torn_bytes);
            log::info!(
                "library {}: {} photos, {replayed} log records replayed{} ({:.1} ms)",
                dir.display(),
                s.catalog.len(),
                if torn > 0 { ", torn tail repaired" } else { "" },
                t0.elapsed().as_secs_f64() * 1000.0
            );
            (s, None)
        }
        Err(e) => {
            log::error!("can't open library {}: {e}", dir.display());
            (unopened(), Some(LibraryProblem::new(dir.to_string_lossy(), e.to_string())))
        }
    }
}

pub(crate) const HELP: &str = "\
lightcraft — photo library + raw developer

USAGE: lightcraft [OPTIONS] [FILES/FOLDERS…]   (files and folders are imported)

OPTIONS:
  --library DIR    open (or create) this library (default: ~/Pictures/LightCraft Library; env LIGHTCRAFT_LIBRARY)
  --no-demo        don't seed a new library with the procedural demo photos
  --memory         throwaway in-memory session (alias --demo); nothing is saved
  --control PORT   serve the JSON-lines control channel on 127.0.0.1:PORT (env LIGHTCRAFT_CONTROL_PORT;
                   see docs/control-protocol.md)
  --version, --help

ENVIRONMENT:
  LIGHTCRAFT_CONFIG_DIR=DIR   use this settings folder instead of the platform default (empty = no settings folder).
                   Without it, a portable.txt file next to the program puts the folder beside it (LightCraftData)
  LIGHTCRAFT_GPU_BACKEND=dx12|vulkan|metal|auto|off   graphics backend (default: DX12 on Windows, Metal on macOS,
                   Vulkan on Linux; off = render on the CPU); else WGPU_BACKEND. LIGHTCRAFT_GPU=0: CPU rendering.
  LIGHTCRAFT_SAM3_DIR=DIR   the SAM 3 model for Object / Describe masks (default: <settings folder>/models/sam3;
                   optional: LightCraft offers to download it when first needed)
  LIGHTCRAFT_SAM3_MIRRORS=URL,…   where to download the SAM 3 model from (base URLs, tried in order)
  LIGHTCRAFT_LOG=info|debug   more log detail from LightCraft itself (default: info from LightCraft, warnings
                   from other crates); else RUST_LOG=<env_logger directives>. Records go to stderr and to
                   <settings folder>/logs/lightcraft.log (previous runs: lightcraft.1.log, lightcraft.2.log)
";

/// The launch arguments' files and folders to import (absolute), and a notice for each one that
/// is not imported (see [`lightcraft_engine::import::launch_path`]; issue #374).
pub(crate) fn launch_imports(args: &[String]) -> (Vec<String>, Vec<String>) {
    let cwd = std::env::current_dir().ok();
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).filter(|h| !h.is_empty()).map(std::path::PathBuf::from);
    launch_imports_in(args, cwd.as_deref(), home.as_deref())
}

pub(crate) fn launch_imports_in(args: &[String], cwd: Option<&std::path::Path>, home: Option<&std::path::Path>) -> (Vec<String>, Vec<String>) {
    use lightcraft_engine::import::{LaunchPath, launch_path};
    let (mut files, mut notices) = (Vec::new(), Vec::new());
    for a in args {
        match launch_path(a, cwd, home, lightcraft_engine::devices::dcim_in) {
            Some(LaunchPath::Import(p)) => files.push(p),
            Some(LaunchPath::Refused { path, why }) => {
                log::warn!("not importing {path} given at launch: {why}");
                notices
                    .push(format!("{path} was not imported: it is {why}. To import from it, choose a folder inside it with File ▸ Import Photos…"));
            }
            None => {}
        }
    }
    (files, notices)
}

/// Where the log files live: `logs` in the settings folder, next to `ui.json` (see `logging`).
/// None with `LIGHTCRAFT_NO_PREFS` (tests, scripts), so those runs don't rotate away the user's logs.
pub(crate) fn log_dir() -> Option<std::path::PathBuf> {
    if std::env::var_os("LIGHTCRAFT_NO_PREFS").is_some() {
        return None;
    }
    config_dir().map(|d| d.join("logs"))
}

/// The control port `value` names; a value that isn't a port is logged and ignored (`what` is
/// the option or variable it came from). Logged before the log file opens, so it is kept for it.
pub(crate) fn control_port_from(what: &str, value: Option<String>) -> Option<u16> {
    let value = value?;
    let port = value.trim().parse().ok();
    if port.is_none() {
        log::warn!("{what} {value:?} is not a port number; no control server");
    }
    port
}

/// An isolated profile keeps the emergency crash log with its other app data. With no
/// override (or an empty override), retain the historical system-temp location.
pub(crate) fn panic_log_path_with(env: impl Fn(&str) -> Option<std::ffi::OsString>, temp: std::path::PathBuf) -> std::path::PathBuf {
    env("LIGHTCRAFT_CONFIG_DIR").filter(|value| !value.is_empty()).map(std::path::PathBuf::from).unwrap_or(temp).join("lightcraft-panics.log")
}
