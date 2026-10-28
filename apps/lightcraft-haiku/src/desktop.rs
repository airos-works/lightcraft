//! The desktop app's start-up helpers that use eframe or rfd there (`launch.rs`, `session.rs` in
//! apps/lightcraft), kept in step with them for the native host. Preferences and the GPU crash
//! markers are the desktop app's own modules (`main.rs`).
use super::prefs::config_dir;
use lightcraft_engine::Session;
use lightcraft_ui_egui::panels::library_problem::{Launch, LibraryProblem};

pub(crate) const HELP: &str = "\
lightcraft — photo library + raw developer

USAGE: lightcraft [OPTIONS] [FILES/FOLDERS…]   (files and folders are imported)

OPTIONS:
  --library DIR    open (or create) this library (env LIGHTCRAFT_LIBRARY; default: the last library opened, or
                   a list of the recent ones with Settings > General > At launch: Ask me or Alt/Option held,
                   else ~/Pictures/LightCraft Library)
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

/// The library session with the folders face models (`<config>/models`) and slideshow templates
/// (`<config>/slideshow-templates`) are kept in. With [`Launch::Ask`] no library is opened yet:
/// the window lists the recent ones.
pub(crate) fn open_session(in_memory: bool, launch: &Launch, seed_demo: bool) -> (Session, Option<LibraryProblem>) {
    let (mut s, problem) = match launch {
        Launch::Ask => (Session::new().with_fs().with_default_denoise_models().with_system_clock(), None),
        Launch::Open { dir, .. } => open_library_session(in_memory, dir.clone(), seed_demo),
    };
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
fn open_library_session(in_memory: bool, dir: Option<std::path::PathBuf>, seed_demo: bool) -> (Session, Option<LibraryProblem>) {
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
