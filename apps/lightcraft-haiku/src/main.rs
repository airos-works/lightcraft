//! The complete LightCraft editor hosted by the native Be API.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![cfg_attr(not(target_os = "haiku"), allow(dead_code))]
mod damage;
mod texture_fit;
mod displays;
mod frame_cache;
mod keys;
mod print;
mod raster;
mod window_recovery;
#[cfg(target_os = "haiku")]
use lightcraft_haiku_host as host;
#[cfg(target_os = "haiku")]
use lightcraft_haiku_host::scanout;
#[cfg(target_os = "haiku")]
#[path = "../../lightcraft/src/alloc_release.rs"]
mod alloc_release;
#[cfg(target_os = "haiku")]
mod desktop;
#[cfg(target_os = "haiku")]
#[path = "../../lightcraft/src/gpu_crash.rs"]
#[allow(dead_code)] // the desktop window's parts
mod gpu_crash;
#[cfg(target_os = "haiku")]
#[path = "../../lightcraft/src/prefs.rs"]
#[allow(dead_code)] // the desktop window's parts
mod prefs;
/// What `gpu_crash` names of the desktop window's renderer: the native host draws its own window.
#[cfg(target_os = "haiku")]
#[allow(dead_code)]
mod window_gpu {
    pub(crate) struct Files;
    impl Files {
        pub(crate) fn new(_dir: Option<std::path::PathBuf>) -> Self {
            Files
        }
    }
}
#[cfg(target_os = "haiku")]
use lightcraft_ui_egui::panels::library_problem::{Launch, LibraryProblem};
#[cfg(target_os = "haiku")]
use lightcraft_ui_egui::{LightcraftApp, UiState};
#[cfg(target_os = "haiku")]
#[path = "../../lightcraft/src/control_server.rs"]
mod control_server;
#[cfg(target_os = "haiku")]
#[path = "../../lightcraft/src/editors.rs"]
mod editors;
#[cfg(target_os = "haiku")]
mod gpu;
#[cfg(target_os = "haiku")]
#[path = "../../lightcraft/src/logging.rs"]
mod logging;
#[cfg(target_os = "haiku")]
mod present;
#[cfg(target_os = "haiku")]
mod services;
#[cfg(target_os = "haiku")]
mod shell;

#[cfg(not(target_os = "haiku"))]
fn main() {
    eprintln!("LightCraft for air/OS requires Haiku; use lightcraft on this platform.");
}
#[cfg(target_os = "haiku")]
fn main() {
    if let Err(e) = native_main() {
        eprintln!("LightCraft: {e}");
        std::process::exit(1);
    }
}
#[cfg(target_os = "haiku")]
fn guard<T>(what: &str, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    lightcraft_engine::guard::catch(what, f).and_then(|r| r)
}
#[cfg(target_os = "haiku")]
fn native_main() -> Result<(), String> {
    use desktop::{HELP, LaunchArgs, control_port_from, launch_imports, launch_library_choice, log_dir, open_session, panic_log_path_with};
    use gpu_crash::{gpu_crash_check, gpu_lost_check_at, gpu_lost_marker_path};
    use prefs::{PrefsWriter, config_dir, load_prefs, prefs_path};
    host::export_vulkan_driver();
    host::export_locale();
    let logger = logging::install();
    lightcraft_engine::guard::install_hook(panic_log_path_with(|n| std::env::var_os(n), std::env::temp_dir()));
    if let Some(logger) = logger {
        logging::record_panics(logger);
    }
    alloc_release::install();
    let mut control_port = control_port_from("LIGHTCRAFT_CONTROL_PORT", std::env::var("LIGHTCRAFT_CONTROL_PORT").ok());
    let (mut files, mut seed_demo, mut in_memory, mut library_dir) = (Vec::new(), true, false, None);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--control" => {
                control_port = Some(args.next().and_then(|s| s.parse::<std::num::NonZeroU16>().ok()).ok_or("--control requires port 1–65535")?.get())
            }
            "--library" => library_dir = Some(std::path::PathBuf::from(args.next().ok_or("--library requires a directory")?)),
            "--no-demo" => seed_demo = false,
            "--memory" | "--demo" => in_memory = true,
            "--version" => {
                println!("LightCraft {} for air/OS ({})", env!("CARGO_PKG_VERSION"), option_env!("LIGHTCRAFT_BUILD_SHA").unwrap_or("dev"));
                return Ok(());
            }
            "--help" | "-h" => {
                println!("{HELP}\nLIGHTCRAFT_RENDERER=auto|cpu|vulkan selects the native window renderer (cpu also disables photo compute).");
                return Ok(());
            }
            _ if arg.starts_with('-') => return Err(format!("Unknown option: {arg}")),
            _ => files.push(arg),
        }
    }
    // Before the log file, the preferences, the GPU markers or the library are touched: a second
    // launch (from Terminal; Tracker hands files to the running copy itself) passes its files
    // and folders to the running copy, which imports them, and ends.
    if host::init(services::SIGNATURE)? == host::Launch::HandedOver {
        return Ok(());
    }
    let mut log_file = None;
    if let Some(logger) = logger {
        match log_dir().filter(|_| !in_memory) {
            Some(dir) => match logger.attach_dir(&dir) {
                Ok(path) => log_file = Some(path),
                Err(e) => log::warn!("No log file: {e}"),
            },
            None => logger.no_file(),
        }
    }
    let LaunchArgs { files, library: launch_library, notices: launch_notices } = launch_imports(&files);
    // a library opened from Tracker (its catalog.snap or folder) is the one that opens, not the last
    // one used; --library and LIGHTCRAFT_LIBRARY still win
    let env_library: Option<std::path::PathBuf> = std::env::var_os("LIGHTCRAFT_LIBRARY").filter(|p| !p.is_empty()).map(Into::into);
    let (library_dir, opened_by_argument) = launch_library_choice(library_dir, env_library.as_ref(), launch_library);
    let (prefs, prefs_warning, keep_prefs_file) = load_prefs(in_memory);
    if let Some(ui) = &prefs {
        lightcraft_ui_egui::i18n::set_language(ui.language);
    }
    // --library, else LIGHTCRAFT_LIBRARY, else the recent libraries if Settings say Ask me, else
    // the library last opened, else the default location (as the desktop app)
    let launch = lightcraft_ui_egui::panels::library_problem::launch_plan(
        in_memory,
        library_dir,
        env_library,
        prefs.as_ref().map(|u| &u.settings),
        lightcraft_engine::library::default_dir(),
    );
    let gpu_crash = gpu_crash_check(in_memory).or_else(|| gpu_lost_check_at(gpu_lost_marker_path().as_deref(), in_memory));
    let window_recovery =
        window_recovery::WindowRecovery::new(config_dir().filter(|_| std::env::var_os("LIGHTCRAFT_NO_PREFS").is_none()), !in_memory);
    let previous_window_failure = window_recovery.previous_failure();
    let renderer = std::env::var("LIGHTCRAFT_RENDERER").unwrap_or_else(|_| "auto".into());
    let force_cpu = renderer == "cpu" || (renderer == "auto" && previous_window_failure.is_some());
    let safe_start = force_cpu || (renderer == "auto" && gpu_crash.is_some());
    let gpu_on = !force_cpu && prefs.as_ref().is_none_or(|u| u.settings.gpu) && gpu_crash.is_none();
    lightcraft_engine::gpu::set_enabled(gpu_on);
    let workers = worker_threads(std::env::var("LIGHTCRAFT_THREADS").ok().as_deref(), std::thread::available_parallelism().map_or(4, std::num::NonZero::get));
    log::info!(target: "lightcraft_haiku", "{workers} worker threads");
    if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(workers).build_global() {
        log::warn!("Worker pool: {e}");
    }
    // Cameras and memory cards (File ▸ Import from Device): the system tells local volumes from
    // the network shares mounted beside them at the root.
    lightcraft_engine::devices::set_volume_lister(host::volumes);
    // As the desktop app's hdr_display::report: app_server composes the screen in 8-bit sRGB
    // (B_RGB32 frame buffers; the drivers set no HDR transfer function or metadata), and the
    // window renderer draws into it, so the window is SDR.
    log::info!(target: "lightcraft_haiku", "display: SDR (air/OS's app_server and graphics drivers have no HDR output)");
    let scale = host::ui_scale();
    let window = prefs.as_ref().and_then(|u| u.window).and_then(lightcraft_ui_egui::window_place::WindowPlace::checked);
    let size = window.map_or(lightcraft_ui_egui::window_place::DEFAULT_SIZE, |w| w.size);
    host::open_window(&host::WindowSpec {
        title: "LightCraft".into(),
        position: None,
        size: [size[0] * scale, size[1] * scale],
        min_size: lightcraft_ui_egui::window_place::MIN_SIZE.map(|n| n * scale),
        zoomed: window.is_some_and(|w| w.zoomed),
    });
    let worker = std::thread::Builder::new()
        .name("LightCraft interface".into())
        .stack_size(16 << 20)
        .spawn(move || {
            let result = guard("Interface", || {
                let ctx = egui::Context::default();
                ctx.set_embed_viewports(true);
                let (session, mut problem) = open_session(in_memory, &launch, seed_demo && files.is_empty());
                let mut app = LightcraftApp::new(session.with_default_sam3_model(), services::native(log_file.as_deref()));
                if let Some(ui) = prefs {
                    app.ui = ui;
                }
                if launch == Launch::Ask {
                    problem = Some(LibraryProblem::choose_later(&app.ui.settings));
                }
                app.alt_asks_for_library = matches!(launch, Launch::Open { alt_asks: true, .. });
                app.integrated_titlebar = false;
                app.gpu_notice = true;
                app.notices.extend(prefs_warning);
                app.notices.extend(launch_notices);
                if problem.is_none() && files.is_empty() {
                    lightcraft_ui_egui::panels::empty_library::open_on_empty_library(&mut app);
                }
                let writer = PrefsWriter {
                    path: if in_memory { None } else { prefs_path() },
                    written: serde_json::to_vec_pretty(&app.ui).unwrap_or_default(),
                    library: app.ui.settings.library_path.clone(),
                    keep_file: keep_prefs_file,
                    ..Default::default()
                };
                // a library opened from its catalog file is the one in use from now on, as Open Library makes it
                // (after the snapshot above, so the writer saves it)
                if opened_by_argument
                    && problem.is_none()
                    && let Some(dir) = app.session.library.as_ref().filter(|l| l.on_disk).map(|l| l.dir.to_string_lossy().into_owned())
                {
                    let settings = &mut app.ui.settings;
                    if settings.library_path != dir && !settings.library_path.is_empty() {
                        let previous = settings.library_path.clone();
                        lightcraft_ui_egui::panels::settings::remember_library(&mut settings.recent_libraries, &previous);
                    }
                    lightcraft_ui_egui::panels::settings::remember_library(&mut settings.recent_libraries, &dir);
                    settings.library_path = dir;
                }
                if let Some(notice) = gpu_crash {
                    app.ui.settings.gpu = false;
                    app.notices.push(notice);
                }
                if force_cpu {
                    app.ui.settings.gpu = false;
                }
                if let Some(reason) = previous_window_failure {
                    app.notices.push(format!("The native GPU window failed last time ({reason}). This launch uses {}. To retry hardware, start with LIGHTCRAFT_RENDERER=vulkan.", if safe_start { "the CPU" } else { "the explicitly selected renderer" }));
                }
                if let Some(port) = control_port {
                    app = app.with_control(control_server::start(port, ctx.clone()));
                }
                if let Some(mut problem) = problem {
                    problem.pending_import = files;
                    app.library_problem = Some(problem);
                } else if !files.is_empty() {
                    if let Err(e) = lightcraft_ui_egui::import::start_paths(&mut app, files) {
                        app.notices.push(e);
                    }
                    app.ui.view = lightcraft_ui_egui::state::ViewMode::PhotoGrid;
                }
                let gpu_lost = gpu_lost_marker_path().filter(|_| !in_memory);
                let mut shell = shell::Shell::new(app, writer, ctx, scale, window_recovery, gpu_lost, safe_start)?;
                shell.run();
                shell.finish();
                Ok(())
            });
            if let Err(e) = &result {
                log::error!("{e}");
                host::alert("LightCraft", e);
            }
            host::quit();
            result
        })
        .map_err(|e| e.to_string())?;
    host::run();
    worker.join().map_err(|_| "Interface thread stopped unexpectedly".to_string())?
}

/// The worker pool's size: `LIGHTCRAFT_THREADS` when it is a number from 1 to 256, else the
/// processors, at most [`MAX_WORKERS`].
fn worker_threads(env: Option<&str>, cpus: usize) -> usize {
    match env.and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(n @ 1..=256) => n,
        _ => cpus.clamp(1, MAX_WORKERS),
    }
}

/// See docs/haiku.md (Performance on air/OS): on the X399 (16 cores) 4, 6 and 8 workers develop
/// and brush alike, 8 export fastest, and 16 are slower at everything.
const MAX_WORKERS: usize = 8;

#[cfg(test)]
mod worker_tests {
    #[test]
    fn worker_threads_take_the_variable_or_the_capped_processor_count() {
        use super::{MAX_WORKERS, worker_threads};
        assert_eq!(worker_threads(None, 64), MAX_WORKERS);
        assert_eq!(worker_threads(None, 2), 2);
        assert_eq!(worker_threads(None, 0), 1);
        assert_eq!(worker_threads(Some("16"), 32), 16);
        assert_eq!(worker_threads(Some(" 3 "), 32), 3);
        assert_eq!(worker_threads(Some("0"), 32), MAX_WORKERS);
        assert_eq!(worker_threads(Some("lots"), 4), 4);
    }
}
