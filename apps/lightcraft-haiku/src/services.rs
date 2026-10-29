//! Be API platform services for the complete LightCraft interface.
use crate::host;
use lightcraft_ui_egui::{
    Services,
    pick::{PickKind, PickRequest},
};
use std::sync::{
    Mutex,
    atomic::{AtomicI32, Ordering},
    mpsc::{self, Receiver, Sender},
};

pub const SIGNATURE: &str = "application/x-vnd.jmgasper-LightCraft";
type Pending = (i32, Sender<Vec<String>>);
static PENDING: Mutex<Option<Pending>> = Mutex::new(None);
static DEFERRED: Mutex<Vec<(host::RawEvent, String)>> = Mutex::new(Vec::new());
static NEXT: AtomicI32 = AtomicI32::new(1);

pub fn clipboard_text() -> Option<String> {
    host::clipboard_get("text/plain").map(|v| String::from_utf8_lossy(&v).trim_end_matches('\0').to_string())
}
pub fn take_deferred() -> Vec<(host::RawEvent, String)> {
    std::mem::take(&mut *DEFERRED.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
}
fn picker(req: PickRequest) -> Result<Receiver<Vec<String>>, String> {
    let (tx, rx) = mpsc::channel();
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut pending = PENDING.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if pending.is_some() {
        return Err("A file panel is already open".into());
    }
    *pending = Some((id, tx));
    drop(pending);
    let mode = match req.kind {
        PickKind::Files => 3,
        PickKind::File => 0,
        PickKind::Folder => 2,
        PickKind::Save => 1,
    };
    let extensions = req.filter.as_ref().map_or(&[][..], |(_, e)| *e);
    let default_title = match req.kind {
        PickKind::Files => "Import Photos",
        PickKind::File => "Choose File",
        PickKind::Folder => "Choose Folder",
        PickKind::Save => "Save File",
    };
    host::file_panel(
        id,
        mode,
        req.file_name.as_deref().unwrap_or(""),
        extensions,
        req.title.as_deref().unwrap_or_else(|| lightcraft_ui_egui::i18n::tr(default_title)),
    );
    Ok(rx)
}
pub fn answer(id: i32, picked: bool, text: &str) {
    let mut pending = PENDING.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if pending.as_ref().is_some_and(|(n, _)| *n == id)
        && let Some((_, tx)) = pending.take()
    {
        let paths = if picked { text.lines().filter(|p| !p.is_empty()).map(str::to_owned).collect() } else { Vec::new() };
        let _ = tx.send(paths);
    }
}
// Embedded dialog buttons still use the synchronous service. The Be API main
// thread remains responsive; preserve unrelated events for the next editor frame.
fn pick_sync(req: PickRequest) -> Vec<String> {
    let Ok(rx) = picker(req) else { return Vec::new() };
    loop {
        match rx.try_recv() {
            Ok(paths) => return paths,
            Err(mpsc::TryRecvError::Disconnected) => return Vec::new(),
            Err(mpsc::TryRecvError::Empty) => {}
        }
        host::wait(Some(std::time::Duration::from_millis(100)));
        while let Some((ev, text)) = host::next_event() {
            if ev.kind == host::kind::DIALOG_ANSWER {
                answer(ev.id, ev.x != 0.0, &text);
            } else {
                DEFERRED.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push((ev, text));
            }
        }
    }
}
fn files(title: &'static str, extensions: &'static [&'static str]) -> lightcraft_ui_egui::PickFiles {
    Box::new(move || pick_sync(PickRequest::files(Some(lightcraft_ui_egui::i18n::tr(title).into()), title, extensions)))
}
fn save(title: &'static str, extensions: &'static [&'static str]) -> lightcraft_ui_egui::SaveFile {
    Box::new(move |name| pick_sync(PickRequest::save(lightcraft_ui_egui::i18n::tr(title), title, extensions, name)).into_iter().next())
}
fn write(path: &str, bytes: &[u8]) -> Result<(), String> {
    lightcraft_engine::export::write_file(path, bytes)?;
    host::type_file(std::path::Path::new(path));
    Ok(())
}
/// [`write`] that never replaces an existing file (screenshots to a new destination).
fn write_new(path: &str, bytes: &[u8]) -> Result<(), String> {
    lightcraft_engine::export::write_file_new(path, bytes)?;
    host::type_file(std::path::Path::new(path));
    Ok(())
}
/// Shows a file selected in its folder, or a folder, in Tracker.
fn reveal(path: &str) -> Result<(), String> {
    if host::reveal(std::path::Path::new(path)) { Ok(()) } else { Err(format!("Could not show {path} in Tracker")) }
}
/// Settings ▸ Display ▸ Choose Profile…, starting in the monitor profiles folder (made so it can
/// be found, as the screen profile's first read does).
fn pick_display_profile() -> Vec<String> {
    if let Some(dir) = crate::displays::profiles_dir() {
        let _ = std::fs::create_dir_all(dir);
    }
    let title = "Choose Monitor Profile";
    let mut req = PickRequest::files(Some(lightcraft_ui_egui::i18n::tr(title).into()), title, &["icc", "icm"]);
    // An Open panel's `file_name` is the folder it starts in.
    req.file_name = crate::prefs::display_profile_dir().map(|d| d.to_string_lossy().into_owned());
    pick_sync(req)
}
/// The profile of the screen showing the window (no screen keeps one on air/OS: the user's file
/// in `~/config/settings/ColorProfiles` named after the monitor, see `displays.rs`).
fn screen_profile() -> lightcraft_ui_egui::ScreenProfileFn {
    let mut reader = crate::displays::ScreenProfile::new(crate::displays::profiles_dir());
    Box::new(move || {
        let w = host::window_info().inner.map(f64::from);
        reader.read(&host::displays(), w)
    })
}
pub fn native(log_file: Option<&std::path::Path>) -> Services {
    Services {
        picker: Some(Box::new(picker)),
        pick_files: Some(files("Import Photos", lightcraft_engine::import::EXTENSIONS)),
        pick_denoise_model: Some(files("Choose Denoise Model", &["onnx"])),
        pick_model_file: Some(files("Choose Face Model", &["onnx"])),
        pick_preset_files: Some(files("Import Presets", lightcraft_ui_egui::pick::PRESET_EXTENSIONS)),
        pick_tracklog: Some(files("Import GPS Tracklog", &["gpx"])),
        pick_display_profile: Some(Box::new(pick_display_profile)),
        screen_profile: Some(screen_profile()),
        save_preset_file: Some(save("Export Presets", &["lcpreset"])),
        pick_keyword_list: Some(files("Import Keywords", &["txt", "utf8"])),
        save_keyword_list: Some(save("Export Keywords", &["txt"])),
        pick_curve_preset_files: Some(files("Import Point Curves", &["lccurve", "json"])),
        save_curve_preset_file: Some(save("Export Point Curve", &["lccurve"])),
        pick_slideshow_templates: Some(files("Import Slideshow Templates", &["json"])),
        save_slideshow_template: Some(save("Export Slideshow Template", &["json"])),
        pick_print_templates: Some(files("Import Print Templates", &[lightcraft_engine::print::template::FILE_EXT])),
        save_print_template: Some(save("Export Print Template", &[lightcraft_engine::print::template::FILE_EXT])),
        pick_lightroom_catalog: Some(Box::new(|| pick_sync(PickRequest::file("Import Lightroom Catalog", "Catalog", &["lrcat"])).into_iter().next())),
        pick_folder: Some(Box::new(|| pick_sync(PickRequest::folder("Choose Folder")).into_iter().next())),
        write: Some(Box::new(write)),
        write_new: Some(Box::new(write_new)),
        write_shared: Some(std::sync::Arc::new(write)),
        png: Some(Box::new(|img| {
            lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(img), &lightcraft_codecs::EncodeMeta::default()).unwrap_or_default()
        })),
        log_file: log_file.map(|p| p.display().to_string()),
        reveal: Some(Box::new(reveal)),
        // The SAM 3 model folder's Show in File Manager (a worker calls it): the folder opens.
        open_model_folder: Some(std::sync::Arc::new(reveal)),
        open_url: Some(Box::new(|url| {
            if !url.starts_with("https://") {
                return Err("Only HTTPS links are opened".into());
            }
            if host::open_url(url) { Ok(()) } else { Err("Could not open the link".into()) }
        })),
        open_with: Some(Box::new(crate::editors::open_with)),
        open_all_with: Some(Box::new(crate::editors::open_all_with)),
        check_editor: Some(Box::new(crate::editors::check)),
        system_theme: Some(Box::new(|_| Some(if host::dark() { egui::Theme::Dark } else { egui::Theme::Light }))),
        ..Services::default()
    }
}
