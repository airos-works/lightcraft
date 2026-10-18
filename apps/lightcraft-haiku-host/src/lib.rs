#![cfg(target_os = "haiku")]
//! The native host (`native/host.cpp`): safe wrappers over its C functions. The only module with
//! `unsafe` code; every call here is a plain C call with owned or borrowed buffers.
#![allow(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::ffi::{CStr, CString, c_char, c_void};

/// An input or window event from the native host. Keep in step with `LcEvent` in `host.cpp`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RawEvent {
    pub kind: i32,
    pub buttons: i32,
    pub clicks: i32,
    pub modifiers: u32,
    pub key: i32,
    pub raw_char: i32,
    pub repeat: i32,
    pub id: i32,
    pub x: f32,
    pub y: f32,
    pub dx: f32,
    pub dy: f32,
    pub text_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct RawWindowInfo {
    inner: [f32; 4],
    outer: [f32; 4],
    screen: [f32; 2],
    minimized: i32,
    active: i32,
    zoomed: i32,
    fullscreen: i32,
}

/// The kinds of [`RawEvent`].
pub mod kind {
    pub const MOUSE_MOVED: i32 = 1;
    pub const MOUSE_DOWN: i32 = 2;
    pub const MOUSE_UP: i32 = 3;
    pub const WHEEL: i32 = 4;
    pub const KEY_DOWN: i32 = 5;
    pub const KEY_UP: i32 = 6;
    pub const RESIZED: i32 = 7;
    pub const FOCUS: i32 = 8;
    pub const CLOSE: i32 = 9;
    pub const OPEN_FILES: i32 = 10;
    pub const DROPPED_FILES: i32 = 11;
    pub const POINTER_GONE: i32 = 12;
    pub const QUIT_REQUESTED: i32 = 13;
    pub const DIALOG_ANSWER: i32 = 14;
    pub const MODIFIERS: i32 = 15;
    pub const MOVED: i32 = 16;
    pub const MINIMIZED: i32 = 17;
    pub const COLORS: i32 = 18;
}

unsafe extern "C" {
    fn lc_host_init(signature: *const c_char) -> i32;
    fn lc_host_open_window(title: *const c_char, x: f32, y: f32, w: f32, h: f32, min_w: f32, min_h: f32, zoomed: i32) -> i32;
    fn lc_host_run();
    fn lc_host_quit();
    fn lc_host_wait(timeout_us: i64) -> i32;
    fn lc_host_wake();
    fn lc_host_next_event(event: *mut RawEvent) -> i32;
    fn lc_host_event_text() -> *const c_char;
    fn lc_host_present(pixels: *const u32, w: i32, h: i32, x0: i32, y0: i32, x1: i32, y1: i32, scale: f32);
    fn lc_host_view_size(w: *mut f32, h: *mut f32);
    fn lc_host_window_info(info: *mut RawWindowInfo);
    fn lc_host_set_title(title: *const c_char);
    fn lc_host_window_command(command: i32, a: f32, b: f32);
    fn lc_host_set_cursor(id: i32);
    fn lc_host_modifiers() -> u32;
    fn lc_host_ui_scale() -> f32;
    fn lc_host_device_scale() -> f32;
    fn lc_host_dark() -> i32;
    fn lc_host_locale(languages: *mut *mut c_char) -> *mut c_char;
    fn lc_free(data: *mut c_void);
    fn lc_clipboard_get(mime: *const c_char, len: *mut usize) -> *mut c_void;
    fn lc_clipboard_set(text: *const c_char, text_len: usize, html: *const c_char, html_len: usize) -> i32;
    fn lc_clipboard_set_picture(mime: *const c_char, data: *const c_void, len: usize) -> i32;
    fn lc_host_file_panel(id: i32, save: i32, name: *const c_char, extensions: *const c_char, title: *const c_char) -> i32;
    fn lc_host_open_url(url: *const c_char) -> i32;
    fn lc_host_alert(title: *const c_char, text: *const c_char);
    fn lc_host_type_file(path: *const c_char) -> i32;
    fn lc_host_claim_types(signature: *const c_char, types: *const c_char) -> i32;
}

/// Text for C: interior NULs (never in what we pass) are dropped rather than failing.
fn c(s: &str) -> CString {
    CString::new(s.replace('\0', "")).unwrap_or_default()
}

/// Takes a `malloc`'d C string from the host.
fn take_string(p: *mut c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: the host returns NUL-terminated strings from strdup.
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    // SAFETY: allocated by the host's malloc.
    unsafe { lc_free(p.cast()) };
    s
}

/// Takes `malloc`'d bytes from the host.
fn take_bytes(p: *mut c_void, len: usize) -> Option<Vec<u8>> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the host allocated `len` bytes at `p`.
    let v = unsafe { std::slice::from_raw_parts(p.cast::<u8>(), len) }.to_vec();
    // SAFETY: allocated by the host's malloc.
    unsafe { lc_free(p) };
    Some(v)
}

/// Creates the BApplication. Call once, on the main thread, before anything else here.
pub fn init(signature: &str) -> Result<(), String> {
    let sig = c(signature);
    // SAFETY: a valid C string.
    match unsafe { lc_host_init(sig.as_ptr()) } {
        0 => Ok(()),
        e => Err(format!("the application couldn't start ({e})")),
    }
}

/// Where and how big the window opens: content top-left (`None`: centred) and content size.
pub struct WindowSpec {
    pub title: String,
    pub position: Option<[f32; 2]>,
    pub size: [f32; 2],
    pub min_size: [f32; 2],
    pub zoomed: bool,
}

pub fn open_window(spec: &WindowSpec) {
    let title = c(&spec.title);
    let [x, y] = spec.position.unwrap_or([-1.0, -1.0]);
    // SAFETY: a valid C string; the host copies it.
    unsafe {
        lc_host_open_window(title.as_ptr(), x, y, spec.size[0], spec.size[1], spec.min_size[0], spec.min_size[1], i32::from(spec.zoomed));
    }
}

/// Runs the application's message loop on this (the main) thread until [`quit`].
pub fn run() {
    // SAFETY: called once, after `init`.
    unsafe { lc_host_run() }
}

/// Ends the message loop (the window closes and [`run`] returns).
pub fn quit() {
    // SAFETY: no arguments.
    unsafe { lc_host_quit() }
}

/// Waits for an event, a [`wake`] or the timeout (`None`: no timeout).
pub fn wait(timeout: Option<std::time::Duration>) {
    let us = timeout.map_or(-1, |t| i64::try_from(t.as_micros()).unwrap_or(i64::MAX));
    // SAFETY: no pointers.
    unsafe { lc_host_wait(us) };
}

/// Ends a [`wait`] early; callable from any thread.
pub fn wake() {
    // SAFETY: no arguments.
    unsafe { lc_host_wake() }
}

/// The next queued event and its text.
pub fn next_event() -> Option<(RawEvent, String)> {
    let mut ev = RawEvent::default();
    // SAFETY: `ev` is a valid, writable `LcEvent`.
    if unsafe { lc_host_next_event(&mut ev) } == 0 {
        return None;
    }
    let text = if ev.text_len == 0 {
        String::new()
    } else {
        // SAFETY: the text stays valid until the next `lc_host_next_event`; it holds `text_len` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(lc_host_event_text().cast::<u8>(), ev.text_len as usize) };
        String::from_utf8_lossy(bytes).into_owned()
    };
    Some((ev, text))
}

/// Shows `pixels` (`width` × `height` device pixels, 0xAARRGGBB, `scale` per logical pixel) in
/// the window; only the `dirty` rectangle ([x0, y0, x1, y1), pixels) is copied and redrawn.
pub fn present(pixels: &[u32], width: usize, height: usize, dirty: [usize; 4], scale: f32) {
    if pixels.len() < width.saturating_mul(height) {
        return;
    }
    let to_i32 = |v: usize| i32::try_from(v).unwrap_or(i32::MAX);
    // SAFETY: `pixels` holds width × height values; the host only reads them.
    unsafe {
        lc_host_present(pixels.as_ptr(), to_i32(width), to_i32(height), to_i32(dirty[0]), to_i32(dirty[1]), to_i32(dirty[2]), to_i32(dirty[3]), scale)
    }
}

/// The canvas size in logical pixels.
pub fn view_size() -> (f32, f32) {
    let (mut w, mut h) = (0.0, 0.0);
    // SAFETY: two writable floats.
    unsafe { lc_host_view_size(&mut w, &mut h) };
    (w, h)
}

/// The window's content and frame rectangles (logical screen pixels), the screen size and its
/// state.
#[derive(Clone, Copy, Debug, Default)]
pub struct WindowInfo {
    pub inner: [f32; 4],
    pub outer: [f32; 4],
    pub screen: [f32; 2],
    pub minimized: bool,
    pub active: bool,
    pub fullscreen: bool,
}

pub fn window_info() -> WindowInfo {
    let mut raw = RawWindowInfo::default();
    // SAFETY: a writable `LcWindowInfo`.
    unsafe { lc_host_window_info(&mut raw) };
    WindowInfo {
        inner: raw.inner,
        outer: raw.outer,
        screen: raw.screen,
        minimized: raw.minimized != 0,
        active: raw.active != 0,
        fullscreen: raw.fullscreen != 0,
    }
}

pub fn set_title(title: &str) {
    let t = c(title);
    // SAFETY: a valid C string; the host copies it.
    unsafe { lc_host_set_title(t.as_ptr()) }
}

/// What [`window_command`] does.
#[derive(Clone, Copy, Debug)]
pub enum WindowCommand {
    /// Frame's top-left corner to (x, y).
    MoveTo(f32, f32),
    /// Content size.
    ResizeTo(f32, f32),
    Minimize(bool),
    Zoom,
    Activate,
    MinSize(f32, f32),
    Fullscreen(bool),
}

pub fn window_command(cmd: WindowCommand) {
    let (id, a, b) = match cmd {
        WindowCommand::MoveTo(x, y) => (1, x, y),
        WindowCommand::ResizeTo(w, h) => (2, w, h),
        WindowCommand::Minimize(m) => (3, if m { 1.0 } else { 0.0 }, 0.0),
        WindowCommand::Zoom => (4, 1.0, 0.0),
        WindowCommand::Activate => (5, 0.0, 0.0),
        WindowCommand::MinSize(w, h) => (6, w, h),
        WindowCommand::Fullscreen(on) => (7, f32::from(u8::from(on)), 0.0),
    };
    // SAFETY: no pointers.
    unsafe { lc_host_window_command(id, a, b) }
}

/// One of Haiku's `BCursorID`s.
pub fn set_cursor(id: i32) {
    // SAFETY: no pointers; unknown ids fall back to the default cursor.
    unsafe { lc_host_set_cursor(id) }
}

/// Haiku's modifier key bits (`modifiers()`).
pub fn modifiers() -> u32 {
    // SAFETY: no arguments.
    unsafe { lc_host_modifiers() }
}

/// The interface scale: the system font size against 12 pt.
pub fn ui_scale() -> f32 {
    // SAFETY: no arguments.
    unsafe { lc_host_ui_scale() }
}

/// Frame buffer pixels per logical pixel: air/OS draws the desktop at its monitors' density
/// (2 at 200 %); 1 elsewhere. `PHOTOCRAFT_DEVICE_SCALE` overrides it.
pub fn device_scale() -> f32 {
    if let Some(s) = std::env::var("PHOTOCRAFT_DEVICE_SCALE").ok().and_then(|s| s.parse::<f32>().ok()).filter(|s| (0.5..=4.0).contains(s)) {
        return s;
    }
    // SAFETY: no arguments.
    let s = unsafe { lc_host_device_scale() };
    if s.is_finite() && (1.0..=4.0).contains(&s) { s } else { 1.0 }
}

/// Whether the system's panel colour is dark.
pub fn dark() -> bool {
    // SAFETY: no arguments.
    unsafe { lc_host_dark() != 0 }
}

/// The system's formatting conventions (`en_US`) and preferred languages (`en`, `de`…).
pub fn locale() -> (String, Vec<String>) {
    let mut langs: *mut c_char = std::ptr::null_mut();
    // SAFETY: a writable pointer; both strings come back from strdup.
    let id = take_string(unsafe { lc_host_locale(&mut langs) });
    let langs = take_string(langs).lines().map(str::to_string).filter(|l| !l.is_empty()).collect();
    (id, langs)
}

/// Sets the locale variables the interface reads (`LANG`, `LC_*`) from the system's settings,
/// where the environment doesn't have them (an app launched from Tracker or the Deskbar).
/// Call before any other thread starts.
pub fn export_locale() {
    if std::env::var_os("LANG").is_some() || std::env::var_os("LC_ALL").is_some() {
        return;
    }
    let (conventions, languages) = locale();
    let safe = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '@');
    let lang = languages.first().filter(|l| safe(l)).map(|l| l.replace('-', "_"));
    let lang = match (lang, safe(&conventions)) {
        // "de" with conventions "de_CH": de_CH; otherwise the language alone.
        (Some(l), true) if conventions.starts_with(&format!("{l}_")) => conventions.clone(),
        (Some(l), _) => l,
        (None, true) => conventions.clone(),
        (None, false) => return,
    };
    // SAFETY: called from `main` before any thread is spawned, so nothing reads the environment
    // concurrently.
    unsafe {
        std::env::set_var("LANG", format!("{lang}.UTF-8"));
        if safe(&conventions) {
            for var in ["LC_NUMERIC", "LC_MONETARY", "LC_PAPER", "LC_MEASUREMENT", "LC_TIME"] {
                if std::env::var_os(var).is_none() {
                    std::env::set_var(var, format!("{conventions}.UTF-8"));
                }
            }
        }
        if languages.len() > 1 && std::env::var_os("LANGUAGE").is_none() {
            std::env::set_var("LANGUAGE", languages.iter().filter(|l| safe(l)).map(|l| l.replace('-', "_")).collect::<Vec<_>>().join(":"));
        }
    }
}

/// The clipboard's data of a MIME type.
pub fn clipboard_get(mime: &str) -> Option<Vec<u8>> {
    let m = c(mime);
    let mut len = 0usize;
    // SAFETY: a valid C string and a writable length.
    let p = unsafe { lc_clipboard_get(m.as_ptr(), &mut len) };
    take_bytes(p, len)
}

/// Replaces the clipboard with `text` and, if given, its HTML.
pub fn clipboard_set(text: &str, html: Option<&str>) -> bool {
    let (h, hl) = html.map_or((std::ptr::null(), 0), |h| (h.as_ptr().cast::<c_char>(), h.len()));
    // SAFETY: the pointers and lengths describe live buffers the host copies.
    unsafe { lc_clipboard_set(text.as_ptr().cast(), text.len(), h, hl) == 0 }
}

/// Replaces the clipboard with an encoded picture.
pub fn clipboard_set_picture(mime: &str, data: &[u8]) -> bool {
    let m = c(mime);
    // SAFETY: valid buffers the host copies.
    unsafe { lc_clipboard_set_picture(m.as_ptr(), data.as_ptr().cast(), data.len()) == 0 }
}

/// Shows an Open panel (offering `extensions`; `*` is any file) or a Save panel (suggesting
/// `name`); the answer comes back as a [`kind::DIALOG_ANSWER`] event with `id`.
pub fn file_panel(id: i32, mode: i32, name: &str, extensions: &[&str], title: &str) {
    let (n, e, t) = (c(name), c(&extensions.join("\n")), c(title));
    // SAFETY: valid C strings; the host copies them.
    unsafe { lc_host_file_panel(id, mode, n.as_ptr(), e.as_ptr(), t.as_ptr()) };
}

/// Opens a URL with its preferred application.
pub fn open_url(url: &str) -> bool {
    let u = c(url);
    // SAFETY: a valid C string.
    unsafe { lc_host_open_url(u.as_ptr()) == 0 }
}

/// Sets a saved file's MIME type (`BEOS:TYPE`) from its extension and content, if it has none.
pub fn type_file(path: &std::path::Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(p) = CString::new(path.as_os_str().as_bytes()) else { return false };
    // SAFETY: a valid C string.
    unsafe { lc_host_type_file(p.as_ptr()) == 0 }
}

/// Makes `signature` the preferred application of those of the MIME types that have none;
/// how many it claimed.
pub fn claim_types(signature: &str, types: &[&str]) -> i32 {
    let (s, t) = (c(signature), c(&types.join("\n")));
    // SAFETY: valid C strings.
    unsafe { lc_host_claim_types(s.as_ptr(), t.as_ptr()) }
}

/// A modal error alert.
pub fn alert(title: &str, text: &str) {
    let (t, x) = (c(title), c(text));
    // SAFETY: valid C strings; the alert copies them.
    unsafe { lc_host_alert(t.as_ptr(), x.as_ptr()) }
}

/// Select an explicitly installed app-private Vulkan driver before threads start.
/// An existing loader override takes precedence. An empty LIGHTCRAFT_VULKAN_ICD
/// opts out of the private driver. Call only during single-threaded startup,
/// alongside `export_locale`, before constructing the application or a device.
pub fn export_vulkan_driver() {
    if std::env::var_os("VK_DRIVER_FILES").is_some() || std::env::var_os("VK_ICD_FILENAMES").is_some() {
        return;
    }
    let manifest = match std::env::var_os("LIGHTCRAFT_VULKAN_ICD") {
        Some(path) if path.is_empty() => return,
        Some(path) => std::path::PathBuf::from(path),
        None => match std::env::var_os("HOME") {
            Some(home) => std::path::PathBuf::from(home).join("config/non-packaged/data/lightcraft/vulkan/icd.json"),
            None => return,
        },
    };
    if !manifest.is_file() {
        return;
    }
    // SAFETY: called at the beginning of main, before locale services, BApplication,
    // the logger, Rayon or any other worker starts. No concurrent environment readers.
    unsafe {
        std::env::set_var("VK_DRIVER_FILES", &manifest);
        std::env::set_var("VK_ICD_FILENAMES", &manifest);
    }
}
