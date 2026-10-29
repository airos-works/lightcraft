//! The Be print kit (`native/print.cpp`): the print server's printers, their drivers' settings and
//! jobs whose pages are drawn as bitmaps. Safe wrappers only; what to print where is decided by
//! `apps/lightcraft-haiku/src/print.rs`. Taken from WordCraft's host (its `wc_print_` functions).

use std::ffi::{CStr, CString, c_char, c_void};

/// What a job asks of a printer's settings. Keep in step with `LcPrintOptions`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RawOptions {
    pub copies: i32,
    pub collate: i32,
    /// -1 the printer's default, 0 one-sided, 1 long edge, 2 short edge.
    pub duplex: i32,
    pub grayscale: i32,
    /// The pages the job will have: its page range is all of them.
    pub pages: i32,
}

/// A started job's paper (left, top, right, bottom in points), resolution and the page range
/// and copies its setup chose. Keep in step with `LcPrintJobInfo`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct JobInfo {
    pub paper: [f32; 4],
    pub printable: [f32; 4],
    pub xres: i32,
    pub yres: i32,
    pub first_page: i32,
    pub last_page: i32,
    pub copies: i32,
}

/// A bitmap of a page: `B_RGB32` (B, G, R, A bytes) or, `gray`, `B_GRAY8`, drawn into
/// `destination` (left, top, right, bottom in the page's points).
pub struct Tile<'a> {
    pub pixels: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub gray: bool,
    pub destination: [f32; 4],
}

/// Keep in step with `LcPrintTile`.
#[repr(C)]
struct RawTile {
    pixels: *const u8,
    width: i32,
    height: i32,
    stride: i32,
    gray: i32,
    destination: [f32; 4],
}

#[repr(C)]
struct RawJob {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn lc_print_free(data: *mut c_void);
    fn lc_print_printers() -> *mut c_char;
    fn lc_print_settings(
        printer: *const c_char,
        mode: i32,
        settings: *const c_void,
        settings_len: usize,
        options: *const RawOptions,
        len: *mut usize,
        status: *mut i32,
    ) -> *mut c_void;
    fn lc_print_system_setup(settings: *const c_void, settings_len: usize, len: *mut usize, status: *mut i32) -> *mut c_void;
    fn lc_print_begin(settings: *const c_void, len: usize, title: *const c_char, info: *mut JobInfo, status: *mut i32) -> *mut RawJob;
    fn lc_print_progress(job: *mut RawJob, done: i32, total: i32);
    fn lc_print_cancelled(job: *mut RawJob) -> i32;
    fn lc_print_page(job: *mut RawJob, tiles: *const RawTile, count: i32, width: f32, height: f32, x: f32, y: f32, filter: i32) -> i32;
    fn lc_print_commit(job: *mut RawJob) -> i32;
    fn lc_print_cancel(job: *mut RawJob);
    fn lc_print_shift_down() -> i32;
}

// The Be API's status codes the native side reports.
pub const CANCELED: i32 = i32::MIN + 12;
pub const NO_INIT: i32 = i32::MIN + 13;
pub const NO_MEMORY: i32 = i32::MIN;
pub const ENTRY_NOT_FOUND: i32 = i32::MIN + 0x6000 + 3;
pub const UNSUPPORTED: i32 = i32::MIN + 0x6000 + 14;

fn c(s: &str) -> CString {
    CString::new(s.replace('\0', "")).unwrap_or_default()
}

/// Takes `len` malloc'd bytes from the native side.
fn take(raw: *mut c_void, len: usize) -> Vec<u8> {
    // SAFETY: `len` bytes at `raw` (not null), freed right after the copy.
    let bytes = unsafe { std::slice::from_raw_parts(raw.cast::<u8>(), len) }.to_vec();
    // SAFETY: allocated by the native side with malloc.
    unsafe { lc_print_free(raw) };
    bytes
}

/// The print server's printers, one a line: name, driver, transport, default (0/1), comments,
/// tab-separated.
pub fn printers() -> String {
    // SAFETY: no arguments; the result is a malloc'd C string or NULL.
    let raw = unsafe { lc_print_printers() };
    if raw.is_null() {
        return String::new();
    }
    // SAFETY: a NUL-terminated string from lc_print_printers, freed right after.
    let text = unsafe { CStr::from_ptr(raw) }.to_string_lossy().into_owned();
    // SAFETY: allocated by the native side with malloc.
    unsafe { lc_print_free(raw.cast()) };
    text
}

/// Whether Shift is held now.
pub fn shift_down() -> bool {
    // SAFETY: no arguments.
    unsafe { lc_print_shift_down() != 0 }
}

/// Which of a printer's settings [`settings`] gives.
#[derive(Clone, Copy, Debug)]
pub enum Mode {
    /// The driver's defaults.
    Defaults = 0,
    /// The driver's page setup dialog, starting from the given settings.
    PageSetup = 1,
    /// The driver's job setup dialog.
    JobSetup = 2,
    /// The given settings with the options applied.
    AsTheyAre = 3,
}

/// A printer's settings (a flattened `BMessage`), or the Be API status why not
/// ([`CANCELED`] for a cancelled dialog, [`UNSUPPORTED`] for a driver without defaults).
pub fn settings(printer: &str, mode: Mode, base: Option<&[u8]>, options: Option<&RawOptions>) -> Result<Vec<u8>, i32> {
    let (ptr, len) = base.map_or((std::ptr::null(), 0), |b| (b.as_ptr().cast::<c_void>(), b.len()));
    let opts = options.map_or(std::ptr::null(), |o| o as *const RawOptions);
    let (mut out_len, mut status) = (0usize, 0i32);
    let name = c(printer);
    // SAFETY: valid C string, buffer and options for the call's duration; the result is
    // malloc'd (`out_len` bytes) or NULL.
    let raw = unsafe { lc_print_settings(name.as_ptr(), mode as i32, ptr, len, opts, &mut out_len, &mut status) };
    if raw.is_null() { Err(status) } else { Ok(take(raw, out_len)) }
}

/// The system's print setup (the print server's window with its printer menu, then that
/// printer's page and job setup), starting from `base`.
pub fn system_setup(base: Option<&[u8]>) -> Result<Vec<u8>, i32> {
    let (ptr, len) = base.map_or((std::ptr::null(), 0), |b| (b.as_ptr().cast::<c_void>(), b.len()));
    let (mut out_len, mut status) = (0usize, 0i32);
    // SAFETY: a valid buffer for the call's duration; the result is malloc'd or NULL.
    let raw = unsafe { lc_print_system_setup(ptr, len, &mut out_len, &mut status) };
    if raw.is_null() { Err(status) } else { Ok(take(raw, out_len)) }
}

/// A job between [`Job::begin`] and its commit; dropped without a commit, it is cancelled.
pub struct Job(*mut RawJob);

impl Job {
    /// Starts a job of `settings` named `title` (a progress window opens), and its paper.
    pub fn begin(settings: &[u8], title: &str) -> Result<(Job, JobInfo), i32> {
        let title = c(title);
        let mut info = JobInfo::default();
        let mut status = 0i32;
        // SAFETY: valid buffer, C string and out-pointers for the call's duration.
        let job = unsafe { lc_print_begin(settings.as_ptr().cast(), settings.len(), title.as_ptr(), &mut info, &mut status) };
        if job.is_null() { Err(status) } else { Ok((Job(job), info)) }
    }

    /// Whether the progress window's Cancel was pressed.
    pub fn cancelled(&self) -> bool {
        // SAFETY: a live job.
        unsafe { lc_print_cancelled(self.0) != 0 }
    }

    pub fn progress(&self, done: usize, total: usize) {
        // SAFETY: a live job.
        unsafe { lc_print_progress(self.0, done.min(i32::MAX as usize) as i32, total.min(i32::MAX as usize) as i32) }
    }

    /// Adds a page of `size` points made of `tiles`, its top left `at` points from the
    /// printable rectangle's; `filter` smooths bitmaps the printer draws larger than they are.
    pub fn page(&self, tiles: &[Tile<'_>], size: (f32, f32), at: (f32, f32), filter: bool) -> Result<(), i32> {
        let raw: Vec<RawTile> = tiles
            .iter()
            .map(|t| RawTile {
                pixels: t.pixels.as_ptr(),
                width: t.width.min(i32::MAX as usize) as i32,
                height: t.height.min(i32::MAX as usize) as i32,
                stride: (t.width * if t.gray { 1 } else { 4 }).min(i32::MAX as usize) as i32,
                gray: i32::from(t.gray),
                destination: t.destination,
            })
            .collect();
        // A tile whose pixels are fewer than its size says would be read past: none is sent.
        if tiles.iter().any(|t| t.pixels.len() < t.width * t.height * if t.gray { 1 } else { 4 }) {
            return Err(NO_MEMORY);
        }
        // SAFETY: `raw` and the pixels it points to (checked above to hold the tiles) live
        // through the call; the native side copies them.
        let status =
            unsafe { lc_print_page(self.0, raw.as_ptr(), raw.len().min(i32::MAX as usize) as i32, size.0, size.1, at.0, at.1, i32::from(filter)) };
        if status == 0 { Ok(()) } else { Err(status) }
    }

    /// Hands the spooled job to the print server.
    pub fn commit(mut self) -> Result<(), i32> {
        let job = std::mem::replace(&mut self.0, std::ptr::null_mut());
        // SAFETY: a live job, freed by the call.
        let status = unsafe { lc_print_commit(job) };
        if status == 0 { Ok(()) } else { Err(status) }
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: a live job that wasn't committed, freed by the call.
            unsafe { lc_print_cancel(self.0) };
        }
    }
}

// SAFETY: the native job is used from one thread at a time (the one printing); its window is
// locked by the native side.
unsafe impl Send for Job {}
