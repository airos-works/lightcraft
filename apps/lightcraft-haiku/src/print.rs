//! File ▸ Print… on air/OS: the Print dialog's composed pages go to a printer through the Be print
//! kit (`lightcraft_haiku_host::print`, `native/print.cpp`) instead of only into a PDF.
//!
//! The system print setup (the print server's window with its printer menu, then that printer's
//! page and job setup, which also asks for copies and the page range) chooses the printer; the
//! print server keeps the answers per application and printer for the next job. Each page, composed at the Print Job panel's resolution,
//! is converted by the print job's colour settings, scaled to the printer's resolution (at most
//! [`MAX_DPI`]), placed on the driver's paper at its real size (centred, shrunk only when it
//! doesn't fit, turned when a landscape page meets portrait paper) and spooled as the strips of
//! rows that aren't white paper, grey ones with one byte a pixel.
//!
//! Colour: the print kit has no colour management. Its bitmaps are device RGB, which drivers
//! (PostScript, PCL, Gutenprint) treat as sRGB. So "Managed by Printer" sends sRGB, also when the
//! panel says Adobe RGB (which only a PDF can carry), and a printer profile converts the pages to
//! the printer's own RGB values, which the driver passes on. A CMYK profile can't be printed this
//! way (the bitmaps are RGB): such a job asks to be saved as a PDF.
//!
//! The placement and strip code is WordCraft's host's (`apps/wordcraft-haiku/src/print.rs`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use lightcraft_engine::print::PageRaster;

/// The highest resolution pages are sent at, in dots per inch. Printers that take more get the
/// bitmaps scaled up by their driver; photos are composed at the Print Job panel's resolution
/// (240 ppi by default) anyway.
pub const MAX_DPI: f32 = 300.0;

/// The largest page bitmap, in pixels (A2 at 300 dpi); bigger pages are sent coarser.
const MAX_PIXELS: f32 = 36_000_000.0;

/// Why a job didn't print.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The user cancelled the print setup, the job's progress window or the activity.
    Cancelled,
    Failed(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Cancelled => f.write_str("printing was cancelled"),
            Error::Failed(message) => f.write_str(message),
        }
    }
}

fn failed(message: impl Into<String>) -> Error {
    Error::Failed(message.into())
}

/// An opaque page bitmap: RGBA, 4 bytes a pixel, rows `width * 4` bytes long.
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

/// The pages of a print, drawn on demand.
pub trait PageSource {
    fn count(&self) -> usize;
    /// The page's size in points (1/72 inch), as it reads.
    fn size(&self, page: usize) -> (f32, f32);
    /// The page `width` x `height` pixels, `scale` pixels a point.
    fn render(&mut self, page: usize, scale: f32, width: usize, height: usize) -> Result<Image, Error>;
}

/// The Print dialog's composed pages, in the print job's colour.
pub struct PhotoPages {
    pages: Vec<PageRaster>,
    color: lightcraft_engine::print_color::PrintColorJob,
    cancel: Arc<AtomicBool>,
}

impl PhotoPages {
    /// `pages` converted as `job` says (see the module's colour notes); `cancel` stops rendering.
    pub fn new(pages: Vec<PageRaster>, job: &lightcraft_engine::print::PrintJob, cancel: Arc<AtomicBool>) -> Result<Self, Error> {
        use lightcraft_engine::print_color::{ManagedSpace, PrintColorJob, PrintProfile};
        let mut settings = job.color.clone();
        if let PrintProfile::Managed { space } = &mut settings.profile {
            *space = ManagedSpace::Srgb;
        }
        let color = PrintColorJob::new(&settings).map_err(failed)?;
        if !matches!(color.channels(), 1 | 3) {
            return Err(failed(
                "the printer profile is CMYK, and air/OS's printer drivers take RGB pages: choose an RGB profile or Managed by Printer, or save a PDF",
            ));
        }
        Ok(PhotoPages { pages, color, cancel })
    }
}

impl PageSource for PhotoPages {
    fn count(&self) -> usize {
        self.pages.len()
    }

    fn size(&self, page: usize) -> (f32, f32) {
        self.pages
            .get(page)
            .filter(|p| p.ppi.is_finite() && p.ppi > 0.0)
            .map_or((612.0, 792.0), |p| (p.width as f32 / p.ppi * 72.0, p.height as f32 / p.ppi * 72.0))
    }

    fn render(&mut self, page: usize, _scale: f32, width: usize, height: usize) -> Result<Image, Error> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let p = self.pages.get(page).ok_or_else(|| failed("the print has fewer pages"))?;
        let device = self.color.convert(&p.rgb).map_err(failed)?;
        let rgba = to_rgba8(&device, self.color.channels());
        let (pw, ph) = (p.width, p.height);
        let image = image::RgbaImage::from_raw(pw, ph, rgba).ok_or_else(|| failed("a page's pixels don't match its size"))?;
        let (w, h) = (u32::try_from(width).unwrap_or(u32::MAX).max(1), u32::try_from(height).unwrap_or(u32::MAX).max(1));
        // Bilinear: the page's white paper stays white next to the photos (no ringing that would
        // spool it as ink), and the composed resolution is near the printer's anyway.
        let image = if (w, h) == (pw, ph) { image } else { image::imageops::resize(&image, w, h, image::imageops::FilterType::Triangle) };
        Ok(Image { width: image.width() as usize, height: image.height() as usize, rgba: image.into_raw() })
    }
}

/// 16-bit samples of `channels` (1 grey or 3 RGB) as opaque 8-bit RGBA.
fn to_rgba8(samples: &[u16], channels: usize) -> Vec<u8> {
    let byte = |v: u16| ((u32::from(v) * 255 + 32767) / 65535) as u8;
    if channels == 1 {
        samples.iter().flat_map(|&v| [byte(v), byte(v), byte(v), 255]).collect()
    } else {
        samples.as_chunks::<3>().0.iter().flat_map(|p| [byte(p[0]), byte(p[1]), byte(p[2]), 255]).collect()
    }
}

/// Where a page goes on the paper, in points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    /// The page is turned a quarter counter-clockwise (its top to the paper's left edge).
    pub rotate: bool,
    /// Its size on paper (after turning and shrinking).
    pub width: f32,
    pub height: f32,
    /// How much it was shrunk to fit (1 = real size).
    pub shrink: f32,
    /// Its top left from the printable rectangle's (where `BPrintJob::DrawView` measures from).
    pub x: f32,
    pub y: f32,
}

/// A page of `page` points on `paper` (left, top, right, bottom points) of which `printable`
/// prints: at its real size, centred, shrunk only when it is bigger than the paper. A landscape
/// page on portrait paper is turned, and the other way round.
pub fn place(page: (f32, f32), paper: [f32; 4], printable: [f32; 4]) -> Placement {
    let (pw, ph) = (page.0.max(1.0), page.1.max(1.0));
    let (paper_w, paper_h) = ((paper[2] - paper[0]).max(1.0), (paper[3] - paper[1]).max(1.0));
    let rotate = (pw > ph) != (paper_w > paper_h) && (pw - ph).abs() > 0.5;
    let (w, h) = if rotate { (ph, pw) } else { (pw, ph) };
    let shrink = (paper_w / w).min(paper_h / h).min(1.0);
    let (w, h) = (w * shrink, h * shrink);
    Placement {
        rotate,
        width: w,
        height: h,
        shrink,
        x: paper[0] + (paper_w - w) / 2.0 - printable[0],
        y: paper[1] + (paper_h - h) / 2.0 - printable[1],
    }
}

/// Pixels a point for a page drawn `width` x `height` points on a printer of `dpi`.
pub fn raster_scale(dpi: f32, width: f32, height: f32) -> f32 {
    let dpi = if dpi.is_finite() && dpi >= 36.0 { dpi.min(MAX_DPI) } else { MAX_DPI };
    let mut scale = dpi / 72.0;
    let pixels = width * height * scale * scale;
    if pixels > MAX_PIXELS {
        scale *= (MAX_PIXELS / pixels).sqrt();
    }
    scale
}

/// `image` turned a quarter counter-clockwise.
pub fn rotate_ccw(image: &Image) -> Image {
    let (w, h) = (image.width, image.height);
    let mut out = vec![255u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            // Source (x, y) lands at (y, w - 1 - x) of the h-wide result.
            let (src, dst) = ((y * w + x) * 4, ((w - 1 - x) * h + y) * 4);
            if let (Some(s), Some(d)) = (image.rgba.get(src..src + 4), out.get_mut(dst..dst + 4)) {
                d.copy_from_slice(s);
            }
        }
    }
    Image { width: h, height: w, rgba: out }
}

/// A strip of a page bitmap to spool: the rows `y..y + height` and the columns that have ink,
/// in the Be API's layout (`B_RGB32`: B, G, R, A bytes; or `B_GRAY8`).
#[derive(Debug, PartialEq, Eq)]
pub struct Strip {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub gray: bool,
    pub pixels: Vec<u8>,
}

/// Rows this many apart or closer go into one strip: fewer, slightly larger bitmaps.
const STRIP_GAP: usize = 8;

/// The strips of `image` that aren't white paper. Strips meet only across white rows, so a
/// rounding at their edges can't show as a line; grey strips take one byte a pixel.
pub fn strips(image: &Image) -> Vec<Strip> {
    let (w, h) = (image.width, image.height);
    let row = |y: usize| image.rgba.get(y * w * 4..(y + 1) * w * 4).unwrap_or(&[]);
    // Each row's first and last inked column.
    let ink: Vec<Option<(usize, usize)>> = (0..h)
        .map(|y| {
            let r = row(y);
            let inked = |x: &usize| r.get(x * 4..x * 4 + 3).is_some_and(|p| p != [255, 255, 255]);
            let first = (0..w).find(inked)?;
            let last = (first..w).rev().find(inked)?;
            Some((first, last))
        })
        .collect();
    let mut out = Vec::new();
    let mut y = 0;
    while y < h {
        let Some(Some((mut left, mut right))) = ink.get(y).copied() else {
            y += 1;
            continue;
        };
        let top = y;
        let mut bottom = y;
        let mut next = y + 1;
        while next < h && next <= bottom + STRIP_GAP {
            if let Some(Some((l, r))) = ink.get(next).copied() {
                left = left.min(l);
                right = right.max(r);
                bottom = next;
            }
            next += 1;
        }
        let (sw, sh) = (right - left + 1, bottom - top + 1);
        let gray = (top..=bottom)
            .all(|y| row(y).get(left * 4..(right + 1) * 4).is_some_and(|r| r.as_chunks::<4>().0.iter().all(|p| p[0] == p[1] && p[1] == p[2])));
        let mut pixels = Vec::with_capacity(sw * sh * if gray { 1 } else { 4 });
        for y in top..=bottom {
            let r = row(y).get(left * 4..(right + 1) * 4).unwrap_or(&[]);
            for p in r.as_chunks::<4>().0 {
                if gray {
                    pixels.push(p[0]);
                } else {
                    pixels.extend_from_slice(&[p[2], p[1], p[0], 255]);
                }
            }
        }
        out.push(Strip { x: left, y: top, width: sw, height: sh, gray, pixels });
        y = bottom + 1;
    }
    out
}

#[cfg(target_os = "haiku")]
pub use native::print;

#[cfg(target_os = "haiku")]
mod native {
    use super::{Error, PageSource, failed, place, raster_scale, rotate_ccw, strips};
    use lightcraft_haiku_host::print::{self as kit, CANCELED, ENTRY_NOT_FOUND, Job, NO_INIT, NO_MEMORY, Tile};

    fn status_text(status: i32) -> String {
        match status {
            ENTRY_NOT_FOUND => "the printer is gone".into(),
            NO_MEMORY => "out of memory".into(),
            s => format!("error {s:#x}"),
        }
    }

    /// Prints `source`'s pages on the printer the system print setup picks, as the job `title`. The receipt the app shows: the printer and
    /// how many pages went. Blocks until the pages are spooled (and while the setup is open):
    /// call it from a worker thread.
    pub fn print(source: &mut dyn PageSource, title: &str) -> Result<String, Error> {
        if source.count() == 0 {
            return Err(failed("there are no pages to print"));
        }
        let settings = kit::system_setup(None)
            .map_err(|s| if s == CANCELED { Error::Cancelled } else { failed(format!("the print setup failed ({})", status_text(s))) })?;
        let (job, info) = Job::begin(&settings, if title.is_empty() { "LightCraft" } else { title }).map_err(|s| {
            failed(if s == NO_INIT {
                "the print server has no default printer; choose one in the Printers preferences".to_owned()
            } else {
                format!("the print job can't start ({})", status_text(s))
            })
        })?;
        // The job setup's page range counts the print's pages.
        let count = source.count();
        let first = usize::try_from(info.first_page.max(1) - 1).unwrap_or(0);
        let last = if info.last_page < 1 { count } else { usize::try_from(info.last_page).unwrap_or(usize::MAX).min(count) };
        let chosen: Vec<usize> = (first..last.max(first)).collect();
        if chosen.is_empty() {
            return Err(failed("the page range has no pages"));
        }
        let device_dpi = info.xres.min(info.yres).max(1) as f32;
        let total = chosen.len();
        for (done, &page) in chosen.iter().enumerate() {
            job.progress(done, total);
            if job.cancelled() {
                return Err(Error::Cancelled);
            }
            let at = place(source.size(page), info.paper, info.printable);
            // The page is rasterised as it reads, then turned.
            let (read_w, read_h) = if at.rotate { (at.height, at.width) } else { (at.width, at.height) };
            let scale = raster_scale(device_dpi, read_w, read_h);
            let (w, h) = ((read_w * scale).round().max(1.0) as usize, (read_h * scale).round().max(1.0) as usize);
            let mut image = source.render(page, scale * at.shrink, w, h)?;
            if at.rotate {
                image = rotate_ccw(&image);
            }
            let strips = strips(&image);
            drop(image);
            // A bitmap pixel is `1 / scale` points; a rectangle's right and bottom edges are the
            // last device dot's.
            let dot = 72.0 / device_dpi;
            let tiles: Vec<Tile<'_>> = strips
                .iter()
                .map(|s| Tile {
                    pixels: &s.pixels,
                    width: s.width,
                    height: s.height,
                    gray: s.gray,
                    destination: [
                        s.x as f32 / scale,
                        s.y as f32 / scale,
                        (s.x + s.width) as f32 / scale - dot,
                        (s.y + s.height) as f32 / scale - dot,
                    ],
                })
                .collect();
            // Bitmaps drawn larger than they are (a 600 dpi printer) are smoothed.
            let filter = scale * 72.0 < device_dpi - 1.0;
            if let Err(status) = job.page(&tiles, (at.width, at.height), (at.x, at.y), filter) {
                if job.cancelled() {
                    return Err(Error::Cancelled);
                }
                return Err(failed(format!("the page can't be spooled ({})", status_text(status))));
            }
        }
        job.progress(total, total);
        if job.cancelled() {
            return Err(Error::Cancelled);
        }
        job.commit().map_err(|s| failed(format!("the print server didn't take the job ({})", status_text(s))))?;
        let copies = info.copies.max(1);
        Ok(format!("{total} page{} × {copies} sent to the printer", if total == 1 { "" } else { "s" }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A4: [f32; 4] = [0.0, 0.0, 595.0, 842.0];

    #[test]
    fn a_page_the_size_of_the_paper_prints_where_it_is() {
        let at = place((595.0, 842.0), A4, [18.0, 18.0, 577.0, 824.0]);
        assert!(!at.rotate);
        assert_eq!((at.width, at.height, at.shrink), (595.0, 842.0, 1.0));
        // Measured from the printable rectangle: the paper's corner is 18 points up and left.
        assert_eq!((at.x, at.y), (-18.0, -18.0));
    }

    #[test]
    fn bigger_pages_shrink_and_smaller_ones_are_centred() {
        let letter = place((612.0, 792.0), A4, A4);
        assert!((letter.shrink - 595.0 / 612.0).abs() < 1e-4);
        assert!((letter.width - 595.0).abs() < 0.01);
        assert!(letter.y > 0.0);
        let a5 = place((420.0, 595.0), A4, A4);
        assert_eq!(a5.shrink, 1.0);
        assert_eq!((a5.x, a5.y), (87.5, 123.5));
    }

    #[test]
    fn landscape_pages_turn_on_portrait_paper() {
        let at = place((842.0, 595.0), A4, A4);
        assert!(at.rotate);
        assert_eq!((at.width, at.height, at.shrink), (595.0, 842.0, 1.0));
        // Landscape paper (the driver's page setup) takes it as it is; square pages never turn.
        assert!(!place((842.0, 595.0), [0.0, 0.0, 842.0, 595.0], [0.0, 0.0, 842.0, 595.0]).rotate);
        assert!(!place((500.0, 500.0), A4, A4).rotate);
    }

    #[test]
    fn rasterising_stays_within_the_caps() {
        assert_eq!(raster_scale(300.0, 595.0, 842.0), 300.0 / 72.0);
        assert_eq!(raster_scale(1200.0, 595.0, 842.0), MAX_DPI / 72.0);
        assert_eq!(raster_scale(150.0, 595.0, 842.0), 150.0 / 72.0);
        assert_eq!(raster_scale(0.0, 595.0, 842.0), MAX_DPI / 72.0);
        let s = raster_scale(300.0, 2384.0, 3370.0);
        assert!(2384.0 * 3370.0 * s * s <= MAX_PIXELS * 1.001);
    }

    fn image(w: usize, h: usize, ink: &[(usize, usize, [u8; 3])]) -> Image {
        let mut rgba = vec![255u8; w * h * 4];
        for &(x, y, c) in ink {
            let i = (y * w + x) * 4;
            rgba[i..i + 3].copy_from_slice(&c);
        }
        Image { width: w, height: h, rgba }
    }

    #[test]
    fn rotation_puts_the_top_on_the_left() {
        let img = image(3, 2, &[(0, 0, [1, 2, 3]), (2, 1, [4, 5, 6])]);
        let r = rotate_ccw(&img);
        assert_eq!((r.width, r.height), (2, 3));
        assert_eq!(&r.rgba[(2 * 2) * 4..(2 * 2) * 4 + 3], &[1, 2, 3]);
        assert_eq!(&r.rgba[4..7], &[4, 5, 6]);
    }

    #[test]
    fn only_inked_rows_are_spooled() {
        assert!(strips(&image(10, 10, &[])).is_empty());
        let img = image(100, 100, &[(10, 5, [0, 0, 0]), (20, 7, [50, 50, 50]), (5, 60, [200, 0, 0])]);
        let s = strips(&img);
        assert_eq!(s.len(), 2);
        // Rows 5 and 7 are close enough to share a strip: columns 10..=20, rows 5..=7, grey.
        assert_eq!((s[0].x, s[0].y, s[0].width, s[0].height, s[0].gray), (10, 5, 11, 3, true));
        assert_eq!(s[0].pixels.len(), 33);
        assert_eq!(s[0].pixels[0], 0);
        // The red dot stays in colour, as B, G, R, A.
        assert_eq!((s[1].x, s[1].y, s[1].width, s[1].height, s[1].gray), (5, 60, 1, 1, false));
        assert_eq!(s[1].pixels, vec![0, 0, 200, 255]);
    }

    fn page(w: u32, h: u32, ppi: f32, v: [u16; 3]) -> PageRaster {
        PageRaster { width: w, height: h, ppi, rgb: (0..w * h).flat_map(|_| v).collect() }
    }

    #[test]
    fn photo_pages_have_their_print_size_and_render_at_the_printers() {
        let job = lightcraft_engine::print::PrintJob::default();
        let mut pages = PhotoPages::new(vec![page(240, 480, 240.0, [65535, 0, 0])], &job, Arc::default()).unwrap();
        assert_eq!(pages.count(), 1);
        // 1 x 2 inches
        assert_eq!(pages.size(0), (72.0, 144.0));
        let img = pages.render(0, 300.0 / 72.0, 300, 600).unwrap();
        assert_eq!((img.width, img.height, img.rgba.len()), (300, 600, 300 * 600 * 4));
        // the page's red stays red in sRGB, opaque
        let p = &img.rgba[(300 * 300 + 150) * 4..(300 * 300 + 150) * 4 + 4];
        assert!(p[0] > 200 && p[1] < 60 && p[2] < 60 && p[3] == 255, "{p:?}");
    }

    #[test]
    fn a_cancelled_print_stops_rendering() {
        let cancel = Arc::new(AtomicBool::new(true));
        let job = lightcraft_engine::print::PrintJob::default();
        let mut pages = PhotoPages::new(vec![page(2, 2, 240.0, [0; 3])], &job, cancel).unwrap();
        assert_eq!(pages.render(0, 1.0, 2, 2).err(), Some(Error::Cancelled));
    }
}
