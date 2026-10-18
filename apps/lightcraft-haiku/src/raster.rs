//! A software rasteriser for egui's tessellated output, for hosts without a GPU path: air/OS
//! draws the frame into a `BBitmap` and blits it. Platform-neutral (tested on any host).
//!
//! egui hands over triangle meshes whose colours and textures are premultiplied sRGB; blending
//! happens in that space (`dst = src + dst × (1 − src.a)`), as egui's own renderers do with a
//! non-sRGB framebuffer. Anti-aliasing comes from egui's feathering (thin triangles fading to
//! transparent), so a pixel belongs to a triangle when its centre does (no coverage maths). Two
//! fast paths cover most of a frame: axis-aligned rectangles (panels, glyphs, pictures, the
//! rendered pages) and flat-coloured triangles. The frame is split into bands of rows drawn in
//! parallel; each band only looks at the shapes that touch it.

use std::collections::HashMap;

use egui::epaint::{ClippedPrimitive, ImageData, ImageDelta, Mesh, Primitive, Vertex};
use egui::{Color32, TextureFilter, TextureId, TexturesDelta};
use rayon::prelude::*;

/// Rows per band.
const BAND: usize = 32;
/// Frames that redraw at most this many bands are drawn on the calling thread.
const SERIAL_BANDS: usize = 3;
/// Shapes hashed on the workers from this many on (a grid of text is ten thousand glyphs, which
/// hash in well under a millisecond here; waking the workers for them every frame costs more).
const PARALLEL_ITEMS: usize = 1 << 17;
/// Largest canvas side drawn, in pixels.
const MAX_SIDE: usize = 16_384;

struct Texture {
    w: usize,
    h: usize,
    /// Premultiplied RGBA.
    px: Vec<[u8; 4]>,
    linear: bool,
    /// Bumped on every change, so a band drawn with the old pixels is drawn again.
    version: u64,
}

/// A clip rectangle in pixels: [x0, x1) × [y0, y1).
#[derive(Clone, Copy, Debug)]
struct Clip {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

#[derive(Clone, Copy, Debug)]
struct V {
    x: f32,
    y: f32,
    u: f32,
    v: f32,
    c: [f32; 4],
}

#[derive(Debug)]
enum Shape {
    /// An axis-aligned rectangle of one colour, textured by `uv` ([u0, v0, u1, v1]) when given.
    Rect { x0: f32, y0: f32, x1: f32, y1: f32, color: [u32; 4], uv: Option<[f32; 4]> },
    /// A triangle: one premultiplied colour (`flat`), or colours (and texture coordinates when
    /// `textured`) interpolated across it.
    Tri { v: [V; 3], flat: Option<[u32; 4]>, textured: bool },
}

#[derive(Debug)]
struct Item {
    shape: Shape,
    clip: Clip,
    texture: TextureId,
    /// Rows the shape touches, clip included: [y0, y1).
    rows: (i32, i32),
}

/// Draws egui frames into a 0xAARRGGBB pixel buffer and keeps the textures egui uploads.
pub struct Rasterizer {
    textures: HashMap<TextureId, Texture>,
    w: usize,
    h: usize,
    buf: Vec<u32>,
    /// The previous frame, to tell what changed.
    prev: Vec<u32>,
    fresh: bool,
    /// What each band was drawn from last time: a band whose shapes, clips and textures are the
    /// same keeps its pixels (a blinking caret redraws one band, not the frame).
    band_keys: Vec<u64>,
    versions: u64,
}

impl Default for Rasterizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Rasterizer {
    pub fn new() -> Self {
        Self { textures: HashMap::new(), w: 0, h: 0, buf: Vec::new(), prev: Vec::new(), fresh: true, band_keys: Vec::new(), versions: 0 }
    }

    /// The last frame: `size()` pixels, row by row, 0xAARRGGBB.
    pub fn pixels(&self) -> &[u32] {
        &self.buf
    }

    #[cfg(test)]
    fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    /// Applies the texture uploads of a frame (before drawing it).
    pub fn set_textures(&mut self, delta: &TexturesDelta) {
        for (id, deltas) in &delta.set {
            for d in deltas {
                self.set_texture(*id, d);
            }
        }
    }

    /// Drops the textures a frame freed (after drawing it).
    pub fn free_textures(&mut self, delta: &TexturesDelta) {
        for id in &delta.free {
            self.textures.remove(id);
        }
    }

    fn set_texture(&mut self, id: TextureId, d: &ImageDelta) {
        let ImageData::Color(image) = &d.image;
        let [w, h] = image.size;
        let linear = d.options.magnification == TextureFilter::Linear;
        let px: Vec<[u8; 4]> = image.pixels.iter().map(|c| c.to_array()).collect();
        if px.len() != w.saturating_mul(h) {
            return;
        }
        self.versions += 1;
        let version = self.versions;
        match d.pos {
            None => {
                self.textures.insert(id, Texture { w, h, px, linear, version });
            }
            Some([x, y]) => {
                let Some(t) = self.textures.get_mut(&id) else { return };
                t.linear = linear;
                t.version = version;
                for row in 0..h {
                    let ty = y + row;
                    if ty >= t.h || x >= t.w {
                        break;
                    }
                    let n = w.min(t.w - x);
                    let (src, dst) = (row * w, ty * t.w + x);
                    if let (Some(s), Some(d)) = (px.get(src..src + n), t.px.get_mut(dst..dst + n)) {
                        d.copy_from_slice(s);
                    }
                }
            }
        }
    }

    /// Draws a frame of `size` pixels over `clear`, returning the rectangle that changed since
    /// the previous frame ([x0, y0, x1, y1)), or `None` when nothing did.
    pub fn render(&mut self, size: [usize; 2], prims: &[ClippedPrimitive], ppp: f32, clear: Color32) -> Option<[usize; 4]> {
        let (w, h) = (size[0].min(MAX_SIDE), size[1].min(MAX_SIDE));
        if w == 0 || h == 0 {
            return None;
        }
        if (w, h) != (self.w, self.h) {
            self.w = w;
            self.h = h;
            self.buf = vec![0; w * h];
            self.prev = vec![0; w * h];
            self.fresh = true;
        }
        let items = prepare(prims, ppp, w as i32, h as i32, &self.textures);
        let missing = items.iter().filter(|i| !self.textures.contains_key(&i.texture)).count();
        if missing > 0 {
            log::debug!(target: "lightcraft_haiku", "{missing} of {} shapes use textures egui hasn't sent", items.len());
        }
        // Which items touch which band.
        let bands = h.div_ceil(BAND);
        let mut bins: Vec<Vec<usize>> = vec![Vec::new(); bands];
        for (i, item) in items.iter().enumerate() {
            let (y0, y1) = item.rows;
            if y1 <= y0 {
                continue;
            }
            let (b0, b1) = (y0 as usize / BAND, ((y1 as usize).saturating_sub(1) / BAND).min(bands - 1));
            for bin in bins.iter_mut().take(b1 + 1).skip(b0) {
                bin.push(i);
            }
        }
        let [cr, cg, cb, _] = clear.to_array();
        let clear_px = 0xff00_0000 | (u32::from(cr) << 16) | (u32::from(cg) << 8) | u32::from(cb);
        let textures = &self.textures;
        let item_keys: Vec<u64> = if items.len() >= PARALLEL_ITEMS {
            items.par_iter().map(|i| i.key(textures)).collect()
        } else {
            items.iter().map(|i| i.key(textures)).collect()
        };
        // A band whose shapes, clips and textures are unchanged keeps its pixels.
        let keys: Vec<u64> = bins
            .iter()
            .map(|bin| {
                let mut key = Fx::default();
                key.add(u64::from(clear_px));
                key.add(w as u64);
                for &i in bin {
                    key.add(item_keys.get(i).copied().unwrap_or(0));
                }
                key.0 | 1
            })
            .collect();
        let fresh = self.fresh || self.band_keys.len() != bands;
        let todo: Vec<bool> = keys.iter().zip(self.band_keys.iter().chain(std::iter::repeat(&0))).map(|(k, old)| fresh || k != old).collect();
        self.band_keys = keys;
        let draw = |band: usize, rows: &mut [u32], prev: &mut [u32]| -> Option<[usize; 4]> {
            if !todo.get(band).copied().unwrap_or(true) {
                return None;
            }
            rows.fill(clear_px);
            let y0 = (band * BAND) as i32;
            let y1 = y0 + (rows.len() / w) as i32;
            let mut target = Band { px: rows, w: w as i32, y0, y1 };
            for &i in bins.get(band).map_or(&[][..], Vec::as_slice) {
                if let Some(item) = items.get(i) {
                    target.draw(item, textures);
                }
            }
            // What changed in this band, and remember it.
            let mut dirty: Option<[usize; 4]> = None;
            for (r, (a, b)) in target.px.chunks(w).zip(prev.chunks(w)).enumerate() {
                let Some(first) = a.iter().zip(b.iter()).position(|(p, q)| p != q) else { continue };
                let last = a.iter().zip(b.iter()).rposition(|(p, q)| p != q).unwrap_or(first);
                let y = band * BAND + r;
                dirty = Some(match dirty {
                    None => [first, y, last + 1, y + 1],
                    Some(d) => [d[0].min(first), d[1], d[2].max(last + 1), y + 1],
                });
            }
            if dirty.is_some() {
                prev.copy_from_slice(target.px);
            }
            dirty
        };
        // A few bands (a caret, a cell, a hover) are drawn here: waking every worker for them
        // costs more processor time than it saves.
        let changed: Vec<Option<[usize; 4]>> = if todo.iter().filter(|t| **t).count() > SERIAL_BANDS {
            self.buf
                .par_chunks_mut(BAND * w)
                .zip(self.prev.par_chunks_mut(BAND * w))
                .enumerate()
                .map(|(band, (rows, prev))| draw(band, rows, prev))
                .collect()
        } else {
            self.buf.chunks_mut(BAND * w).zip(self.prev.chunks_mut(BAND * w)).enumerate().map(|(band, (rows, prev))| draw(band, rows, prev)).collect()
        };
        let dirty = if fresh {
            Some([0, 0, w, h])
        } else {
            changed.into_iter().flatten().reduce(|a, b| [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])])
        };
        self.fresh = false;
        dirty
    }

    /// The last frame as an egui image (for the control channel's screenshots).
    pub fn image(&self) -> egui::ColorImage {
        let pixels = self.buf.iter().map(|&p| Color32::from_rgb((p >> 16) as u8, (p >> 8) as u8, p as u8)).collect();
        egui::ColorImage::new([self.w, self.h], pixels)
    }
}

/// A small, fast hash (FxHash's mix) for telling whether a band's shapes changed.
#[derive(Default)]
struct Fx(u64);

impl Fx {
    #[inline]
    fn add(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }

    fn f(&mut self, v: f32) {
        self.add(u64::from(v.to_bits()));
    }
}

impl Item {
    /// Everything that decides the item's pixels: its shape, clip and texture contents.
    fn key(&self, textures: &HashMap<TextureId, Texture>) -> u64 {
        let mut k = Fx::default();
        for v in [self.clip.x0, self.clip.y0, self.clip.x1, self.clip.y1] {
            k.add(v as u64);
        }
        k.add(match self.texture {
            TextureId::Managed(n) => n,
            TextureId::User(n) => n ^ (1 << 63),
        });
        k.add(textures.get(&self.texture).map_or(0, |t| t.version));
        match &self.shape {
            Shape::Rect { x0, y0, x1, y1, color, uv } => {
                k.add(1);
                for f in [*x0, *y0, *x1, *y1] {
                    k.f(f);
                }
                for c in color {
                    k.add(u64::from(*c));
                }
                for f in uv.unwrap_or([-1.0; 4]) {
                    k.f(f);
                }
            }
            Shape::Tri { v, flat, textured } => {
                k.add(2 + u64::from(*textured));
                for p in v {
                    for f in [p.x, p.y, p.u, p.v, p.c[0], p.c[1], p.c[2], p.c[3]] {
                        k.f(f);
                    }
                }
                for c in flat.unwrap_or([256; 4]) {
                    k.add(u64::from(c));
                }
            }
        }
        k.0
    }
}

/// Turns egui's meshes into shapes in pixel space, with their clip rectangles.
fn prepare(prims: &[ClippedPrimitive], ppp: f32, w: i32, h: i32, textures: &HashMap<TextureId, Texture>) -> Vec<Item> {
    let mut items = Vec::new();
    for p in prims {
        let Primitive::Mesh(mesh) = &p.primitive else { continue };
        let r = p.clip_rect;
        let clip = Clip {
            x0: ((r.min.x * ppp).round() as i32).clamp(0, w),
            y0: ((r.min.y * ppp).round() as i32).clamp(0, h),
            x1: ((r.max.x * ppp).round() as i32).clamp(0, w),
            y1: ((r.max.y * ppp).round() as i32).clamp(0, h),
        };
        if clip.x1 <= clip.x0 || clip.y1 <= clip.y0 {
            continue;
        }
        mesh_items(mesh, ppp, clip, textures.get(&mesh.texture_id), &mut items);
    }
    items
}

fn mesh_items(mesh: &Mesh, ppp: f32, clip: Clip, texture: Option<&Texture>, out: &mut Vec<Item>) {
    let idx = &mesh.indices;
    let vert = |i: u32| -> Option<&Vertex> { mesh.vertices.get(i as usize) };
    let mut t = 0;
    while t + 3 <= idx.len() {
        // egui's rectangles: (i, i+1, i+2) and (i+2, i+1, i+3) over top-left, top-right,
        // bottom-left and bottom-right corners.
        if t + 6 <= idx.len()
            && let Some(item) = quad(mesh, &idx[t..t + 6], ppp, clip, texture)
        {
            out.push(item);
            t += 6;
            continue;
        }
        let (Some(a), Some(b), Some(c)) = (vert(idx[t]), vert(idx[t + 1]), vert(idx[t + 2])) else {
            t += 3;
            continue;
        };
        t += 3;
        let v = [a, b, c].map(|v| V { x: v.pos.x * ppp, y: v.pos.y * ppp, u: v.uv.x, v: v.uv.y, c: v.color.to_array().map(f32::from) });
        let same_color = a.color == b.color && b.color == c.color;
        let same_uv = a.uv == b.uv && b.uv == c.uv;
        if same_color && a.color == Color32::TRANSPARENT {
            continue;
        }
        let flat = (same_color && same_uv).then(|| tint(a.color, texture.map(|t| t.sample(a.uv.x, a.uv.y))));
        if let Some(f) = flat
            && f == [0; 4]
        {
            continue;
        }
        let ymin = v.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let ymax = v.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
        let rows = (((ymin - 0.5).ceil() as i32).max(clip.y0), ((ymax - 0.5).ceil() as i32).min(clip.y1));
        out.push(Item { shape: Shape::Tri { v, flat, textured: texture.is_some() && !same_uv }, clip, texture: mesh.texture_id, rows });
    }
}

/// The rectangle six indices draw, when they are one of egui's axis-aligned, single-coloured
/// rectangles.
fn quad(mesh: &Mesh, idx: &[u32], ppp: f32, clip: Clip, texture: Option<&Texture>) -> Option<Item> {
    let [i0, i1, i2, j0, j1, i3] = <[u32; 6]>::try_from(idx).ok()?;
    if j0 != i2 || j1 != i1 {
        return None;
    }
    let [tl, tr, bl, br] = [i0, i1, i2, i3].map(|i| mesh.vertices.get(i as usize));
    let (tl, tr, bl, br) = (tl?, tr?, bl?, br?);
    let aligned = tl.pos.y == tr.pos.y
        && bl.pos.y == br.pos.y
        && tl.pos.x == bl.pos.x
        && tr.pos.x == br.pos.x
        && tl.uv.y == tr.uv.y
        && bl.uv.y == br.uv.y
        && tl.uv.x == bl.uv.x
        && tr.uv.x == br.uv.x;
    let same_color = tl.color == tr.color && tl.color == bl.color && tl.color == br.color;
    if !aligned || !same_color || tl.pos.x >= tr.pos.x || tl.pos.y >= bl.pos.y {
        return None;
    }
    let (x0, y0, x1, y1) = (tl.pos.x * ppp, tl.pos.y * ppp, br.pos.x * ppp, br.pos.y * ppp);
    let rows = (((y0 - 0.5).ceil() as i32).max(clip.y0), ((y1 - 0.5).ceil() as i32).min(clip.y1));
    let flat_uv = tl.uv == br.uv;
    let (color, uv) = match texture {
        Some(t) if flat_uv => (tint(tl.color, Some(t.sample(tl.uv.x, tl.uv.y))), None),
        Some(_) => (tl.color.to_array().map(u32::from), Some([tl.uv.x, tl.uv.y, br.uv.x, br.uv.y])),
        None => (tl.color.to_array().map(u32::from), None),
    };
    Some(Item { shape: Shape::Rect { x0, y0, x1, y1, color, uv }, clip, texture: mesh.texture_id, rows })
}

/// A premultiplied vertex colour times a texel (or alone).
fn tint(c: Color32, texel: Option<[u32; 4]>) -> [u32; 4] {
    let c = c.to_array().map(u32::from);
    match texel {
        Some(t) => [mul(c[0], t[0]), mul(c[1], t[1]), mul(c[2], t[2]), mul(c[3], t[3])],
        None => c,
    }
}

/// x × y / 255, rounded.
#[inline]
fn mul(x: u32, y: u32) -> u32 {
    let t = x * y + 128;
    (t + (t >> 8)) >> 8
}

#[inline]
fn blend(dst: &mut u32, s: [u32; 4]) {
    let a = s[3].min(255);
    if a == 255 {
        *dst = 0xff00_0000 | (s[0].min(255) << 16) | (s[1].min(255) << 8) | s[2].min(255);
        return;
    }
    if a == 0 && s[0] == 0 && s[1] == 0 && s[2] == 0 {
        return;
    }
    let inv = 255 - a;
    let d = *dst;
    let r = (s[0] + mul((d >> 16) & 255, inv)).min(255);
    let g = (s[1] + mul((d >> 8) & 255, inv)).min(255);
    let b = (s[2] + mul(d & 255, inv)).min(255);
    *dst = 0xff00_0000 | (r << 16) | (g << 8) | b;
}

impl Texture {
    /// The premultiplied texel at (u, v), filtered by the texture's option; clamped to the edge.
    #[inline]
    fn sample(&self, u: f32, v: f32) -> [u32; 4] {
        if self.w == 0 || self.h == 0 {
            return [0; 4];
        }
        let (wf, hf) = (self.w as f32, self.h as f32);
        let at = |x: usize, y: usize| -> [u32; 4] { self.px.get(y * self.w + x).copied().unwrap_or([0; 4]).map(u32::from) };
        if !self.linear {
            let x = ((u * wf) as isize).clamp(0, self.w as isize - 1) as usize;
            let y = ((v * hf) as isize).clamp(0, self.h as isize - 1) as usize;
            return at(x, y);
        }
        let fx = u * wf - 0.5;
        let fy = v * hf - 0.5;
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (((fx - x0) * 256.0) as u32, ((fy - y0) * 256.0) as u32);
        let clampx = |x: f32| (x as isize).clamp(0, self.w as isize - 1) as usize;
        let clampy = |y: f32| (y as isize).clamp(0, self.h as isize - 1) as usize;
        let (xa, xb, ya, yb) = (clampx(x0), clampx(x0 + 1.0), clampy(y0), clampy(y0 + 1.0));
        if tx == 0 && ty == 0 {
            return at(xa, ya);
        }
        let (p00, p10, p01, p11) = (at(xa, ya), at(xb, ya), at(xa, yb), at(xb, yb));
        let mut out = [0; 4];
        for (i, o) in out.iter_mut().enumerate() {
            let top = p00[i] * (256 - tx) + p10[i] * tx;
            let bottom = p01[i] * (256 - tx) + p11[i] * tx;
            *o = (top * (256 - ty) + bottom * ty + (1 << 15)) >> 16;
        }
        out
    }
}

/// The rows of the frame one thread draws.
struct Band<'a> {
    px: &'a mut [u32],
    w: i32,
    y0: i32,
    y1: i32,
}

impl Band<'_> {
    fn draw(&mut self, item: &Item, textures: &HashMap<TextureId, Texture>) {
        let rows = (item.rows.0.max(self.y0), item.rows.1.min(self.y1));
        if rows.1 <= rows.0 {
            return;
        }
        match &item.shape {
            Shape::Rect { x0, y0, x1, y1, color, uv } => {
                let ix0 = ((x0 - 0.5).ceil() as i32).max(item.clip.x0);
                let ix1 = ((x1 - 0.5).ceil() as i32).min(item.clip.x1);
                if ix1 <= ix0 {
                    return;
                }
                let texture = uv.and_then(|uv| Some((uv, textures.get(&item.texture)?)));
                for y in rows.0..rows.1 {
                    let row = self.row(y, ix0, ix1);
                    match texture {
                        None => {
                            for p in row {
                                blend(p, *color);
                            }
                        }
                        Some((uv, t)) => {
                            let fy = (y as f32 + 0.5 - y0) / (y1 - y0);
                            let v = uv[1] + fy * (uv[3] - uv[1]);
                            let du = (uv[2] - uv[0]) / (x1 - x0);
                            let mut u = uv[0] + (ix0 as f32 + 0.5 - x0) * du;
                            for p in row {
                                let s = t.sample(u, v);
                                blend(p, [mul(color[0], s[0]), mul(color[1], s[1]), mul(color[2], s[2]), mul(color[3], s[3])]);
                                u += du;
                            }
                        }
                    }
                }
            }
            Shape::Tri { v, flat, textured } => self.triangle(v, *flat, if *textured { textures.get(&item.texture) } else { None }, item.clip, rows),
        }
    }

    /// Pixels [x0, x1) of row y (within the band).
    fn row(&mut self, y: i32, x0: i32, x1: i32) -> &mut [u32] {
        let start = ((y - self.y0) * self.w) as usize;
        self.px.get_mut(start + x0 as usize..start + x1 as usize).unwrap_or(&mut [])
    }

    fn triangle(&mut self, v: &[V; 3], flat: Option<[u32; 4]>, texture: Option<&Texture>, clip: Clip, rows: (i32, i32)) {
        let mut s = *v;
        s.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));
        let [a, b, c] = s;
        // Twice the signed area; attribute gradients come from it.
        let area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
        if area.abs() < 1e-6 {
            return;
        }
        let grad = |fa: f32, fb: f32, fc: f32| -> (f32, f32) {
            let dx = ((fb - fa) * (c.y - a.y) - (fc - fa) * (b.y - a.y)) / area;
            let dy = ((fc - fa) * (b.x - a.x) - (fb - fa) * (c.x - a.x)) / area;
            (dx, dy)
        };
        let color_grads: [(f32, f32); 4] = std::array::from_fn(|i| grad(a.c[i], b.c[i], c.c[i]));
        let (ugx, ugy) = grad(a.u, b.u, c.u);
        let (vgx, vgy) = grad(a.v, b.v, c.v);
        // x where an edge (endpoints in y order, so both triangles of a shared edge agree) meets
        // the horizontal line y.
        let edge_x = |p: &V, q: &V, y: f32| p.x + (y - p.y) * (q.x - p.x) / (q.y - p.y);
        for y in rows.0..rows.1 {
            let yc = y as f32 + 0.5;
            if yc < a.y || yc >= c.y {
                continue;
            }
            let long = edge_x(&a, &c, yc);
            let short = if yc < b.y { edge_x(&a, &b, yc) } else { edge_x(&b, &c, yc) };
            let (l, r) = if long < short { (long, short) } else { (short, long) };
            let ix0 = ((l - 0.5).ceil() as i32).max(clip.x0);
            let ix1 = ((r - 0.5).ceil() as i32).min(clip.x1);
            if ix1 <= ix0 {
                continue;
            }
            let row = self.row(y, ix0, ix1);
            if let Some(color) = flat {
                for p in row {
                    blend(p, color);
                }
                continue;
            }
            let x = ix0 as f32 + 0.5;
            let (dx, dy) = (x - a.x, yc - a.y);
            let mut col: [f32; 4] = std::array::from_fn(|i| a.c[i] + color_grads[i].0 * dx + color_grads[i].1 * dy);
            let mut u = a.u + ugx * dx + ugy * dy;
            let mut vv = a.v + vgx * dx + vgy * dy;
            for p in row {
                let c4 = col.map(|f| f.clamp(0.0, 255.0) as u32);
                let px = match texture {
                    Some(t) => {
                        let s = t.sample(u, vv);
                        [mul(c4[0], s[0]), mul(c4[1], s[1]), mul(c4[2], s[2]), mul(c4[3], s[3])]
                    }
                    None => c4,
                };
                blend(p, px);
                for (ch, g) in col.iter_mut().zip(&color_grads) {
                    *ch += g.0;
                }
                u += ugx;
                vv += vgx;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::epaint::{Mesh, Vertex};
    use egui::{Rect, pos2};

    fn prim(mesh: Mesh) -> ClippedPrimitive {
        ClippedPrimitive { clip_rect: Rect::from_min_max(pos2(0.0, 0.0), pos2(1000.0, 1000.0)), primitive: Primitive::Mesh(mesh) }
    }

    fn px(r: &Rasterizer, x: usize, y: usize) -> u32 {
        r.pixels()[y * r.size().0 + x]
    }

    #[test]
    fn a_rectangle_fills_exactly_its_pixels() {
        let mut mesh = Mesh::default();
        mesh.add_colored_rect(Rect::from_min_max(pos2(2.0, 3.0), pos2(6.0, 5.0)), Color32::RED);
        let mut r = Rasterizer::new();
        let dirty = r.render([10, 10], &[prim(mesh)], 1.0, Color32::BLACK);
        assert_eq!(dirty, Some([0, 0, 10, 10]));
        assert_eq!(px(&r, 2, 3), 0xffff0000);
        assert_eq!(px(&r, 5, 4), 0xffff0000);
        assert_eq!(px(&r, 6, 4), 0xff000000);
        assert_eq!(px(&r, 1, 3), 0xff000000);
        assert_eq!(px(&r, 2, 5), 0xff000000);
    }

    #[test]
    fn half_transparent_shapes_blend_and_shared_edges_draw_once() {
        // Two triangles of a square sharing a diagonal, half-transparent: no seam of double blend.
        let mut mesh = Mesh::default();
        let c = Color32::from_rgba_premultiplied(0, 0, 128, 128);
        for p in [pos2(0.0, 0.0), pos2(8.0, 0.0), pos2(0.0, 8.0), pos2(8.0, 8.0)] {
            mesh.vertices.push(Vertex { pos: p, uv: egui::epaint::WHITE_UV, color: c });
        }
        mesh.indices = vec![0, 1, 3, 0, 3, 2];
        let mut r = Rasterizer::new();
        r.render([8, 8], &[prim(mesh)], 1.0, Color32::WHITE);
        let expect = px(&r, 1, 6);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(px(&r, x, y), expect, "({x}, {y})");
            }
        }
        // White under half-transparent blue: 255 × 127/255 + 0 for red.
        assert_eq!((expect >> 16) & 255, 127);
        assert_eq!(expect & 255, 255);
    }

    #[test]
    fn only_what_changed_is_reported() {
        let mut r = Rasterizer::new();
        let mut a = Mesh::default();
        a.add_colored_rect(Rect::from_min_max(pos2(0.0, 0.0), pos2(4.0, 4.0)), Color32::RED);
        r.render([16, 16], &[prim(a.clone())], 1.0, Color32::BLACK);
        assert_eq!(r.render([16, 16], &[prim(a)], 1.0, Color32::BLACK), None);
        let mut b = Mesh::default();
        b.add_colored_rect(Rect::from_min_max(pos2(0.0, 0.0), pos2(4.0, 4.0)), Color32::RED);
        b.add_colored_rect(Rect::from_min_max(pos2(9.0, 10.0), pos2(11.0, 12.0)), Color32::GREEN);
        assert_eq!(r.render([16, 16], &[prim(b)], 1.0, Color32::BLACK), Some([9, 10, 11, 12]));
    }

    #[test]
    fn textured_rectangles_sample_the_texture() {
        let mut r = Rasterizer::new();
        let id = TextureId::Managed(7);
        let image = egui::ColorImage::new([2, 1], vec![Color32::RED, Color32::BLUE]);
        let mut delta = TexturesDelta::default();
        delta.push(id, ImageDelta::full(image, egui::TextureOptions::NEAREST));
        r.set_textures(&delta);
        delta.clear();
        let mut mesh = Mesh::with_texture(id);
        mesh.add_rect_with_uv(Rect::from_min_max(pos2(0.0, 0.0), pos2(4.0, 2.0)), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
        r.render([4, 2], &[prim(mesh)], 1.0, Color32::BLACK);
        assert_eq!(px(&r, 0, 0), 0xffff0000);
        assert_eq!(px(&r, 1, 1), 0xffff0000);
        assert_eq!(px(&r, 2, 0), 0xff0000ff);
        assert_eq!(px(&r, 3, 1), 0xff0000ff);
    }

    /// Twelve bars, one per band: green, except the bands `marked` says, in `color`.
    fn bars(marked: impl Fn(usize) -> bool, color: Color32) -> Vec<ClippedPrimitive> {
        (0..12)
            .map(|i| {
                let mut mesh = Mesh::default();
                let y = i as f32 * 32.0 + 4.0;
                let fill = if marked(i) { color } else { Color32::from_rgb(20, 120, 60) };
                mesh.add_colored_rect(Rect::from_min_max(pos2(3.0, y), pos2(90.0, y + 20.0)), fill);
                prim(mesh)
            })
            .collect()
    }

    #[test]
    fn partial_frames_match_a_fresh_render_on_the_serial_and_parallel_paths() {
        let mut r = Rasterizer::new();
        r.render([96, 384], &bars(|_| false, Color32::RED), 1.0, Color32::WHITE);
        // One band changes: drawn on this thread.
        let one = bars(|i| i == 5, Color32::BLUE);
        assert_eq!(r.render([96, 384], &one, 1.0, Color32::WHITE), Some([3, 164, 90, 184]));
        let mut fresh = Rasterizer::new();
        fresh.render([96, 384], &one, 1.0, Color32::WHITE);
        assert_eq!(r.pixels(), fresh.pixels());
        // Six change: drawn on the workers.
        let six = bars(|i| i % 2 == 1, Color32::GOLD);
        assert_eq!(r.render([96, 384], &six, 1.0, Color32::WHITE), Some([3, 36, 90, 376]));
        let mut fresh = Rasterizer::new();
        fresh.render([96, 384], &six, 1.0, Color32::WHITE);
        assert_eq!(r.pixels(), fresh.pixels());
        assert_eq!(r.render([96, 384], &six, 1.0, Color32::WHITE), None, "nothing changed");
    }

    #[test]
    fn scaled_frames_and_clipping() {
        let mut mesh = Mesh::default();
        mesh.add_colored_rect(Rect::from_min_max(pos2(1.0, 1.0), pos2(3.0, 3.0)), Color32::RED);
        let p = ClippedPrimitive { clip_rect: Rect::from_min_max(pos2(0.0, 0.0), pos2(2.0, 10.0)), primitive: Primitive::Mesh(mesh) };
        let mut r = Rasterizer::new();
        r.render([8, 8], &[p], 2.0, Color32::BLACK);
        assert_eq!(px(&r, 2, 2), 0xffff0000);
        assert_eq!(px(&r, 3, 5), 0xffff0000);
        assert_eq!(px(&r, 4, 2), 0xff000000, "clipped at 2 pt = 4 px");
        assert_eq!(px(&r, 2, 6), 0xff000000);
    }
}
