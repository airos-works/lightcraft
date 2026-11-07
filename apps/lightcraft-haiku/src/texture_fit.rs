//! egui texture uploads as a small GPU takes them. egui is told the device's largest texture side
//! and sizes its own textures (the font atlas) by it, but an image the app sized itself can still
//! be larger on a GPU with 4096 px textures (the Raspberry Pi 4's V3DV), where wgpu would raise a
//! validation error and the windows would lose the device. Such an image is uploaded at a fraction
//! of its size instead: egui's texture coordinates are relative, so it draws the same picture, less
//! sharply.
use std::borrow::Cow;
use std::collections::HashSet;

use egui::epaint::ImageDelta;

/// The textures uploaded smaller than egui made them.
#[derive(Default)]
pub struct TextureFit {
    shrunk: HashSet<egui::TextureId>,
}

impl TextureFit {
    /// `image`, an upload to texture `id`, as a device whose textures are at most `max` px a side
    /// takes it, or `None` to drop it: a partial update of a texture uploaded smaller (its place in
    /// the smaller texture is not worth working out; egui's own partial updates, the font atlas's,
    /// are never that large).
    pub fn fit<'a>(&mut self, id: egui::TextureId, image: &'a ImageDelta, max: usize) -> Option<Cow<'a, ImageDelta>> {
        if image.pos.is_some() {
            return (!self.shrunk.contains(&id)).then_some(Cow::Borrowed(image));
        }
        let [w, h] = image.image.size();
        let max = max.max(1);
        if w <= max && h <= max {
            self.shrunk.remove(&id);
            return Some(Cow::Borrowed(image));
        }
        let egui::ImageData::Color(color) = &image.image;
        let small = shrink(color, max);
        log::warn!(
            "texture {id:?} is {w}x{h} px, over the GPU's {max} px: uploaded at {}x{}",
            small.size[0],
            small.size[1]
        );
        self.shrunk.insert(id);
        Some(Cow::Owned(ImageDelta::full(small, image.options)))
    }

    /// Texture `id` was freed.
    pub fn free(&mut self, id: egui::TextureId) {
        self.shrunk.remove(&id);
    }
}

/// `image` box-filtered by the smallest whole factor that brings both sides to `max` or less
/// (egui's colours are premultiplied, so averaging them is right).
fn shrink(image: &egui::ColorImage, max: usize) -> egui::ColorImage {
    let [w, h] = image.size;
    let k = w.max(h).div_ceil(max.max(1)).max(1);
    let (sw, sh) = (w.div_ceil(k), h.div_ceil(k));
    let mut pixels = Vec::with_capacity(sw * sh);
    for y in 0..sh {
        for x in 0..sw {
            let (mut sum, mut n) = ([0u64; 4], 0u64);
            for sy in y * k..((y + 1) * k).min(h) {
                for p in image.pixels.get(sy * w + x * k..sy * w + ((x + 1) * k).min(w)).unwrap_or_default() {
                    for (s, c) in sum.iter_mut().zip(p.to_array()) {
                        *s += u64::from(c);
                    }
                    n += 1;
                }
            }
            let [r, g, b, a] = sum.map(|s| u8::try_from(s / n.max(1)).unwrap_or(u8::MAX));
            pixels.push(egui::Color32::from_rgba_premultiplied(r, g, b, a));
        }
    }
    egui::ColorImage::new([sw, sh], pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full(size: [usize; 2], f: impl Fn(usize, usize) -> egui::Color32) -> ImageDelta {
        let pixels = (0..size[1]).flat_map(|y| (0..size[0]).map(move |x| (x, y))).map(|(x, y)| f(x, y)).collect();
        ImageDelta::full(egui::ColorImage::new(size, pixels), egui::TextureOptions::LINEAR)
    }

    fn size(d: &ImageDelta) -> [usize; 2] {
        d.image.size()
    }

    #[test]
    fn textures_within_the_limit_go_up_as_they_are() {
        let mut fit = TextureFit::default();
        let image = full([64, 32], |_, _| egui::Color32::RED);
        assert!(matches!(fit.fit(egui::TextureId::Managed(1), &image, 64), Some(Cow::Borrowed(_))));
    }

    #[test]
    fn larger_textures_are_box_filtered_to_fit_and_their_partial_updates_dropped() {
        let mut fit = TextureFit::default();
        let id = egui::TextureId::Managed(7);
        // A one-pixel checkerboard of white and transparent: grey (premultiplied) when halved.
        let image = full([10, 4], |x, y| if (x + y) % 2 == 0 { egui::Color32::WHITE } else { egui::Color32::TRANSPARENT });
        let small = fit.fit(id, &image, 5).unwrap();
        assert_eq!(size(&small), [5, 2]);
        let egui::ImageData::Color(c) = &small.image;
        assert!(c.pixels.iter().all(|p| p.to_array() == [127, 127, 127, 127]), "{:?}", c.pixels);
        // Odd sizes round up: 11 px at a factor of 3 is 4 px, the last of one column.
        assert_eq!(size(&fit.fit(id, &full([11, 3], |_, _| egui::Color32::BLUE), 4).unwrap()), [4, 1]);
        let patch = ImageDelta::partial([0, 0], egui::ColorImage::new([1, 1], vec![egui::Color32::RED]), egui::TextureOptions::LINEAR);
        assert!(fit.fit(id, &patch, 5).is_none(), "the patch's place in the smaller texture is unknown");
        assert!(fit.fit(egui::TextureId::Managed(8), &patch, 5).is_some(), "other textures' patches go up");
        // Set again within the limit, or freed: its patches go up again.
        assert!(fit.fit(id, &full([5, 5], |_, _| egui::Color32::RED), 5).is_some());
        assert!(fit.fit(id, &patch, 5).is_some());
        fit.fit(id, &image, 5);
        fit.free(id);
        assert!(fit.fit(id, &patch, 5).is_some());
    }
}
