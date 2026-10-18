//! Reuse a completed native bitmap when egui produces identical draw commands.
//! Exact mesh comparisons avoid hash collisions; callbacks are always redrawn.
pub struct Frame {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub size: [u32; 2],
    pub ppp: f32,
    pub clear: egui::Color32,
}
impl Frame {
    pub fn matches(&self, size: [u32; 2], primitives: &[egui::ClippedPrimitive], ppp: f32, clear: egui::Color32) -> bool {
        self.size == size
            && self.ppp == ppp
            && self.clear == clear
            && self.primitives.len() == primitives.len()
            && self.primitives.iter().zip(primitives).all(|(a, b)| {
                a.clip_rect == b.clip_rect
                    && match (&a.primitive, &b.primitive) {
                        (egui::epaint::Primitive::Mesh(a), egui::epaint::Primitive::Mesh(b)) => a == b,
                        _ => false,
                    }
            })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn changes_to_pixels_or_viewport_invalidate_the_completed_frame() {
        let mut mesh = egui::Mesh::default();
        mesh.add_colored_rect(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(8.0, 8.0)), egui::Color32::RED);
        let primitives = vec![egui::ClippedPrimitive { clip_rect: egui::Rect::EVERYTHING, primitive: egui::epaint::Primitive::Mesh(mesh) }];
        let f = Frame { primitives: primitives.clone(), size: [16, 16], ppp: 1.0, clear: egui::Color32::BLACK };
        assert!(f.matches([16, 16], &primitives, 1.0, egui::Color32::BLACK));
        assert!(!f.matches([17, 16], &primitives, 1.0, egui::Color32::BLACK));
        assert!(!f.matches([16, 16], &primitives, 2.0, egui::Color32::BLACK));
        assert!(!f.matches([16, 16], &primitives, 1.0, egui::Color32::WHITE));
        assert!(!f.matches([16, 16], &[], 1.0, egui::Color32::BLACK));
        let mut changed = primitives.clone();
        changed[0].clip_rect = egui::Rect::NOTHING;
        assert!(!f.matches([16, 16], &changed, 1.0, egui::Color32::BLACK));
        changed = primitives.clone();
        if let egui::epaint::Primitive::Mesh(mesh) = &mut changed[0].primitive {
            mesh.vertices[0].color = egui::Color32::BLUE;
        }
        assert!(!f.matches([16, 16], &changed, 1.0, egui::Color32::BLACK));
    }
}
