//! Which parts of a frame changed, as the presenter (`present.rs`) reads them back: tiles marked
//! by its GPU comparison, joined into a few rectangles. Plain Rust, so it is tested anywhere.

/// Tiles the comparison works in (pixels a side). Readback rectangles are made of whole tiles.
pub const TILE: u32 = 64;
/// More changed areas than this are read back as the one rectangle around them.
const MAX_RECTS: usize = 32;

/// The changed tiles (`flags`, one per tile, row by row) as a few rectangles of whole tiles
/// (`[x0, y0, x1, y1)` pixels, within `size`): runs along each row of tiles, joined with the
/// same run in the row above. Past [`MAX_RECTS`], the one rectangle around them all.
pub fn changed_rects(flags: &[u32], tiles: [u32; 2], size: [u32; 2]) -> Vec<[u32; 4]> {
    let [tw, th] = tiles;
    let mut done: Vec<[u32; 4]> = Vec::new();
    // Rectangles (in tiles) reaching the row above, and those reaching this one.
    let mut open: Vec<[u32; 4]> = Vec::new();
    let mut next = Vec::new();
    for ty in 0..th {
        let mut tx = 0;
        while tx < tw {
            if flags.get((ty * tw + tx) as usize).copied().unwrap_or(0) == 0 {
                tx += 1;
                continue;
            }
            let start = tx;
            while tx < tw && flags.get((ty * tw + tx) as usize).copied().unwrap_or(0) != 0 {
                tx += 1;
            }
            match open.iter().position(|r| r[0] == start && r[2] == tx) {
                Some(i) => {
                    let mut r = open.swap_remove(i);
                    r[3] = ty + 1;
                    next.push(r);
                }
                None => next.push([start, ty, tx, ty + 1]),
            }
        }
        done.append(&mut open);
        std::mem::swap(&mut open, &mut next);
    }
    done.append(&mut open);
    let mut rects: Vec<[u32; 4]> = done
        .iter()
        .map(|r| [r[0] * TILE, r[1] * TILE, (r[2] * TILE).min(size[0]), (r[3] * TILE).min(size[1])])
        .filter(|r| r[0] < r[2] && r[1] < r[3])
        .collect();
    if rects.len() > MAX_RECTS {
        let around = rects.iter().fold([u32::MAX, u32::MAX, 0, 0], |a, r| [a[0].min(r[0]), a[1].min(r[1]), a[2].max(r[2]), a[3].max(r[3])]);
        rects = vec![around];
    }
    rects
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changed_tiles_become_few_rectangles_clipped_to_the_frame() {
        // 4 x 3 tiles of a 200 x 150 frame:
        // . X X .
        // . X X .
        // X . . X
        let flags = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1];
        let mut rects = changed_rects(&flags, [4, 3], [200, 150]);
        rects.sort_unstable();
        assert_eq!(rects, vec![[0, 128, 64, 150], [64, 0, 192, 128], [192, 128, 200, 150]]);
        assert!(changed_rects(&[0; 12], [4, 3], [200, 150]).is_empty());
    }

    #[test]
    fn too_many_changed_areas_read_back_as_one() {
        // A checkerboard of 8 x 8 tiles: 32 runs of one.
        let flags: Vec<u32> = (0..64).map(|i| ((i / 8 + i % 8) % 2) as u32).collect();
        assert_eq!(changed_rects(&flags, [8, 8], [512, 512]).len(), 32);
        let flags: Vec<u32> = (0..100).map(|i| ((i / 10 + i % 10) % 2) as u32).collect();
        assert_eq!(changed_rects(&flags, [10, 10], [640, 600]), vec![[0, 0, 640, 600]]);
    }
}
