//! Tile grid math. Documents are split into square tiles that are allocated
//! sparsely: a tile that was never touched costs nothing.

/// Edge length of a tile in pixels.
pub const TILE_SIZE: u32 = 256;
/// Number of mip levels of a tile (256 → 1).
pub const TILE_MIP_LEVELS: u32 = TILE_SIZE.ilog2() + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileCoord {
    pub x: i32,
    pub y: i32,
}

impl TileCoord {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Tile containing the given document pixel position.
    pub fn containing(px: f64, py: f64) -> Self {
        let s = TILE_SIZE as f64;
        Self::new((px / s).floor() as i32, (py / s).floor() as i32)
    }

    /// Top-left document pixel of this tile.
    pub fn origin(self) -> (i64, i64) {
        (
            self.x as i64 * TILE_SIZE as i64,
            self.y as i64 * TILE_SIZE as i64,
        )
    }
}

/// Number of tiles needed to cover `w × h` pixels.
pub fn grid_size(w: u32, h: u32) -> (u32, u32) {
    (w.div_ceil(TILE_SIZE), h.div_ceil(TILE_SIZE))
}

/// All tiles intersecting the axis-aligned rectangle `[min, max)` in document
/// pixels, clipped to the document grid of `doc_w × doc_h`.
pub fn tiles_in_rect(
    min: (f64, f64),
    max: (f64, f64),
    doc_w: u32,
    doc_h: u32,
) -> impl Iterator<Item = TileCoord> {
    let (gw, gh) = grid_size(doc_w, doc_h);
    let a = TileCoord::containing(min.0, min.1);
    // `max` is exclusive: a rect ending exactly on a tile edge does not touch the next tile.
    let b = TileCoord::containing((max.0 - 1e-9).max(min.0), (max.1 - 1e-9).max(min.1));
    let x0 = a.x.max(0);
    let y0 = a.y.max(0);
    let x1 = b.x.min(gw as i32 - 1);
    let y1 = b.y.min(gh as i32 - 1);
    (y0..=y1).flat_map(move |y| (x0..=x1).map(move |x| TileCoord::new(x, y)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_levels() {
        assert_eq!(TILE_MIP_LEVELS, 9);
    }

    #[test]
    fn containing_handles_negative() {
        assert_eq!(TileCoord::containing(-0.5, 0.0), TileCoord::new(-1, 0));
        assert_eq!(TileCoord::containing(255.9, 256.0), TileCoord::new(0, 1));
    }

    #[test]
    fn rect_is_clipped_to_document() {
        let v: Vec<_> = tiles_in_rect((-100.0, -100.0), (300.0, 10.0), 600, 600).collect();
        assert_eq!(v, vec![TileCoord::new(0, 0), TileCoord::new(1, 0)]);
        let v: Vec<_> = tiles_in_rect((0.0, 0.0), (256.0, 256.0), 1000, 1000).collect();
        assert_eq!(v, vec![TileCoord::new(0, 0)]);
        assert_eq!(
            tiles_in_rect((800.0, 0.0), (900.0, 10.0), 600, 600).count(),
            0
        );
    }
}
