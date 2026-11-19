//! imagine-core: GPU-agnostic building blocks (tile math, view transform,
//! colour conversion, stroke → dab generation).

pub mod brush;
pub mod color;
pub mod tile;
pub mod view;

pub use brush::{BrushSettings, Dab, InputSample, Stroke};
pub use tile::{TILE_MIP_LEVELS, TILE_SIZE, TileCoord};
pub use view::View;
