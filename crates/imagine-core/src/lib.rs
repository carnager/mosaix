//! imagine-core: GPU-agnostic building blocks (document model, blend modes,
//! history, tile math, view transform, colour conversion, stroke → dabs).

pub mod blend;
pub mod brush;
pub mod color;
pub mod doc;
pub mod history;
pub mod tile;
pub mod view;

pub use blend::{BlendMode, BlendSpace};
pub use brush::{BrushSettings, Dab, InputSample, Stroke};
pub use doc::{Content, Document, Layer, LayerId, Op};
pub use history::History;
pub use tile::{TILE_MIP_LEVELS, TILE_SIZE, TileCoord};
pub use view::View;
