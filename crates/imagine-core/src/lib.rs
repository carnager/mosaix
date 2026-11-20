//! imagine-core: GPU-agnostic building blocks (document model, blend modes,
//! history, tile math, view transform, colour conversion, stroke → dabs).

pub mod adjust;
pub mod blend;
pub mod brush;
pub mod color;
pub mod doc;
pub mod filter;
pub mod history;
pub mod mask;
pub mod tile;
pub mod view;

pub use adjust::Adjustment;
pub use filter::Filter;
pub use blend::{BlendMode, BlendSpace};
pub use brush::{BrushSettings, Dab, InputSample, Stroke, TIP_NAMES, tip_from_name};
pub use doc::{Content, Document, Layer, LayerId, Mask, MaskRef, MaskTile, Op};
pub use history::History;
pub use tile::{TILE_MIP_LEVELS, TILE_SIZE, TileCoord};
pub use view::View;
