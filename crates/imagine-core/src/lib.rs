//! imagine-core: GPU-agnostic building blocks (document model, blend modes,
//! history, tile math, view transform, colour conversion, stroke → dabs).

pub mod adjust;
pub mod blend;
pub mod brush;
pub mod color;
pub mod doc;
pub mod fill;
pub mod gradient;
pub mod filter;
pub mod heal;
pub mod history;
pub mod i18n;
pub mod mask;
pub mod pattern;
pub mod plugin;
pub mod raw;
pub mod style;
pub mod tile;
pub mod vector;
pub mod view;

pub use adjust::Adjustment;
pub use blend::{BlendMode, BlendSpace};
pub use brush::{BrushSettings, Dab, InputSample, Stroke, TIP_NAMES, tip_from_name, tip_label};
pub use doc::{Content, Document, Layer, LayerId, Mask, MaskRef, MaskTile, Op};
pub use filter::{Filter, SmartFilter};
pub use history::History;
pub use i18n::tr;
pub use tile::{TILE_MIP_LEVELS, TILE_SIZE, TileCoord};
pub use vector::VectorLayer;
pub use view::View;
