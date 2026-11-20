//! Pixel filters (applied to a layer, undoable). Evaluated on the GPU.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Filter {
    /// Radius ≈ standard deviation in px (like Photoshop), 0.1…1000.
    GaussianBlur { radius: f32 },
    /// amount in % (1…500), radius in px, threshold in levels (0…255).
    UnsharpMask {
        amount: f32,
        radius: f32,
        #[serde(default)]
        threshold: f32,
    },
    /// Keeps detail above the radius, grey elsewhere (for sharpening in overlay mode).
    HighPass { radius: f32 },
    /// amount in % (0…100) of the value range.
    AddNoise {
        amount: f32,
        #[serde(default)]
        monochrome: bool,
        #[serde(default = "yes")]
        gaussian: bool,
        #[serde(default)]
        seed: u32,
    },
}

/// A filter kept on a pixel layer and applied when compositing
/// (non-destructive; the layer's own pixels stay untouched).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmartFilter {
    pub filter: Filter,
    #[serde(default = "yes")]
    pub enabled: bool,
}

impl SmartFilter {
    pub fn new(filter: Filter) -> Self {
        Self {
            filter,
            enabled: true,
        }
    }
}

fn yes() -> bool {
    true
}

impl Filter {
    pub fn defaults() -> Vec<Filter> {
        vec![
            Filter::GaussianBlur { radius: 5.0 },
            Filter::UnsharpMask {
                amount: 100.0,
                radius: 2.0,
                threshold: 0.0,
            },
            Filter::HighPass { radius: 10.0 },
            Filter::AddNoise {
                amount: 10.0,
                monochrome: false,
                gaussian: true,
                seed: 0,
            },
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Filter::GaussianBlur { .. } => crate::tr("Gaussian Blur"),
            Filter::UnsharpMask { .. } => crate::tr("Unsharp Mask"),
            Filter::HighPass { .. } => crate::tr("High Pass"),
            Filter::AddNoise { .. } => crate::tr("Add Noise"),
        }
    }

    /// Blur radius involved (0 for point filters).
    pub fn radius(&self) -> f32 {
        match self {
            Filter::GaussianBlur { radius }
            | Filter::UnsharpMask { radius, .. }
            | Filter::HighPass { radius } => radius.clamp(0.1, 1000.0),
            Filter::AddNoise { .. } => 0.0,
        }
    }
}
