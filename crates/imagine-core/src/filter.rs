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
    /// A WGSL filter plugin. `source` is kept with the filter, so documents
    /// work without the plugin installed; empty = look it up by `name`.
    Plugin {
        name: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        source: String,
        /// Parameter values by id (numbers, bools as 0/1, colours as 3 linear floats).
        #[serde(default, deserialize_with = "plugin_values")]
        values: std::collections::BTreeMap<String, Vec<f32>>,
    },
}

/// Accepts numbers, bools, "#rrggbb" and float lists as parameter values.
fn plugin_values<'de, D: serde::Deserializer<'de>>(d: D) -> Result<std::collections::BTreeMap<String, Vec<f32>>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum V {
        N(f32),
        B(bool),
        L(Vec<f32>),
        S(String),
    }
    let m: std::collections::BTreeMap<String, V> = Deserialize::deserialize(d)?;
    m.into_iter()
        .map(|(k, v)| {
            let v = match v {
                V::N(n) => vec![n],
                V::B(b) => vec![b as u8 as f32],
                V::L(l) => l,
                V::S(s) => {
                    let h = s.trim_start_matches('#');
                    let c = (h.len() == 6).then(|| u32::from_str_radix(h, 16).ok()).flatten().ok_or_else(|| {
                        serde::de::Error::custom(format!("{k}: expected a number, bool, list or #rrggbb"))
                    })?;
                    [16, 8, 0].iter().map(|sh| crate::color::srgb_to_linear(((c >> sh) & 255) as f32 / 255.0)).collect()
                }
            };
            Ok((k, v))
        })
        .collect()
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

    pub fn name(&self) -> std::borrow::Cow<'static, str> {
        match self {
            Filter::GaussianBlur { .. } => crate::tr("Gaussian Blur").into(),
            Filter::UnsharpMask { .. } => crate::tr("Unsharp Mask").into(),
            Filter::HighPass { .. } => crate::tr("High Pass").into(),
            Filter::AddNoise { .. } => crate::tr("Add Noise").into(),
            Filter::Plugin { name, .. } => name.clone().into(),
        }
    }

    /// A plugin filter with its default values.
    pub fn plugin(source: &str) -> Result<Filter, String> {
        let spec = crate::plugin::parse(source)?;
        Ok(Filter::Plugin {
            name: spec.name.clone(),
            source: source.to_string(),
            values: spec.params.iter().map(|p| (p.id.clone(), p.default.clone())).collect(),
        })
    }

    /// Blur radius involved (0 for point filters).
    pub fn radius(&self) -> f32 {
        match self {
            Filter::GaussianBlur { radius }
            | Filter::UnsharpMask { radius, .. }
            | Filter::HighPass { radius } => radius.clamp(0.1, 1000.0),
            Filter::AddNoise { .. } | Filter::Plugin { .. } => 0.0,
        }
    }
}
