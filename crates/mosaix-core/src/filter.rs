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
    /// Brightens dark and darkens bright areas, judged by their
    /// surroundings within `radius` px; amounts in % (0…100).
    ShadowsHighlights {
        #[serde(default = "shadows_default")]
        shadows: f32,
        #[serde(default)]
        highlights: f32,
        #[serde(default = "sh_radius")]
        radius: f32,
    },
    /// Blur along a line: angle in degrees (0 = horizontal, counter-
    /// clockwise like Photoshop), distance in px (1…1000).
    MotionBlur {
        #[serde(default)]
        angle: f32,
        #[serde(default = "ten")]
        distance: f32,
    },
    /// Each pixel becomes the median of its surroundings (radius 1…10 px):
    /// removes specks and dust, keeps edges.
    Median {
        #[serde(default = "two")]
        radius: f32,
    },
    /// Mosaic: square cells of `size` px (2…500) filled with their average.
    Pixelate {
        #[serde(default = "ten")]
        size: f32,
    },
    /// Smooths noise but keeps edges (bilateral): strength and preserved
    /// detail in % (0…100), colour noise in % (0…100).
    ReduceNoise {
        #[serde(default = "fifty")]
        strength: f32,
        #[serde(default = "fifty")]
        detail: f32,
        #[serde(default = "fifty")]
        color: f32,
    },
    /// Blurs flat areas, keeps edges: radius px (1…100), threshold in
    /// levels (2…255) – like Photoshop's Surface Blur.
    SurfaceBlur {
        #[serde(default = "five")]
        radius: f32,
        #[serde(default = "fifteen")]
        threshold: f32,
    },
    /// Sharpening without halos or sharpened noise: amount % (1…500),
    /// radius px, noise threshold % (0…100).
    SmartSharpen {
        #[serde(default = "hundred")]
        amount: f32,
        #[serde(default = "one")]
        radius: f32,
        #[serde(default = "ten")]
        noise: f32,
    },
    /// Out-of-focus blur with a round aperture: radius px (1…200), bright
    /// spots bloom into discs (`highlights` %, above `threshold` levels).
    LensBlur {
        #[serde(default = "ten")]
        radius: f32,
        #[serde(default)]
        highlights: f32,
        #[serde(default = "two_hundred")]
        threshold: f32,
    },
    /// Average over a square of radius px (1…500).
    BoxBlur {
        #[serde(default = "five")]
        radius: f32,
    },
    /// Blur towards (zoom) or around (spin) a centre; amount 1…100,
    /// centre as a share of the canvas (0…1).
    RadialBlur {
        #[serde(default = "ten")]
        amount: f32,
        #[serde(default)]
        spin: bool,
        #[serde(default = "half")]
        center_x: f32,
        #[serde(default = "half")]
        center_y: f32,
    },
    /// Ripples: amplitude and wavelength in px, horizontal and/or vertical.
    Wave {
        #[serde(default = "ten")]
        amplitude: f32,
        #[serde(default = "hundred")]
        wavelength: f32,
        #[serde(default = "yes")]
        horizontal: bool,
        #[serde(default)]
        vertical: bool,
    },
    /// Turns the middle more than the edge: angle in degrees (−999…999),
    /// within a circle of `radius` % of the shorter canvas side around the centre.
    Twirl {
        #[serde(default = "fifty")]
        angle: f32,
        #[serde(default = "hundred")]
        radius: f32,
        #[serde(default = "half")]
        center_x: f32,
        #[serde(default = "half")]
        center_y: f32,
    },
    /// Bulges (+) or pinches (−) a circle in the middle: amount −100…100.
    Spherize {
        #[serde(default = "fifty")]
        amount: f32,
    },
    /// Rectangular ↔ polar coordinates (e.g. a panorama into a little planet).
    PolarCoordinates {
        #[serde(default = "yes")]
        to_polar: bool,
    },
    /// Shifts the layer by dx, dy px; `wrap` brings what leaves back in on the other side.
    Offset {
        #[serde(default)]
        dx: f32,
        #[serde(default)]
        dy: f32,
        #[serde(default = "yes")]
        wrap: bool,
    },
    /// Grey relief lit from `angle` degrees; height px (1…10), amount % (1…500).
    Emboss {
        #[serde(default = "one_thirty_five")]
        angle: f32,
        #[serde(default = "three")]
        height: f32,
        #[serde(default = "hundred")]
        amount: f32,
    },
    /// Edges as dark lines on white.
    FindEdges,
    /// Painterly smoothing (Kuwahara): brush radius px (1…10).
    OilPaint {
        #[serde(default = "four")]
        radius: f32,
    },
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
fn ten() -> f32 {
    10.0
}
fn two() -> f32 {
    2.0
}
fn one() -> f32 {
    1.0
}
fn five() -> f32 {
    5.0
}
fn fifteen() -> f32 {
    15.0
}
fn fifty() -> f32 {
    50.0
}
fn hundred() -> f32 {
    100.0
}
fn two_hundred() -> f32 {
    200.0
}
fn half() -> f32 {
    0.5
}
fn three() -> f32 {
    3.0
}
fn four() -> f32 {
    4.0
}
fn one_thirty_five() -> f32 {
    135.0
}
fn shadows_default() -> f32 {
    35.0
}
fn sh_radius() -> f32 {
    30.0
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
            Filter::ShadowsHighlights { shadows: 35.0, highlights: 0.0, radius: 30.0 },
            Filter::MotionBlur { angle: 0.0, distance: 20.0 },
            Filter::Median { radius: 2.0 },
            Filter::ReduceNoise { strength: 50.0, detail: 50.0, color: 50.0 },
            Filter::SurfaceBlur { radius: 5.0, threshold: 15.0 },
            Filter::SmartSharpen { amount: 100.0, radius: 1.0, noise: 10.0 },
            Filter::LensBlur { radius: 10.0, highlights: 0.0, threshold: 200.0 },
            Filter::BoxBlur { radius: 5.0 },
            Filter::RadialBlur { amount: 10.0, spin: false, center_x: 0.5, center_y: 0.5 },
            Filter::Pixelate { size: 10.0 },
            Filter::Wave { amplitude: 10.0, wavelength: 100.0, horizontal: true, vertical: false },
            Filter::Twirl { angle: 50.0, radius: 100.0, center_x: 0.5, center_y: 0.5 },
            Filter::Spherize { amount: 50.0 },
            Filter::PolarCoordinates { to_polar: true },
            Filter::Offset { dx: 0.0, dy: 0.0, wrap: true },
            Filter::Emboss { angle: 135.0, height: 3.0, amount: 100.0 },
            Filter::FindEdges,
            Filter::OilPaint { radius: 4.0 },
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
            Filter::ShadowsHighlights { .. } => crate::tr("Shadows/Highlights").into(),
            Filter::MotionBlur { .. } => crate::tr("Motion Blur").into(),
            Filter::Median { .. } => crate::tr("Median").into(),
            Filter::Pixelate { .. } => crate::tr("Mosaic").into(),
            Filter::ReduceNoise { .. } => crate::tr("Reduce Noise").into(),
            Filter::SurfaceBlur { .. } => crate::tr("Surface Blur").into(),
            Filter::SmartSharpen { .. } => crate::tr("Smart Sharpen").into(),
            Filter::LensBlur { .. } => crate::tr("Lens Blur").into(),
            Filter::BoxBlur { .. } => crate::tr("Box Blur").into(),
            Filter::RadialBlur { .. } => crate::tr("Radial Blur").into(),
            Filter::Wave { .. } => crate::tr("Wave").into(),
            Filter::Twirl { .. } => crate::tr("Twirl").into(),
            Filter::Spherize { .. } => crate::tr("Spherize").into(),
            Filter::PolarCoordinates { .. } => crate::tr("Polar Coordinates").into(),
            Filter::Offset { .. } => crate::tr("Offset").into(),
            Filter::Emboss { .. } => crate::tr("Emboss").into(),
            Filter::FindEdges => crate::tr("Find Edges").into(),
            Filter::OilPaint { .. } => crate::tr("Oil Paint").into(),
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
            | Filter::HighPass { radius }
            | Filter::ShadowsHighlights { radius, .. }
            | Filter::SmartSharpen { radius, .. } => radius.clamp(0.1, 1000.0),
            Filter::ReduceNoise { .. }
            | Filter::SurfaceBlur { .. }
            | Filter::LensBlur { .. }
            | Filter::BoxBlur { .. }
            | Filter::RadialBlur { .. }
            | Filter::Wave { .. }
            | Filter::Twirl { .. }
            | Filter::Spherize { .. }
            | Filter::PolarCoordinates { .. }
            | Filter::Offset { .. }
            | Filter::Emboss { .. }
            | Filter::FindEdges
            | Filter::OilPaint { .. } => 0.0,
            Filter::AddNoise { .. }
            | Filter::Plugin { .. }
            | Filter::MotionBlur { .. }
            | Filter::Median { .. }
            | Filter::Pixelate { .. } => 0.0,
        }
    }
}
