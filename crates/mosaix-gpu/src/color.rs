//! Colour profiles: converting embedded ICC profiles to linear sRGB.

use mosaix_core::tr;

pub type RowConverter = Box<dyn Fn(&mut [f32]) + Send + Sync>;

/// Converter from an embedded ICC profile to linear sRGB, unless the
/// profile already is sRGB (or cannot be used, then the image is taken as sRGB).
pub fn color_transform(icc: &[u8]) -> Option<(String, RowConverter)> {
    use moxcms::{ColorProfile, Layout, ToneReprCurve, TransformOptions};
    let src = match ColorProfile::new_from_slice(icc) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(
                "{}",
                tr!(
                    "Embedded color profile unreadable ({error}), assuming sRGB",
                    error = format!("{e:?}")
                )
            );
            return None;
        }
    };
    let name = match &src.description {
        Some(moxcms::ProfileText::Localizable(v)) => {
            v.first().map(|s| s.value.clone()).unwrap_or_default()
        }
        Some(moxcms::ProfileText::PlainString(s)) => s.clone(),
        _ => String::new(),
    };
    tracing::debug!("embedded color profile: {name:?}");
    let srgb = ColorProfile::new_srgb();
    // sRGB primaries (up to rounding in the profile).
    let close = |a: moxcms::Xyzd, b: moxcms::Xyzd| (a.x - b.x).abs() + (a.y - b.y).abs() + (a.z - b.z).abs() < 0.003;
    let same_primaries = close(src.red_colorant, srgb.red_colorant)
        && close(src.green_colorant, srgb.green_colorant)
        && close(src.blue_colorant, srgb.blue_colorant);
    // The name alone is not enough: "…P3_Srgb" is P3 with sRGB's curve.
    if same_primaries && (name.to_lowercase().contains("srgb") || src.red_trc == srgb.red_trc) {
        return None;
    }
    let mut linear = ColorProfile::new_srgb();
    let identity = ToneReprCurve::Lut(vec![]);
    linear.red_trc = Some(identity.clone());
    linear.green_trc = Some(identity.clone());
    linear.blue_trc = Some(identity);
    linear.cicp = None;
    let opts = TransformOptions {
        prefer_fixed_point: false,
        allow_extended_range_rgb_xyz: true,
        ..Default::default()
    };
    match src.create_transform_f32(Layout::Rgb, &linear, Layout::Rgb, opts) {
        Ok(t) => Some((
            if name.is_empty() {
                tr("untitled").into()
            } else {
                name
            },
            Box::new(move |rgb: &mut [f32]| {
                let input = rgb.to_vec();
                if t.transform(&input, rgb).is_err() {
                    rgb.iter_mut()
                        .for_each(|v| *v = mosaix_core::color::srgb_to_linear(*v));
                }
            }),
        )),
        Err(e) => {
            tracing::warn!(
                "{}",
                tr!(
                    "Color profile “{name}” cannot be converted ({error}), assuming sRGB",
                    name = name,
                    error = format!("{e:?}")
                )
            );
            None
        }
    }
}

