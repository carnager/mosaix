//! Rasterises vector layers (text, shapes) into document tiles: premultiplied
//! linear RGBA f16, like painted layers. Text is laid out with parley
//! (system fonts via fontique), everything is drawn with vello_cpu in chunks
//! of tiles, in parallel, and only where the content is.

use std::sync::{Mutex, OnceLock};

use half::f16;
use imagine_core::color::{linear_to_srgb, srgb_to_linear};
use imagine_core::vector::{Affine as Mat, Geometry, ShapeSpec, TextAlign, TextSpec, VectorKind, VectorLayer};
use imagine_core::{TILE_SIZE, TileCoord};
use parley::{Alignment, AlignmentOptions, FontContext, FontStyle, FontWeight, LayoutContext, LineHeight, PositionedLayoutItem, StyleProperty};
use rayon::prelude::*;
use vello_cpu::kurbo::{Affine, BezPath, Ellipse, Rect, RoundedRect, Shape, Stroke};
use vello_cpu::peniko::FontData;
use vello_cpu::peniko::color::{AlphaColor, Srgb};
use vello_cpu::{Pixmap, RenderContext, Resources};

/// Pixels of one rendered tile (TILE_SIZE² × RGBA).
pub type TilePixels = Vec<f16>;

/// Tiles per side of one render chunk.
const CHUNK_TILES: u32 = 4;
/// Refuse absurd areas (document px per side).
const MAX_EXTENT: f64 = 65536.0;

struct Fonts {
    font: FontContext,
    layout: LayoutContext<()>,
}

fn fonts() -> &'static Mutex<Fonts> {
    static F: OnceLock<Mutex<Fonts>> = OnceLock::new();
    F.get_or_init(|| Mutex::new(Fonts { font: FontContext::new(), layout: LayoutContext::new() }))
}

/// Installed font families, sorted.
pub fn font_families() -> Vec<String> {
    let mut f = fonts().lock().unwrap();
    let mut v: Vec<String> = f.font.collection.family_names().map(str::to_owned).collect();
    v.sort_by_key(|s| s.to_lowercase());
    v.dedup();
    v
}

/// One installable face of a font group: render it with `family` + `weight`
/// + `italic`.
#[derive(Debug, Clone, PartialEq)]
pub struct FontFace {
    pub family: String,
    pub weight: f32,
    pub italic: bool,
    pub label: String,
}

/// A font family as users think of it ("IBM Plex Mono"), with all its faces
/// even when the system lists them as separate families ("… Medm", "… SmBld").
#[derive(Debug, Clone, PartialEq)]
pub struct FontGroup {
    pub name: String,
    pub kind: FontKind,
    /// Has Latin letters (other scripts are hidden by default).
    pub latin: bool,
    /// Sorted by weight, upright before italic.
    pub faces: Vec<FontFace>,
}

impl FontGroup {
    /// The face to show by default (closest to regular, upright).
    pub fn regular(&self) -> Option<&FontFace> {
        self.faces.iter().min_by(|a, b| {
            let k = |f: &FontFace| (f.italic as u8, (f.weight - 400.0).abs() as u32);
            k(a).cmp(&k(b))
        })
    }

    /// The face of this group closest to (family, weight, italic).
    pub fn face(&self, family: &str, weight: f32, italic: bool) -> Option<&FontFace> {
        self.faces.iter().min_by(|a, b| {
            let k = |f: &FontFace| (f.family != family) as u32 * 2000 + (f.italic != italic) as u32 * 1000 + (f.weight - weight).abs() as u32;
            k(a).cmp(&k(b))
        })
    }
}

/// Rough classification for filtering the font list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FontKind {
    Sans,
    Serif,
    Mono,
    Script,
    Display,
    Symbol,
}

impl FontKind {
    pub const ALL: [Self; 6] = [Self::Sans, Self::Serif, Self::Mono, Self::Script, Self::Display, Self::Symbol];

    pub fn name(self) -> &'static str {
        match self {
            Self::Sans => "Sans",
            Self::Serif => "Serif",
            Self::Mono => "Mono",
            Self::Script => "Schreibschrift",
            Self::Display => "Display",
            Self::Symbol => "Symbole",
        }
    }
}

/// Classifies a font from its OS/2 family class, PANOSE and fixed pitch,
/// with the name as a fallback.
fn classify(name: &str, data: &[u8], index: u32) -> FontKind {
    use read_fonts::{FontRef, TableProvider};
    let n = name.to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| n.contains(w));
    if has(&["emoji", "symbol", "icon", "dingbat", "awesome", "material", "wingding", "webding"]) {
        return FontKind::Symbol;
    }
    let font = FontRef::from_index(data, index).ok();
    let mono = font.as_ref().and_then(|f| f.post().ok()).is_some_and(|p| p.is_fixed_pitch() != 0);
    if mono || has(&["mono", "code", "courier", "console", "terminal"]) {
        return FontKind::Mono;
    }
    let os2 = font.as_ref().and_then(|f| f.os2().ok());
    let class = os2.as_ref().map_or(0, |o| (o.s_family_class() >> 8) as u8);
    let panose: Vec<u8> = os2.as_ref().map(|o| o.panose_10().to_vec()).unwrap_or_default();
    let p0 = panose.first().copied().unwrap_or(0);
    let p1 = panose.get(1).copied().unwrap_or(0);
    if class == 12 || p0 == 5 {
        return FontKind::Symbol;
    }
    if class == 10 || p0 == 3 || has(&["script", "hand", "brush", "callig", "signature"]) {
        return FontKind::Script;
    }
    if class == 9 || p0 == 4 || has(&["display", "decor", "poster", "stencil"]) {
        return FontKind::Display;
    }
    if has(&["sans"]) || class == 8 || (p0 == 2 && (11..=15).contains(&p1)) {
        return FontKind::Sans;
    }
    if has(&["serif"]) || (1..=7).contains(&class) || (p0 == 2 && (2..=10).contains(&p1)) {
        return FontKind::Serif;
    }
    FontKind::Sans
}

/// Whether the font maps the basic Latin letters.
fn has_latin(data: &[u8], index: u32) -> bool {
    use read_fonts::{FontRef, TableProvider};
    let Some(cmap) = FontRef::from_index(data, index).ok().and_then(|f| f.cmap().ok()) else { return true };
    "AaZz".chars().all(|c| cmap.map_codepoint(c).is_some_and(|g| g.to_u32() != 0))
}

/// Style words that some fonts put into the family name.
const STYLE_WORDS: &[&str] = &[
    "thin", "hairline", "extralight", "extlt", "ultralight", "light", "lt", "text", "regular", "book", "normal", "medium", "medm", "md",
    "semibold", "smbld", "demibold", "demi", "bold", "bd", "extrabold", "extbd", "ultrabold", "heavy", "black", "blk", "italic", "oblique",
];

fn base_name(family: &str) -> String {
    let mut words: Vec<&str> = family.split_whitespace().collect();
    while words.len() > 1 && STYLE_WORDS.contains(&words.last().unwrap().to_lowercase().as_str()) {
        words.pop();
    }
    words.join(" ")
}

fn weight_label(w: f32) -> &'static str {
    match w.round() as i32 {
        ..=149 => "Thin",
        150..=249 => "ExtraLight",
        250..=349 => "Light",
        350..=424 => "Regular",
        425..=474 => "Text",
        475..=549 => "Medium",
        550..=649 => "SemiBold",
        650..=749 => "Bold",
        750..=849 => "ExtraBold",
        _ => "Black",
    }
}

/// Installed fonts grouped into families with their faces, sorted by name.
pub fn font_groups() -> Vec<FontGroup> {
    let mut f = fonts().lock().unwrap();
    let ids: Vec<_> = f.font.collection.family_ids().collect();
    // Lower-case base name → (display name, faces, (kind, has Latin)).
    type Entry = (String, Vec<FontFace>, Option<(FontKind, bool)>);
    let mut groups: std::collections::BTreeMap<String, Entry> = Default::default();
    for id in ids {
        let Some(fam) = f.font.collection.family(id) else { continue };
        let name = fam.name().to_string();
        let base = base_name(&name);
        let entry = groups.entry(base.to_lowercase()).or_insert_with(|| (base.clone(), Vec::new(), None));
        if entry.2.is_none()
            && let Some(info) = fam.fonts().first()
            && let Some(blob) = info.load(None)
        {
            entry.2 = Some((classify(&base, blob.data(), info.index()), has_latin(blob.data(), info.index())));
        }
        for info in fam.fonts() {
            let weight = info.weight().value();
            let italic = !matches!(info.style(), parley::FontStyle::Normal);
            // Same weight and slope from another alias ("Medm" vs "Medium"): keep one.
            if entry.1.iter().any(|x| (x.weight - weight).abs() < 1.0 && x.italic == italic) {
                continue;
            }
            let label = format!("{}{}", weight_label(weight), if italic { " Italic" } else { "" });
            entry.1.push(FontFace { family: name.clone(), weight, italic, label });
        }
    }
    let mut out: Vec<FontGroup> = groups
        .into_values()
        .filter(|(_, faces, _)| !faces.is_empty())
        .map(|(name, mut faces, kind)| {
            faces.sort_by(|a, b| a.weight.total_cmp(&b.weight).then(a.italic.cmp(&b.italic)));
            let (kind, latin) = kind.unwrap_or((FontKind::Sans, true));
            FontGroup { name, kind, latin, faces }
        })
        .collect();
    out.sort_by_key(|g| g.name.to_lowercase());
    out
}

/// A small preview of `text` in `family` (white on transparent, straight
/// RGBA 8 bit): width, height, pixels. For font lists.
pub fn font_preview(family: &str, weight: f32, italic: bool, text: &str, size: f32, max_width: u16) -> Option<(u16, u16, Vec<u8>)> {
    let mut spec = TextSpec::new(text, 0.0, 0.0, size, [1.0; 4]);
    spec.font = family.into();
    spec.weight = weight;
    spec.italic = italic;
    let l = layout_text(&spec);
    if l.runs.is_empty() {
        return None;
    }
    let w = (l.width.ceil() as u16 + 2).clamp(1, max_width);
    let h = (l.height.ceil() as u16 + 2).max(1);
    let mut ctx = RenderContext::new(w, h);
    let mut res = Resources::new();
    ctx.set_transform(Affine::translate((1.0, 1.0)));
    draw(&mut ctx, &mut res, &Prepared::Text(spec, l));
    ctx.flush();
    let mut pix = Pixmap::new(w, h);
    ctx.render(&mut pix, &mut res);
    let rgba = pix.data().iter().flat_map(|p| [255, 255, 255, p.a]).collect();
    Some((w, h, rgba))
}

/// One run of glyphs with the same font (positions relative to the text box).
struct Run {
    font: FontData,
    size: f32,
    coords: Vec<i16>,
    glyphs: Vec<vello_cpu::Glyph>,
}

struct TextLayout {
    runs: Vec<Run>,
    width: f32,
    height: f32,
}

fn layout_text(t: &TextSpec) -> TextLayout {
    let mut guard = fonts().lock().unwrap();
    let Fonts { font, layout } = &mut *guard;
    let mut b = layout.ranged_builder(font, &t.text, 1.0, false);
    b.push_default(StyleProperty::FontFamily(parley::FontFamily::Source(t.font.as_str().into())));
    b.push_default(StyleProperty::FontSize(t.size));
    b.push_default(StyleProperty::FontWeight(FontWeight::new(t.weight)));
    if t.italic {
        b.push_default(StyleProperty::FontStyle(FontStyle::Italic));
    }
    b.push_default(StyleProperty::LineHeight(LineHeight::FontSizeRelative(t.line_height)));
    if t.letter_spacing != 0.0 {
        b.push_default(StyleProperty::LetterSpacing(t.letter_spacing));
    }
    let mut l = b.build(&t.text);
    let align = match t.align {
        TextAlign::Left => Alignment::Left,
        TextAlign::Center => Alignment::Center,
        TextAlign::Right => Alignment::Right,
        TextAlign::Justify => Alignment::Justify,
    };
    l.set_alignment(align, AlignmentOptions::default());
    l.break_all_lines(t.width);
    let mut runs = Vec::new();
    for line in l.lines() {
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(g) = item {
                let run = g.run();
                runs.push(Run {
                    font: run.font().clone(),
                    size: run.font_size(),
                    coords: run.normalized_coords().iter().map(|c| c.to_bits()).collect(),
                    glyphs: g.positioned_glyphs().map(|p| vello_cpu::Glyph { id: p.id, x: p.x, y: p.y }).collect(),
                });
            }
        }
    }
    TextLayout { runs, width: t.width.unwrap_or(l.width()).max(l.width()), height: l.height() }
}

fn affine(m: &Mat) -> Affine {
    Affine::new(*m)
}

fn geometry_path(g: &Geometry) -> Option<BezPath> {
    Some(match g {
        Geometry::Rect { x, y, w, h, radius } => {
            let r = Rect::new(*x, *y, x + w, y + h);
            if *radius > 0.0 { RoundedRect::from_rect(r, *radius).to_path(0.1) } else { r.to_path(0.1) }
        }
        Geometry::Ellipse { cx, cy, rx, ry } => Ellipse::new((*cx, *cy), (*rx, *ry), 0.0).to_path(0.1),
        Geometry::Polygon { points, closed } => {
            let mut p = BezPath::new();
            for (i, q) in points.iter().enumerate() {
                if i == 0 {
                    p.move_to((q[0], q[1]));
                } else {
                    p.line_to((q[0], q[1]));
                }
            }
            if *closed && points.len() > 2 {
                p.close_path();
            }
            p
        }
        Geometry::Path { d } => BezPath::from_svg(d).ok()?,
    })
}

fn color(c: [f32; 4]) -> AlphaColor<Srgb> {
    AlphaColor::new([linear_to_srgb(c[0].max(0.0)), linear_to_srgb(c[1].max(0.0)), linear_to_srgb(c[2].max(0.0)), c[3].clamp(0.0, 1.0)])
}

/// What to draw, prepared once for all chunks.
enum Prepared {
    Text(TextSpec, TextLayout),
    Shape(ShapeSpec, Option<BezPath>),
}

fn prepare(v: &VectorLayer) -> Prepared {
    match &v.kind {
        VectorKind::Text(t) => Prepared::Text(t.clone(), layout_text(t)),
        VectorKind::Shape(s) => Prepared::Shape(s.clone(), geometry_path(&s.geometry)),
    }
}

/// Bounding box (x0, y0, x1, y1) before the layer transform.
fn local_bounds(p: &Prepared) -> Option<Rect> {
    match p {
        Prepared::Text(t, l) => {
            if l.runs.is_empty() {
                return None;
            }
            // Generous margin for overhanging glyphs (italics, swashes).
            let m = t.size as f64 * 0.5 + t.letter_spacing.abs() as f64;
            Some(Rect::new(t.x - m, t.y - m, t.x + l.width as f64 + m, t.y + l.height as f64 + m))
        }
        Prepared::Shape(s, path) => {
            let b = path.as_ref()?.bounding_box();
            let pad = s.stroke.as_ref().map_or(1.0, |st| st.width as f64 * 2.0 + 1.0);
            Some(b.inflate(pad, pad))
        }
    }
}

/// Bounding box of the rendered content in document px, after the transform.
pub fn bounds(v: &VectorLayer) -> Option<[f64; 4]> {
    let p = prepare(v);
    let r = affine(&v.transform).transform_rect_bbox(local_bounds(&p)?);
    Some([r.x0, r.y0, r.x1, r.y1])
}

fn draw(ctx: &mut RenderContext, res: &mut Resources, p: &Prepared) {
    match p {
        Prepared::Text(t, l) => {
            ctx.set_paint(color(t.color));
            let base = *ctx.transform();
            ctx.set_transform(base * Affine::translate((t.x, t.y)));
            for run in &l.runs {
                let _ = ctx
                    .glyph_run(res, &run.font)
                    .font_size(run.size)
                    .normalized_coords(&run.coords)
                    .hint(false)
                    .fill_glyphs(run.glyphs.iter().copied());
            }
            ctx.set_transform(base);
        }
        Prepared::Shape(s, Some(path)) => {
            if let Some(f) = s.fill {
                ctx.set_paint(color(f));
                ctx.fill_path(path);
            }
            if let Some(st) = &s.stroke {
                ctx.set_paint(color(st.color));
                ctx.set_stroke(Stroke::new(st.width as f64));
                ctx.stroke_path(path);
            }
        }
        Prepared::Shape(_, None) => {}
    }
}

/// sRGB-encoded 8-bit → linear.
fn lut() -> &'static [f32; 256] {
    static L: OnceLock<[f32; 256]> = OnceLock::new();
    L.get_or_init(|| std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0)))
}

/// Renders the layer into tiles (premultiplied linear f16); empty tiles are left out.
pub fn render(v: &VectorLayer) -> Vec<(TileCoord, TilePixels)> {
    let p = prepare(v);
    let Some(local) = local_bounds(&p) else { return Vec::new() };
    let m = affine(&v.transform);
    let b = m.transform_rect_bbox(local);
    if !(b.width() < MAX_EXTENT && b.height() < MAX_EXTENT) {
        return Vec::new();
    }
    let ts = TILE_SIZE as i64;
    let (tx0, ty0) = ((b.x0.floor() as i64).div_euclid(ts), (b.y0.floor() as i64).div_euclid(ts));
    let (tx1, ty1) = ((b.x1.ceil() as i64).div_euclid(ts), (b.y1.ceil() as i64).div_euclid(ts));
    let ct = CHUNK_TILES as i64;
    let chunks: Vec<(i64, i64)> = (ty0..=ty1)
        .step_by(ct as usize)
        .flat_map(|cy| (tx0..=tx1).step_by(ct as usize).map(move |cx| (cx, cy)))
        .collect();
    let size = (TILE_SIZE * CHUNK_TILES) as u16;
    chunks
        .par_iter()
        .map_init(
            || (RenderContext::new(size, size), Resources::new(), Pixmap::new(size, size)),
            |(ctx, res, pix), &(cx, cy)| {
                ctx.reset();
                let origin = (cx * ts, cy * ts);
                ctx.set_transform(Affine::translate((-origin.0 as f64, -origin.1 as f64)) * m);
                draw(ctx, res, &p);
                ctx.flush();
                pix.data_as_u8_slice_mut().fill(0);
                ctx.render(&mut *pix, res);
                let data = pix.data();
                let lut = lut();
                let mut out = Vec::new();
                for ty in 0..ct.min(ty1 - cy + 1) {
                    for tx in 0..ct.min(tx1 - cx + 1) {
                        let mut px = vec![f16::ZERO; (TILE_SIZE * TILE_SIZE * 4) as usize];
                        let mut any = false;
                        for y in 0..TILE_SIZE as usize {
                            let row = (ty as usize * TILE_SIZE as usize + y) * size as usize + tx as usize * TILE_SIZE as usize;
                            for x in 0..TILE_SIZE as usize {
                                let s = data[row + x];
                                if s.a == 0 {
                                    continue;
                                }
                                any = true;
                                let a = s.a as f32 / 255.0;
                                let lin = |c: u8| {
                                    // Un-premultiply in 8 bit, linearise, premultiply again.
                                    let straight = ((c as u32 * 255 + s.a as u32 / 2) / s.a as u32).min(255) as usize;
                                    f16::from_f32(lut[straight] * a)
                                };
                                let o = (y * TILE_SIZE as usize + x) * 4;
                                px[o] = lin(s.r);
                                px[o + 1] = lin(s.g);
                                px[o + 2] = lin(s.b);
                                px[o + 3] = f16::from_f32(a);
                            }
                        }
                        if any {
                            out.push((TileCoord::new((cx + tx) as i32, (cy + ty) as i32), px));
                        }
                    }
                }
                out
            },
        )
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use imagine_core::vector::{IDENTITY, StrokeStyle};

    fn px(tiles: &[(TileCoord, TilePixels)], x: i64, y: i64) -> [f32; 4] {
        let ts = TILE_SIZE as i64;
        let c = TileCoord::new(x.div_euclid(ts) as i32, y.div_euclid(ts) as i32);
        match tiles.iter().find(|(t, _)| *t == c) {
            Some((_, p)) => {
                let o = ((y.rem_euclid(ts) * ts + x.rem_euclid(ts)) * 4) as usize;
                [p[o].to_f32(), p[o + 1].to_f32(), p[o + 2].to_f32(), p[o + 3].to_f32()]
            }
            None => [0.0; 4],
        }
    }

    #[test]
    fn rect_fills_exactly_its_area_across_tiles() {
        let red = [1.0, 0.0, 0.0, 1.0];
        let v = VectorLayer::shape(ShapeSpec { geometry: Geometry::Rect { x: 200.0, y: 100.0, w: 400.0, h: 50.0, radius: 0.0 }, fill: Some(red), stroke: None });
        let t = render(&v);
        assert_eq!(px(&t, 300, 120), red);
        assert_eq!(px(&t, 599, 149), red);
        assert_eq!(px(&t, 600, 120)[3], 0.0);
        assert_eq!(px(&t, 199, 120)[3], 0.0);
        // Spans tiles 0…2 horizontally, one row.
        assert!(t.iter().all(|(c, _)| c.y == 0) && t.len() == 3, "{:?}", t.iter().map(|(c, _)| *c).collect::<Vec<_>>());
        let b = bounds(&v).unwrap();
        assert!(b[0] <= 200.0 && b[2] >= 600.0);
    }

    #[test]
    fn transform_and_stroke() {
        let blue = [0.0, 0.0, 1.0, 1.0];
        let mut v = VectorLayer::shape(ShapeSpec {
            geometry: Geometry::Ellipse { cx: 0.0, cy: 0.0, rx: 50.0, ry: 50.0 },
            fill: None,
            stroke: Some(StrokeStyle { color: blue, width: 8.0 }),
        });
        v.transform = [1.0, 0.0, 0.0, 1.0, 1000.0, 1000.0];
        let t = render(&v);
        assert_eq!(px(&t, 1050, 1000), blue, "ring at radius 50");
        assert_eq!(px(&t, 1000, 1000)[3], 0.0, "hollow");
        assert_ne!(v.transform, IDENTITY);
    }

    #[test]
    fn groups_fold_style_families() {
        assert_eq!(base_name("IBM Plex Mono SmBld"), "IBM Plex Mono");
        assert_eq!(base_name("IBM Plex Sans Condensed Light"), "IBM Plex Sans Condensed");
        assert_eq!(base_name("Black"), "Black");
        let groups = font_groups();
        assert!(!groups.is_empty());
        let names: std::collections::HashSet<_> = groups.iter().map(|g| g.name.to_lowercase()).collect();
        assert_eq!(names.len(), groups.len(), "no duplicate groups");
        for g in &groups {
            assert!(g.regular().is_some());
        }
    }

    #[test]
    fn text_renders_glyphs() {
        let fams = font_families();
        assert!(!fams.is_empty(), "no system fonts");
        let white = [1.0, 1.0, 1.0, 1.0];
        let v = VectorLayer::text(TextSpec::new("Hello imagine", 10.0, 10.0, 64.0, white));
        let t = render(&v);
        assert!(!t.is_empty());
        let ink: usize = t.iter().map(|(_, p)| p.chunks(4).filter(|q| q[3].to_f32() > 0.5).count()).sum();
        assert!(ink > 2000, "ink pixels {ink}");
        let b = bounds(&v).unwrap();
        assert!(b[2] - b[0] > 300.0, "text width {b:?}");
    }
}
