//! Rasterises vector layers (text, shapes) into document tiles: premultiplied
//! linear RGBA f16, like painted layers. Text is laid out with parley
//! (system fonts via fontique), everything is drawn with vello_cpu in chunks
//! of tiles, in parallel, and only where the content is.

use std::sync::{Mutex, OnceLock};

use half::f16;
use mosaix_core::color::{linear_to_srgb, srgb_to_linear};
use mosaix_core::vector::{
    Affine as Mat, Geometry, Paint, ShapeSpec, TextAlign, TextSpec, VectorKind, VectorLayer,
};
use mosaix_core::{TILE_SIZE, TileCoord};
use parley::{
    Alignment, AlignmentOptions, FontContext, FontStyle, FontWeight, LayoutContext, LineHeight,
    PositionedLayoutItem, StyleProperty,
};
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
    F.get_or_init(|| {
        Mutex::new(Fonts {
            font: FontContext::new(),
            layout: LayoutContext::new(),
        })
    })
}

/// Installed font families, sorted.
pub fn font_families() -> Vec<String> {
    let mut f = fonts().lock().unwrap();
    let mut v: Vec<String> = f
        .font
        .collection
        .family_names()
        .map(str::to_owned)
        .collect();
    v.sort_by_key(|s| s.to_lowercase());
    v.dedup();
    v
}

// ---- editing text on the canvas -------------------------------------------------

/// A text selection (anchor and focus, with the column remembered for
/// up / down). Collapsed = a caret.
pub use parley::Selection as TextSelection;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextMove {
    Left,
    Right,
    WordLeft,
    WordRight,
    Up,
    Down,
    LineStart,
    LineEnd,
    Start,
    End,
}

/// How a click selects: a caret, a word (double click) or a line (triple).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClickKind {
    Caret,
    Word,
    Line,
}

/// Selection at a point (text-box coordinates); with `extend`, the old
/// selection grows to it (Shift+click, dragging).
pub fn text_select_at(
    t: &TextSpec,
    x: f32,
    y: f32,
    kind: ClickKind,
    extend: Option<&TextSelection>,
) -> TextSelection {
    let l = build_layout(t);
    match (extend, kind) {
        (Some(old), _) => old.refresh(&l).extend_to_point(&l, x, y),
        (None, ClickKind::Caret) => TextSelection::from_point(&l, x, y),
        (None, ClickKind::Word) => TextSelection::word_from_point(&l, x, y),
        (None, ClickKind::Line) => TextSelection::line_from_point(&l, x, y),
    }
}

/// A selection from byte offsets (clamped to char boundaries).
pub fn text_range(t: &TextSpec, anchor: usize, focus: usize) -> TextSelection {
    let l = build_layout(t);
    let fix = |i: usize| {
        let mut i = i.min(t.text.len());
        while !t.text.is_char_boundary(i) {
            i -= 1;
        }
        i
    };
    let a = parley::Cursor::from_byte_index(&l, fix(anchor), parley::Affinity::Downstream);
    let f = parley::Cursor::from_byte_index(&l, fix(focus), parley::Affinity::Downstream);
    TextSelection::new(a, f)
}

/// Moves the caret (or extends the selection with `extend`).
pub fn text_move(t: &TextSpec, sel: &TextSelection, m: TextMove, extend: bool) -> TextSelection {
    let l = build_layout(t);
    let s = sel.refresh(&l);
    match m {
        TextMove::Left => s.previous_visual(&l, extend),
        TextMove::Right => s.next_visual(&l, extend),
        TextMove::WordLeft => s.previous_visual_word(&l, extend),
        TextMove::WordRight => s.next_visual_word(&l, extend),
        TextMove::Up => s.previous_line(&l, extend),
        TextMove::Down => s.next_line(&l, extend),
        TextMove::LineStart => s.line_start(&l, extend),
        TextMove::LineEnd => s.line_end(&l, extend),
        TextMove::Start | TextMove::End => {
            let i = if m == TextMove::Start {
                0
            } else {
                t.text.len()
            };
            let c = parley::Cursor::from_byte_index(&l, i, parley::Affinity::Downstream);
            if extend {
                s.extend(c)
            } else {
                TextSelection::new(c, c)
            }
        }
    }
}

/// Size of the laid-out text box (width, height).
pub fn text_box_size(t: &TextSpec) -> (f32, f32) {
    let l = build_layout(t);
    (
        t.width.unwrap_or(0.0).max(l.width()),
        l.height().max(t.size * t.line_height),
    )
}

/// Selection rectangles and the caret (x0, y0, x1, y1) in text-box coordinates.
pub fn text_geometry(t: &TextSpec, sel: &TextSelection) -> (Vec<[f32; 4]>, [f32; 4]) {
    let l = build_layout(t);
    let s = sel.refresh(&l);
    let rects = s
        .geometry(&l)
        .into_iter()
        .map(|(b, _)| [b.x0 as f32, b.y0 as f32, b.x1 as f32, b.y1 as f32])
        .collect();
    let c = s.focus().geometry(&l, (t.size / 24.0).max(1.5));
    // An empty text has no line: a caret of the font's height at the start.
    let caret = if c.height() < 1.0 {
        [0.0, 0.0, (t.size / 24.0).max(1.5), t.size * t.line_height]
    } else {
        [c.x0 as f32, c.y0 as f32, c.x1 as f32, c.y1 as f32]
    };
    (rects, caret)
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
            let k = |f: &FontFace| {
                (f.family != family) as u32 * 2000
                    + (f.italic != italic) as u32 * 1000
                    + (f.weight - weight).abs() as u32
            };
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
    pub const ALL: [Self; 6] = [
        Self::Sans,
        Self::Serif,
        Self::Mono,
        Self::Script,
        Self::Display,
        Self::Symbol,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Sans => "Sans",
            Self::Serif => "Serif",
            Self::Mono => "Mono",
            Self::Script => mosaix_core::tr("Script"),
            Self::Display => "Display",
            Self::Symbol => mosaix_core::tr("Symbols"),
        }
    }
}

/// Classifies a font from its OS/2 family class, PANOSE and fixed pitch,
/// with the name as a fallback.
fn classify(name: &str, data: &[u8], index: u32) -> FontKind {
    use read_fonts::{FontRef, TableProvider};
    let n = name.to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| n.contains(w));
    if has(&[
        "emoji", "symbol", "icon", "dingbat", "awesome", "material", "wingding", "webding",
    ]) {
        return FontKind::Symbol;
    }
    let font = FontRef::from_index(data, index).ok();
    let mono = font
        .as_ref()
        .and_then(|f| f.post().ok())
        .is_some_and(|p| p.is_fixed_pitch() != 0);
    if mono || has(&["mono", "code", "courier", "console", "terminal"]) {
        return FontKind::Mono;
    }
    let os2 = font.as_ref().and_then(|f| f.os2().ok());
    let class = os2.as_ref().map_or(0, |o| (o.s_family_class() >> 8) as u8);
    let panose: Vec<u8> = os2
        .as_ref()
        .map(|o| o.panose_10().to_vec())
        .unwrap_or_default();
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
    let Some(cmap) = FontRef::from_index(data, index)
        .ok()
        .and_then(|f| f.cmap().ok())
    else {
        return true;
    };
    "AaZz"
        .chars()
        .all(|c| cmap.map_codepoint(c).is_some_and(|g| g.to_u32() != 0))
}

/// Style words that some fonts put into the family name.
const STYLE_WORDS: &[&str] = &[
    "thin",
    "hairline",
    "extralight",
    "extlt",
    "ultralight",
    "light",
    "lt",
    "text",
    "regular",
    "book",
    "normal",
    "medium",
    "medm",
    "md",
    "semibold",
    "smbld",
    "demibold",
    "demi",
    "bold",
    "bd",
    "extrabold",
    "extbd",
    "ultrabold",
    "heavy",
    "black",
    "blk",
    "italic",
    "oblique",
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
        let Some(fam) = f.font.collection.family(id) else {
            continue;
        };
        let name = fam.name().to_string();
        let base = base_name(&name);
        let entry = groups
            .entry(base.to_lowercase())
            .or_insert_with(|| (base.clone(), Vec::new(), None));
        if entry.2.is_none()
            && let Some(info) = fam.fonts().first()
            && let Some(blob) = info.load(None)
        {
            entry.2 = Some((
                classify(&base, blob.data(), info.index()),
                has_latin(blob.data(), info.index()),
            ));
        }
        for info in fam.fonts() {
            let weight = info.weight().value();
            let italic = !matches!(info.style(), parley::FontStyle::Normal);
            // Same weight and slope from another alias ("Medm" vs "Medium"): keep one.
            if entry
                .1
                .iter()
                .any(|x| (x.weight - weight).abs() < 1.0 && x.italic == italic)
            {
                continue;
            }
            let label = format!(
                "{}{}",
                weight_label(weight),
                if italic { " Italic" } else { "" }
            );
            entry.1.push(FontFace {
                family: name.clone(),
                weight,
                italic,
                label,
            });
        }
    }
    let mut out: Vec<FontGroup> = groups
        .into_values()
        .filter(|(_, faces, _)| !faces.is_empty())
        .map(|(name, mut faces, kind)| {
            faces.sort_by(|a, b| a.weight.total_cmp(&b.weight).then(a.italic.cmp(&b.italic)));
            let (kind, latin) = kind.unwrap_or((FontKind::Sans, true));
            FontGroup {
                name,
                kind,
                latin,
                faces,
            }
        })
        .collect();
    out.sort_by_key(|g| g.name.to_lowercase());
    out
}

/// A small preview of `text` in `family` (white on transparent, straight
/// RGBA 8 bit): width, height, pixels. For font lists.
pub fn font_preview(
    family: &str,
    weight: f32,
    italic: bool,
    text: &str,
    size: f32,
    max_width: u16,
) -> Option<(u16, u16, Vec<u8>)> {
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
    let rgba = pix
        .data()
        .iter()
        .flat_map(|p| [255, 255, 255, p.a])
        .collect();
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
    /// Underline and strikethrough bars (x0, y0, x1, y1 in the text box).
    decorations: Vec<[f64; 4]>,
    width: f32,
    height: f32,
    /// Text on a path: per glyph (run, index) its transform onto the curve.
    on_path: Option<Vec<(usize, usize, Affine)>>,
}

/// A path flattened into points with cumulative lengths.
struct Polyline {
    pts: Vec<(f64, f64)>,
    len: Vec<f64>,
}

impl Polyline {
    fn new(nodes: &[mosaix_core::vector::PathNode], closed: bool, flip: bool) -> Option<Self> {
        let path = geometry_path(&Geometry::Bezier { nodes: nodes.to_vec(), closed })?;
        let mut pts = Vec::new();
        vello_cpu::kurbo::flatten(path, 0.1, |el| match el {
            vello_cpu::kurbo::PathEl::MoveTo(p) | vello_cpu::kurbo::PathEl::LineTo(p) => pts.push((p.x, p.y)),
            vello_cpu::kurbo::PathEl::ClosePath => {
                if let Some(&f) = pts.first() {
                    pts.push(f);
                }
            }
            _ => {}
        });
        if flip {
            pts.reverse();
        }
        if pts.len() < 2 {
            return None;
        }
        let mut len = vec![0.0];
        for w in pts.windows(2) {
            let d = (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1);
            len.push(len.last().unwrap() + d);
        }
        Some(Self { pts, len })
    }

    fn total(&self) -> f64 {
        *self.len.last().unwrap()
    }

    /// Arc length of the point on the curve nearest to (x, y), and the
    /// signed distance from it along the normal (rotate(direction) · (0, 1)).
    fn project(&self, x: f64, y: f64) -> (f64, f64) {
        let mut best = (f64::MAX, 0.0, 0.0);
        for (i, w) in self.pts.windows(2).enumerate() {
            let (a, b) = (w[0], w[1]);
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            let l2 = (dx * dx + dy * dy).max(1e-12);
            let t = (((x - a.0) * dx + (y - a.1) * dy) / l2).clamp(0.0, 1.0);
            let (px, py) = (a.0 + dx * t, a.1 + dy * t);
            let d2 = (x - px).powi(2) + (y - py).powi(2);
            if d2 < best.0 {
                let l = l2.sqrt();
                // Normal (−sin, cos) of the segment direction.
                let side = ((x - px) * -dy + (y - py) * dx) / l;
                best = (d2, self.len[i] + l * t, side);
            }
        }
        (best.1, best.2)
    }

    /// Point and direction (radians) at arc length `s`.
    fn at(&self, s: f64) -> (f64, f64, f64) {
        let i = self.len.partition_point(|&l| l <= s).clamp(1, self.pts.len() - 1);
        let (a, b) = (self.pts[i - 1], self.pts[i]);
        let seg = (self.len[i] - self.len[i - 1]).max(1e-9);
        let t = ((s - self.len[i - 1]) / seg).clamp(0.0, 1.0);
        (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, (b.1 - a.1).atan2(b.0 - a.0))
    }
}

/// Text on a path: the curve, where the text starts on it and the baseline.
fn path_frame(t: &TextSpec) -> Option<(Polyline, f64, f64, bool)> {
    let p = t.path.as_ref()?;
    let line = Polyline::new(&p.nodes, p.closed, p.flip)?;
    let l = build_layout(t);
    let baseline = l.lines().next().map_or(t.size as f64 * 0.8, |ln| ln.metrics().baseline as f64);
    let start = p.offset.clamp(0.0, 1.0) as f64 * line.total();
    Some((line, start, baseline, p.closed))
}

/// Text on a path: a point in the layer's coordinates → the flat line
/// layout's coordinates (for clicks). `None` if the text has no path.
pub fn path_to_text(t: &TextSpec, x: f64, y: f64) -> Option<(f32, f32)> {
    let (line, start, baseline, closed) = path_frame(t)?;
    let (s, d) = line.project(x, y);
    let mut fx = s - start;
    if closed {
        fx = fx.rem_euclid(line.total());
    }
    Some((fx.max(0.0) as f32, (baseline + d) as f32))
}

/// Text on a path: flat layout coordinates → points in the layer's
/// coordinates (for the caret and selection). Each input point is mapped.
pub fn text_to_path(t: &TextSpec, pts: &[(f32, f32)]) -> Option<Vec<(f64, f64)>> {
    let (line, start, baseline, closed) = path_frame(t)?;
    let total = line.total();
    Some(
        pts.iter()
            .map(|&(x, y)| {
                let mut s = start + x as f64;
                if closed {
                    s = s.rem_euclid(total);
                }
                let (cx, cy, a) = line.at(s.min(total));
                let d = y as f64 - baseline;
                (cx - a.sin() * d, cy + a.cos() * d)
            })
            .collect(),
    )
}

/// The parley layout of a text spec (positions relative to the text box).
fn build_layout(t: &TextSpec) -> parley::Layout<()> {
    let mut guard = fonts().lock().unwrap();
    let Fonts { font, layout } = &mut *guard;
    let mut b = layout.ranged_builder(font, &t.text, 1.0, false);
    b.push_default(StyleProperty::FontFamily(parley::FontFamily::Source(
        t.font.as_str().into(),
    )));
    b.push_default(StyleProperty::FontSize(t.size));
    b.push_default(StyleProperty::FontWeight(FontWeight::new(t.weight)));
    if t.italic {
        b.push_default(StyleProperty::FontStyle(FontStyle::Italic));
    }
    b.push_default(StyleProperty::LineHeight(LineHeight::FontSizeRelative(
        t.line_height,
    )));
    if t.letter_spacing != 0.0 {
        b.push_default(StyleProperty::LetterSpacing(t.letter_spacing));
    }
    let css = t.feature_css();
    if !css.is_empty() {
        b.push_default(StyleProperty::FontFeatures(parley::FontFeatures::Source(css.as_str().into())));
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
    l
}

/// PostScript name of the font a text is set in ("LiberationSans-Bold"),
/// as Photoshop names fonts.
pub fn font_postscript_name(t: &TextSpec) -> Option<String> {
    use read_fonts::types::NameId;
    use read_fonts::{FontRef, TableProvider};
    let mut probe = t.clone();
    probe.text = "a".into();
    let l = build_layout(&probe);
    let line = l.lines().next()?;
    for item in line.items() {
        if let PositionedLayoutItem::GlyphRun(g) = item {
            let fd = g.run().font();
            let f = FontRef::from_index(fd.data.data(), fd.index).ok()?;
            let name = f.name().ok()?;
            let data = name.string_data();
            return name
                .name_record()
                .iter()
                .filter(|r| r.name_id() == NameId::POSTSCRIPT_NAME)
                .find_map(|r| Some(r.string(data).ok()?.chars().collect::<String>()))
                .filter(|s| !s.is_empty());
        }
    }
    None
}

/// Where Photoshop anchors a text, in the layer's own coordinates: point
/// text at the first baseline (left end, centre or right end by alignment),
/// paragraph text at the box's top left.
pub fn text_anchor(t: &TextSpec) -> [f64; 2] {
    if t.width.is_some() {
        return [t.x, t.y];
    }
    let l = build_layout(t);
    let baseline = l.lines().next().map_or(t.size * 0.8, |ln| ln.metrics().baseline) as f64;
    let w = l.width() as f64;
    let x = match t.align {
        TextAlign::Center => t.x + w * 0.5,
        TextAlign::Right => t.x + w,
        _ => t.x,
    };
    [x, t.y + baseline]
}

/// OpenType feature tags (GSUB and GPOS) of the font a text is set in.
pub fn font_feature_tags(t: &TextSpec) -> std::collections::BTreeSet<String> {
    use read_fonts::{FontRef, TableProvider};
    let mut probe = t.clone();
    probe.text = "a".into();
    probe.features.clear();
    let l = build_layout(&probe);
    let mut out = std::collections::BTreeSet::new();
    for line in l.lines() {
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(g) = item {
                let fd = g.run().font();
                let Ok(f) = FontRef::from_index(fd.data.data(), fd.index) else { continue };
                if let Ok(gsub) = f.gsub()
                    && let Ok(list) = gsub.feature_list()
                {
                    out.extend(list.feature_records().iter().map(|r| r.feature_tag().to_string()));
                }
                if let Ok(gpos) = f.gpos()
                    && let Ok(list) = gpos.feature_list()
                {
                    out.extend(list.feature_records().iter().map(|r| r.feature_tag().to_string()));
                }
            }
        }
    }
    out
}

fn layout_text(t: &TextSpec) -> TextLayout {
    // Text on a path is one line.
    let single;
    let t = match &t.path {
        Some(_) => {
            let mut s = t.clone();
            s.text = s.text.replace(['\n', '\r'], " ");
            s.width = None;
            single = s;
            &single
        }
        None => t,
    };
    let l = build_layout(t);
    let mut runs = Vec::new();
    let mut decorations = Vec::new();
    for line in l.lines() {
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(g) = item {
                let run = g.run();
                // Bars from the font's own metrics (offsets up from the baseline).
                // Offsets are to the top of the bar, upwards from the baseline.
                let m = run.font_metrics();
                let (x0, x1, base) = (g.offset() as f64, (g.offset() + g.advance()) as f64, g.baseline() as f64);
                if t.underline {
                    let (off, size) = (m.underline_offset as f64, (m.underline_size as f64).max(1.0));
                    decorations.push([x0, base - off, x1, base - off + size]);
                }
                if t.strikethrough {
                    let (off, size) = (m.strikethrough_offset as f64, (m.strikethrough_size as f64).max(1.0));
                    decorations.push([x0, base - off, x1, base - off + size]);
                }
                runs.push(Run {
                    font: run.font().clone(),
                    size: run.font_size(),
                    coords: run
                        .normalized_coords()
                        .iter()
                        .map(|c| c.to_bits())
                        .collect(),
                    glyphs: g
                        .positioned_glyphs()
                        .map(|p| vello_cpu::Glyph {
                            id: p.id,
                            x: p.x,
                            y: p.y,
                        })
                        .collect(),
                });
            }
        }
    }
    let on_path = t.path.as_ref().and_then(|p| {
        let line = Polyline::new(&p.nodes, p.closed, p.flip)?;
        let total = line.total();
        let baseline = l.lines().next().map_or(t.size as f64 * 0.8, |ln| ln.metrics().baseline as f64);
        let start = p.offset.clamp(0.0, 1.0) as f64 * total;
        let mut out = Vec::new();
        for (ri, run) in runs.iter().enumerate() {
            let n = run.glyphs.len();
            for (gi, g) in run.glyphs.iter().enumerate() {
                // Advance ≈ distance to the next glyph (the last: a quarter em).
                let next = if gi + 1 < n { run.glyphs[gi + 1].x } else { g.x + run.size * 0.5 };
                let mid = g.x as f64 + ((next - g.x) as f64).max(0.0) * 0.5;
                let mut s = start + mid;
                if p.closed {
                    s = s.rem_euclid(total);
                } else if s > total {
                    continue; // does not fit on the path
                }
                let (x, y, a) = line.at(s);
                let m = Affine::translate((x, y)) * Affine::rotate(a) * Affine::translate((-mid, -baseline));
                out.push((ri, gi, m));
            }
        }
        Some(out)
    });
    TextLayout {
        runs,
        decorations,
        width: t.width.unwrap_or(l.width()).max(l.width()),
        height: l.height(),
        on_path,
    }
}

fn affine(m: &Mat) -> Affine {
    Affine::new(*m)
}

fn geometry_path(g: &Geometry) -> Option<BezPath> {
    Some(match g {
        Geometry::Rect { x, y, w, h, radius } => {
            let r = Rect::new(*x, *y, x + w, y + h);
            if *radius > 0.0 {
                RoundedRect::from_rect(r, *radius).to_path(0.1)
            } else {
                r.to_path(0.1)
            }
        }
        Geometry::Ellipse { cx, cy, rx, ry } => {
            // kurbo leaves it open (ends on its start): close it, so strokes
            // can sit inside/outside and exports see a closed path.
            let mut p = Ellipse::new((*cx, *cy), (*rx, *ry), 0.0).to_path(0.1);
            p.close_path();
            p
        }
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
        Geometry::Bezier { nodes, closed } => {
            let mut p = BezPath::new();
            let first = nodes.first()?;
            p.move_to((first.p[0], first.p[1]));
            let seg = |p: &mut BezPath,
                       a: &mosaix_core::vector::PathNode,
                       b: &mosaix_core::vector::PathNode| {
                if a.h_out == a.p && b.h_in == b.p {
                    p.line_to((b.p[0], b.p[1]));
                } else {
                    p.curve_to(
                        (a.h_out[0], a.h_out[1]),
                        (b.h_in[0], b.h_in[1]),
                        (b.p[0], b.p[1]),
                    );
                }
            };
            for w in nodes.windows(2) {
                seg(&mut p, &w[0], &w[1]);
            }
            if *closed && nodes.len() > 2 {
                seg(&mut p, nodes.last().unwrap(), first);
                p.close_path();
            }
            p
        }
    })
}

/// Flattens a path into closed polygons (for the boolean operations).
fn contours(path: &BezPath) -> Vec<Vec<[f64; 2]>> {
    let mut out: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut cur: Vec<[f64; 2]> = Vec::new();
    vello_cpu::kurbo::flatten(path.iter(), 0.05, |el| {
        use vello_cpu::kurbo::PathEl;
        match el {
            PathEl::MoveTo(p) => {
                if cur.len() > 2 {
                    out.push(std::mem::take(&mut cur));
                }
                cur.clear();
                cur.push([p.x, p.y]);
            }
            PathEl::LineTo(p) => cur.push([p.x, p.y]),
            PathEl::ClosePath => {
                if cur.len() > 2 {
                    out.push(std::mem::take(&mut cur));
                }
                cur.clear();
            }
            _ => {}
        }
    });
    if cur.len() > 2 {
        out.push(cur);
    }
    out
}

/// The outline of a shape: its geometry combined with its parts.
fn shape_path(s: &ShapeSpec) -> Option<BezPath> {
    let base = geometry_path(&s.geometry);
    if s.parts.is_empty() {
        return base;
    }
    use i_overlay::core::fill_rule::FillRule;
    use i_overlay::core::overlay_rule::OverlayRule;
    use i_overlay::float::single::SingleFloatOverlay;
    use mosaix_core::vector::PathOp;
    let mut acc: Vec<Vec<Vec<[f64; 2]>>> = base.as_ref().map_or(Vec::new(), |b| vec![contours(b)]);
    for part in &s.parts {
        let clip = geometry_path(&part.geometry).map_or(Vec::new(), |p| contours(&p));
        let rule = match part.op {
            PathOp::Unite => OverlayRule::Union,
            PathOp::Subtract => OverlayRule::Difference,
            PathOp::Intersect => OverlayRule::Intersect,
            PathOp::Exclude => OverlayRule::Xor,
        };
        acc = acc.overlay(&vec![clip], rule, FillRule::NonZero);
    }
    let mut path = BezPath::new();
    for shape in &acc {
        for c in shape {
            for (i, q) in c.iter().enumerate() {
                if i == 0 {
                    path.move_to((q[0], q[1]));
                } else {
                    path.line_to((q[0], q[1]));
                }
            }
            path.close_path();
        }
    }
    Some(path)
}

fn color(c: [f32; 4]) -> AlphaColor<Srgb> {
    AlphaColor::new([
        linear_to_srgb(c[0].max(0.0)),
        linear_to_srgb(c[1].max(0.0)),
        linear_to_srgb(c[2].max(0.0)),
        c[3].clamp(0.0, 1.0),
    ])
}

/// What to draw, prepared once for all chunks.
enum Prepared {
    Text(TextSpec, TextLayout),
    Shape(ShapeSpec, Option<BezPath>),
}

fn prepare(v: &VectorLayer) -> Prepared {
    match &v.kind {
        VectorKind::Text(t) => Prepared::Text(t.clone(), layout_text(t)),
        VectorKind::Shape(s) => Prepared::Shape(s.clone(), shape_path(s)),
    }
}

/// Bounding box (x0, y0, x1, y1) before the layer transform.
fn local_bounds(p: &Prepared) -> Option<Rect> {
    match p {
        Prepared::Text(t, l) => {
            if l.runs.is_empty() {
                return None;
            }
            if let Some(gs) = &l.on_path {
                // Around every placed glyph.
                let m = t.size as f64 * 1.2;
                let mut r: Option<Rect> = None;
                for (ri, gi, a) in gs {
                    let g = l.runs[*ri].glyphs[*gi];
                    let p = *a * vello_cpu::kurbo::Point::new(g.x as f64, g.y as f64);
                    let b = Rect::new(p.x - m, p.y - m, p.x + m, p.y + m);
                    r = Some(r.map_or(b, |r| r.union(b)));
                }
                return r;
            }
            // Generous margin for overhanging glyphs (italics, swashes).
            let m = t.size as f64 * 0.5 + t.letter_spacing.abs() as f64;
            Some(Rect::new(
                t.x - m,
                t.y - m,
                t.x + l.width as f64 + m,
                t.y + l.height as f64 + m,
            ))
        }
        Prepared::Shape(s, path) => {
            let b = path.as_ref()?.bounding_box();
            let pad = s
                .stroke
                .as_ref()
                .map_or(1.0, |st| st.width as f64 * 2.0 + 1.0);
            Some(b.inflate(pad, pad))
        }
    }
}

/// The outline of a shape layer in document px (after its transform) as
/// Bézier subpaths: nodes and whether the subpath is closed. Quadratic
/// segments become cubics. `None` for text layers.
pub fn shape_outline(v: &VectorLayer) -> Option<Vec<(Vec<mosaix_core::vector::PathNode>, bool)>> {
    use mosaix_core::vector::PathNode;
    use vello_cpu::kurbo::{PathEl, Point};
    let VectorKind::Shape(s) = &v.kind else { return None };
    let mut path = shape_path(s)?;
    path.apply_affine(affine(&v.transform));
    let pt = |p: Point| [p.x, p.y];
    let mut out: Vec<(Vec<PathNode>, bool)> = Vec::new();
    let mut cur: Vec<PathNode> = Vec::new();
    let flush = |cur: &mut Vec<PathNode>, out: &mut Vec<(Vec<PathNode>, bool)>, closed: bool| {
        let mut nodes = std::mem::take(cur);
        if closed && nodes.len() > 1 {
            // A closing segment back onto the start: merge the duplicate node.
            let (f, l) = (nodes[0], *nodes.last().unwrap());
            if (f.p[0] - l.p[0]).abs() < 1e-4 && (f.p[1] - l.p[1]).abs() < 1e-4 {
                nodes[0].h_in = l.h_in;
                nodes.pop();
            }
        }
        if !nodes.is_empty() {
            out.push((nodes, closed));
        }
    };
    for el in path.elements() {
        match *el {
            PathEl::MoveTo(p) => {
                flush(&mut cur, &mut out, false);
                cur.push(PathNode::corner(pt(p)));
            }
            PathEl::LineTo(p) => cur.push(PathNode::corner(pt(p))),
            PathEl::QuadTo(c, p) => {
                let a = cur.last().map_or(pt(p), |n| n.p);
                let c1 = [a[0] + 2.0 / 3.0 * (c.x - a[0]), a[1] + 2.0 / 3.0 * (c.y - a[1])];
                let c2 = [p.x + 2.0 / 3.0 * (c.x - p.x), p.y + 2.0 / 3.0 * (c.y - p.y)];
                if let Some(n) = cur.last_mut() {
                    n.h_out = c1;
                }
                cur.push(PathNode { p: pt(p), h_in: c2, h_out: pt(p) });
            }
            PathEl::CurveTo(c1, c2, p) => {
                if let Some(n) = cur.last_mut() {
                    n.h_out = pt(c1);
                }
                cur.push(PathNode { p: pt(p), h_in: pt(c2), h_out: pt(p) });
            }
            PathEl::ClosePath => flush(&mut cur, &mut out, true),
        }
    }
    flush(&mut cur, &mut out, false);
    Some(out)
}

/// `path` (document px) as a geometry in the local space of a layer with
/// transform `t` (unchanged when `t` is the identity).
fn to_local(path: BezPath, g: Option<&Geometry>, t: &Mat) -> Option<Geometry> {
    let identity = *t == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    if identity && let Some(g) = g {
        return Some(g.clone());
    }
    let mut path = path;
    path.apply_affine(affine(t).inverse());
    Some(Geometry::Path { d: path.to_svg() })
}

/// Adds a shape drawn in document px to a shape layer with a path
/// operation (Combine / Subtract / Intersect / Exclude). False for text.
pub fn add_shape_part(v: &mut VectorLayer, g: &Geometry, op: mosaix_core::vector::PathOp) -> bool {
    let t = v.transform;
    let VectorKind::Shape(s) = &mut v.kind else { return false };
    let Some(local) = geometry_path(g).and_then(|p| to_local(p, Some(g), &t)) else { return false };
    s.parts.push(mosaix_core::vector::ShapePart { op, geometry: local });
    true
}

/// Layer › Combine Shapes: `others` (above `base`, bottom → top) become
/// parts of `base` with `op`; `base` keeps its fill and stroke. None if
/// one of them is not a shape.
pub fn combine_shapes(base: &VectorLayer, others: &[&VectorLayer], op: mosaix_core::vector::PathOp) -> Option<VectorLayer> {
    let mut out = base.clone();
    let VectorKind::Shape(_) = &out.kind else { return None };
    for o in others {
        let VectorKind::Shape(os) = &o.kind else { return None };
        // The other layer's whole outline, in document px.
        let mut path = shape_path(os)?;
        path.apply_affine(affine(&o.transform));
        let geometry = to_local(path, None, &base.transform)?;
        let VectorKind::Shape(s) = &mut out.kind else { unreachable!() };
        s.parts.push(mosaix_core::vector::ShapePart { op, geometry });
    }
    Some(out)
}

/// Bounding box of the rendered content in document px, after the transform.
pub fn bounds(v: &VectorLayer) -> Option<[f64; 4]> {
    let p = prepare(v);
    let r = affine(&v.transform).transform_rect_bbox(local_bounds(&p)?);
    Some([r.x0, r.y0, r.x1, r.y1])
}

/// Which parts of a shape `draw` paints.
#[derive(Clone, Copy, PartialEq)]
enum Part {
    All,
    /// The fill in white (its coverage, for a paint).
    FillMask,
    Stroke,
}

fn draw(ctx: &mut RenderContext, res: &mut Resources, p: &Prepared) {
    draw_part(ctx, res, p, Part::All)
}

fn draw_part(ctx: &mut RenderContext, res: &mut Resources, p: &Prepared, part: Part) {
    match p {
        Prepared::Text(t, l) => {
            ctx.set_paint(color(t.color));
            let base = *ctx.transform();
            if let Some(gs) = &l.on_path {
                for (ri, gi, a) in gs {
                    let run = &l.runs[*ri];
                    ctx.set_transform(base * *a);
                    let _ = ctx
                        .glyph_run(res, &run.font)
                        .font_size(run.size)
                        .normalized_coords(&run.coords)
                        .hint(false)
                        .fill_glyphs(std::iter::once(run.glyphs[*gi]));
                }
                ctx.set_transform(base);
                return;
            }
            ctx.set_transform(base * Affine::translate((t.x, t.y)));
            for run in &l.runs {
                let _ = ctx
                    .glyph_run(res, &run.font)
                    .font_size(run.size)
                    .normalized_coords(&run.coords)
                    .hint(false)
                    .fill_glyphs(run.glyphs.iter().copied());
            }
            for d in &l.decorations {
                ctx.fill_rect(&Rect::new(d[0], d[1], d[2], d[3]));
            }
            ctx.set_transform(base);
        }
        Prepared::Shape(s, Some(path)) => {
            if let Some(f) = s.fill.filter(|_| part != Part::Stroke) {
                ctx.set_paint(color(if part == Part::FillMask { [1.0; 4] } else { f }));
                ctx.fill_path(path);
            }
            if part == Part::FillMask {
                return;
            }
            if let Some(st) = &s.stroke {
                use mosaix_core::vector::{StrokeAlign, StrokeCap, StrokeJoin};
                use vello_cpu::kurbo::{Cap, Join};
                use vello_cpu::peniko::{BlendMode, Compose, Mix};
                let closed = path.elements().iter().any(|e| matches!(e, vello_cpu::kurbo::PathEl::ClosePath));
                let align = if closed { st.align } else { StrokeAlign::Center };
                // Inside / outside: a stroke twice as wide, cut to one side of the path.
                let w = st.width as f64 * if align == StrokeAlign::Center { 1.0 } else { 2.0 };
                let stroke = Stroke::new(w)
                    .with_join(match st.join {
                        StrokeJoin::Round => Join::Round,
                        StrokeJoin::Miter => Join::Miter,
                        StrokeJoin::Bevel => Join::Bevel,
                    })
                    .with_caps(match st.cap {
                        StrokeCap::Round => Cap::Round,
                        StrokeCap::Butt => Cap::Butt,
                        StrokeCap::Square => Cap::Square,
                    });
                ctx.set_paint(color(st.color));
                ctx.set_stroke(stroke);
                match align {
                    StrokeAlign::Center => ctx.stroke_path(path),
                    StrokeAlign::Inside => {
                        ctx.push_clip_layer(path);
                        ctx.stroke_path(path);
                        ctx.pop_layer();
                    }
                    StrokeAlign::Outside => {
                        ctx.push_layer(None, None, None, None, None);
                        ctx.stroke_path(path);
                        // Erase the inside (also hides inner outlines of combined paths).
                        ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::DestOut));
                        ctx.set_paint(color([0.0, 0.0, 0.0, 1.0]));
                        ctx.fill_path(path);
                        ctx.pop_layer();
                        ctx.pop_layer();
                    }
                }
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

/// A shape's gradient or pattern, ready to evaluate per document pixel.
enum Painter {
    Gradient { lut: mosaix_core::gradient::GradientLut, pos: mosaix_core::gradient::Positions, inv: Affine },
    Pattern { pattern: mosaix_core::pattern::Pattern, k: f32, offset: [f32; 2] },
}

impl Painter {
    fn new(p: &Prepared, m: Affine) -> Option<Self> {
        let Prepared::Shape(s, Some(path)) = p else {
            return None;
        };
        s.fill?;
        Some(match s.paint.as_ref()? {
            Paint::Gradient { gradient, style, angle, scale, reverse, offset, bounds } => {
                let bounds = bounds.unwrap_or_else(|| {
                    let b = path.bounding_box();
                    [b.x0 as f32, b.y0 as f32, b.x1 as f32, b.y1 as f32]
                });
                let size = [bounds[2] - bounds[0], bounds[3] - bounds[1]];
                let geo = mosaix_core::gradient::GradientGeometry {
                    style: *style,
                    angle: *angle,
                    scale: *scale,
                    reverse: *reverse,
                    bounds,
                    offset: [offset[0] * size[0], offset[1] * size[1]],
                };
                Self::Gradient { lut: gradient.lut(), pos: geo.positions(), inv: m.inverse() }
            }
            Paint::Pattern { pattern, scale, offset } => {
                Self::Pattern { pattern: pattern.clone(), k: 1.0 / scale.max(0.01), offset: *offset }
            }
        })
    }

    /// Straight linear RGB + alpha at the document point `(x, y)`.
    fn at(&self, x: f64, y: f64) -> [f32; 4] {
        match self {
            Self::Gradient { lut, pos, inv } => {
                // Photoshop samples gradients at the pixel corner.
                let p = *inv * vello_cpu::kurbo::Point::new(x - 0.5, y - 0.5);
                lut.at(pos.at(p.x as f32, p.y as f32))
            }
            Self::Pattern { pattern, k, offset } => {
                let p = pattern.sample((x as f32 - offset[0]) * k, (y as f32 - offset[1]) * k);
                [srgb_to_linear(p[0]), srgb_to_linear(p[1]), srgb_to_linear(p[2]), p[3]]
            }
        }
    }
}

/// Renders the layer into tiles (premultiplied linear f16); empty tiles are left out.
pub fn render(v: &VectorLayer) -> Vec<(TileCoord, TilePixels)> {
    let p = prepare(v);
    let Some(local) = local_bounds(&p) else {
        return Vec::new();
    };
    let m = affine(&v.transform);
    let b = m.transform_rect_bbox(local);
    if !(b.width() < MAX_EXTENT && b.height() < MAX_EXTENT) {
        return Vec::new();
    }
    let ts = TILE_SIZE as i64;
    let (tx0, ty0) = (
        (b.x0.floor() as i64).div_euclid(ts),
        (b.y0.floor() as i64).div_euclid(ts),
    );
    let (tx1, ty1) = (
        (b.x1.ceil() as i64).div_euclid(ts),
        (b.y1.ceil() as i64).div_euclid(ts),
    );
    let ct = CHUNK_TILES as i64;
    let chunks: Vec<(i64, i64)> = (ty0..=ty1)
        .step_by(ct as usize)
        .flat_map(|cy| (tx0..=tx1).step_by(ct as usize).map(move |cx| (cx, cy)))
        .collect();
    let size = (TILE_SIZE * CHUNK_TILES) as u16;
    let painter = Painter::new(&p, m);
    chunks
        .par_iter()
        .map_init(
            || {
                (
                    RenderContext::new(size, size),
                    Resources::new(),
                    Pixmap::new(size, size),
                    Pixmap::new(if painter.is_some() { size } else { 1 }, if painter.is_some() { size } else { 1 }),
                )
            },
            |(ctx, res, pix, cover), &(cx, cy)| {
                let origin = (cx * ts, cy * ts);
                let place = Affine::translate((-origin.0 as f64, -origin.1 as f64)) * m;
                let mut pass = |part: Part, into: &mut Pixmap| {
                    ctx.reset();
                    ctx.set_transform(place);
                    draw_part(ctx, res, &p, part);
                    ctx.flush();
                    into.data_as_u8_slice_mut().fill(0);
                    ctx.render(&mut *into, res);
                };
                if painter.is_some() {
                    pass(Part::FillMask, cover);
                    pass(Part::Stroke, pix);
                } else {
                    pass(Part::All, pix);
                }
                let data = pix.data();
                let cov = painter.as_ref().map(|_| cover.data());
                let lut = lut();
                let mut out = Vec::new();
                for ty in 0..ct.min(ty1 - cy + 1) {
                    for tx in 0..ct.min(tx1 - cx + 1) {
                        let mut px = vec![f16::ZERO; (TILE_SIZE * TILE_SIZE * 4) as usize];
                        let mut any = false;
                        for y in 0..TILE_SIZE as usize {
                            let row = (ty as usize * TILE_SIZE as usize + y) * size as usize
                                + tx as usize * TILE_SIZE as usize;
                            for x in 0..TILE_SIZE as usize {
                                let s = data[row + x];
                                if let (Some(pt), Some(cov)) = (&painter, cov) {
                                    // The paint under the stroke.
                                    let c = cov[row + x].a as f32 / 255.0;
                                    if c == 0.0 && s.a == 0 {
                                        continue;
                                    }
                                    let doc = (
                                        (origin.0 + tx * ts) as f64 + x as f64 + 0.5,
                                        (origin.1 + ty * ts) as f64 + y as f64 + 0.5,
                                    );
                                    let f = if c > 0.0 { pt.at(doc.0, doc.1) } else { [0.0; 4] };
                                    let fa = f[3] * c;
                                    let sa = s.a as f32 / 255.0;
                                    let sl = |v: u8| {
                                        if s.a == 0 {
                                            0.0
                                        } else {
                                            lut[((v as u32 * 255 + s.a as u32 / 2) / s.a as u32).min(255) as usize] * sa
                                        }
                                    };
                                    let st = [sl(s.r), sl(s.g), sl(s.b)];
                                    let a = sa + fa * (1.0 - sa);
                                    if a <= 0.0 {
                                        continue;
                                    }
                                    any = true;
                                    let o = (y * TILE_SIZE as usize + x) * 4;
                                    for k in 0..3 {
                                        px[o + k] = f16::from_f32(st[k] + f[k] * fa * (1.0 - sa));
                                    }
                                    px[o + 3] = f16::from_f32(a);
                                    continue;
                                }
                                if s.a == 0 {
                                    continue;
                                }
                                any = true;
                                let a = s.a as f32 / 255.0;
                                let lin = |c: u8| {
                                    // Un-premultiply in 8 bit, linearise, premultiply again.
                                    let straight = ((c as u32 * 255 + s.a as u32 / 2) / s.a as u32)
                                        .min(255)
                                        as usize;
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
    use mosaix_core::vector::{IDENTITY, StrokeStyle};

    fn px(tiles: &[(TileCoord, TilePixels)], x: i64, y: i64) -> [f32; 4] {
        let ts = TILE_SIZE as i64;
        let c = TileCoord::new(x.div_euclid(ts) as i32, y.div_euclid(ts) as i32);
        match tiles.iter().find(|(t, _)| *t == c) {
            Some((_, p)) => {
                let o = ((y.rem_euclid(ts) * ts + x.rem_euclid(ts)) * 4) as usize;
                [
                    p[o].to_f32(),
                    p[o + 1].to_f32(),
                    p[o + 2].to_f32(),
                    p[o + 3].to_f32(),
                ]
            }
            None => [0.0; 4],
        }
    }

    #[test]
    fn shape_parts_combine() {
        use mosaix_core::vector::{PathOp, ShapePart};
        let circle = |cx: f64, r: f64| Geometry::Ellipse { cx, cy: 100.0, rx: r, ry: r };
        let shape = |op: PathOp, part: Geometry| {
            VectorLayer::shape(ShapeSpec {
                geometry: circle(100.0, 60.0),
                fill: Some([1.0, 0.0, 0.0, 1.0]),
                stroke: None,
                paint: None,
                parts: vec![ShapePart { op, geometry: part }],
            })
        };
        let a = |v: &VectorLayer, x: i64| px(&render(v), x, 100)[3];
        // Ring: a hole in the middle.
        let ring = shape(PathOp::Subtract, circle(100.0, 30.0));
        assert!(a(&ring, 100) < 0.01 && a(&ring, 55) > 0.99, "ring");
        // Two overlapping circles (100 ± 60 and 160 ± 60, overlap 100…160).
        let other = circle(160.0, 60.0);
        let u = shape(PathOp::Unite, other.clone());
        assert!(a(&u, 60) > 0.99 && a(&u, 130) > 0.99 && a(&u, 200) > 0.99);
        let i = shape(PathOp::Intersect, other.clone());
        assert!(a(&i, 60) < 0.01 && a(&i, 130) > 0.99 && a(&i, 200) < 0.01);
        let x = shape(PathOp::Exclude, other.clone());
        assert!(a(&x, 60) > 0.99 && a(&x, 130) < 0.01 && a(&x, 200) > 0.99);
        let d = shape(PathOp::Subtract, other);
        assert!(a(&d, 60) > 0.99 && a(&d, 130) < 0.01 && a(&d, 200) < 0.01);
        // A rectangle base and a part from path data (as Combine Shapes makes it).
        let mut rx = VectorLayer::shape(ShapeSpec {
            geometry: Geometry::Rect { x: 220.0, y: 40.0, w: 120.0, h: 120.0, radius: 0.0 },
            fill: Some([1.0, 0.0, 0.0, 1.0]),
            stroke: Some(mosaix_core::vector::StrokeStyle::new([0.0, 0.0, 0.0, 1.0], 4.0)),
            paint: None,
            parts: Vec::new(),
        });
        let green = VectorLayer::shape(ShapeSpec {
            geometry: Geometry::Ellipse { cx: 330.0, cy: 70.0, rx: 50.0, ry: 50.0 },
            fill: Some([0.0, 1.0, 0.0, 1.0]),
            stroke: None,
            paint: None,
            parts: Vec::new(),
        });
        rx = combine_shapes(&rx, &[&green], PathOp::Exclude).unwrap();
        let t = render(&rx);
        assert!(px(&t, 320, 70)[3] < 0.01, "overlap empty: {:?}", px(&t, 320, 70));
        assert!(px(&t, 250, 140)[3] > 0.99 && px(&t, 370, 70)[3] > 0.99);
        // The outline follows the result: the ring has two closed subpaths.
        assert_eq!(shape_outline(&ring).unwrap().len(), 2);
    }

    #[test]
    fn rect_fills_exactly_its_area_across_tiles() {
        let red = [1.0, 0.0, 0.0, 1.0];
        let v = VectorLayer::shape(ShapeSpec {
            geometry: Geometry::Rect {
                x: 200.0,
                y: 100.0,
                w: 400.0,
                h: 50.0,
                radius: 0.0,
            },
            fill: Some(red),
            stroke: None,
            paint: None,
            parts: Vec::new(),
        });
        let t = render(&v);
        assert_eq!(px(&t, 300, 120), red);
        assert_eq!(px(&t, 599, 149), red);
        assert_eq!(px(&t, 600, 120)[3], 0.0);
        assert_eq!(px(&t, 199, 120)[3], 0.0);
        // Spans tiles 0…2 horizontally, one row.
        assert!(
            t.iter().all(|(c, _)| c.y == 0) && t.len() == 3,
            "{:?}",
            t.iter().map(|(c, _)| *c).collect::<Vec<_>>()
        );
        let b = bounds(&v).unwrap();
        assert!(b[0] <= 200.0 && b[2] >= 600.0);
    }

    #[test]
    fn text_follows_a_path() {
        use mosaix_core::vector::{PathNode, TextPath};
        // An arc over the top: from (100, 300) up through (300, 100) to (500, 300).
        let mut a = PathNode::corner([100.0, 300.0]);
        a.h_out = [100.0, 150.0];
        let mut b = PathNode::corner([500.0, 300.0]);
        b.h_in = [500.0, 150.0];
        let mut t = TextSpec::new("Hello curved world", 0.0, 0.0, 40.0, [0.0, 0.0, 0.0, 1.0]);
        t.path = Some(TextPath { nodes: vec![a, b], closed: false, offset: 0.0, flip: false });
        let v = VectorLayer::text(t);
        let [x0, y0, x1, y1] = bounds(&v).unwrap();
        assert!(x0 < 150.0 && x1 > 400.0 && y0 < 200.0 && y1 > 280.0, "follows the arc: {:?}", [x0, y0, x1, y1]);
        let tiles = render(&v);
        let ink = |x: i64, y: i64| tiles.iter().any(|(c, px)| {
            let (ox, oy) = c.origin();
            let ts = TILE_SIZE as i64;
            (ox..ox + ts).contains(&x) && (oy..oy + ts).contains(&y) && {
                // Any ink within 25 px.
                (-25..=25).any(|dy| (-25..=25).any(|dx| {
                    let (lx, ly) = (x + dx - ox, y + dy - oy);
                    (0..ts).contains(&lx) && (0..ts).contains(&ly) && px[((ly * ts + lx) * 4 + 3) as usize].to_f32() > 0.5
                }))
            }
        });
        assert!(ink(130, 230), "start of the arc, left side going up");
        assert!(!ink(300, 300), "nothing where straight text would sit");
        // Clicks and carets map between the curve and the flat line: there and back.
        let t = match &v.kind {
            mosaix_core::vector::VectorKind::Text(t) => t.clone(),
            _ => unreachable!(),
        };
        for &(x, y) in &[(40.0f32, 20.0f32), (200.0, 35.0), (320.0, 10.0)] {
            let p = text_to_path(&t, &[(x, y)]).unwrap()[0];
            let (bx, by) = path_to_text(&t, p.0, p.1).unwrap();
            assert!((bx - x).abs() < 0.5 && (by - y).abs() < 0.5, "({x},{y}) → {p:?} → ({bx},{by})");
        }
    }

    #[test]
    fn opentype_features_change_shaping() {
        let glyphs = |t: &TextSpec| -> usize {
            let l = build_layout(t);
            l.lines()
                .flat_map(|line| line.items().collect::<Vec<_>>())
                .map(|i| match i {
                    PositionedLayoutItem::GlyphRun(g) => g.positioned_glyphs().count(),
                    _ => 0,
                })
                .sum()
        };
        // Any installed serif with standard ligatures will do.
        let mut t = TextSpec::new("office", 0.0, 0.0, 40.0, [0.0, 0.0, 0.0, 1.0]);
        for family in ["Liberation Serif", "DejaVu Serif", "Noto Serif", "serif"] {
            t.font = family.into();
            if font_feature_tags(&t).contains("liga") {
                break;
            }
        }
        if !font_feature_tags(&t).contains("liga") {
            eprintln!("skipping: no font with ligatures");
            return;
        }
        let with = glyphs(&t);
        t.set_feature("liga", false);
        assert_eq!(t.feature_css(), "\"liga\" 0");
        let without = glyphs(&t);
        assert!(with < without, "ffi ligature: {with} vs {without} glyphs");
        t.set_feature("liga", true);
        assert!(t.features.is_empty(), "back to default, nothing stored");
    }

    #[test]
    fn bezier_path_fills_a_closed_curve() {
        use mosaix_core::vector::PathNode;
        let red = [1.0, 0.0, 0.0, 1.0];
        // A closed square with one rounded side.
        let mut n = vec![
            PathNode::corner([100.0, 100.0]),
            PathNode::corner([300.0, 100.0]),
            PathNode::corner([300.0, 300.0]),
            PathNode::corner([100.0, 300.0]),
        ];
        n[1].h_out = [400.0, 150.0];
        n[2].h_in = [400.0, 250.0];
        let v = VectorLayer::shape(ShapeSpec {
            geometry: Geometry::Bezier {
                nodes: n,
                closed: true,
            },
            fill: Some(red),
            stroke: None,
            paint: None,
            parts: Vec::new(),
        });
        let t = render(&v);
        assert_eq!(px(&t, 200, 200), red);
        assert_eq!(px(&t, 360, 200), red, "bulging side");
        assert_eq!(px(&t, 360, 120)[3], 0.0);
    }

    #[test]
    fn transform_and_stroke() {
        let blue = [0.0, 0.0, 1.0, 1.0];
        let mut v = VectorLayer::shape(ShapeSpec {
            geometry: Geometry::Ellipse {
                cx: 0.0,
                cy: 0.0,
                rx: 50.0,
                ry: 50.0,
            },
            fill: None,
            stroke: Some(StrokeStyle::new(blue, 8.0)),
            paint: None,
            parts: Vec::new(),
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
        assert_eq!(
            base_name("IBM Plex Sans Condensed Light"),
            "IBM Plex Sans Condensed"
        );
        assert_eq!(base_name("Black"), "Black");
        let groups = font_groups();
        assert!(!groups.is_empty());
        let names: std::collections::HashSet<_> =
            groups.iter().map(|g| g.name.to_lowercase()).collect();
        assert_eq!(names.len(), groups.len(), "no duplicate groups");
        for g in &groups {
            assert!(g.regular().is_some());
        }
    }

    #[test]
    fn text_editing_moves_and_measures() {
        let t = TextSpec::new("Hallo Welt\nzweite Zeile", 0.0, 0.0, 40.0, [1.0; 4]);
        let s = text_select_at(&t, 0.0, 10.0, ClickKind::Caret, None);
        assert_eq!(s.focus().index(), 0);
        let s = text_move(&t, &s, TextMove::Right, false);
        assert_eq!(s.focus().index(), 1);
        let w = text_move(&t, &s, TextMove::WordRight, false);
        assert!(w.focus().index() >= 5, "{}", w.focus().index());
        let down = text_move(&t, &s, TextMove::Down, false);
        assert!(down.focus().index() > "Hallo Welt".len());
        let all = text_move(
            &t,
            &text_move(&t, &s, TextMove::Start, false),
            TextMove::End,
            true,
        );
        assert_eq!(all.text_range(), 0..t.text.len());
        let (rects, caret) = text_geometry(&t, &all);
        assert_eq!(rects.len(), 2, "one rectangle per line");
        assert!(caret[1] > 30.0, "caret on the second line: {caret:?}");
        let (_, c1) = text_geometry(&t, &text_range(&t, 1, 1));
        let (_, c3) = text_geometry(&t, &text_range(&t, 3, 3));
        assert!(c3[0] > c1[0]);
        let word = text_select_at(&t, 160.0, 10.0, ClickKind::Word, None);
        assert_eq!(&t.text[word.text_range()], "Welt");
    }

    #[test]
    fn text_renders_glyphs() {
        let fams = font_families();
        assert!(!fams.is_empty(), "no system fonts");
        let white = [1.0, 1.0, 1.0, 1.0];
        let v = VectorLayer::text(TextSpec::new("Hello mosaix", 10.0, 10.0, 64.0, white));
        let t = render(&v);
        assert!(!t.is_empty());
        let ink: usize = t
            .iter()
            .map(|(_, p)| p.chunks(4).filter(|q| q[3].to_f32() > 0.5).count())
            .sum();
        assert!(ink > 2000, "ink pixels {ink}");
        let b = bounds(&v).unwrap();
        assert!(b[2] - b[0] > 300.0, "text width {b:?}");
    }

    #[test]
    fn gradient_and_pattern_paint() {
        use mosaix_core::gradient::Gradient;
        let red = [1.0, 0.0, 0.0, 1.0];
        let mut spec = ShapeSpec {
            geometry: Geometry::Rect { x: 0.0, y: 0.0, w: 100.0, h: 20.0, radius: 0.0 },
            fill: Some(red),
            stroke: None,
            paint: Some(Paint::Gradient {
                gradient: Gradient::two([0.0; 3], [1.0; 3]),
                style: Default::default(),
                angle: 0.0,
                scale: 1.0,
                reverse: false,
                offset: [0.0; 2],
                bounds: None,
            }),
            parts: Vec::new(),
        };
        let tiles = render(&VectorLayer::shape(spec.clone()));
        // Left → right, black → white, the fill colour not used.
        let (l, m, r) = (px(&tiles, 0, 10), px(&tiles, 50, 10), px(&tiles, 99, 10));
        assert!(l[0] < 0.01 && r[0] > 0.95 && (l[3] - 1.0).abs() < 0.01, "{l:?} {r:?}");
        assert!((linear_to_srgb(m[0]) - 0.5).abs() < 0.03 && (m[0] - m[1]).abs() < 1e-3, "{m:?}");
        // Moved with the layer.
        let mut v = VectorLayer::shape(spec.clone());
        v.transform = [1.0, 0.0, 0.0, 1.0, 50.0, 0.0];
        let t = render(&v);
        assert!(px(&t, 50, 10)[0] < 0.01 && px(&t, 149, 10)[0] > 0.95);
        // A pattern, tiled in document coordinates.
        let pattern = mosaix_core::pattern::Pattern::new("2", 2, 1, vec![255, 255, 255, 255, 0, 0, 0, 255]).unwrap();
        spec.paint = Some(Paint::Pattern { pattern, scale: 1.0, offset: [0.0; 2] });
        let t = render(&VectorLayer::shape(spec));
        assert!(px(&t, 10, 5)[0] > 0.95 && px(&t, 11, 5)[0] < 0.05, "{:?} {:?}", px(&t, 10, 5), px(&t, 11, 5));
    }
}

#[cfg(test)]
mod outline_tests {
    use super::*;
    use mosaix_core::vector::{Geometry, ShapeSpec};

    #[test]
    fn ellipse_outline_has_four_nodes() {
        let v = VectorLayer::shape(ShapeSpec {
            geometry: Geometry::Ellipse { cx: 50.0, cy: 50.0, rx: 40.0, ry: 30.0 },
            fill: Some([1.0; 4]),
            stroke: None,
            paint: None,
            parts: Vec::new(),
        });
        let o = shape_outline(&v).unwrap();
        assert_eq!(o.len(), 1);
        assert!(o[0].1, "closed");
        assert_eq!(o[0].0.len(), 4, "{:?}", o[0].0);
    }


}
