//! The order, grouping and icons of the menu bar. The commands themselves
//! (labels, shortcuts) live in [`super::palette::COMMANDS`].

use egui::{Align2, Atom, FontId, Vec2};

use super::glyphs as g;
use super::panels::ToolId;

#[derive(Debug, Clone, Copy)]
pub enum Icon {
    None,
    Glyph(&'static str),
    /// The tool's own icon from the tool bar.
    Tool(ToolId),
}

/// Entries that are not plain commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Special {
    Recent,
    /// Free transform in a given mode (scale, skew, distort, perspective).
    Transform,
    ColorRange,
    /// Expand, contract, border, smooth.
    Modify,
    SaveSelection,
    LoadSelection,
    GenerativeFill,
    PlaceEmbedded,
    OpenAsLayers,
    Revert,
    /// Edit › Stroke.
    StrokeSelection,
    TransformSelection,
    SelectAndMask,
    Preferences,
    SelectionToShape,
    /// Layer › Delete Hidden Layers.
    DeleteHidden,
    CombineShapes,
    PuppetWarp,
    ContentAwareScale,
    ConvertToSmart,
    NewGuide,
    GridSettings,
    Actions,
    Adjustments,
    /// Image › Adjustments: the other adjustments, applied to the pixels.
    MoreAdjustments,
    /// Select › Layer Pixels (as Ctrl+click on a layer thumbnail).
    LayerPixels,
    /// The built-in filters of one group (see [`FILTER_GROUPS`]).
    FilterGroup(u8),
    /// Filter › Plugins.
    Plugins,
    CanvasSize,
    ImageSize,
    Upscale,
    Messages,
}

#[derive(Debug, Clone, Copy)]
pub enum Item {
    /// A command by its (English) label.
    Cmd(&'static str, Icon),
    Special(Special, Icon),
    /// A submenu.
    Sub(&'static str, Icon, &'static [Item]),
    Sep,
}

use Icon::{Glyph as G, Tool as T};
use Item::{Cmd as C, Sep, Special as S, Sub};

const ADJUSTMENTS: &[Item] = &[
    C("Invert", G(g::CIRCLE_HALF)),
    C("Desaturate", Icon::None),
    C("Hue/Saturation …", Icon::None),
    C("Levels …", G(g::CHART_BAR)),
    C("Curves …", G(g::WAVE_SINE)),
    Sep,
    C("Auto Tone", Icon::None),
    C("Auto Contrast", Icon::None),
    C("Auto Color", Icon::None),
    Sep,
    S(Special::MoreAdjustments, Icon::None),
];

const FILE: &[Item] = &[
    C("New …", G(g::FILE_PLUS)),
    C("Open …", G(g::FOLDER_OPEN)),
    S(Special::Recent, G(g::CLOCK_COUNTER_CLOCKWISE)),
    S(Special::OpenAsLayers, G(g::STACK_PLUS)),
    S(Special::PlaceEmbedded, G(g::IMAGE_SQUARE)),
    S(Special::Revert, G(g::ARROW_COUNTER_CLOCKWISE)),
    Sep,
    C("Close", G(g::SIGN_OUT)),
    C("Save", G(g::FLOPPY_DISK)),
    C("Save As …", G(g::FLOPPY_DISK_BACK)),
    C("Export As …", G(g::EXPORT)),
    C("Export as PNG", G(g::FILE_IMAGE)),
    Sep,
    S(Special::Actions, G(g::LIGHTNING)),
    Sep,
    C("Quit", G(g::SIGN_OUT)),
];

const EDIT: &[Item] = &[
    C("Undo", G(g::ARROW_U_UP_LEFT)),
    C("Redo", G(g::ARROW_U_UP_RIGHT)),
    Sep,
    C("Cut", G(g::SCISSORS)),
    C("Copy", G(g::COPY)),
    C("Copy Merged", G(g::STACK)),
    C("Paste", G(g::CLIPBOARD)),
    Sep,
    C("Free Transform", G(g::BOUNDING_BOX)),
    S(Special::Transform, G(g::PERSPECTIVE)),
    S(Special::PuppetWarp, Icon::None),
    S(Special::ContentAwareScale, G(g::ARROWS_OUT)),
    C("Fill Area", G(g::PAINT_BUCKET)),
    S(Special::StrokeSelection, G(g::SELECTION)),
    C("Delete Selection (Pixels)", G(g::TRASH)),
    Sep,
    C("Content-Aware Fill", G(g::SQUARE_HALF)),
    C("Content-Aware Fill (All Layers)", Icon::None),
    C("Remove (AI)", G(g::SPARKLE)),
    S(Special::GenerativeFill, G(g::STAR_FOUR)),
    Sep,
    C("Import Brushes …", G(g::PAINT_BRUSH)),
    C("Define Pattern", G(g::CHECKERBOARD)),
    C("Import Patterns …", G(g::GRID_FOUR)),
    Sep,
    C("Keyboard Shortcuts …", G(g::KEYBOARD)),
    S(Special::Preferences, G(g::SLIDERS_HORIZONTAL)),
];

const IMAGE: &[Item] = &[
    S(Special::ImageSize, G(g::ARROWS_OUT)),
    S(Special::CanvasSize, G(g::FRAME_CORNERS)),
    S(Special::Upscale, G(g::SPARKLE)),
    C("Crop Image", G(g::CROP)),
    Sep,
    C("Rotate Image 90° Clockwise", G(g::ARROW_CLOCKWISE)),
    C("Rotate Image 90° Counter-Clockwise", G(g::ARROW_COUNTER_CLOCKWISE)),
    C("Rotate Image 180°", G(g::ARROWS_CLOCKWISE)),
    C("Flip Image Horizontal", G(g::FLIP_HORIZONTAL)),
    C("Flip Image Vertical", G(g::FLIP_VERTICAL)),
    Sep,
    Sub("Adjustments", G(g::CIRCLE_HALF_TILT), ADJUSTMENTS),
    Sep,
    C("Blend Space sRGB / Linear", G(g::CIRCLES_THREE)),
];

/// Filter › Blur, Sharpen, Noise, Pixelate, Other: filter kinds (serde names).
pub const FILTER_GROUPS: [(&str, &[&str]); 7] = [
    ("Blur", &["gaussian_blur", "box_blur", "motion_blur", "radial_blur", "lens_blur", "surface_blur"]),
    ("Sharpen", &["unsharp_mask", "smart_sharpen", "high_pass"]),
    ("Noise", &["reduce_noise", "median", "add_noise"]),
    ("Pixelate", &["pixelate"]),
    ("Other", &["shadows_highlights", "offset"]),
    ("Distort", &["wave", "twirl", "spherize", "polar_coordinates"]),
    ("Stylize", &["emboss", "find_edges", "oil_paint"]),
];

const FILTER: &[Item] = &[
    C("Last Filter", G(g::ARROWS_CLOCKWISE)),
    Sep,
    Sub("Blur", G(g::DROP), &[S(Special::FilterGroup(0), Icon::None)]),
    Sub("Distort", G(g::WAVE_SINE), &[S(Special::FilterGroup(5), Icon::None)]),
    Sub("Sharpen", G(g::CROSSHAIR), &[S(Special::FilterGroup(1), Icon::None)]),
    Sub("Noise", G(g::DOTS_NINE), &[S(Special::FilterGroup(2), Icon::None)]),
    Sub("Pixelate", G(g::SQUARES_FOUR), &[S(Special::FilterGroup(3), Icon::None)]),
    Sub("Stylize", G(g::PALETTE), &[S(Special::FilterGroup(6), Icon::None)]),
    Sub("Other", G(g::SLIDERS_HORIZONTAL), &[S(Special::FilterGroup(4), Icon::None)]),
    Sep,
    C("Liquify …", G(g::WAVES)),
    Sep,
    S(Special::Plugins, G(g::PUZZLE_PIECE)),
];

const LAYER: &[Item] = &[
    C("New Layer", G(g::PLUS)),
    C("Duplicate Layer", G(g::COPY_SIMPLE)),
    C("Layer via Cut", G(g::SCISSORS)),
    S(Special::ConvertToSmart, G(g::FRAME_CORNERS)),
    S(Special::Adjustments, G(g::CIRCLE_HALF)),
    Sep,
    C("Group Layers", G(g::FOLDER_SIMPLE_PLUS)),
    C("Ungroup Layers", G(g::FOLDER_SIMPLE_MINUS)),
    C("Merge Down", G(g::ARROWS_IN_SIMPLE)),
    C("Merge Visible", G(g::STACK)),
    C("Flatten Image", G(g::IMAGE)),
    S(Special::DeleteHidden, G(g::TRASH)),
    S(Special::CombineShapes, Icon::None),
    C("Create/Release Clipping Mask", G(g::ARROW_ELBOW_DOWN_RIGHT)),
    Sep,
    C("Add Layer Mask", G(g::MASK_HAPPY)),
    C("Paint Mask / Pixels", Icon::None),
    C("Toggle Layer Mask", Icon::None),
    C("Apply Layer Mask", Icon::None),
    C("Remove Background (AI)", G(g::SPARKLE)),
    Sep,
    C("Free Transform / Rotate", G(g::BOUNDING_BOX)),
    C("Rotate Layer 90° Clockwise", G(g::ARROW_CLOCKWISE)),
    C("Rotate Layer 90° Counter-Clockwise", G(g::ARROW_COUNTER_CLOCKWISE)),
    C("Rotate Layer 180°", G(g::ARROWS_CLOCKWISE)),
    C("Flip Layer Horizontal", G(g::FLIP_HORIZONTAL)),
    C("Flip Layer Vertical", G(g::FLIP_VERTICAL)),
    C("Rasterize (Text/Shape/RAW to Pixels)", G(g::SQUARES_FOUR)),
    Sep,
    C("Toggle Layer Visibility", G(g::EYE)),
    C("Bring Layer Forward", G(g::ARROW_UP)),
    C("Send Layer Backward", G(g::ARROW_DOWN)),
    C("Next Blend Mode", G(g::CIRCLES_THREE)),
];

const SELECT: &[Item] = &[
    C("Select All", G(g::SELECTION_ALL)),
    C("Deselect", G(g::SELECTION_SLASH)),
    C("Inverse", G(g::SELECTION_INVERSE)),
    Sep,
    S(Special::ColorRange, G(g::EYEDROPPER_SAMPLE)),
    C("Subject (AI)", G(g::USER_FOCUS)),
    S(Special::LayerPixels, G(g::SELECTION_FOREGROUND)),
    Sep,
    S(Special::Modify, G(g::SELECTION_PLUS)),
    S(Special::TransformSelection, G(g::BOUNDING_BOX)),
    S(Special::SelectAndMask, G(g::USER_FOCUS)),
    S(Special::SelectionToShape, G(g::POLYGON)),
    C("Feather …", G(g::SELECTION_FOREGROUND)),
    Sep,
    S(Special::SaveSelection, G(g::FLOPPY_DISK)),
    S(Special::LoadSelection, G(g::FOLDER_OPEN)),
    Sep,
    C("Quick Mask", G(g::MASK_HAPPY)),
];

const TOOL: &[Item] = &[
    C("Move", T(ToolId::Move)),
    C("Rectangular / Elliptical Marquee", T(ToolId::Marquee)),
    C("Lasso / Polygonal Lasso", T(ToolId::Lasso)),
    C("Magic Wand", T(ToolId::Wand)),
    C("Object Selection (AI)", T(ToolId::ObjectSelect)),
    C("Crop Tool", T(ToolId::Crop)),
    Sep,
    C("Brush / Pencil", T(ToolId::Brush)),
    C("Smudge", T(ToolId::Smudge)),
    C("Dodge / Burn / Sponge", T(ToolId::Dodge)),
    C("Eraser / Background Eraser", T(ToolId::Eraser)),
    C("Clone Stamp", T(ToolId::Clone)),
    C("History Brush", T(ToolId::HistoryBrush)),
    C("Eyedropper / Ruler", T(ToolId::Eyedropper)),
    C("Spot Healing / Remove (AI) / Healing Brush", T(ToolId::SpotHeal)),
    C("Gradient / Paint Bucket", T(ToolId::Gradient)),
    Sep,
    C("Text", T(ToolId::Text)),
    C("Shape (Rectangle / Ellipse)", T(ToolId::Shape)),
    C("Pen", T(ToolId::Pen)),
    Sep,
    C("Switch Brush Tip", G(g::PAINT_BRUSH)),
    C("Toggle Brush Dynamics", G(g::GAUGE)),
    C("Next Palette Color", G(g::PALETTE)),
    C("Black / White", G(g::CIRCLE_HALF)),
];

const VIEW: &[Item] = &[
    C("Fit on Screen", G(g::CORNERS_OUT)),
    C("100 %", G(g::MAGNIFYING_GLASS)),
    C("Reset Rotation", G(g::ARROW_COUNTER_CLOCKWISE)),
    Sep,
    C("Rulers", Icon::None),
    C("Guides", Icon::None),
    C("Grid", Icon::None),
    C("Snap", Icon::None),
    S(Special::NewGuide, G(g::PLUS)),
    C("Clear Guides", G(g::TRASH)),
    S(Special::GridSettings, G(g::GRID_NINE)),
    Sep,
    C("Next Document", G(g::ARROW_RIGHT)),
    C("Previous Document", G(g::ARROW_LEFT)),
    C("Toggle Interface", G(g::SIDEBAR)),
    C("Command Palette", G(g::TERMINAL_WINDOW)),
    S(Special::Messages, G(g::CHAT_TEXT)),
    Sep,
    C("Benchmark", G(g::GAUGE)),
];

/// The entries of `menu` (an English name from [`super::palette::MENUS`]).
pub fn layout(menu: &str) -> &'static [Item] {
    match menu {
        "File" => FILE,
        "Edit" => EDIT,
        "Image" => IMAGE,
        "Layer" => LAYER,
        "Select" => SELECT,
        "Filter" => FILTER,
        "Tool" => TOOL,
        "View" => VIEW,
        _ => &[],
    }
}

pub const ICON_SIZE: f32 = 16.0;

/// A fixed-size icon cell, so the labels line up (empty for `Icon::None`).
pub fn atom<'a>(icon: Icon) -> Atom<'a> {
    Atom::paint(Vec2::splat(ICON_SIZE), move |ui, a| match icon {
        Icon::None => {}
        Icon::Glyph(gl) => {
            ui.painter().text(
                a.rect.center(),
                Align2::CENTER_CENTER,
                gl,
                FontId::proportional(ICON_SIZE - 1.0),
                a.fallback_text_color,
            );
        }
        Icon::Tool(t) => {
            super::icons::paint(ui.painter(), a.rect, super::icons::Icon::Tool(t), a.fallback_text_color);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::super::palette::{COMMANDS, MENUS};
    use super::*;

    /// Every command is in its menu exactly once, and every entry exists.
    #[test]
    fn layouts_cover_the_commands() {
        for menu in MENUS {
            fn cmds(items: &[Item], out: &mut Vec<&'static str>) {
                for i in items {
                    match i {
                        Item::Cmd(l, _) => out.push(l),
                        Item::Sub(_, _, sub) => cmds(sub, out),
                        _ => {}
                    }
                }
            }
            let mut labels = Vec::new();
            cmds(layout(menu), &mut labels);
            for l in &labels {
                assert!(COMMANDS.iter().any(|c| c.menu == menu && c.label == *l), "{menu}: no command {l:?}");
                assert_eq!(labels.iter().filter(|x| *x == l).count(), 1, "{menu}: {l:?} twice");
            }
            for c in COMMANDS.iter().filter(|c| c.menu == menu) {
                assert!(labels.contains(&c.label), "{menu}: {:?} is missing", c.label);
            }
        }
    }
}
