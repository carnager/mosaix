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
    NewGuide,
    GridSettings,
    Actions,
    Filters,
    Adjustments,
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
    Sep,
}

use Icon::{Glyph as G, Tool as T};
use Item::{Cmd as C, Sep, Special as S};

const FILE: &[Item] = &[
    C("New …", G(g::FILE_PLUS)),
    C("Open …", G(g::FOLDER_OPEN)),
    S(Special::Recent, G(g::CLOCK_COUNTER_CLOCKWISE)),
    Sep,
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
    C("Fill Area", G(g::PAINT_BUCKET)),
    C("Delete Selection (Pixels)", G(g::TRASH)),
    Sep,
    C("Content-Aware Fill", G(g::SQUARE_HALF)),
    C("Content-Aware Fill (All Layers)", Icon::None),
    C("Remove (AI)", G(g::SPARKLE)),
    Sep,
    C("Import Brushes …", G(g::PAINT_BRUSH)),
    C("Define Pattern", G(g::CHECKERBOARD)),
    C("Import Patterns …", G(g::GRID_FOUR)),
    Sep,
    C("Keyboard Shortcuts …", G(g::KEYBOARD)),
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
    S(Special::Filters, G(g::SLIDERS_HORIZONTAL)),
    C("Liquify …", G(g::WAVES)),
    Sep,
    C("Blend Space sRGB / Linear", G(g::CIRCLES_THREE)),
];

const LAYER: &[Item] = &[
    C("New Layer", G(g::PLUS)),
    C("Duplicate Layer", G(g::COPY_SIMPLE)),
    S(Special::Adjustments, G(g::CIRCLE_HALF)),
    Sep,
    C("Group Layers", G(g::FOLDER_SIMPLE_PLUS)),
    C("Ungroup Layers", G(g::FOLDER_SIMPLE_MINUS)),
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
    Sep,
    S(Special::Modify, G(g::SELECTION_PLUS)),
    C("Feather …", G(g::SELECTION_FOREGROUND)),
    Sep,
    S(Special::SaveSelection, G(g::FLOPPY_DISK)),
    S(Special::LoadSelection, G(g::FOLDER_OPEN)),
    Sep,
    C("Quick Mask View", G(g::MASK_HAPPY)),
];

const TOOL: &[Item] = &[
    C("Move", T(ToolId::Move)),
    C("Rectangular / Elliptical Marquee", T(ToolId::Marquee)),
    C("Lasso", T(ToolId::Lasso)),
    C("Magic Wand", T(ToolId::Wand)),
    C("Object Selection (AI)", T(ToolId::ObjectSelect)),
    C("Crop Tool", T(ToolId::Crop)),
    Sep,
    C("Brush", T(ToolId::Brush)),
    C("Smudge", T(ToolId::Smudge)),
    C("Eraser", T(ToolId::Eraser)),
    C("Clone Stamp", T(ToolId::Clone)),
    C("Spot Healing / Remove (AI) / Healing Brush", T(ToolId::SpotHeal)),
    C("Gradient", T(ToolId::Gradient)),
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
            let labels: Vec<&str> = layout(menu)
                .iter()
                .filter_map(|i| match i {
                    Item::Cmd(l, _) => Some(*l),
                    _ => None,
                })
                .collect();
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
