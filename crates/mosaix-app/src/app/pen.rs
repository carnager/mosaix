//! The pen tool: drawing Bézier paths (click = corner, drag = smooth node,
//! click on the first node closes) and editing the nodes and handles of a
//! path layer. A finished closed path becomes a filled shape layer, an open
//! one a stroked line.

use glam::{DAffine2, DVec2};
use mosaix_core::vector::{Geometry, PathNode, ShapeSpec, StrokeStyle, VectorKind, VectorLayer};
use mosaix_core::{LayerId, tr};
use smithay_client_toolkit::seat::keyboard::{KeyEvent, Keysym};

use super::{App, Drag};
use crate::ui::panels::PathOverlay;

/// Grab distance for nodes and handles (screen px).
const GRAB: f64 = 9.0;

#[derive(Default)]
pub(super) struct Pen {
    /// A path being drawn (document coordinates), not yet a layer.
    pub drawing: Vec<PathNode>,
    /// Selected node of the edited path layer.
    pub selected: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum PenDrag {
    /// Pulling the handles out of the node just placed.
    NewNode(usize),
    /// Moving a node of the edited layer (last pointer position, local coords).
    Node { idx: usize, last: DVec2 },
    /// Moving a handle; `free` breaks the symmetry (Alt).
    Handle { idx: usize, out: bool, free: bool },
}

fn affine(m: &[f64; 6]) -> DAffine2 {
    DAffine2::from_cols_array(m)
}

fn v(p: [f64; 2]) -> DVec2 {
    DVec2::new(p[0], p[1])
}

fn a(p: DVec2) -> [f64; 2] {
    [p.x, p.y]
}

impl App {
    /// The active layer if it is an editable path: (id, nodes, closed, transform).
    fn edited_path(&self) -> Option<(LayerId, Vec<PathNode>, bool, DAffine2)> {
        let l = self.session.as_ref()?.engine.doc().active_layer()?;
        let vl = l.vector()?;
        match &vl.kind {
            VectorKind::Shape(ShapeSpec {
                geometry: Geometry::Bezier { nodes, closed },
                ..
            }) => Some((l.id, nodes.clone(), *closed, affine(&vl.transform))),
            _ => None,
        }
    }

    /// Writes new nodes into the edited path layer (merged into one undo step per drag).
    fn set_path_nodes(&mut self, id: LayerId, nodes: Vec<PathNode>) {
        let Some(mut vl) = self
            .session
            .as_ref()
            .and_then(|s| s.engine.doc().find(id))
            .and_then(|l| l.vector().cloned())
        else {
            return;
        };
        if let VectorKind::Shape(ShapeSpec {
            geometry: Geometry::Bezier { nodes: n, .. },
            ..
        }) = &mut vl.kind
        {
            *n = nodes;
        }
        if let Some(e) = self.engine() {
            let _ = e.set_vector(id, vl, tr("Edit Path"), Some(&format!("path-{id}")));
        }
        self.touch();
    }

    pub(super) fn pen_press(&mut self, screen: DVec2, doc: DVec2) -> Drag {
        let near = |a: DVec2, b: DVec2, view: &mosaix_core::View| {
            view.doc_to_screen(a).distance(view.doc_to_screen(b)) <= GRAB
        };
        // Drawing: close on the first node, else add a node.
        if !self.pen.drawing.is_empty() {
            let first = v(self.pen.drawing[0].p);
            if self.pen.drawing.len() >= 2 && near(first, doc, &self.view) {
                self.pen_finish(true);
                return Drag::None;
            }
            self.pen.drawing.push(PathNode::corner(a(doc)));
            return Drag::Pen(PenDrag::NewNode(self.pen.drawing.len() - 1));
        }
        // Editing the active path layer: handles of the selected node, then nodes.
        if let Some((id, mut nodes, closed, m)) = self.edited_path() {
            let local = m.inverse().transform_point2(doc);
            let to_doc = |p: [f64; 2]| m.transform_point2(v(p));
            if let Some(i) = self.pen.selected.filter(|i| *i < nodes.len()) {
                for out in [true, false] {
                    let h = if out { nodes[i].h_out } else { nodes[i].h_in };
                    if h != nodes[i].p && near(to_doc(h), doc, &self.view) {
                        return Drag::Pen(PenDrag::Handle {
                            idx: i,
                            out,
                            free: self.mods.alt,
                        });
                    }
                }
            }
            if let Some(i) = nodes
                .iter()
                .position(|n| near(to_doc(n.p), doc, &self.view))
            {
                self.pen.selected = Some(i);
                if self.mods.alt {
                    // Alt+click: corner ↔ smooth.
                    let n = &mut nodes[i];
                    if n.is_smooth() {
                        *n = PathNode::corner(n.p);
                    } else {
                        // Handles along the neighbours' direction.
                        mosaix_core::vector::smooth_node(&mut nodes, closed, i);
                    }
                    self.set_path_nodes(id, nodes);
                    if let Some(e) = self.engine() {
                        e.end_merge();
                    }
                    return Drag::None;
                }
                return Drag::Pen(PenDrag::Node {
                    idx: i,
                    last: local,
                });
            }
        }
        // Anywhere else: start a new path.
        let _ = screen;
        self.pen.selected = None;
        self.pen.drawing = vec![PathNode::corner(a(doc))];
        tracing::info!(
            "{}",
            tr(
                "Pen: click = corner, drag = curve, on the start point = close, Enter/Esc = finish open, Backspace = remove last point"
            )
        );
        Drag::Pen(PenDrag::NewNode(0))
    }

    pub(super) fn pen_drag(&mut self, d: PenDrag, doc: DVec2) -> PenDrag {
        match d {
            PenDrag::NewNode(i) => {
                if let Some(n) = self.pen.drawing.get_mut(i) {
                    let p = v(n.p);
                    n.h_out = a(doc);
                    n.h_in = a(p * 2.0 - doc);
                }
                d
            }
            PenDrag::Node { idx, last } => {
                let Some((id, mut nodes, _, m)) = self.edited_path() else {
                    return d;
                };
                let local = m.inverse().transform_point2(doc);
                let delta = local - last;
                if let Some(n) = nodes.get_mut(idx) {
                    n.p = a(v(n.p) + delta);
                    n.h_in = a(v(n.h_in) + delta);
                    n.h_out = a(v(n.h_out) + delta);
                }
                self.set_path_nodes(id, nodes);
                PenDrag::Node { idx, last: local }
            }
            PenDrag::Handle { idx, out, free } => {
                let Some((id, mut nodes, _, m)) = self.edited_path() else {
                    return d;
                };
                let local = m.inverse().transform_point2(doc);
                if let Some(n) = nodes.get_mut(idx) {
                    let p = v(n.p);
                    let (this, other) = if out {
                        (&mut n.h_out, &mut n.h_in)
                    } else {
                        (&mut n.h_in, &mut n.h_out)
                    };
                    *this = a(local);
                    if !free {
                        // Smooth: the opposite handle keeps its length, opposite direction.
                        let len = (v(*other) - p).length();
                        let dir = (p - local).normalize_or_zero();
                        if len > 0.0 && dir != DVec2::ZERO {
                            *other = a(p + dir * len);
                        }
                    }
                }
                self.set_path_nodes(id, nodes);
                d
            }
        }
    }

    pub(super) fn pen_release(&mut self) {
        if let Some(e) = self.engine() {
            e.end_merge();
        }
    }

    /// Ends drawing: a path with at least two nodes becomes a shape layer.
    pub(super) fn pen_finish(&mut self, closed: bool) {
        let nodes = std::mem::take(&mut self.pen.drawing);
        if nodes.len() < 2 {
            return;
        }
        let c = self.brush.color;
        let colour = [c[0], c[1], c[2], 1.0];
        let spec = if closed {
            ShapeSpec {
                geometry: Geometry::Bezier {
                    nodes,
                    closed: true,
                },
                fill: Some(colour),
                stroke: None,
                paint: None,
            }
        } else {
            let width = (2.0 / self.view.zoom as f32).clamp(1.0, 50.0).max(2.0);
            ShapeSpec {
                geometry: Geometry::Bezier {
                    nodes,
                    closed: false,
                },
                fill: None,
                stroke: Some(StrokeStyle::new(colour, width)),
                paint: None,
            }
        };
        if let Some(e) = self.engine()
            && e.add_vector(VectorLayer::shape(spec), tr("Path")).is_some()
        {
            self.touch();
        }
        self.pen.selected = None;
        self.needs_redraw = true;
    }

    /// Keys for the pen. True if used.
    pub(super) fn pen_key(&mut self, ev: &KeyEvent) -> bool {
        if !self.pen.drawing.is_empty() {
            match ev.keysym {
                Keysym::Return | Keysym::KP_Enter | Keysym::Escape => self.pen_finish(false),
                Keysym::BackSpace => {
                    self.pen.drawing.pop();
                }
                _ => return false,
            }
            self.needs_redraw = true;
            return true;
        }
        if matches!(ev.keysym, Keysym::Delete | Keysym::BackSpace)
            && let (Some(i), Some((id, mut nodes, _, _))) = (self.pen.selected, self.edited_path())
            && i < nodes.len()
            && nodes.len() > 2
        {
            nodes.remove(i);
            self.pen.selected = None;
            self.set_path_nodes(id, nodes);
            if let Some(e) = self.engine() {
                e.end_merge();
            }
            return true;
        }
        false
    }

    /// The path being drawn or edited, in screen points (for the UI).
    pub(super) fn path_overlay(&self) -> Option<PathOverlay> {
        let (nodes, closed, m, drawing) = if !self.pen.drawing.is_empty() {
            (self.pen.drawing.clone(), false, DAffine2::IDENTITY, true)
        } else {
            let (_, n, c, m) = self.edited_path()?;
            (n, c, m, false)
        };
        let cursor = if matches!(self.drag, Drag::Pen(_)) {
            None
        } else {
            self.cursor
        };
        Some(overlay(
            &nodes,
            closed,
            m,
            drawing,
            self.pen.selected,
            cursor,
            &self.view,
            self.scale,
        ))
    }
}

/// Screen geometry of a path: curve, nodes, handles, rubber band.
#[allow(clippy::too_many_arguments)]
pub fn overlay(
    nodes: &[PathNode],
    closed: bool,
    m: DAffine2,
    drawing: bool,
    selected: Option<usize>,
    cursor: Option<DVec2>,
    view: &mosaix_core::View,
    scale: f64,
) -> PathOverlay {
    let k = 1.0 / scale;
    let s = |p: DVec2| -> [f32; 2] {
        let q = view.doc_to_screen(p) * k;
        [q.x as f32, q.y as f32]
    };
    let doc = |p: [f64; 2]| m.transform_point2(v(p));
    let mut curve = Vec::new();
    let mut seg = |a: &PathNode, b: &PathNode| {
        let (p0, p1, p2, p3) = (doc(a.p), doc(a.h_out), doc(b.h_in), doc(b.p));
        for i in 0..=24 {
            let t = i as f64 / 24.0;
            let u = 1.0 - t;
            curve.push(s(p0 * (u * u * u)
                + p1 * (3.0 * u * u * t)
                + p2 * (3.0 * u * t * t)
                + p3 * (t * t * t)));
        }
    };
    for w in nodes.windows(2) {
        seg(&w[0], &w[1]);
    }
    if closed && nodes.len() > 2 {
        seg(nodes.last().unwrap(), &nodes[0]);
    }
    let selected = if drawing {
        Some(nodes.len().saturating_sub(1))
    } else {
        selected
    };
    let mut handles = Vec::new();
    for (i, n) in nodes.iter().enumerate() {
        if Some(i) == selected || drawing {
            for h in [n.h_in, n.h_out] {
                if h != n.p {
                    handles.push((s(doc(n.p)), s(doc(h))));
                }
            }
        }
    }
    let rubber = if drawing {
        cursor.and_then(|c| {
            nodes
                .last()
                .map(|l| (s(doc(l.p)), s(view.screen_to_doc(c))))
        })
    } else {
        None
    };
    PathOverlay {
        curve,
        nodes: nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (s(doc(n.p)), Some(i) == selected, n.is_smooth()))
            .collect(),
        handles,
        rubber,
        closing: drawing
            && nodes.len() >= 2
            && cursor.is_some_and(|c| view.doc_to_screen(v(nodes[0].p)).distance(c) <= GRAB),
    }
}
