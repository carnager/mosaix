//! Document model: a tree of layers whose pixel data is a sparse tile map.
//!
//! The model is generic over the tile handle `T` so it stays GPU-agnostic:
//! the engine uses ref-counted GPU slots, files use blob references, tests use
//! plain values. Tile maps sit behind `Arc`, so cloning a whole document (one
//! undo step) is cheap and unchanged tiles are shared between steps.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::adjust::Adjustment;
use crate::blend::{BlendMode, BlendSpace};
use crate::filter::{Filter, SmartFilter};
use crate::tile::TileCoord;

pub type LayerId = u32;
pub type TileMap<T> = HashMap<TileCoord, T>;

#[derive(Debug, Clone)]
pub enum Content<T> {
    Pixels(Arc<TileMap<T>>),
    /// Children, bottom → top.
    Group(Vec<Layer<T>>),
    /// Non-destructive correction of everything below.
    Adjustment(Adjustment),
    /// Text or shape, rasterised when composited (the engine substitutes
    /// the rendered pixels; `program` skips it in the raw document).
    Vector(crate::vector::VectorLayer),
}

/// One tile of a GPU-side mask.
#[derive(Debug, Clone, PartialEq)]
pub enum MaskTile<T> {
    Solid(u8),
    Pixels(T),
}

/// A sparse 8-bit mask (selection or layer mask). Tiles not in the map have
/// the value `default` (0 = hidden / not selected, 255 = visible / selected).
#[derive(Debug, Clone)]
pub struct Mask<T> {
    pub tiles: Arc<TileMap<MaskTile<T>>>,
    pub default: u8,
}

/// The mask value relevant for one tile.
#[derive(Debug, PartialEq)]
pub enum MaskRef<'a, T> {
    Solid(u8),
    Tile(&'a T),
}

impl<T> Clone for MaskRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for MaskRef<'_, T> {}

impl<T> Mask<T> {
    pub fn filled(v: u8) -> Self {
        Self {
            tiles: Arc::new(TileMap::new()),
            default: v,
        }
    }

    pub fn at(&self, c: TileCoord) -> MaskRef<'_, T> {
        match self.tiles.get(&c) {
            Some(MaskTile::Solid(v)) => MaskRef::Solid(*v),
            Some(MaskTile::Pixels(t)) => MaskRef::Tile(t),
            None => MaskRef::Solid(self.default),
        }
    }

    pub fn map<U>(&self, f: &mut impl FnMut(&T) -> U) -> Mask<U> {
        Mask {
            tiles: Arc::new(
                self.tiles
                    .iter()
                    .map(|(c, t)| {
                        (
                            *c,
                            match t {
                                MaskTile::Solid(v) => MaskTile::Solid(*v),
                                MaskTile::Pixels(p) => MaskTile::Pixels(f(p)),
                            },
                        )
                    })
                    .collect(),
            ),
            default: self.default,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Layer<T> {
    pub id: LayerId,
    pub name: String,
    pub opacity: f32,
    pub blend: BlendMode,
    pub visible: bool,
    pub content: Content<T>,
    pub mask: Option<Mask<T>>,
    pub mask_enabled: bool,
    /// Smart filters on a pixel layer, applied in order when compositing.
    pub filters: Vec<SmartFilter>,
    /// Layer styles (stroke, shadows, glows, bevel …).
    pub effects: Vec<crate::style::Effect>,
    /// Clipping mask: shown only where the layers below it (down to the
    /// first unclipped "base" layer) have pixels.
    pub clip: bool,
    /// Groups: "Pass Through" – the children blend directly with what is
    /// below the group (Photoshop's default); otherwise the group is
    /// composited on its own first (isolated, "Normal").
    pub pass_through: bool,
    /// Photoshop's "Fill": opacity of the layer's own pixels only; layer
    /// styles keep their strength (1 = fully opaque).
    pub fill_opacity: f32,
}

impl<T> Layer<T> {
    pub fn is_group(&self) -> bool {
        matches!(self.content, Content::Group(_))
    }

    pub fn tiles(&self) -> Option<&Arc<TileMap<T>>> {
        match &self.content {
            Content::Pixels(t) => Some(t),
            _ => None,
        }
    }

    /// Whether the layer has pixels (it can be painted on).
    pub fn is_pixels(&self) -> bool {
        matches!(self.content, Content::Pixels(_))
    }

    pub fn vector(&self) -> Option<&crate::vector::VectorLayer> {
        match &self.content {
            Content::Vector(v) => Some(v),
            _ => None,
        }
    }

    pub fn adjustment(&self) -> Option<&Adjustment> {
        match &self.content {
            Content::Adjustment(a) => Some(a),
            _ => None,
        }
    }

    /// The active mask's value at a tile, `None` if the layer has no (enabled) mask.
    pub fn mask_at(&self, c: TileCoord) -> Option<MaskRef<'_, T>> {
        self.mask
            .as_ref()
            .filter(|_| self.mask_enabled)
            .map(|m| m.at(c))
    }
}

/// One step of the per-tile compositing program.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op<'a, T> {
    /// Composite a pixel layer. `tile` is `None` if the layer has no tile at
    /// this coordinate (only emitted when `extra` asked for it, e.g. for a
    /// stroke on an empty area).
    /// `mask` is `None` when the layer is not masked here (no mask, or fully revealing).
    Layer {
        id: LayerId,
        tile: Option<&'a T>,
        opacity: f32,
        blend: BlendMode,
        mask: Option<MaskRef<'a, T>>,
        /// Clipped: covers only where something is already drawn (source-atop).
        clip: bool,
    },
    /// Start a group: isolated (empty), or `pass` (Pass Through: starts
    /// from what is below).
    Push { pass: bool },
    /// Composite the group onto what is below.
    Pop {
        opacity: f32,
        blend: BlendMode,
        mask: Option<MaskRef<'a, T>>,
        clip: bool,
        /// Pass Through: mix with what was below by opacity × mask.
        pass: bool,
    },
    /// Apply an adjustment to what is below (mixed by opacity × mask).
    Adjust {
        id: LayerId,
        adj: &'a Adjustment,
        opacity: f32,
        blend: BlendMode,
        mask: Option<MaskRef<'a, T>>,
    },
}

#[derive(Debug, Clone)]
pub struct Document<T> {
    pub width: u32,
    pub height: u32,
    pub blend_space: BlendSpace,
    /// Top-level layers, bottom → top.
    pub layers: Vec<Layer<T>>,
    pub active: LayerId,
    /// `None` = nothing selected, everything is editable.
    pub selection: Option<Mask<T>>,
    next_id: LayerId,
}

impl<T: Clone> Document<T> {
    /// Document with a single empty "Hintergrund" layer.
    pub fn new(width: u32, height: u32) -> Self {
        let mut d = Self::empty(width, height);
        let id = d.alloc_id();
        d.layers.push(Layer::pixels(id, crate::tr("Background")));
        d.active = id;
        d
    }

    /// Document without layers (for building from files).
    pub fn empty(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            blend_space: BlendSpace::default(),
            layers: Vec::new(),
            active: 0,
            selection: None,
            next_id: 1,
        }
    }

    pub fn alloc_id(&mut self) -> LayerId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Makes sure future ids don't collide with ids loaded from a file.
    pub fn fix_next_id(&mut self) {
        let mut max = 0;
        self.for_each(&mut |l, _| max = max.max(l.id));
        self.next_id = self.next_id.max(max + 1);
    }

    /// Depth-first walk, bottom → top, with nesting depth.
    pub fn for_each(&self, f: &mut impl FnMut(&Layer<T>, usize)) {
        fn walk<T>(ls: &[Layer<T>], d: usize, f: &mut impl FnMut(&Layer<T>, usize)) {
            for l in ls {
                f(l, d);
                if let Content::Group(c) = &l.content {
                    walk(c, d + 1, f);
                }
            }
        }
        walk(&self.layers, 0, f);
    }

    pub fn find(&self, id: LayerId) -> Option<&Layer<T>> {
        fn walk<T>(ls: &[Layer<T>], id: LayerId) -> Option<&Layer<T>> {
            ls.iter().find_map(|l| {
                if l.id == id {
                    Some(l)
                } else if let Content::Group(c) = &l.content {
                    walk(c, id)
                } else {
                    None
                }
            })
        }
        walk(&self.layers, id)
    }

    pub fn find_mut(&mut self, id: LayerId) -> Option<&mut Layer<T>> {
        let (list, i) = self.locate_mut(id)?;
        Some(&mut list[i])
    }

    pub fn active_layer(&self) -> Option<&Layer<T>> {
        self.find(self.active)
    }

    /// The list containing `id` and its index in it.
    fn locate_mut(&mut self, id: LayerId) -> Option<(&mut Vec<Layer<T>>, usize)> {
        fn walk<T>(ls: &mut Vec<Layer<T>>, id: LayerId) -> Option<(&mut Vec<Layer<T>>, usize)> {
            if let Some(i) = ls.iter().position(|l| l.id == id) {
                return Some((ls, i));
            }
            for l in ls.iter_mut() {
                if let Content::Group(c) = &mut l.content
                    && let Some(r) = walk(c, id)
                {
                    return Some(r);
                }
            }
            None
        }
        walk(&mut self.layers, id)
    }

    /// Mutable access to a pixel layer's tile map (copy-on-write).
    pub fn tiles_mut(&mut self, id: LayerId) -> Option<&mut TileMap<T>> {
        match &mut self.find_mut(id)?.content {
            Content::Pixels(t) => Some(Arc::make_mut(t)),
            _ => None,
        }
    }

    /// Inserts a layer above the active one (inside it if the active layer is a group).
    pub fn insert_above_active(&mut self, layer: Layer<T>) -> LayerId {
        let id = layer.id;
        let active = self.active;
        match self.locate_mut(active) {
            Some((list, i)) => {
                if let Content::Group(children) = &mut list[i].content {
                    children.push(layer);
                } else {
                    list.insert(i + 1, layer);
                }
            }
            None => self.layers.push(layer),
        }
        self.active = id;
        id
    }

    /// Adds an adjustment layer above the active one (with a revealing mask
    /// slot left empty; masks are added like for any layer).
    pub fn add_adjustment(&mut self, adj: Adjustment) -> LayerId {
        let id = self.alloc_id();
        let mut l = Layer::pixels(id, adj.name());
        l.content = Content::Adjustment(adj);
        self.insert_above_active(l)
    }

    /// The document as it looks underneath layer `id`: the layer itself and
    /// everything stacked above it are hidden (the input of an adjustment).
    pub fn below(&self, id: LayerId) -> Document<T> {
        fn walk<T>(ls: &mut [Layer<T>], id: LayerId, seen: &mut bool) {
            for l in ls {
                if l.id == id {
                    *seen = true;
                }
                if *seen {
                    l.visible = false;
                } else if let Content::Group(c) = &mut l.content {
                    walk(c, id, seen);
                }
            }
        }
        let mut d = self.clone();
        walk(&mut d.layers, id, &mut false);
        d
    }

    /// Adds a text or shape layer above the active one.
    pub fn add_vector(&mut self, v: crate::vector::VectorLayer) -> LayerId {
        let id = self.alloc_id();
        let mut l = Layer::pixels(id, &v.default_name());
        l.content = Content::Vector(v);
        self.insert_above_active(l)
    }

    pub fn add_pixel_layer(&mut self) -> LayerId {
        let id = self.alloc_id();
        let n = self.count_layers() + 1;
        self.insert_above_active(Layer::pixels(id, &crate::tr!("Layer {n}", n = n)))
    }

    /// Duplicates the active layer (tiles are shared, not copied).
    pub fn duplicate_active(&mut self) -> Option<LayerId> {
        let src = self.active_layer()?.clone();
        let mut copy = self.reassign_ids(src);
        copy.name = crate::tr!("{name} copy", name = copy.name);
        let id = copy.id;
        let (list, i) = self.locate_mut(self.active)?;
        list.insert(i + 1, copy);
        self.active = id;
        Some(id)
    }

    fn reassign_ids(&mut self, mut l: Layer<T>) -> Layer<T> {
        l.id = self.alloc_id();
        if let Content::Group(children) = l.content {
            l.content =
                Content::Group(children.into_iter().map(|c| self.reassign_ids(c)).collect());
        }
        l
    }

    /// Removes the active layer; refuses to remove the last remaining layer.
    pub fn remove_active(&mut self) -> bool {
        if self.count_layers() <= 1 {
            return false;
        }
        let order = self.display_order();
        let pos = order.iter().position(|&id| id == self.active);
        let Some((list, i)) = self.locate_mut(self.active) else {
            return false;
        };
        let removed = list.remove(i);
        let mut gone = Vec::new();
        Self::collect_ids(&removed, &mut gone);
        // Select the next layer below in display order, else the one above.
        let next = pos.and_then(|p| {
            order[p + 1..]
                .iter()
                .chain(order[..p].iter().rev())
                .find(|id| !gone.contains(id))
                .copied()
        });
        if self.layers.is_empty() {
            let id = self.alloc_id();
            self.layers.push(Layer::pixels(id, crate::tr("Background")));
            self.active = id;
        } else if let Some(n) = next {
            self.active = n;
        }
        true
    }

    fn collect_ids(l: &Layer<T>, out: &mut Vec<LayerId>) {
        out.push(l.id);
        if let Content::Group(c) = &l.content {
            for c in c {
                Self::collect_ids(c, out);
            }
        }
    }

    /// Moves the active layer one step up (`+1`) or down (`-1`) within its list.
    pub fn move_active(&mut self, step: i32) -> bool {
        let Some((list, i)) = self.locate_mut(self.active) else {
            return false;
        };
        let j = i as i32 + step;
        if j < 0 || j >= list.len() as i32 {
            return false;
        }
        list.swap(i, j as usize);
        true
    }

    /// Moves layer `id` next to `target` (`above` in panel order), or into
    /// the group `target` as its topmost child when `into` is set. Refuses to
    /// move a group into itself.
    pub fn move_layer(&mut self, id: LayerId, target: LayerId, above: bool, into: bool) -> bool {
        if id == target {
            return false;
        }
        let mut subtree = Vec::new();
        match self.find(id) {
            Some(l) => Self::collect_ids(l, &mut subtree),
            None => return false,
        }
        if subtree.contains(&target)
            || self.find(target).is_none()
            || (into && !self.find(target).unwrap().is_group())
        {
            return false;
        }
        let Some((list, i)) = self.locate_mut(id) else {
            return false;
        };
        let layer = list.remove(i);
        if into {
            let Some(Layer {
                content: Content::Group(children),
                ..
            }) = self.find_mut(target)
            else {
                unreachable!()
            };
            children.push(layer);
        } else {
            let (list, j) = self.locate_mut(target).expect("target exists");
            // Lists are stored bottom → top; "above" in the panel = later in the list.
            list.insert(if above { j + 1 } else { j }, layer);
        }
        true
    }

    /// Wraps the active layer in a new group.
    pub fn group_active(&mut self) -> Option<LayerId> {
        let gid = self.alloc_id();
        let n = self.count_groups() + 1;
        let (list, i) = self.locate_mut(self.active)?;
        let inner = list.remove(i);
        let group = Layer {
            id: gid,
            name: crate::tr!("Group {n}", n = n),
            opacity: 1.0,
            blend: BlendMode::Normal,
            visible: true,
            content: Content::Group(vec![inner]),
            pass_through: true,
            fill_opacity: 1.0,
            mask: None,
            mask_enabled: true,
            filters: Vec::new(),
            effects: Vec::new(),
            clip: false,
        };
        list.insert(i, group);
        self.active = gid;
        Some(gid)
    }

    /// Dissolves the active group into its parent.
    /// Clips the active layer to the one below it, or releases it. False if
    /// it is the lowest layer of its group (nothing to clip to).
    pub fn toggle_clip_active(&mut self) -> bool {
        let id = self.active;
        fn find<T>(ls: &mut [Layer<T>], id: LayerId) -> Option<(&mut [Layer<T>], usize)> {
            if let Some(i) = ls.iter().position(|l| l.id == id) {
                return Some((ls, i));
            }
            for l in ls.iter_mut() {
                if let Content::Group(c) = &mut l.content
                    && let Some(r) = find(c, id)
                {
                    return Some(r);
                }
            }
            None
        }
        match find(&mut self.layers, id) {
            Some((ls, i)) if i > 0 || ls[i].clip => {
                ls[i].clip = !ls[i].clip;
                true
            }
            _ => false,
        }
    }

    pub fn ungroup_active(&mut self) -> bool {
        let Some((list, i)) = self.locate_mut(self.active) else {
            return false;
        };
        if !list[i].is_group() {
            return false;
        }
        let g = list.remove(i);
        let Content::Group(children) = g.content else {
            unreachable!()
        };
        let top = children.last().map(|c| c.id);
        for (k, c) in children.into_iter().enumerate() {
            list.insert(i + k, c);
        }
        if let Some(t) = top {
            self.active = t;
        }
        true
    }

    /// Layer ids in panel order: top → bottom, depth-first.
    pub fn display_order(&self) -> Vec<LayerId> {
        self.panel_rows().into_iter().map(|(id, _)| id).collect()
    }

    /// (id, depth) in panel order: top → bottom.
    pub fn panel_rows(&self) -> Vec<(LayerId, usize)> {
        fn walk<T>(ls: &[Layer<T>], d: usize, out: &mut Vec<(LayerId, usize)>) {
            for l in ls.iter().rev() {
                out.push((l.id, d));
                if let Content::Group(c) = &l.content {
                    walk(c, d + 1, out);
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.layers, 0, &mut out);
        out
    }

    /// Selects the neighbouring layer in panel order (`-1` = above, `+1` = below).
    pub fn select_relative(&mut self, step: i32) {
        let order = self.display_order();
        if let Some(p) = order.iter().position(|&id| id == self.active) {
            let q = (p as i32 + step).clamp(0, order.len() as i32 - 1) as usize;
            self.active = order[q];
        }
    }

    pub fn count_layers(&self) -> usize {
        let mut n = 0;
        self.for_each(&mut |l, _| n += usize::from(!l.is_group()));
        n
    }

    fn count_groups(&self) -> usize {
        let mut n = 0;
        self.for_each(&mut |l, _| n += usize::from(l.is_group()));
        n
    }

    /// Union of all tile coordinates that have pixel data in any layer.
    pub fn all_tile_coords(&self) -> std::collections::HashSet<TileCoord> {
        let mut s = std::collections::HashSet::new();
        self.for_each(&mut |l, _| {
            if let Some(t) = l.tiles() {
                s.extend(t.keys().copied());
            }
        });
        s
    }

    /// Builds the compositing program for one tile. Invisible layers, empty
    /// groups, zero-opacity layers and layers fully hidden by their mask are
    /// skipped. `force` names a layer that is emitted even without a tile here
    /// (the target of an active stroke).
    pub fn program(&self, coord: TileCoord, force: Option<LayerId>) -> Vec<Op<'_, T>> {
        /// `None` = no masking needed here.
        fn mask<T>(l: &Layer<T>, c: TileCoord) -> Option<MaskRef<'_, T>> {
            l.mask_at(c).filter(|m| !matches!(m, MaskRef::Solid(255)))
        }
        /// One layer's ops. `look`: opacity and blend to use (a clipping
        /// base is drawn plainly, its own values apply to the clipping group).
        fn emit<'a, T>(
            l: &'a Layer<T>,
            c: TileCoord,
            force: Option<LayerId>,
            out: &mut Vec<Op<'a, T>>,
            (opacity, blend): (f32, BlendMode),
            clip: bool,
            base: bool,
        ) {
            let m = mask(l, c);
            // Fully masked out – unless a stroke is being painted into this layer (or its mask).
            if matches!(m, Some(MaskRef::Solid(0))) && force != Some(l.id) {
                return;
            }
            match &l.content {
                Content::Pixels(tiles) => {
                    let tile = tiles.get(&c);
                    if tile.is_some() || force == Some(l.id) {
                        out.push(Op::Layer {
                            id: l.id,
                            tile,
                            opacity,
                            blend,
                            mask: m,
                            clip,
                        });
                    }
                }
                Content::Vector(_) => {}
                Content::Adjustment(adj) => {
                    // Nothing below in this group: nothing to adjust.
                    if !out.is_empty() && !matches!(out.last(), Some(Op::Push { pass: false })) {
                        out.push(Op::Adjust {
                            id: l.id,
                            adj,
                            opacity,
                            blend,
                            mask: m,
                        });
                    }
                }
                Content::Group(children) => {
                    // A clipping base or a clipped group is composited isolated.
                    let pass = l.pass_through && !clip && !base;
                    let start = out.len();
                    out.push(Op::Push { pass });
                    walk(children, c, force, out);
                    if out.len() == start + 1 {
                        out.pop(); // empty group
                    } else {
                        out.push(Op::Pop {
                            opacity,
                            blend,
                            mask: m,
                            clip,
                            pass,
                        });
                    }
                }
            }
        }
        fn shown<T>(l: &Layer<T>) -> bool {
            l.visible && l.opacity > 0.0
        }
        fn walk<'a, T>(
            ls: &'a [Layer<T>],
            c: TileCoord,
            force: Option<LayerId>,
            out: &mut Vec<Op<'a, T>>,
        ) {
            let mut i = 0;
            while i < ls.len() {
                let base = &ls[i];
                // The clipped layers resting on this one.
                let mut j = i + 1;
                while j < ls.len() && ls[j].clip {
                    j += 1;
                }
                let clipped = &ls[i + 1..j];
                if !shown(base) {
                    // A hidden base hides its clipping group too.
                    i = j;
                    continue;
                }
                if clipped.iter().all(|l| !shown(l)) {
                    emit(
                        base,
                        c,
                        force,
                        out,
                        (base.opacity, base.blend),
                        false,
                        false,
                    );
                    i = j;
                    continue;
                }
                // Clipping group: base plainly, clipped layers atop, then the
                // base's opacity and blend mode for the whole group.
                let start = out.len();
                out.push(Op::Push { pass: false });
                emit(base, c, force, out, (1.0, BlendMode::Normal), false, true);
                if out.len() == start + 1 {
                    // The base has nothing here: neither do the clipped layers.
                    out.pop();
                    i = j;
                    continue;
                }
                for l in clipped.iter().filter(|l| shown(l)) {
                    emit(l, c, force, out, (l.opacity, l.blend), true, false);
                }
                out.push(Op::Pop {
                    opacity: base.opacity,
                    blend: base.blend,
                    mask: None,
                    clip: false,
                    pass: false,
                });
                i = j;
            }
        }
        let mut out = Vec::new();
        walk(&self.layers, coord, force, &mut out);
        out
    }

    /// Visits every tile handle: layer pixels, layer masks and the selection.
    pub fn for_each_tile(&self, f: &mut impl FnMut(&T)) {
        self.for_each(&mut |l, _| {
            if let Some(t) = l.tiles() {
                t.values().for_each(&mut *f);
            }
            if let Some(m) = &l.mask {
                for t in m.tiles.values() {
                    if let MaskTile::Pixels(p) = t {
                        f(p);
                    }
                }
            }
        });
        if let Some(sel) = &self.selection {
            for t in sel.tiles.values() {
                if let MaskTile::Pixels(p) = t {
                    f(p);
                }
            }
        }
    }

    /// Converts tile handles, keeping structure and ids.
    pub fn map_tiles<U>(&self, f: &mut impl FnMut(&T) -> U) -> Document<U> {
        fn map<T, U>(ls: &[Layer<T>], f: &mut impl FnMut(&T) -> U) -> Vec<Layer<U>> {
            ls.iter()
                .map(|l| Layer {
                    id: l.id,
                    name: l.name.clone(),
                    opacity: l.opacity,
                    blend: l.blend,
                    visible: l.visible,
                    content: match &l.content {
                        Content::Pixels(t) => {
                            Content::Pixels(Arc::new(t.iter().map(|(c, v)| (*c, f(v))).collect()))
                        }
                        Content::Group(c) => Content::Group(map(c, f)),
                        Content::Adjustment(a) => Content::Adjustment(a.clone()),
                        Content::Vector(v) => Content::Vector(v.clone()),
                    },
                    mask: l.mask.as_ref().map(|m| m.map(f)),
                    mask_enabled: l.mask_enabled,
                    filters: l.filters.clone(),
                    effects: l.effects.clone(),
                    clip: l.clip,
                    pass_through: l.pass_through,
                    fill_opacity: l.fill_opacity,
                })
                .collect()
        }
        Document {
            width: self.width,
            height: self.height,
            blend_space: self.blend_space,
            layers: map(&self.layers, f),
            active: self.active,
            selection: self.selection.as_ref().map(|m| m.map(f)),
            next_id: self.next_id,
        }
    }

    /// Human-readable layer list (top → bottom), for the terminal panel.
    pub fn describe(&self) -> String {
        let mut s = String::new();
        for (id, depth) in self.panel_rows() {
            let l = self.find(id).unwrap();
            let marker = if id == self.active { "▶" } else { " " };
            let vis = if l.visible { "●" } else { "○" };
            let icon = if l.is_group() {
                "▾ "
            } else if l.adjustment().is_some() {
                "◐ "
            } else {
                ""
            };
            let mask = match (&l.mask, l.mask_enabled) {
                (Some(_), true) => format!(" [{}]", crate::tr("Mask")),
                (Some(_), false) => format!(" [{}]", crate::tr("Mask off")),
                (None, _) => String::new(),
            };
            let clip = if l.clip { "↳ " } else { "" };
            let name = format!("{}{clip}{icon}{}{mask}", "  ".repeat(depth), l.name);
            s += &format!(
                " {marker} {vis} {name:<28} {:<26} {:>3} %\n",
                l.blend.name(),
                (l.opacity * 100.0).round()
            );
        }
        s
    }
}

impl<T> Layer<T> {
    pub fn pixels(id: LayerId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            opacity: 1.0,
            blend: BlendMode::Normal,
            visible: true,
            content: Content::Pixels(Arc::new(TileMap::new())),
            mask: None,
            mask_enabled: true,
            filters: Vec::new(),
            effects: Vec::new(),
            clip: false,
            pass_through: false,
            fill_opacity: 1.0,
        }
    }

    /// The layer styles that are switched on.
    pub fn active_effects(&self) -> Vec<crate::style::Effect> {
        self.effects.iter().filter(|e| e.enabled).cloned().collect()
    }

    /// The smart filters that are switched on.
    pub fn active_filters(&self) -> impl Iterator<Item = &Filter> {
        self.filters.iter().filter(|f| f.enabled).map(|f| &f.filter)
    }
}

/// Hashes the parts of an op that influence the result (tile identity via `key`).
pub fn hash_program<T>(ops: &[Op<'_, T>], key: impl Fn(&T) -> u64, h: &mut impl Hasher) {
    for op in ops {
        match op {
            Op::Layer {
                id,
                tile,
                opacity,
                blend,
                mask,
                clip,
            } => {
                0u8.hash(h);
                clip.hash(h);
                id.hash(h);
                tile.map(&key).hash(h);
                opacity.to_bits().hash(h);
                (*blend as u32).hash(h);
                hash_mask(mask, &key, h);
            }
            Op::Push { pass } => {
                1u8.hash(h);
                pass.hash(h);
            }
            Op::Adjust {
                id,
                adj,
                opacity,
                blend,
                mask,
            } => {
                3u8.hash(h);
                id.hash(h);
                // Parameters change the result: hash their serialised form.
                format!("{adj:?}").hash(h);
                opacity.to_bits().hash(h);
                (*blend as u32).hash(h);
                hash_mask(mask, &key, h);
            }
            Op::Pop {
                opacity,
                blend,
                mask,
                clip,
                pass,
            } => {
                2u8.hash(h);
                clip.hash(h);
                pass.hash(h);
                opacity.to_bits().hash(h);
                (*blend as u32).hash(h);
                hash_mask(mask, &key, h);
            }
        }
    }
}

fn hash_mask<T>(m: &Option<MaskRef<'_, T>>, key: impl Fn(&T) -> u64, h: &mut impl Hasher) {
    match m {
        None => 0u8.hash(h),
        Some(MaskRef::Solid(v)) => (1u8, *v).hash(h),
        Some(MaskRef::Tile(t)) => (2u8, key(t)).hash(h),
    }
}

#[cfg(test)]
mod adjust_tests {
    use super::*;

    #[test]
    fn adjustments_apply_to_what_is_below_only() {
        let mut d = Document::<u32>::new(512, 256);
        let bg = d.active;
        d.tiles_mut(bg).unwrap().insert(TileCoord::new(0, 0), 7);
        let adj = d.add_adjustment(Adjustment::Invert);
        assert!(d.find(adj).unwrap().tiles().is_none() && !d.find(adj).unwrap().is_pixels());
        let p = d.program(TileCoord::new(0, 0), None);
        assert!(matches!(
            p.as_slice(),
            [Op::Layer { .. }, Op::Adjust { .. }]
        ));
        // Nothing below at (1, 0): no adjust op (and no program).
        assert!(d.program(TileCoord::new(1, 0), None).is_empty());
        // Inside a Pass Through group (the default): still adjusts what is below.
        d.active = adj;
        let g = d.group_active().unwrap();
        let p = d.program(TileCoord::new(0, 0), None);
        assert!(
            matches!(
                p.as_slice(),
                [
                    Op::Layer { .. },
                    Op::Push { pass: true },
                    Op::Adjust { .. },
                    Op::Pop { pass: true, .. }
                ]
            ),
            "{:?}",
            p.len()
        );
        // In an isolated group as first child: nothing below within the group.
        d.find_mut(g).unwrap().pass_through = false;
        let p = d.program(TileCoord::new(0, 0), None);
        assert!(matches!(p.as_slice(), [Op::Layer { .. }]), "{:?}", p.len());
    }
}

#[cfg(test)]
mod move_tests {
    use super::*;

    #[test]
    fn move_layer_reorders_and_nests() {
        let mut d = Document::<()>::new(10, 10);
        let bg = d.active;
        let a = d.add_pixel_layer();
        let b = d.add_pixel_layer();
        assert_eq!(d.display_order(), vec![b, a, bg]);
        assert!(d.move_layer(bg, b, true, false));
        assert_eq!(d.display_order(), vec![bg, b, a]);
        d.active = a;
        let g = d.group_active().unwrap();
        assert!(d.move_layer(b, g, false, true));
        assert_eq!(d.panel_rows(), vec![(bg, 0), (g, 0), (b, 1), (a, 1)]);
        assert!(!d.move_layer(g, b, true, false), "group into its own child");
        assert!(
            !d.move_layer(a, bg, false, true),
            "only groups take children"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc() -> Document<u32> {
        let mut d = Document::new(1000, 1000);
        d.tiles_mut(d.active)
            .unwrap()
            .insert(TileCoord::new(0, 0), 7);
        d
    }

    #[test]
    fn add_select_move_remove() {
        let mut d = doc();
        let bg = d.active;
        let a = d.add_pixel_layer();
        let b = d.add_pixel_layer();
        assert_eq!(d.display_order(), vec![b, a, bg]);
        d.select_relative(1);
        assert_eq!(d.active, a);
        assert!(d.move_active(1));
        assert_eq!(d.display_order(), vec![a, b, bg]);
        assert!(d.remove_active());
        assert_eq!(d.display_order(), vec![b, bg]);
        assert_eq!(d.active, b);
        assert!(d.remove_active());
        assert!(!d.remove_active(), "last layer stays");
    }

    #[test]
    fn duplicate_shares_tiles_and_cow_separates() {
        let mut d = doc();
        let bg = d.active;
        let dup = d.duplicate_active().unwrap();
        let p1 = Arc::as_ptr(d.find(bg).unwrap().tiles().unwrap());
        let p2 = Arc::as_ptr(d.find(dup).unwrap().tiles().unwrap());
        assert_eq!(p1, p2);
        d.tiles_mut(dup).unwrap().insert(TileCoord::new(1, 0), 9);
        assert_eq!(d.find(bg).unwrap().tiles().unwrap().len(), 1);
        assert_eq!(d.find(dup).unwrap().tiles().unwrap().len(), 2);
    }

    #[test]
    fn snapshot_is_independent() {
        let mut d = doc();
        let before = d.clone();
        d.tiles_mut(d.active)
            .unwrap()
            .insert(TileCoord::new(0, 0), 8);
        assert_eq!(
            before.find(before.active).unwrap().tiles().unwrap()[&TileCoord::new(0, 0)],
            7
        );
    }

    #[test]
    fn group_ungroup_and_program() {
        let mut d = doc();
        let bg = d.active;
        let top = d.add_pixel_layer();
        d.tiles_mut(top).unwrap().insert(TileCoord::new(0, 0), 1);
        let g = d.group_active().unwrap();
        d.find_mut(g).unwrap().opacity = 0.5;
        let p = d.program(TileCoord::new(0, 0), None);
        assert_eq!(p.len(), 4);
        assert!(matches!(p[0], Op::Layer { id, .. } if id == bg));
        assert!(matches!(p[1], Op::Push { .. }));
        assert!(matches!(p[3], Op::Pop { opacity, .. } if opacity == 0.5));
        // Masks: solid 0 hides, a pixel tile becomes part of the op, solid 255 is a no-op.
        let mut m = Mask::filled(255);
        Arc::make_mut(&mut m.tiles).insert(TileCoord::new(0, 0), MaskTile::Pixels(42));
        d.find_mut(top).unwrap().mask = Some(m);
        let p = d.program(TileCoord::new(0, 0), None);
        assert!(matches!(
            p[2],
            Op::Layer {
                mask: Some(MaskRef::Tile(&42)),
                ..
            }
        ));
        d.find_mut(top).unwrap().mask = Some(Mask::filled(0));
        assert_eq!(
            d.program(TileCoord::new(0, 0), None).len(),
            1,
            "group with fully hidden layer vanishes"
        );
        let forced = d.program(TileCoord::new(0, 0), Some(top));
        assert!(
            matches!(
                forced[2],
                Op::Layer {
                    mask: Some(MaskRef::Solid(0)),
                    ..
                }
            ),
            "mask painting needs the op"
        );
        d.find_mut(top).unwrap().mask_enabled = false;
        assert_eq!(
            d.program(TileCoord::new(0, 0), None).len(),
            4,
            "disabled mask is ignored"
        );
        d.find_mut(top).unwrap().mask = None;
        // Empty coordinate inside the group: group disappears from the program.
        assert!(d.program(TileCoord::new(3, 3), None).is_empty());
        // Forced layer appears without a tile.
        let p = d.program(TileCoord::new(3, 3), Some(top));
        assert_eq!(p.len(), 3);
        // Hidden layers vanish.
        d.find_mut(bg).unwrap().visible = false;
        assert_eq!(d.program(TileCoord::new(0, 0), None).len(), 3);
        assert!(d.ungroup_active());
        assert_eq!(d.display_order(), vec![top, bg]);
    }

    #[test]
    fn insert_into_group_and_remove_group() {
        let mut d = doc();
        let g = d.group_active().unwrap();
        let inner = d.add_pixel_layer();
        assert!(matches!(&d.find(g).unwrap().content, Content::Group(c) if c.len() == 2));
        let rows = d.panel_rows();
        assert_eq!(rows[0], (g, 0));
        assert_eq!(rows[1], (inner, 1));
        d.active = g;
        let other = d.add_pixel_layer(); // goes into the group, too
        d.active = g;
        d.layers.push(Layer::pixels(99, "x"));
        d.fix_next_id();
        assert!(d.remove_active());
        assert!(d.find(other).is_none());
        assert_eq!(d.active, 99);
        assert!(d.alloc_id() > 99);
    }

    #[test]
    fn map_preserves_structure() {
        let d = doc();
        let mut d = d;
        let mut sel = Mask::filled(0);
        Arc::make_mut(&mut sel.tiles).insert(TileCoord::new(1, 1), MaskTile::Pixels(5));
        d.selection = Some(sel);
        let mut all = Vec::new();
        d.for_each_tile(&mut |t| all.push(*t));
        all.sort();
        assert_eq!(all, vec![5, 7]);
        let m = d.map_tiles(&mut |v| *v as u64 * 2);
        assert!(matches!(
            m.selection.as_ref().unwrap().at(TileCoord::new(1, 1)),
            MaskRef::Tile(&10)
        ));
        assert_eq!(
            m.find(d.active).unwrap().tiles().unwrap()[&TileCoord::new(0, 0)],
            14
        );
    }
}

#[cfg(test)]
mod below_tests {
    use super::*;

    #[test]
    fn below_hides_layer_and_everything_above() {
        let mut d: Document<u32> = Document::new(10, 10);
        let bg = d.active;
        let a = d.add_pixel_layer();
        let g = d.group_active().unwrap();
        d.active = a;
        let adj = d.add_adjustment(Adjustment::Invert);
        let a2 = d.add_pixel_layer(); // above the adjustment, inside the group
        d.active = g;
        let top = d.add_pixel_layer();
        let b = d.below(adj);
        let vis = |id| b.find(id).unwrap().visible;
        assert!(vis(bg) && vis(g) && vis(a));
        assert!(!vis(adj) && !vis(a2) && !vis(top));
        assert!(d.find(top).unwrap().visible);
    }
}
