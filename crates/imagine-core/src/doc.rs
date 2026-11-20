//! Document model: a tree of layers whose pixel data is a sparse tile map.
//!
//! The model is generic over the tile handle `T` so it stays GPU-agnostic:
//! the engine uses ref-counted GPU slots, files use blob references, tests use
//! plain values. Tile maps sit behind `Arc`, so cloning a whole document (one
//! undo step) is cheap and unchanged tiles are shared between steps.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::blend::{BlendMode, BlendSpace};
use crate::tile::TileCoord;

pub type LayerId = u32;
pub type TileMap<T> = HashMap<TileCoord, T>;

#[derive(Debug, Clone)]
pub enum Content<T> {
    Pixels(Arc<TileMap<T>>),
    /// Children, bottom → top.
    Group(Vec<Layer<T>>),
}

#[derive(Debug, Clone)]
pub struct Layer<T> {
    pub id: LayerId,
    pub name: String,
    pub opacity: f32,
    pub blend: BlendMode,
    pub visible: bool,
    pub content: Content<T>,
}

impl<T> Layer<T> {
    pub fn is_group(&self) -> bool {
        matches!(self.content, Content::Group(_))
    }

    pub fn tiles(&self) -> Option<&Arc<TileMap<T>>> {
        match &self.content {
            Content::Pixels(t) => Some(t),
            Content::Group(_) => None,
        }
    }
}

/// One step of the per-tile compositing program.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op<'a, T> {
    /// Composite a pixel layer. `tile` is `None` if the layer has no tile at
    /// this coordinate (only emitted when `extra` asked for it, e.g. for a
    /// stroke on an empty area).
    Layer { id: LayerId, tile: Option<&'a T>, opacity: f32, blend: BlendMode },
    /// Start an isolated group.
    Push,
    /// Composite the group onto what is below.
    Pop { opacity: f32, blend: BlendMode },
}

#[derive(Debug, Clone)]
pub struct Document<T> {
    pub width: u32,
    pub height: u32,
    pub blend_space: BlendSpace,
    /// Top-level layers, bottom → top.
    pub layers: Vec<Layer<T>>,
    pub active: LayerId,
    next_id: LayerId,
}

impl<T: Clone> Document<T> {
    /// Document with a single empty "Hintergrund" layer.
    pub fn new(width: u32, height: u32) -> Self {
        let mut d = Self::empty(width, height);
        let id = d.alloc_id();
        d.layers.push(Layer::pixels(id, "Hintergrund"));
        d.active = id;
        d
    }

    /// Document without layers (for building from files).
    pub fn empty(width: u32, height: u32) -> Self {
        Self { width, height, blend_space: BlendSpace::default(), layers: Vec::new(), active: 0, next_id: 1 }
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
            Content::Group(_) => None,
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

    pub fn add_pixel_layer(&mut self) -> LayerId {
        let id = self.alloc_id();
        let n = self.count_layers() + 1;
        self.insert_above_active(Layer::pixels(id, &format!("Ebene {n}")))
    }

    /// Duplicates the active layer (tiles are shared, not copied).
    pub fn duplicate_active(&mut self) -> Option<LayerId> {
        let src = self.active_layer()?.clone();
        let mut copy = self.reassign_ids(src);
        copy.name = format!("{} Kopie", copy.name);
        let id = copy.id;
        let (list, i) = self.locate_mut(self.active)?;
        list.insert(i + 1, copy);
        self.active = id;
        Some(id)
    }

    fn reassign_ids(&mut self, mut l: Layer<T>) -> Layer<T> {
        l.id = self.alloc_id();
        if let Content::Group(children) = l.content {
            l.content = Content::Group(children.into_iter().map(|c| self.reassign_ids(c)).collect());
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
        let Some((list, i)) = self.locate_mut(self.active) else { return false };
        let removed = list.remove(i);
        let mut gone = Vec::new();
        Self::collect_ids(&removed, &mut gone);
        // Select the next layer below in display order, else the one above.
        let next = pos.and_then(|p| {
            order[p + 1..].iter().chain(order[..p].iter().rev()).find(|id| !gone.contains(id)).copied()
        });
        if self.layers.is_empty() {
            let id = self.alloc_id();
            self.layers.push(Layer::pixels(id, "Hintergrund"));
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
        let Some((list, i)) = self.locate_mut(self.active) else { return false };
        let j = i as i32 + step;
        if j < 0 || j >= list.len() as i32 {
            return false;
        }
        list.swap(i, j as usize);
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
            name: format!("Gruppe {n}"),
            opacity: 1.0,
            blend: BlendMode::Normal,
            visible: true,
            content: Content::Group(vec![inner]),
        };
        list.insert(i, group);
        self.active = gid;
        Some(gid)
    }

    /// Dissolves the active group into its parent.
    pub fn ungroup_active(&mut self) -> bool {
        let Some((list, i)) = self.locate_mut(self.active) else { return false };
        if !list[i].is_group() {
            return false;
        }
        let g = list.remove(i);
        let Content::Group(children) = g.content else { unreachable!() };
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
    /// groups and zero-opacity layers are skipped. `force` names a layer that is
    /// emitted even without a tile here (the target of an active stroke).
    pub fn program(&self, coord: TileCoord, force: Option<LayerId>) -> Vec<Op<'_, T>> {
        fn walk<'a, T>(ls: &'a [Layer<T>], c: TileCoord, force: Option<LayerId>, out: &mut Vec<Op<'a, T>>) {
            for l in ls {
                if !l.visible || l.opacity <= 0.0 {
                    continue;
                }
                match &l.content {
                    Content::Pixels(tiles) => {
                        let tile = tiles.get(&c);
                        if tile.is_some() || force == Some(l.id) {
                            out.push(Op::Layer { id: l.id, tile, opacity: l.opacity, blend: l.blend });
                        }
                    }
                    Content::Group(children) => {
                        let start = out.len();
                        out.push(Op::Push);
                        walk(children, c, force, out);
                        if out.len() == start + 1 {
                            out.pop(); // empty group
                        } else {
                            out.push(Op::Pop { opacity: l.opacity, blend: l.blend });
                        }
                    }
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.layers, coord, force, &mut out);
        out
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
                    },
                })
                .collect()
        }
        Document {
            width: self.width,
            height: self.height,
            blend_space: self.blend_space,
            layers: map(&self.layers, f),
            active: self.active,
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
            let icon = if l.is_group() { "▾ " } else { "" };
            let name = format!("{}{icon}{}", "  ".repeat(depth), l.name);
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
        }
    }
}

/// Hashes the parts of an op that influence the result (tile identity via `key`).
pub fn hash_program<T>(ops: &[Op<'_, T>], key: impl Fn(&T) -> u64, h: &mut impl Hasher) {
    for op in ops {
        match op {
            Op::Layer { id, tile, opacity, blend } => {
                0u8.hash(h);
                id.hash(h);
                tile.map(&key).hash(h);
                opacity.to_bits().hash(h);
                (*blend as u32).hash(h);
            }
            Op::Push => 1u8.hash(h),
            Op::Pop { opacity, blend } => {
                2u8.hash(h);
                opacity.to_bits().hash(h);
                (*blend as u32).hash(h);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc() -> Document<u32> {
        let mut d = Document::new(1000, 1000);
        d.tiles_mut(d.active).unwrap().insert(TileCoord::new(0, 0), 7);
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
        d.tiles_mut(d.active).unwrap().insert(TileCoord::new(0, 0), 8);
        assert_eq!(before.find(before.active).unwrap().tiles().unwrap()[&TileCoord::new(0, 0)], 7);
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
        assert!(matches!(p[1], Op::Push));
        assert!(matches!(p[3], Op::Pop { opacity, .. } if opacity == 0.5));
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
        let m = d.map_tiles(&mut |v| *v as u64 * 2);
        assert_eq!(m.find(d.active).unwrap().tiles().unwrap()[&TileCoord::new(0, 0)], 14);
    }
}
