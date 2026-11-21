//! Text and shape layers: adding, editing and rasterising. The pixels are
//! produced in `filter::refresh_vectors` when the document is composited.

use mosaix_core::doc::Content;
use mosaix_core::{LayerId, VectorLayer};

use super::Engine;

impl Engine {
    /// Adds a text or shape layer above the active one (one undo step).
    pub fn add_vector(&mut self, v: VectorLayer, label: &str) -> Option<LayerId> {
        let mut id = None;
        self.edit(label, |d| {
            id = Some(d.add_vector(v));
            true
        });
        id
    }

    /// Replaces a vector layer's description. With a `merge` key, repeated
    /// changes (typing, dragging a slider) form one undo step.
    pub fn set_vector(
        &mut self,
        layer: LayerId,
        v: VectorLayer,
        label: &str,
        merge: Option<&str>,
    ) -> Result<(), &'static str> {
        if self.doc.find(layer).is_none_or(|l| l.vector().is_none()) {
            return Err(mosaix_core::tr("not a text or shape layer"));
        }
        let f = |d: &mut mosaix_core::Document<crate::pool::Tile>| {
            let l = d.find_mut(layer).unwrap();
            let changed = l.vector() != Some(&v);
            l.content = Content::Vector(v);
            changed
        };
        match merge {
            Some(key) => self.edit_merged(label, key, f),
            None => self.edit(label, f),
        };
        Ok(())
    }

    /// Converts a vector or RAW layer into ordinary pixels.
    pub fn rasterize(&mut self, layer: LayerId, label: &str) -> Result<(), &'static str> {
        if self.doc.find(layer).is_none_or(|l| !l.is_rendered()) {
            return Err(mosaix_core::tr("not a text, shape or RAW layer"));
        }
        self.refresh_smart();
        let tiles = self
            .vectors
            .get(&layer)
            .map(|(_, t)| t.clone())
            .unwrap_or_default();
        self.edit(label, |d| {
            d.find_mut(layer).unwrap().content = Content::Pixels(tiles);
            true
        });
        Ok(())
    }

    /// Shows a text or shape layer with another description without changing
    /// the document (e.g. while browsing fonts). `None` ends it.
    pub fn set_vector_preview(
        &mut self,
        p: Option<(LayerId, &VectorLayer)>,
    ) -> Result<(), &'static str> {
        match p {
            None => self.set_filter_preview(None),
            Some((id, v)) => {
                let kind = super::transform::PreviewKind::Vector(id, Box::new(v.clone()));
                if self.preview.as_ref().is_some_and(|p| p.kind == kind) {
                    return Ok(());
                }
                self.compute_preview(kind)
            }
        }
    }
}
