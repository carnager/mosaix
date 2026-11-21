//! Camera RAW layers: the original file kept in the document plus how it is
//! developed (crate imagine-raw), so white balance and exposure stay editable.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::vector::{Affine, IDENTITY};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum WhiteBalance {
    /// What the camera recorded.
    AsShot,
    /// Grey world over the unclipped pixels.
    Auto,
    /// Neutral under D65.
    Daylight,
    /// A colour temperature in kelvin; positive tint is more magenta.
    Custom { temperature: f32, tint: f32 },
}

/// How to develop a RAW file.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Develop {
    pub white_balance: WhiteBalance,
    /// In stops.
    pub exposure: f32,
}

impl Default for Develop {
    fn default() -> Self {
        Self { white_balance: WhiteBalance::AsShot, exposure: 0.0 }
    }
}

/// The bytes of a RAW file. Compared and cached by `id` (a content hash with
/// the top bit set, so it never collides with tile serials when saving).
#[derive(Clone)]
pub struct RawData {
    pub bytes: Arc<Vec<u8>>,
    pub id: u64,
    /// The original file name (decoders may look at the extension).
    pub name: String,
}

impl RawData {
    pub fn new(bytes: Vec<u8>, name: String) -> Self {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut h);
        Self { id: h.finish() | 1 << 63, bytes: Arc::new(bytes), name }
    }
}

impl std::fmt::Debug for RawData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RawData({:?}, {} bytes, {:016x})", self.name, self.bytes.len(), self.id)
    }
}

impl PartialEq for RawData {
    fn eq(&self, o: &Self) -> bool {
        self.id == o.id
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RawLayer {
    pub data: RawData,
    pub develop: Develop,
    /// Placement of the developed image (moves and free transforms stay editable).
    pub transform: Affine,
}

impl RawLayer {
    pub fn new(data: RawData, develop: Develop) -> Self {
        Self { data, develop, transform: IDENTITY }
    }
}
