//! Boolean operations on solids

use crate::brep::{Solid, Shape};
use crate::{Result, CascadeError};

/// Fuse (union) two solids
pub fn fuse(solid1: &Solid, solid2: &Solid) -> Result<Solid> {
    // TODO: Implement boolean union
    Err(CascadeError::NotImplemented("boolean::fuse".into()))
}

/// Cut (difference) solid2 from solid1
pub fn cut(solid1: &Solid, solid2: &Solid) -> Result<Solid> {
    // TODO: Implement boolean difference
    Err(CascadeError::NotImplemented("boolean::cut".into()))
}

/// Common (intersection) of two solids
pub fn common(solid1: &Solid, solid2: &Solid) -> Result<Solid> {
    // TODO: Implement boolean intersection
    Err(CascadeError::NotImplemented("boolean::common".into()))
}

/// Section - intersection of solid with plane, returns wire/face
pub fn section(solid: &Solid, plane_origin: [f64; 3], plane_normal: [f64; 3]) -> Result<Shape> {
    // TODO: Implement section
    Err(CascadeError::NotImplemented("boolean::section".into()))
}
