//! Shape modification operations

use crate::brep::Solid;
use crate::{Result, CascadeError};

pub fn fillet(solid: &Solid, edges: &[usize], radius: f64) -> Result<Solid> {
    Err(CascadeError::NotImplemented("modify::fillet".into()))
}

pub fn chamfer(solid: &Solid, edges: &[usize], distance: f64) -> Result<Solid> {
    Err(CascadeError::NotImplemented("modify::chamfer".into()))
}

pub fn offset(solid: &Solid, distance: f64) -> Result<Solid> {
    Err(CascadeError::NotImplemented("modify::offset".into()))
}

pub fn transform(solid: &Solid, matrix: [[f64; 4]; 4]) -> Result<Solid> {
    Err(CascadeError::NotImplemented("modify::transform".into()))
}
