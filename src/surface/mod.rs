//! Surface types

use crate::{Result, CascadeError};

pub struct Plane { pub origin: [f64; 3], pub normal: [f64; 3] }
pub struct CylindricalSurface { pub origin: [f64; 3], pub axis: [f64; 3], pub radius: f64 }
pub struct SphericalSurface { pub center: [f64; 3], pub radius: f64 }
pub struct BSplineSurface { /* TODO */ }
pub struct BezierSurface { /* TODO */ }

// TODO: Implement surface evaluation, normal calculation, etc.
