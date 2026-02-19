//! Curve types

use crate::{Result, CascadeError};

pub struct Line { pub start: [f64; 3], pub end: [f64; 3] }
pub struct Circle { pub center: [f64; 3], pub normal: [f64; 3], pub radius: f64 }
pub struct Ellipse { pub center: [f64; 3], pub major_axis: [f64; 3], pub minor_radius: f64 }
pub struct BSplineCurve { pub control_points: Vec<[f64; 3]>, pub knots: Vec<f64>, pub degree: usize }
pub struct BezierCurve { pub control_points: Vec<[f64; 3]> }

// TODO: Implement curve evaluation, intersection, etc.
