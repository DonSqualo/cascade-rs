//! 2D direction vector (unit vector).
//!
//! Port of OCCT's gp_Dir2d class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Dir2d.hxx

use crate::gp::XY;
use crate::precision;
use std::ops::{Mul, Neg};

/// A 2D unit direction vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dir2d {
    xy: XY,
}

impl Dir2d {
    /// Creates a direction along positive X axis.
    #[inline]
    pub const fn new() -> Self {
        Self { xy: XY::from_coords(1.0, 0.0) }
    }

    /// Creates a direction from two coordinates (auto-normalized).
    /// Panics if the vector is too small.
    pub fn from_coords(x: f64, y: f64) -> Self {
        let xy = XY::from_coords(x, y);
        let d = xy.modulus();
        if d <= precision::CONFUSION {
            panic!("Dir2d::from_coords: zero-length direction");
        }
        Self { xy: xy.divided(d) }
    }

    /// Creates a direction from an XY (auto-normalized).
    /// Panics if the XY is too small.
    pub fn from_xy(xy: XY) -> Self {
        let d = xy.modulus();
        if d <= precision::CONFUSION {
            panic!("Dir2d::from_xy: zero-length direction");
        }
        Self { xy: xy.divided(d) }
    }

    /// Returns the X component.
    #[inline]
    pub const fn x(&self) -> f64 {
        self.xy.x()
    }

    /// Returns the Y component.
    #[inline]
    pub const fn y(&self) -> f64 {
        self.xy.y()
    }

    /// Returns coordinates as tuple.
    #[inline]
    pub const fn coords(&self) -> (f64, f64) {
        self.xy.coords()
    }

    /// Returns the underlying XY.
    #[inline]
    pub const fn xy(&self) -> XY {
        self.xy
    }

    /// Computes the dot product with another direction.
    #[inline]
    pub const fn dot(&self, other: &Dir2d) -> f64 {
        self.xy.dot(&other.xy)
    }

    /// Computes the 2D cross product with another direction.
    #[inline]
    pub const fn crossed(&self, other: &Dir2d) -> f64 {
        self.xy.crossed(&other.xy)
    }

    /// Computes the angle from this direction to another (in radians).
    pub fn angle_to(&self, other: &Dir2d) -> f64 {
        // Using atan2 for proper quadrant handling
        let cross = self.crossed(other);
        let dot = self.dot(other);
        cross.atan2(dot)
    }

    /// Checks if this direction is equal to another within tolerance.
    #[inline]
    pub fn is_equal(&self, other: &Dir2d, tolerance: f64) -> bool {
        self.xy.is_equal(&other.xy, tolerance)
    }

    /// Checks if this direction is opposite to another.
    #[inline]
    pub fn is_opposite(&self, other: &Dir2d, tolerance: f64) -> bool {
        let neg_other = other.xy.reversed();
        self.xy.is_equal(&neg_other, tolerance)
    }

    /// Checks if this direction is perpendicular to another.
    #[inline]
    pub fn is_perpendicular(&self, other: &Dir2d, tolerance: f64) -> bool {
        self.dot(other).abs() < tolerance
    }

    /// Reverses direction.
    #[inline]
    pub fn reversed(&self) -> Dir2d {
        Dir2d { xy: self.xy.reversed() }
    }
}

impl Neg for Dir2d {
    type Output = Dir2d;
    #[inline]
    fn neg(self) -> Dir2d {
        self.reversed()
    }
}

impl Mul<f64> for Dir2d {
    type Output = crate::gp::Vec2d;
    #[inline]
    fn mul(self, scalar: f64) -> crate::gp::Vec2d {
        crate::gp::Vec2d::from_xy(self.xy.multiplied(scalar))
    }
}

impl Mul<Dir2d> for f64 {
    type Output = crate::gp::Vec2d;
    #[inline]
    fn mul(self, dir: Dir2d) -> crate::gp::Vec2d {
        crate::gp::Vec2d::from_xy(dir.xy.multiplied(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dir2d_new() {
        let d = Dir2d::new();
        assert_eq!(d.x(), 1.0);
        assert_eq!(d.y(), 0.0);
    }

    #[test]
    fn test_dir2d_from_coords() {
        let d = Dir2d::from_coords(3.0, 4.0);
        assert!((d.x() - 0.6).abs() < 1e-10);
        assert!((d.y() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_dir2d_dot() {
        let a = Dir2d::new(); // (1, 0)
        let b = Dir2d::from_coords(0.0, 1.0); // (0, 1)
        assert!(a.dot(&b).abs() < 1e-10);
    }

    #[test]
    fn test_dir2d_crossed() {
        let a = Dir2d::new(); // (1, 0)
        let b = Dir2d::from_coords(0.0, 1.0); // (0, 1)
        assert!((a.crossed(&b) - 1.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic]
    fn test_dir2d_zero_vector() {
        Dir2d::from_coords(0.0, 0.0);
    }
}
