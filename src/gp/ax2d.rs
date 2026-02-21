//! 2D axis (point + direction).
//!
//! Port of OCCT's gp_Ax2d class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Ax2d.hxx

use crate::gp::{Pnt2d, Dir2d};

/// A 2D axis defined by a point (origin) and a direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ax2d {
    origin: Pnt2d,
    direction: Dir2d,
}

impl Ax2d {
    /// Creates an axis at origin with direction along X.
    #[inline]
    pub fn new() -> Self {
        Self {
            origin: Pnt2d::new(),
            direction: Dir2d::new(),
        }
    }

    /// Creates an axis with given origin and direction.
    #[inline]
    pub fn from_origin_direction(origin: Pnt2d, direction: Dir2d) -> Self {
        Self { origin, direction }
    }

    /// Returns the origin (reference point).
    #[inline]
    pub fn origin(&self) -> Pnt2d {
        self.origin
    }

    /// Sets the origin.
    #[inline]
    pub fn set_origin(&mut self, origin: Pnt2d) {
        self.origin = origin;
    }

    /// Returns the direction.
    #[inline]
    pub fn direction(&self) -> Dir2d {
        self.direction
    }

    /// Sets the direction.
    #[inline]
    pub fn set_direction(&mut self, direction: Dir2d) {
        self.direction = direction;
    }

    /// Returns a perpendicular direction (rotated 90° counterclockwise).
    #[inline]
    pub fn perpendicular_direction(&self) -> Dir2d {
        // Rotate direction 90° counterclockwise: (x, y) -> (-y, x)
        Dir2d::from_coords(-self.direction.y(), self.direction.x())
    }
}

impl Default for Ax2d {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ax2d_new() {
        let ax = Ax2d::new();
        assert_eq!(ax.origin().x(), 0.0);
        assert_eq!(ax.direction().x_val(), 1.0);
    }

    #[test]
    fn test_ax2d_from_origin_direction() {
        let origin = Pnt2d::from_coords(1.0, 2.0);
        let direction = Dir2d::from_coords(3.0, 4.0);
        let ax = Ax2d::from_origin_direction(origin, direction);
        assert_eq!(ax.origin().x(), 1.0);
        assert!((ax.direction().x_val() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_ax2d_perpendicular() {
        let ax = Ax2d::new(); // Direction is (1, 0)
        let perp = ax.perpendicular_direction();
        assert!(perp.x().abs() < 1e-10);
        assert_eq!(perp.y(), 1.0);
    }
}
