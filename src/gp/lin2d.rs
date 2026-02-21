//! 2D line.
//!
//! Port of OCCT's gp_Lin2d class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Lin2d.hxx

use crate::gp::{Pnt2d, Dir2d, Ax2d};

/// An infinite 2D line defined by a point and a direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lin2d {
    position: Ax2d,
}

impl Lin2d {
    /// Creates a line through origin along X axis.
    #[inline]
    pub fn new() -> Self {
        Self { position: Ax2d::new() }
    }

    /// Creates a line with given origin and direction.
    #[inline]
    pub fn from_origin_direction(origin: Pnt2d, direction: Dir2d) -> Self {
        Self { position: Ax2d::from_origin_direction(origin, direction) }
    }

    /// Creates a line through two points.
    pub fn from_two_points(p1: Pnt2d, p2: Pnt2d) -> Self {
        let dir = Dir2d::from_coords(p2.x() - p1.x(), p2.y() - p1.y());
        Self { position: Ax2d::from_origin_direction(p1, dir) }
    }

    /// Creates a line from an axis.
    #[inline]
    pub fn from_axis(axis: Ax2d) -> Self {
        Self { position: axis }
    }

    /// Returns the position axis.
    #[inline]
    pub fn position(&self) -> Ax2d {
        self.position
    }

    /// Returns the origin (a point on the line).
    #[inline]
    pub fn origin(&self) -> Pnt2d {
        self.position.origin()
    }

    /// Returns the direction.
    #[inline]
    pub fn direction(&self) -> Dir2d {
        self.position.direction()
    }

    /// Computes the distance from a point to the line.
    pub fn distance(&self, point: Pnt2d) -> f64 {
        let v = crate::gp::Vec2d::from_coords(
            point.x() - self.origin().x(),
            point.y() - self.origin().y(),
        );
        // Convert direction to Vec2d for cross product
        let dir_vec = crate::gp::Vec2d::from_coords(
            self.direction().x(),
            self.direction().y(),
        );
        let perp = dir_vec.crossed(&v);
        perp.abs()
    }

    /// Computes the parameter for the closest point on the line to a given point.
    /// The closest point is: origin + parameter * direction
    pub fn parameter(&self, point: Pnt2d) -> f64 {
        let v = crate::gp::Vec2d::from_coords(
            point.x() - self.origin().x(),
            point.y() - self.origin().y(),
        );
        // Convert direction to Vec2d for dot product
        let dir_vec = crate::gp::Vec2d::from_coords(
            self.direction().x(),
            self.direction().y(),
        );
        v.dot(&dir_vec)
    }

    /// Computes the closest point on the line to a given point.
    pub fn closest_point(&self, point: Pnt2d) -> Pnt2d {
        let t = self.parameter(point);
        let dir = self.direction();
        self.origin() + (dir * t)
    }

    /// Checks if a point lies on the line within tolerance.
    pub fn contains(&self, point: Pnt2d, tolerance: f64) -> bool {
        self.distance(point) <= tolerance
    }
}

impl Default for Lin2d {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lin2d_new() {
        let line = Lin2d::new();
        assert_eq!(line.origin().x(), 0.0);
        assert_eq!(line.direction().x(), 1.0);
    }

    #[test]
    fn test_lin2d_from_two_points() {
        let p1 = Pnt2d::from_coords(0.0, 0.0);
        let p2 = Pnt2d::from_coords(3.0, 4.0);
        let line = Lin2d::from_two_points(p1, p2);
        assert_eq!(line.origin().x(), 0.0);
        assert!((line.direction().x() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_lin2d_distance() {
        let line = Lin2d::new(); // Along X axis
        let p = Pnt2d::from_coords(0.0, 5.0);
        assert_eq!(line.distance(p), 5.0);
    }

    #[test]
    fn test_lin2d_contains() {
        let line = Lin2d::new();
        let p = Pnt2d::from_coords(10.0, 0.0);
        assert!(line.contains(p, 1e-10));

        let p2 = Pnt2d::from_coords(0.0, 0.1);
        assert!(!line.contains(p2, 1e-10));
    }

    #[test]
    fn test_lin2d_closest_point() {
        let line = Lin2d::new();
        let p = Pnt2d::from_coords(5.0, 3.0);
        let closest = line.closest_point(p);
        assert_eq!(closest.x(), 5.0);
        assert_eq!(closest.y(), 0.0);
    }
}
