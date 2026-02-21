//! 2D circle.
//!
//! Port of OCCT's gp_Circ2d class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Circ2d.hxx

use crate::gp::{Pnt2d, Dir2d, Ax2d, Ax22d};

/// A 2D circle defined by center, radius, and axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Circ2d {
    position: Ax22d,
    radius: f64,
}

impl Circ2d {
    /// Creates a circle centered at origin with radius 1.
    #[inline]
    pub fn new(center: Pnt2d, radius: f64) -> Self {
        Self {
            position: Ax22d::from_origin_x_direction(center, Dir2d::new()),
            radius,
        }
    }

    /// Creates a circle with axis.
    #[inline]
    pub fn from_axis22d(axis: Ax22d, radius: f64) -> Self {
        Self { position: axis, radius }
    }

    /// Returns the center.
    #[inline]
    pub fn center(&self) -> Pnt2d {
        self.position.origin()
    }

    /// Returns the radius.
    #[inline]
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Sets the center.
    #[inline]
    pub fn set_center(&mut self, center: Pnt2d) {
        let mut pos = self.position;
        pos.set_origin(center);
        self.position = pos;
    }

    /// Sets the radius.
    #[inline]
    pub fn set_radius(&mut self, radius: f64) {
        self.radius = radius.abs();
    }

    /// Returns the area of the circle.
    #[inline]
    pub fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }

    /// Returns the circumference of the circle.
    #[inline]
    pub fn circumference(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius
    }

    /// Computes the distance from a point to the circle.
    /// Positive if outside, negative if inside, zero on the circle.
    pub fn distance(&self, point: Pnt2d) -> f64 {
        let dx = point.x() - self.center().x();
        let dy = point.y() - self.center().y();
        let dist = (dx * dx + dy * dy).sqrt();
        dist - self.radius
    }

    /// Checks if a point is on the circle within tolerance.
    pub fn contains(&self, point: Pnt2d, tolerance: f64) -> bool {
        self.distance(point).abs() <= tolerance
    }

    /// Returns position axis.
    #[inline]
    pub fn position(&self) -> Ax22d {
        self.position
    }

    /// Sets position axis.
    #[inline]
    pub fn set_position(&mut self, axis: Ax22d) {
        self.position = axis;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circ2d_new() {
        let center = Pnt2d::from_coords(1.0, 2.0);
        let circle = Circ2d::new(center, 3.0);
        assert_eq!(circle.radius(), 3.0);
        assert_eq!(circle.center().x(), 1.0);
    }

    #[test]
    fn test_circ2d_area() {
        let center = Pnt2d::new();
        let circle = Circ2d::new(center, 1.0);
        assert!((circle.area() - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_circ2d_circumference() {
        let center = Pnt2d::new();
        let circle = Circ2d::new(center, 1.0);
        assert!((circle.circumference() - 2.0 * std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_circ2d_contains() {
        let center = Pnt2d::from_coords(0.0, 0.0);
        let circle = Circ2d::new(center, 5.0);

        let on_circle = Pnt2d::from_coords(5.0, 0.0);
        assert!(circle.contains(on_circle, 1e-10));

        let outside = Pnt2d::from_coords(10.0, 0.0);
        assert!(!circle.contains(outside, 1e-10));
    }

    #[test]
    fn test_circ2d_distance() {
        let center = Pnt2d::from_coords(0.0, 0.0);
        let circle = Circ2d::new(center, 5.0);

        let on_circle = Pnt2d::from_coords(5.0, 0.0);
        assert!(circle.distance(on_circle).abs() < 1e-10);

        let outside = Pnt2d::from_coords(10.0, 0.0);
        assert!((circle.distance(outside) - 5.0).abs() < 1e-10);
    }
}
