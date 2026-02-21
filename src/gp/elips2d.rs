//! 2D ellipse.
//!
//! Port of OCCT's gp_Elips2d class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Elips2d.hxx

use crate::gp::{Pnt2d, Dir2d, Ax22d};

/// A 2D ellipse defined by center, major/minor axes, and orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Elips2d {
    position: Ax22d,
    major_radius: f64,
    minor_radius: f64,
}

impl Elips2d {
    /// Creates an ellipse with given center, major and minor radii.
    #[inline]
    pub fn new(center: Pnt2d, major_radius: f64, minor_radius: f64) -> Self {
        Self {
            position: Ax22d::from_origin_x_direction(center, Dir2d::new()),
            major_radius: major_radius.abs(),
            minor_radius: minor_radius.abs(),
        }
    }

    /// Creates an ellipse with axis.
    #[inline]
    pub fn from_axis22d(axis: Ax22d, major_radius: f64, minor_radius: f64) -> Self {
        Self {
            position: axis,
            major_radius: major_radius.abs(),
            minor_radius: minor_radius.abs(),
        }
    }

    /// Returns the center.
    #[inline]
    pub fn center(&self) -> Pnt2d {
        self.position.origin()
    }

    /// Returns the major radius.
    #[inline]
    pub fn major_radius(&self) -> f64 {
        self.major_radius
    }

    /// Returns the minor radius.
    #[inline]
    pub fn minor_radius(&self) -> f64 {
        self.minor_radius
    }

    /// Sets the center.
    #[inline]
    pub fn set_center(&mut self, center: Pnt2d) {
        let mut pos = self.position;
        pos.set_origin(center);
        self.position = pos;
    }

    /// Sets the major radius.
    #[inline]
    pub fn set_major_radius(&mut self, radius: f64) {
        self.major_radius = radius.abs();
    }

    /// Sets the minor radius.
    #[inline]
    pub fn set_minor_radius(&mut self, radius: f64) {
        self.minor_radius = radius.abs();
    }

    /// Returns the eccentricity.
    pub fn eccentricity(&self) -> f64 {
        if self.major_radius == 0.0 {
            return 0.0;
        }
        let c_sq = self.major_radius * self.major_radius - self.minor_radius * self.minor_radius;
        if c_sq < 0.0 {
            0.0
        } else {
            c_sq.sqrt() / self.major_radius
        }
    }

    /// Returns the area of the ellipse.
    #[inline]
    pub fn area(&self) -> f64 {
        std::f64::consts::PI * self.major_radius * self.minor_radius
    }

    /// Returns an approximation of the perimeter.
    /// Uses Ramanujan's approximation.
    pub fn perimeter(&self) -> f64 {
        let h = ((self.major_radius - self.minor_radius).powi(2))
            / ((self.major_radius + self.minor_radius).powi(2));
        std::f64::consts::PI * (self.major_radius + self.minor_radius)
            * (1.0 + 3.0 * h / (10.0 + (4.0 - 3.0 * h).sqrt()))
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
    fn test_elips2d_new() {
        let center = Pnt2d::from_coords(1.0, 2.0);
        let ellipse = Elips2d::new(center, 5.0, 3.0);
        assert_eq!(ellipse.major_radius(), 5.0);
        assert_eq!(ellipse.minor_radius(), 3.0);
    }

    #[test]
    fn test_elips2d_area() {
        let center = Pnt2d::new();
        let ellipse = Elips2d::new(center, 2.0, 1.0);
        assert!((ellipse.area() - 2.0 * std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_elips2d_eccentricity_circle() {
        let center = Pnt2d::new();
        let ellipse = Elips2d::new(center, 1.0, 1.0);
        assert!(ellipse.eccentricity().abs() < 1e-10); // Circle has eccentricity 0
    }

    #[test]
    fn test_elips2d_eccentricity_line() {
        let center = Pnt2d::new();
        let ellipse = Elips2d::new(center, 1.0, 0.0);
        assert!((ellipse.eccentricity() - 1.0).abs() < 1e-10); // Degenerate ellipse
    }
}
