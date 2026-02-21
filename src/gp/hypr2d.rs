//! 2D hyperbola.
//!
//! Port of OCCT's gp_Hypr2d class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Hypr2d.hxx

use crate::gp::{Pnt2d, Dir2d, Ax22d};

/// A 2D hyperbola defined by center, semi-major and semi-minor axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hypr2d {
    position: Ax22d,
    major_radius: f64,
    minor_radius: f64,
}

impl Hypr2d {
    /// Creates a hyperbola with given center, semi-major and semi-minor axes.
    #[inline]
    pub fn new(center: Pnt2d, major_radius: f64, minor_radius: f64) -> Self {
        Self {
            position: Ax22d::from_origin_x_direction(center, Dir2d::new()),
            major_radius: major_radius.abs(),
            minor_radius: minor_radius.abs(),
        }
    }

    /// Creates a hyperbola with axis.
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

    /// Returns the semi-major axis (distance from center to vertex).
    #[inline]
    pub fn major_radius(&self) -> f64 {
        self.major_radius
    }

    /// Returns the semi-minor axis.
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

    /// Sets the semi-major axis.
    #[inline]
    pub fn set_major_radius(&mut self, radius: f64) {
        self.major_radius = radius.abs();
    }

    /// Sets the semi-minor axis.
    #[inline]
    pub fn set_minor_radius(&mut self, radius: f64) {
        self.minor_radius = radius.abs();
    }

    /// Returns the eccentricity (always > 1 for hyperbola).
    pub fn eccentricity(&self) -> f64 {
        if self.major_radius == 0.0 {
            return f64::INFINITY;
        }
        let c_sq = self.major_radius * self.major_radius + self.minor_radius * self.minor_radius;
        c_sq.sqrt() / self.major_radius
    }

    /// Returns the distance from center to focus.
    #[inline]
    pub fn focal_distance(&self) -> f64 {
        (self.major_radius * self.major_radius + self.minor_radius * self.minor_radius).sqrt()
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
    fn test_hypr2d_new() {
        let center = Pnt2d::from_coords(1.0, 2.0);
        let hyperbola = Hypr2d::new(center, 3.0, 4.0);
        assert_eq!(hyperbola.major_radius(), 3.0);
        assert_eq!(hyperbola.minor_radius(), 4.0);
    }

    #[test]
    fn test_hypr2d_eccentricity() {
        let center = Pnt2d::new();
        let hyperbola = Hypr2d::new(center, 1.0, 1.0);
        assert!((hyperbola.eccentricity() - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_hypr2d_focal_distance() {
        let center = Pnt2d::new();
        let hyperbola = Hypr2d::new(center, 3.0, 4.0);
        assert!((hyperbola.focal_distance() - 5.0).abs() < 1e-10); // 3-4-5 triangle
    }
}
