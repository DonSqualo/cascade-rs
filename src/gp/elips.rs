//! Ellipse in 3D space.
//! Port of OCCT's gp_Elips class.

use super::{Ax1, Ax2, Pnt, Trsf, Vec3};
use std::f64::consts::PI;

/// An ellipse in 3D space with major and minor radii.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Elips {
    pos: Ax2,
    major_radius: f64,
    minor_radius: f64,
}

impl Elips {
    /// Creates an indefinite ellipse.
    #[inline]
    pub const fn new() -> Self {
        Self {
            pos: Ax2::standard(),
            major_radius: f64::MAX,
            minor_radius: f64::MIN,
        }
    }

    /// Creates an ellipse from coordinate system and radii.
    #[inline]
    pub fn from_ax2(ax2: Ax2, major_radius: f64, minor_radius: f64) -> Self {
        if minor_radius < 0.0 || major_radius < minor_radius {
            panic!("Invalid ellipse parameters");
        }
        Self {
            pos: ax2,
            major_radius,
            minor_radius,
        }
    }

    /// Returns the major radius.
    #[inline]
    pub const fn major_radius(&self) -> f64 {
        self.major_radius
    }

    /// Returns the minor radius.
    #[inline]
    pub const fn minor_radius(&self) -> f64 {
        self.minor_radius
    }

    /// Sets the major radius.
    #[inline]
    pub fn set_major_radius(&mut self, r: f64) {
        if r < self.minor_radius {
            panic!("Major radius must be >= minor radius");
        }
        self.major_radius = r;
    }

    /// Sets the minor radius.
    #[inline]
    pub fn set_minor_radius(&mut self, r: f64) {
        if r < 0.0 || self.major_radius < r {
            panic!("Minor radius must be in valid range");
        }
        self.minor_radius = r;
    }

    /// Returns the location (center).
    #[inline]
    pub const fn location(&self) -> Pnt {
        self.pos.location()
    }

    /// Sets the location.
    #[inline]
    pub fn set_location(&mut self, p: Pnt) {
        self.pos.set_location(p);
    }

    /// Returns the position (axis system).
    #[inline]
    pub const fn position(&self) -> Ax2 {
        self.pos
    }

    /// Computes the area of the ellipse.
    #[inline]
    pub fn area(&self) -> f64 {
        PI * self.major_radius * self.minor_radius
    }

    /// Computes the eccentricity (0 to 1).
    pub fn eccentricity(&self) -> f64 {
        if self.major_radius == 0.0 {
            0.0
        } else {
            ((self.major_radius * self.major_radius - self.minor_radius * self.minor_radius)
                .sqrt())
                / self.major_radius
        }
    }

    /// Computes the focal distance (distance between two foci).
    #[inline]
    pub fn focal(&self) -> f64 {
        2.0 * (self.major_radius * self.major_radius - self.minor_radius * self.minor_radius)
            .sqrt()
    }

    /// Returns the first focus.
    pub fn focus1(&self) -> Pnt {
        let c = (self.major_radius * self.major_radius - self.minor_radius * self.minor_radius)
            .sqrt();
        let loc = self.pos.location();
        let xdir = self.pos.xdirection();
        Pnt::from_coords(
            loc.x() + c * xdir.x(),
            loc.y() + c * xdir.y(),
            loc.z() + c * xdir.z(),
        )
    }

    /// Returns the second focus.
    pub fn focus2(&self) -> Pnt {
        let c = (self.major_radius * self.major_radius - self.minor_radius * self.minor_radius)
            .sqrt();
        let loc = self.pos.location();
        let xdir = self.pos.xdirection();
        Pnt::from_coords(
            loc.x() - c * xdir.x(),
            loc.y() - c * xdir.y(),
            loc.z() - c * xdir.z(),
        )
    }

    /// Returns the parameter (semi-latus rectum).
    pub fn parameter(&self) -> f64 {
        if self.major_radius == 0.0 {
            0.0
        } else {
            (1.0 - self.eccentricity() * self.eccentricity()) * self.major_radius
        }
    }

    /// Returns the X axis (major axis).
    #[inline]
    pub fn x_axis(&self) -> Ax1 {
        Ax1::from_pnt_dir(self.pos.location(), self.pos.xdirection())
    }

    /// Returns the Y axis (minor axis).
    #[inline]
    pub fn y_axis(&self) -> Ax1 {
        Ax1::from_pnt_dir(self.pos.location(), self.pos.ydirection())
    }

    /// Rotate the ellipse.
    pub fn rotate(&mut self, ax: &Ax1, ang: f64) {
        self.pos.rotate(ax, ang);
    }

    /// Returns a rotated ellipse.
    pub fn rotated(&self, ax: &Ax1, ang: f64) -> Elips {
        let mut result = *self;
        result.rotate(ax, ang);
        result
    }

    /// Scale the ellipse.
    pub fn scale(&mut self, p: &Pnt, s: f64) {
        self.major_radius *= s.abs();
        self.minor_radius *= s.abs();
        self.pos.scale(p, s);
    }

    /// Returns a scaled ellipse.
    pub fn scaled(&self, p: &Pnt, s: f64) -> Elips {
        let mut result = *self;
        result.scale(p, s);
        result
    }

    /// Transform the ellipse.
    pub fn transform(&mut self, t: &Trsf) {
        let s = t.scale_factor();
        self.major_radius *= s;
        self.minor_radius *= s;
        self.pos.transform(t);
    }

    /// Returns a transformed ellipse.
    pub fn transformed(&self, t: &Trsf) -> Elips {
        let mut result = *self;
        result.transform(t);
        result
    }

    /// Translate the ellipse.
    pub fn translate(&mut self, v: &Vec3) {
        self.pos.translate(v);
    }

    /// Returns a translated ellipse.
    pub fn translated(&self, v: &Vec3) -> Elips {
        let mut result = *self;
        result.translate(v);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elips_basic() {
        let elips = Elips::from_ax2(Ax2::standard(), 5.0, 3.0);
        assert_eq!(elips.major_radius(), 5.0);
        assert_eq!(elips.minor_radius(), 3.0);
    }

    #[test]
    fn test_elips_area() {
        let elips = Elips::from_ax2(Ax2::standard(), 5.0, 3.0);
        let area = elips.area();
        assert!((area - PI * 5.0 * 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_elips_eccentricity() {
        let elips = Elips::from_ax2(Ax2::standard(), 5.0, 3.0);
        let e = elips.eccentricity();
        assert!(e > 0.0 && e < 1.0);
    }

    #[test]
    fn test_elips_focal() {
        let elips = Elips::from_ax2(Ax2::standard(), 5.0, 3.0);
        let f = elips.focal();
        assert!(f > 0.0);
    }

    #[test]
    fn test_elips_foci() {
        let elips = Elips::from_ax2(Ax2::standard(), 5.0, 3.0);
        let f1 = elips.focus1();
        let f2 = elips.focus2();
        assert_eq!(f1.x() > 0.0, f2.x() < 0.0);
    }

    #[test]
    fn test_elips_scale() {
        let elips = Elips::from_ax2(Ax2::standard(), 5.0, 3.0);
        let scaled = elips.scaled(&Pnt::new(), 2.0);
        assert_eq!(scaled.major_radius(), 10.0);
        assert_eq!(scaled.minor_radius(), 6.0);
    }
}
