//! Torus in 3D space.
//! Port of OCCT's gp_Torus class.

use super::{Ax1, Ax3, Pnt, Trsf, Vec3};
use std::f64::consts::PI;

/// A toroidal surface in 3D space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Torus {
    pos: Ax3,
    major_radius: f64,
    minor_radius: f64,
}

impl Torus {
    /// Creates an indefinite torus.
    #[inline]
    pub const fn new() -> Self {
        Self {
            pos: Ax3::new(),
            major_radius: f64::MAX,
            minor_radius: f64::MAX,
        }
    }

    /// Creates a torus from coordinate system and radii.
    #[inline]
    pub fn from_ax3(ax3: Ax3, major_radius: f64, minor_radius: f64) -> Self {
        if major_radius < 0.0 || minor_radius < 0.0 {
            panic!("Radii must be non-negative");
        }
        Self {
            pos: ax3,
            major_radius,
            minor_radius,
        }
    }

    /// Returns the major radius (distance from center to tube center).
    #[inline]
    pub const fn major_radius(&self) -> f64 {
        self.major_radius
    }

    /// Returns the minor radius (tube radius).
    #[inline]
    pub const fn minor_radius(&self) -> f64 {
        self.minor_radius
    }

    /// Sets the major radius.
    #[inline]
    pub fn set_major_radius(&mut self, r: f64) {
        if r < 0.0 {
            panic!("Major radius must be non-negative");
        }
        self.major_radius = r;
    }

    /// Sets the minor radius.
    #[inline]
    pub fn set_minor_radius(&mut self, r: f64) {
        if r < 0.0 {
            panic!("Minor radius must be non-negative");
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
    pub const fn position(&self) -> Ax3 {
        self.pos
    }

    /// Sets the position.
    #[inline]
    pub fn set_position(&mut self, ax3: Ax3) {
        self.pos = ax3;
    }

    /// Computes the area of the torus.
    #[inline]
    pub fn area(&self) -> f64 {
        4.0 * PI * PI * self.major_radius * self.minor_radius
    }

    /// Computes the volume of the torus.
    #[inline]
    pub fn volume(&self) -> f64 {
        2.0 * PI * PI * self.major_radius * self.minor_radius * self.minor_radius
    }

    /// Returns true if the local coordinate system is right-handed.
    #[inline]
    pub const fn direct(&self) -> bool {
        self.pos.direct()
    }

    /// Returns the X axis.
    #[inline]
    pub fn x_axis(&self) -> Ax1 {
        Ax1::from_pnt_dir(self.pos.location(), self.pos.x_direction())
    }

    /// Returns the Y axis.
    #[inline]
    pub fn y_axis(&self) -> Ax1 {
        Ax1::from_pnt_dir(self.pos.location(), self.pos.y_direction())
    }

    /// Reverses the U parametrization.
    #[inline]
    pub fn u_reverse(&mut self) {
        self.pos.y_reverse();
    }

    /// Reverses the V parametrization.
    #[inline]
    pub fn v_reverse(&mut self) {
        self.pos.z_reverse();
    }

    /// Rotate the torus.
    pub fn rotate(&mut self, ax: &Ax1, ang: f64) {
        self.pos.rotate(ax, ang);
    }

    /// Returns a rotated torus.
    pub fn rotated(&self, ax: &Ax1, ang: f64) -> Torus {
        let mut result = *self;
        result.rotate(ax, ang);
        result
    }

    /// Scale the torus.
    pub fn scale(&mut self, p: &Pnt, s: f64) {
        self.major_radius *= s.abs();
        self.minor_radius *= s.abs();
        self.pos.scale(p, s);
    }

    /// Returns a scaled torus.
    pub fn scaled(&self, p: &Pnt, s: f64) -> Torus {
        let mut result = *self;
        result.scale(p, s);
        result
    }

    /// Transform the torus.
    pub fn transform(&mut self, t: &Trsf) {
        self.major_radius *= t.scale_factor().abs();
        self.minor_radius *= t.scale_factor().abs();
        self.pos.transform(t);
    }

    /// Returns a transformed torus.
    pub fn transformed(&self, t: &Trsf) -> Torus {
        let mut result = *self;
        result.transform(t);
        result
    }

    /// Translate the torus.
    pub fn translate(&mut self, v: &Vec3) {
        self.pos.translate(v);
    }

    /// Returns a translated torus.
    pub fn translated(&self, v: &Vec3) -> Torus {
        let mut result = *self;
        result.translate(v);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_torus_basic() {
        let torus = Torus::from_ax3(Ax3::new(), 5.0, 2.0);
        assert_eq!(torus.major_radius(), 5.0);
        assert_eq!(torus.minor_radius(), 2.0);
    }

    #[test]
    fn test_torus_area() {
        let torus = Torus::from_ax3(Ax3::new(), 1.0, 1.0);
        let area = torus.area();
        assert!((area - 4.0 * PI * PI).abs() < 1e-10);
    }

    #[test]
    fn test_torus_volume() {
        let torus = Torus::from_ax3(Ax3::new(), 1.0, 1.0);
        let vol = torus.volume();
        assert!((vol - 2.0 * PI * PI).abs() < 1e-10);
    }

    #[test]
    fn test_torus_direct() {
        let torus = Torus::from_ax3(Ax3::new(), 5.0, 2.0);
        assert!(torus.direct());
    }

    #[test]
    fn test_torus_scale() {
        let torus = Torus::from_ax3(Ax3::new(), 5.0, 2.0);
        let scaled = torus.scaled(&Pnt::new(), 2.0);
        assert_eq!(scaled.major_radius(), 10.0);
        assert_eq!(scaled.minor_radius(), 4.0);
    }

    #[test]
    fn test_torus_translate() {
        let torus = Torus::from_ax3(Ax3::new(), 5.0, 2.0);
        let v = Vec3::from_coords(1.0, 2.0, 3.0);
        let trans = torus.translated(&v);
        assert!((trans.location().x() - 1.0).abs() < 1e-10);
    }
}
