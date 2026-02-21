//! Sphere in 3D space.
//! Port of OCCT's gp_Sphere class.

use super::{Ax1, Ax3, Pnt, Trsf, Vec3};
use std::f64::consts::PI;

/// A spherical surface in 3D space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sphere {
    pos: Ax3,
    radius: f64,
}

impl Sphere {
    /// Creates an indefinite sphere.
    #[inline]
    pub const fn new() -> Self {
        Self {
            pos: Ax3::standard(),
            radius: f64::MAX,
        }
    }

    /// Creates a sphere from coordinate system and radius.
    #[inline]
    pub fn from_ax3(ax3: Ax3, radius: f64) -> Self {
        if radius < 0.0 {
            panic!("Radius must be non-negative");
        }
        Self { pos: ax3, radius }
    }

    /// Returns the radius.
    #[inline]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// Sets the radius.
    #[inline]
    pub fn set_radius(&mut self, r: f64) {
        if r < 0.0 {
            panic!("Radius must be non-negative");
        }
        self.radius = r;
    }

    /// Returns the location (center).
    #[inline]
    pub fn location(&self) -> Pnt {
        *self.pos.location()
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

    /// Computes the area of the sphere.
    #[inline]
    pub fn area(&self) -> f64 {
        4.0 * PI * self.radius * self.radius
    }

    /// Computes the volume of the sphere.
    #[inline]
    pub fn volume(&self) -> f64 {
        (4.0 * PI * self.radius * self.radius * self.radius) / 3.0
    }

    /// Returns true if the local coordinate system is right-handed.
    #[inline]
    pub const fn direct(&self) -> bool {
        self.pos.direct()
    }

    /// Returns the X axis.
    #[inline]
    pub fn x_axis(&self) -> Ax1 {
        Ax1::from_pnt_dir(*self.pos.location(), *self.pos.xdirection())
    }

    /// Returns the Y axis.
    #[inline]
    pub fn y_axis(&self) -> Ax1 {
        Ax1::from_pnt_dir(*self.pos.location(), *self.pos.ydirection())
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

    /// Rotate the sphere.
    pub fn rotate(&mut self, ax: &Ax1, ang: f64) {
        self.pos.rotate(ax, ang);
    }

    /// Returns a rotated sphere.
    pub fn rotated(&self, ax: &Ax1, ang: f64) -> Sphere {
        let mut result = *self;
        result.rotate(ax, ang);
        result
    }

    /// Scale the sphere.
    pub fn scale(&mut self, p: &Pnt, s: f64) {
        self.radius *= s.abs();
        self.pos.scale(p, s);
    }

    /// Returns a scaled sphere.
    pub fn scaled(&self, p: &Pnt, s: f64) -> Sphere {
        let mut result = *self;
        result.scale(p, s);
        result
    }

    /// Transform the sphere.
    pub fn transform(&mut self, t: &Trsf) {
        self.radius *= t.scale_factor().abs();
        self.pos.transform(t);
    }

    /// Returns a transformed sphere.
    pub fn transformed(&self, t: &Trsf) -> Sphere {
        let mut result = *self;
        result.transform(t);
        result
    }

    /// Translate the sphere.
    pub fn translate(&mut self, v: &Vec3) {
        self.pos.translate(v);
    }

    /// Returns a translated sphere.
    pub fn translated(&self, v: &Vec3) -> Sphere {
        let mut result = *self;
        result.translate(v);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_basic() {
        let sphere = Sphere::from_ax3(Ax3::standard(), 5.0);
        assert_eq!(sphere.radius(), 5.0);
    }

    #[test]
    fn test_sphere_area() {
        let sphere = Sphere::from_ax3(Ax3::standard(), 1.0);
        let area = sphere.area();
        assert!((area - 4.0 * PI).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_volume() {
        let sphere = Sphere::from_ax3(Ax3::standard(), 1.0);
        let vol = sphere.volume();
        assert!((vol - 4.0 * PI / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_set_radius() {
        let mut sphere = Sphere::from_ax3(Ax3::standard(), 5.0);
        sphere.set_radius(10.0);
        assert_eq!(sphere.radius(), 10.0);
    }

    #[test]
    fn test_sphere_direct() {
        let sphere = Sphere::from_ax3(Ax3::standard(), 5.0);
        assert!(sphere.direct());
    }

    #[test]
    fn test_sphere_scale() {
        let sphere = Sphere::from_ax3(Ax3::standard(), 5.0);
        let scaled = sphere.scaled(&Pnt::new(), 2.0);
        assert_eq!(scaled.radius(), 10.0);
    }

    #[test]
    fn test_sphere_translate() {
        let sphere = Sphere::from_ax3(Ax3::standard(), 5.0);
        let v = Vec3::from_coords(1.0, 2.0, 3.0);
        let trans = sphere.translated(&v);
        assert!((trans.location().x() - 1.0).abs() < 1e-10);
    }
}
