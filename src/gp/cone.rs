//! Cone in 3D space.
//! Port of OCCT's gp_Cone class.

use super::{Ax1, Ax3, Pnt, Trsf, Vec3};
use crate::precision;

/// A conical surface in 3D space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cone {
    pos: Ax3,
    radius: f64,
    semi_angle: f64,
}

impl Cone {
    /// Creates an indefinite cone.
    #[inline]
    pub const fn new() -> Self {
        Self {
            pos: Ax3::standard(),
            radius: f64::MAX,
            semi_angle: std::f64::consts::FRAC_PI_4,
        }
    }

    /// Creates a cone from coordinate system, semi-angle, and radius.
    #[inline]
    pub fn from_ax3(ax3: Ax3, semi_angle: f64, radius: f64) -> Self {
        if radius < 0.0 {
            panic!("Radius must be non-negative");
        }
        if semi_angle.abs() <= precision::RESOLUTION
            || semi_angle.abs() >= std::f64::consts::FRAC_PI_2 - precision::RESOLUTION
        {
            panic!("Semi-angle out of range");
        }
        Self {
            pos: ax3,
            radius,
            semi_angle,
        }
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

    /// Returns the semi-angle.
    #[inline]
    pub const fn semi_angle(&self) -> f64 {
        self.semi_angle
    }

    /// Sets the semi-angle.
    #[inline]
    pub fn set_semi_angle(&mut self, ang: f64) {
        if ang.abs() <= precision::RESOLUTION
            || ang.abs() >= std::f64::consts::FRAC_PI_2 - precision::RESOLUTION
        {
            panic!("Semi-angle out of range");
        }
        self.semi_angle = ang;
    }

    /// Returns the apex point.
    pub fn apex(&self) -> Pnt {
        let loc = self.pos.location();
        let dir = self.pos.direction();
        let dist = -self.radius / self.semi_angle.tan();
        Pnt::from_coords(
            loc.x() + dist * dir.x(),
            loc.y() + dist * dir.y(),
            loc.z() + dist * dir.z(),
        )
    }

    /// Returns the location.
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

    /// Returns the axis.
    #[inline]
    pub const fn axis(&self) -> Ax1 {
        self.pos.axis()
    }

    /// Sets the axis.
    pub fn set_axis(&mut self, ax: &Ax1) -> Result<(), String> {
        self.pos.set_axis(ax)
    }

    /// Returns true if the local coordinate system is right-handed.
    #[inline]
    pub const fn direct(&self) -> bool {
        self.pos.direct()
    }

    /// Returns the X axis.
    #[inline]
    pub fn x_axis(&self) -> Ax1 {
        Ax1::from_pnt_dir(self.pos.location(), self.posxdirection())
    }

    /// Returns the Y axis.
    #[inline]
    pub fn y_axis(&self) -> Ax1 {
        Ax1::from_pnt_dir(self.pos.location(), self.posydirection())
    }

    /// Reverses the U parametrization.
    #[inline]
    pub fn u_reverse(&mut self) {
        self.posy_reverse();
    }

    /// Reverses the V parametrization.
    #[inline]
    pub fn v_reverse(&mut self) {
        self.posz_reverse();
        self.semi_angle = -self.semi_angle;
    }

    /// Rotate the cone.
    pub fn rotate(&mut self, ax: &Ax1, ang: f64) {
        self.pos.rotate(ax, ang);
    }

    /// Returns a rotated cone.
    pub fn rotated(&self, ax: &Ax1, ang: f64) -> Cone {
        let mut result = *self;
        result.rotate(ax, ang);
        result
    }

    /// Scale the cone.
    pub fn scale(&mut self, p: &Pnt, s: f64) {
        self.radius *= s.abs();
        self.pos.scale(p, s);
    }

    /// Returns a scaled cone.
    pub fn scaled(&self, p: &Pnt, s: f64) -> Cone {
        let mut result = *self;
        result.scale(p, s);
        result
    }

    /// Transform the cone.
    pub fn transform(&mut self, t: &Trsf) {
        self.radius *= t.scale_factor().abs();
        self.pos.transform(t);
    }

    /// Returns a transformed cone.
    pub fn transformed(&self, t: &Trsf) -> Cone {
        let mut result = *self;
        result.transform(t);
        result
    }

    /// Translate the cone.
    pub fn translate(&mut self, v: &Vec3) {
        self.pos.translate(v);
    }

    /// Returns a translated cone.
    pub fn translated(&self, v: &Vec3) -> Cone {
        let mut result = *self;
        result.translate(v);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cone_basic() {
        let cone = Cone::from_ax3(Ax3::standard(), std::f64::consts::FRAC_PI_4, 5.0);
        assert_eq!(cone.radius(), 5.0);
        assert_eq!(cone.semi_angle(), std::f64::consts::FRAC_PI_4);
    }

    #[test]
    fn test_cone_apex() {
        let cone = Cone::from_ax3(Ax3::standard(), std::f64::consts::FRAC_PI_4, 5.0);
        let apex = cone.apex();
        assert!(apex.z() < 0.0); // apex should be below reference plane
    }

    #[test]
    fn test_cone_scale() {
        let cone = Cone::from_ax3(Ax3::standard(), std::f64::consts::FRAC_PI_4, 5.0);
        let scaled = cone.scaled(&Pnt::new(), 2.0);
        assert_eq!(scaled.radius(), 10.0);
    }

    #[test]
    fn test_cone_direct() {
        let cone = Cone::from_ax3(Ax3::standard(), std::f64::consts::FRAC_PI_4, 5.0);
        assert!(cone.direct());
    }

    #[test]
    fn test_cone_translate() {
        let cone = Cone::from_ax3(Ax3::standard(), std::f64::consts::FRAC_PI_4, 5.0);
        let v = Vec3::from_coords(1.0, 2.0, 3.0);
        let trans = cone.translated(&v);
        assert!((trans.location().x() - 1.0).abs() < 1e-10);
    }
}
