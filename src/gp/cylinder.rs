//! Cylinder in 3D space.
//! Port of OCCT's gp_Cylinder class.

use super::{Ax1, Ax3, Pnt, Trsf, Vec3};

/// A cylindrical surface in 3D space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cylinder {
    pos: Ax3,
    radius: f64,
}

impl Cylinder {
    /// Creates an indefinite cylinder.
    #[inline]
    pub const fn new() -> Self {
        Self {
            pos: Ax3::standard(),
            radius: f64::MAX,
        }
    }

    /// Creates a cylinder from coordinate system and radius.
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

    /// Reverses the U parametrization (reverses Y axis).
    #[inline]
    pub fn u_reverse(&mut self) {
        self.posy_reverse();
    }

    /// Reverses the V parametrization (reverses Z axis).
    #[inline]
    pub fn v_reverse(&mut self) {
        self.posz_reverse();
    }

    /// Rotate the cylinder.
    pub fn rotate(&mut self, ax: &Ax1, ang: f64) {
        self.pos.rotate(ax, ang);
    }

    /// Returns a rotated cylinder.
    pub fn rotated(&self, ax: &Ax1, ang: f64) -> Cylinder {
        let mut result = *self;
        result.rotate(ax, ang);
        result
    }

    /// Scale the cylinder.
    pub fn scale(&mut self, p: &Pnt, s: f64) {
        self.radius *= s.abs();
        self.pos.scale(p, s);
    }

    /// Returns a scaled cylinder.
    pub fn scaled(&self, p: &Pnt, s: f64) -> Cylinder {
        let mut result = *self;
        result.scale(p, s);
        result
    }

    /// Transform the cylinder.
    pub fn transform(&mut self, t: &Trsf) {
        self.radius *= t.scale_factor().abs();
        self.pos.transform(t);
    }

    /// Returns a transformed cylinder.
    pub fn transformed(&self, t: &Trsf) -> Cylinder {
        let mut result = *self;
        result.transform(t);
        result
    }

    /// Translate the cylinder.
    pub fn translate(&mut self, v: &Vec3) {
        self.pos.translate(v);
    }

    /// Returns a translated cylinder.
    pub fn translated(&self, v: &Vec3) -> Cylinder {
        let mut result = *self;
        result.translate(v);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cylinder_basic() {
        let cyl = Cylinder::from_ax3(Ax3::standard(), 5.0);
        assert_eq!(cyl.radius(), 5.0);
    }

    #[test]
    fn test_cylinder_set_radius() {
        let mut cyl = Cylinder::from_ax3(Ax3::standard(), 5.0);
        cyl.set_radius(10.0);
        assert_eq!(cyl.radius(), 10.0);
    }

    #[test]
    fn test_cylinder_direct() {
        let cyl = Cylinder::from_ax3(Ax3::standard(), 5.0);
        assert!(cyl.direct());
    }

    #[test]
    fn test_cylinder_scale() {
        let cyl = Cylinder::from_ax3(Ax3::standard(), 5.0);
        let scaled = cyl.scaled(&Pnt::new(), 2.0);
        assert_eq!(scaled.radius(), 10.0);
    }

    #[test]
    fn test_cylinder_translate() {
        let cyl = Cylinder::from_ax3(Ax3::standard(), 5.0);
        let v = Vec3::from_coords(1.0, 2.0, 3.0);
        let trans = cyl.translated(&v);
        assert!((trans.location().x() - 1.0).abs() < 1e-10);
    }
}
