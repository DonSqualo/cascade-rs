//! Parabola in 3D space.
//! Port of OCCT's gp_Parab class.

use super::{Ax1, Ax2, Pnt, Trsf, Vec3};

/// A parabola in 3D space defined by focal length.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Parab {
    pos: Ax2,
    focal_length: f64,
}

impl Parab {
    /// Creates an indefinite parabola.
    #[inline]
    pub const fn new() -> Self {
        Self {
            pos: Ax2::standard(),
            focal_length: f64::MAX,
        }
    }

    /// Creates a parabola from coordinate system and focal length.
    #[inline]
    pub fn from_ax2(ax2: Ax2, focal_length: f64) -> Self {
        if focal_length < 0.0 {
            panic!("Focal length must be >= 0");
        }
        Self { pos: ax2, focal_length }
    }

    /// Returns the focal length (distance from apex to focus).
    #[inline]
    pub const fn focal(&self) -> f64 {
        self.focal_length
    }

    /// Sets the focal length.
    #[inline]
    pub fn set_focal(&mut self, f: f64) {
        if f < 0.0 {
            panic!("Focal length must be >= 0");
        }
        self.focal_length = f;
    }

    /// Returns the parameter (2 * focal length).
    #[inline]
    pub const fn parameter(&self) -> f64 {
        2.0 * self.focal_length
    }

    /// Returns the location (apex).
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

    /// Returns the focus point.
    pub fn focus(&self) -> Pnt {
        let loc = self.pos.location();
        let xdir = self.pos.xdirection();
        Pnt::from_coords(
            loc.x() + self.focal_length * xdir.x(),
            loc.y() + self.focal_length * xdir.y(),
            loc.z() + self.focal_length * xdir.z(),
        )
    }

    /// Returns the X axis (symmetry axis).
    #[inline]
    pub fn x_axis(&self) -> Ax1 {
        Ax1::from_pnt_dir(self.pos.location(), self.pos.xdirection())
    }

    /// Returns the Y axis (parallel to directrix).
    #[inline]
    pub fn y_axis(&self) -> Ax1 {
        Ax1::from_pnt_dir(self.pos.location(), self.pos.ydirection())
    }

    /// Rotate the parabola.
    pub fn rotate(&mut self, ax: &Ax1, ang: f64) {
        self.pos.rotate(ax, ang);
    }

    /// Returns a rotated parabola.
    pub fn rotated(&self, ax: &Ax1, ang: f64) -> Parab {
        let mut result = *self;
        result.rotate(ax, ang);
        result
    }

    /// Scale the parabola.
    pub fn scale(&mut self, p: &Pnt, s: f64) {
        self.focal_length *= s.abs();
        self.pos.scale(p, s);
    }

    /// Returns a scaled parabola.
    pub fn scaled(&self, p: &Pnt, s: f64) -> Parab {
        let mut result = *self;
        result.scale(p, s);
        result
    }

    /// Transform the parabola.
    pub fn transform(&mut self, t: &Trsf) {
        self.focal_length *= t.scale_factor();
        self.pos.transform(t);
    }

    /// Returns a transformed parabola.
    pub fn transformed(&self, t: &Trsf) -> Parab {
        let mut result = *self;
        result.transform(t);
        result
    }

    /// Translate the parabola.
    pub fn translate(&mut self, v: &Vec3) {
        self.pos.translate(v);
    }

    /// Returns a translated parabola.
    pub fn translated(&self, v: &Vec3) -> Parab {
        let mut result = *self;
        result.translate(v);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parab_basic() {
        let parab = Parab::from_ax2(Ax2::standard(), 5.0);
        assert_eq!(parab.focal(), 5.0);
    }

    #[test]
    fn test_parab_parameter() {
        let parab = Parab::from_ax2(Ax2::standard(), 5.0);
        assert_eq!(parab.parameter(), 10.0);
    }

    #[test]
    fn test_parab_focus() {
        let parab = Parab::from_ax2(Ax2::standard(), 5.0);
        let focus = parab.focus();
        assert!((focus.x() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_parab_scale() {
        let parab = Parab::from_ax2(Ax2::standard(), 5.0);
        let scaled = parab.scaled(&Pnt::new(), 2.0);
        assert_eq!(scaled.focal(), 10.0);
    }

    #[test]
    fn test_parab_translate() {
        let parab = Parab::from_ax2(Ax2::standard(), 5.0);
        let v = Vec3::from_coords(1.0, 2.0, 3.0);
        let trans = parab.translated(&v);
        assert!((trans.location().x() - 1.0).abs() < 1e-10);
    }
}
