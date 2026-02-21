//! Hyperbola in 3D space.
//! Port of OCCT's gp_Hypr class.

use super::{Ax1, Ax2, Pnt, Trsf, Vec3};

/// A hyperbola in 3D space with major and minor radii.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Hypr {
    pos: Ax2,
    major_radius: f64,
    minor_radius: f64,
}

impl Hypr {
    /// Creates an indefinite hyperbola.
    #[inline]
    pub const fn new() -> Self {
        Self {
            pos: Ax2::new(),
            major_radius: f64::MAX,
            minor_radius: f64::MIN,
        }
    }

    /// Creates a hyperbola from coordinate system and radii.
    #[inline]
    pub fn from_ax2(ax2: Ax2, major_radius: f64, minor_radius: f64) -> Self {
        if minor_radius < 0.0 || major_radius < 0.0 {
            panic!("Invalid hyperbola parameters");
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
        if r < 0.0 {
            panic!("Major radius must be >= 0");
        }
        self.major_radius = r;
    }

    /// Sets the minor radius.
    #[inline]
    pub fn set_minor_radius(&mut self, r: f64) {
        if r < 0.0 {
            panic!("Minor radius must be >= 0");
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

    /// Computes the eccentricity (> 1).
    pub fn eccentricity(&self) -> f64 {
        if self.major_radius == 0.0 {
            1.0
        } else {
            ((self.major_radius * self.major_radius + self.minor_radius * self.minor_radius)
                .sqrt())
                / self.major_radius
        }
    }

    /// Computes the focal distance.
    #[inline]
    pub fn focal(&self) -> f64 {
        2.0 * (self.major_radius * self.major_radius + self.minor_radius * self.minor_radius)
            .sqrt()
    }

    /// Returns the first focus.
    pub fn focus1(&self) -> Pnt {
        let c = (self.major_radius * self.major_radius + self.minor_radius * self.minor_radius)
            .sqrt();
        let loc = self.pos.location();
        let xdir = self.pos.x_direction();
        Pnt::from_coords(
            loc.x() + c * xdir.x(),
            loc.y() + c * xdir.y(),
            loc.z() + c * xdir.z(),
        )
    }

    /// Returns the second focus.
    pub fn focus2(&self) -> Pnt {
        let c = (self.major_radius * self.major_radius + self.minor_radius * self.minor_radius)
            .sqrt();
        let loc = self.pos.location();
        let xdir = self.pos.x_direction();
        Pnt::from_coords(
            loc.x() - c * xdir.x(),
            loc.y() - c * xdir.y(),
            loc.z() - c * xdir.z(),
        )
    }

    /// Returns the conjugate branch 1 (on positive Y side).
    #[inline]
    pub fn conjugate_branch1(&self) -> Hypr {
        Hypr {
            pos: Ax2::from_pnt_dir(
                self.pos.location(),
                self.pos.direction(),
                self.pos.y_direction(),
            ),
            major_radius: self.minor_radius,
            minor_radius: self.major_radius,
        }
    }

    /// Returns the conjugate branch 2 (on negative Y side).
    #[inline]
    pub fn conjugate_branch2(&self) -> Hypr {
        let mut ydir = self.pos.y_direction();
        ydir.reverse();
        Hypr {
            pos: Ax2::from_pnt_dir(self.pos.location(), self.pos.direction(), ydir),
            major_radius: self.minor_radius,
            minor_radius: self.major_radius,
        }
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

    /// Rotate the hyperbola.
    pub fn rotate(&mut self, ax: &Ax1, ang: f64) {
        self.pos.rotate(ax, ang);
    }

    /// Returns a rotated hyperbola.
    pub fn rotated(&self, ax: &Ax1, ang: f64) -> Hypr {
        let mut result = *self;
        result.rotate(ax, ang);
        result
    }

    /// Scale the hyperbola.
    pub fn scale(&mut self, p: &Pnt, s: f64) {
        self.major_radius *= s.abs();
        self.minor_radius *= s.abs();
        self.pos.scale(p, s);
    }

    /// Returns a scaled hyperbola.
    pub fn scaled(&self, p: &Pnt, s: f64) -> Hypr {
        let mut result = *self;
        result.scale(p, s);
        result
    }

    /// Transform the hyperbola.
    pub fn transform(&mut self, t: &Trsf) {
        let s = t.scale_factor();
        self.major_radius *= s;
        self.minor_radius *= s;
        self.pos.transform(t);
    }

    /// Returns a transformed hyperbola.
    pub fn transformed(&self, t: &Trsf) -> Hypr {
        let mut result = *self;
        result.transform(t);
        result
    }

    /// Translate the hyperbola.
    pub fn translate(&mut self, v: &Vec3) {
        self.pos.translate(v);
    }

    /// Returns a translated hyperbola.
    pub fn translated(&self, v: &Vec3) -> Hypr {
        let mut result = *self;
        result.translate(v);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypr_basic() {
        let hypr = Hypr::from_ax2(Ax2::new(), 5.0, 3.0);
        assert_eq!(hypr.major_radius(), 5.0);
        assert_eq!(hypr.minor_radius(), 3.0);
    }

    #[test]
    fn test_hypr_eccentricity() {
        let hypr = Hypr::from_ax2(Ax2::new(), 5.0, 3.0);
        let e = hypr.eccentricity();
        assert!(e > 1.0);
    }

    #[test]
    fn test_hypr_focal() {
        let hypr = Hypr::from_ax2(Ax2::new(), 5.0, 3.0);
        let f = hypr.focal();
        assert!(f > 0.0);
    }

    #[test]
    fn test_hypr_foci() {
        let hypr = Hypr::from_ax2(Ax2::new(), 5.0, 3.0);
        let f1 = hypr.focus1();
        let f2 = hypr.focus2();
        let dist = f1.distance(&f2);
        assert!((dist - hypr.focal()).abs() < 1e-10);
    }

    #[test]
    fn test_hypr_conjugate() {
        let hypr = Hypr::from_ax2(Ax2::new(), 5.0, 3.0);
        let conj1 = hypr.conjugate_branch1();
        assert_eq!(conj1.major_radius(), 3.0);
        assert_eq!(conj1.minor_radius(), 5.0);
    }

    #[test]
    fn test_hypr_scale() {
        let hypr = Hypr::from_ax2(Ax2::new(), 5.0, 3.0);
        let scaled = hypr.scaled(&Pnt::new(), 2.0);
        assert_eq!(scaled.major_radius(), 10.0);
        assert_eq!(scaled.minor_radius(), 6.0);
    }
}
