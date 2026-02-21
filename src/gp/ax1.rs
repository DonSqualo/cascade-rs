//! Axis (point + direction).
//!
//! Port of OCCT's gp_Ax1 class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Ax1.hxx

use super::{Pnt, Dir, Vec3};

/// An axis in 3D space: a point (location) and a direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ax1 {
    loc: Pnt,
    vdir: Dir,
}

impl Default for Ax1 {
    fn default() -> Self {
        Self::oz()
    }
}

impl Ax1 {
    /// Creates an axis at origin pointing in Z direction.
    pub const fn oz() -> Self {
        Self {
            loc: Pnt::new(),
            vdir: Dir::z(),
        }
    }

    /// Creates an axis at origin pointing in X direction.
    pub const fn ox() -> Self {
        Self {
            loc: Pnt::new(),
            vdir: Dir::x(),
        }
    }

    /// Creates an axis at origin pointing in Y direction.
    pub const fn oy() -> Self {
        Self {
            loc: Pnt::new(),
            vdir: Dir::y(),
        }
    }

    /// Creates an axis from point and direction.
    pub const fn new(p: Pnt, v: Dir) -> Self {
        Self { loc: p, vdir: v }
    }

    /// Sets the direction.
    #[inline]
    pub fn set_direction(&mut self, v: Dir) {
        self.vdir = v;
    }

    /// Sets the location.
    #[inline]
    pub fn set_location(&mut self, p: Pnt) {
        self.loc = p;
    }

    /// Returns the direction.
    #[inline]
    pub const fn direction(&self) -> &Dir {
        &self.vdir
    }

    /// Returns the location.
    #[inline]
    pub const fn location(&self) -> &Pnt {
        &self.loc
    }

    /// Returns true if coaxial with other within tolerances.
    pub fn is_coaxial(&self, other: &Ax1, angular_tol: f64, linear_tol: f64) -> bool {
        // Directions must be parallel
        if !self.vdir.is_parallel(&other.vdir, angular_tol) {
            return false;
        }
        // Other location must be on this axis
        let v = Vec3::from_points(&self.loc, other.location());
        let cross = v.crossed(&Vec3::from_xyz(*self.vdir.xyz()));
        cross.magnitude() <= linear_tol
    }

    /// Returns true if parallel to other within angular tolerance.
    #[inline]
    pub fn is_parallel(&self, other: &Ax1, angular_tol: f64) -> bool {
        self.vdir.is_parallel(&other.vdir, angular_tol)
    }

    /// Returns true if opposite to other within angular tolerance.
    #[inline]
    pub fn is_opposite(&self, other: &Ax1, angular_tol: f64) -> bool {
        self.vdir.is_opposite(&other.vdir, angular_tol)
    }

    /// Returns true if perpendicular to other within angular tolerance.
    #[inline]
    pub fn is_normal(&self, other: &Ax1, angular_tol: f64) -> bool {
        self.vdir.is_normal(&other.vdir, angular_tol)
    }

    /// Returns angle to other axis (radians, 0 to PI).
    #[inline]
    pub fn angle(&self, other: &Ax1) -> f64 {
        self.vdir.angle(&other.vdir)
    }

    /// Reverses direction in place.
    #[inline]
    pub fn reverse(&mut self) {
        self.vdir.reverse();
    }

    /// Returns axis with reversed direction.
    #[inline]
    pub fn reversed(&self) -> Ax1 {
        Ax1 {
            loc: self.loc,
            vdir: self.vdir.reversed(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision;

    #[test]
    fn test_ax1_default() {
        let a = Ax1::default();
        assert_eq!(a.location().x(), 0.0);
        assert!((a.direction().z_val() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ax1_is_parallel() {
        let a1 = Ax1::oz();
        let a2 = Ax1::new(Pnt::from_coords(1.0, 0.0, 0.0), Dir::z());
        assert!(a1.is_parallel(&a2, precision::ANGULAR));
    }

    #[test]
    fn test_ax1_reversed() {
        let a = Ax1::oz();
        let r = a.reversed();
        assert!((r.direction().z_val() + 1.0).abs() < 1e-10);
    }
}
