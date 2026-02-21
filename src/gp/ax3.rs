//! Right-handed or left-handed coordinate system.
//!
//! Port of OCCT's gp_Ax3 class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Ax3.hxx

use super::{Pnt, Dir, Ax1, Ax2};

/// A coordinate system that can be right-handed or left-handed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ax3 {
    axis: Ax1,
    vxdir: Dir,
    vydir: Dir,
    direct: bool, // true = right-handed
}

impl Default for Ax3 {
    fn default() -> Self {
        Self::standard()
    }
}

impl Ax3 {
    /// Standard right-handed coordinate system at origin.
    pub const fn standard() -> Self {
        Self {
            axis: Ax1::new(Pnt::new(), Dir::z()),
            vxdir: Dir::x(),
            vydir: Dir::y(),
            direct: true,
        }
    }

    /// Creates from Ax2 (always right-handed).
    pub fn from_ax2(a2: &Ax2) -> Self {
        Self {
            axis: *a2.axis(),
            vxdir: *a2.xdirection(),
            vydir: *a2.ydirection(),
            direct: true,
        }
    }

    /// Creates from point and N direction (right-handed).
    pub fn new(p: Pnt, n: Dir) -> Self {
        let a2 = Ax2::new(p, n);
        Self::from_ax2(&a2)
    }

    /// Creates from direction at origin (right-handed).
    pub fn from_dir(n: Dir) -> Self {
        Self::new(Pnt::new(), n)
    }

    /// Creates from point, N direction, and X direction.
    pub fn new_with_x(p: Pnt, n: Dir, vx: Dir) -> Self {
        let a2 = Ax2::new_with_x(p, n, vx);
        Self::from_ax2(&a2)
    }

    /// Returns true if right-handed.
    #[inline]
    pub const fn direct(&self) -> bool {
        self.direct
    }

    /// Returns the main axis.
    #[inline]
    pub const fn axis(&self) -> &Ax1 {
        &self.axis
    }

    /// Returns the N direction.
    #[inline]
    pub const fn direction(&self) -> &Dir {
        self.axis.direction()
    }

    /// Returns the location.
    #[inline]
    pub const fn location(&self) -> &Pnt {
        self.axis.location()
    }

    /// Returns the X direction.
    #[inline]
    pub const fn xdirection(&self) -> &Dir {
        &self.vxdir
    }

    /// Returns the Y direction.
    #[inline]
    pub const fn ydirection(&self) -> &Dir {
        &self.vydir
    }

    /// Converts to Ax2 (always right-handed).
    /// If this is left-handed, reverses X direction.
    pub fn to_ax2(&self) -> Ax2 {
        if self.direct {
            Ax2::new_with_x(*self.location(), *self.direction(), self.vxdir)
        } else {
            Ax2::new_with_x(*self.location(), *self.direction(), self.vxdir.reversed())
        }
    }

    /// Sets the axis.
    pub fn set_axis(&mut self, a1: Ax1) {
        self.axis = a1;
        self.recompute_y();
    }

    /// Sets the N direction.
    pub fn set_direction(&mut self, v: Dir) {
        self.axis.set_direction(v);
        self.recompute_y();
    }

    /// Sets the location.
    pub fn set_location(&mut self, p: Pnt) {
        self.axis.set_location(p);
    }

    /// Sets the X direction.
    pub fn set_xdirection(&mut self, vx: Dir) {
        // Project onto plane perpendicular to N
        let n = self.axis.direction();
        let dot = n.dot(&vx);
        let mut proj = *vx.xyz();
        proj.subtract(&n.xyz().multiplied(dot));
        if let Some(new_vx) = Dir::from_xyz(proj) {
            self.vxdir = new_vx;
            self.recompute_y();
        }
    }

    /// Sets the Y direction.
    pub fn set_ydirection(&mut self, vy: Dir) {
        // Compute X from N × Y
        let n = self.axis.direction();
        if let Some(vx) = vy.crossed(n) {
            self.vxdir = vx;
            self.recompute_y();
        }
    }

    /// Makes right-handed if not already.
    pub fn make_direct(&mut self) {
        if !self.direct {
            self.vxdir.reverse();
            self.direct = true;
        }
    }

    /// Reverses handedness.
    pub fn reverse(&mut self) {
        self.vydir.reverse();
        self.direct = !self.direct;
    }

    /// Reverses X direction.
    pub fn x_reverse(&mut self) {
        self.vxdir.reverse();
        self.direct = !self.direct;
    }

    /// Reverses Y direction.
    pub fn y_reverse(&mut self) {
        self.vydir.reverse();
        self.direct = !self.direct;
    }

    /// Returns reversed.
    pub fn reversed(&self) -> Ax3 {
        let mut result = *self;
        result.reverse();
        result
    }

    fn recompute_y(&mut self) {
        if self.direct {
            if let Some(vy) = Dir::from_xyz(self.axis.direction().xyz().crossed(self.vxdir.xyz())) {
                self.vydir = vy;
            }
        } else {
            if let Some(vy) = Dir::from_xyz(self.vxdir.xyz().crossed(self.axis.direction().xyz())) {
                self.vydir = vy;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ax3_standard() {
        let a = Ax3::standard();
        assert!(a.direct());
        assert!((a.direction().z_val() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ax3_to_ax2() {
        let a3 = Ax3::standard();
        let a2 = a3.to_ax2();
        assert!((a2.direction().z_val() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ax3_reversed() {
        let a = Ax3::standard();
        let r = a.reversed();
        assert!(!r.direct());
    }
}
