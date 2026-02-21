//! Coordinate system (axis + reference directions).
//!
//! Port of OCCT's gp_Ax2 class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Ax2.hxx

use super::{Pnt, Dir, Ax1};

/// A right-handed coordinate system in 3D space.
/// Defined by a point (origin), a main direction (N), and an X direction.
/// Y direction is computed as N × X.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ax2 {
    axis: Ax1,  // N direction
    vxdir: Dir, // X direction
    vydir: Dir, // Y direction (computed)
}

impl Default for Ax2 {
    fn default() -> Self {
        Self::standard()
    }
}

impl Ax2 {
    /// Standard coordinate system at origin.
    pub const fn standard() -> Self {
        Self {
            axis: Ax1::new(Pnt::new(), Dir::z()),
            vxdir: Dir::x(),
            vydir: Dir::y(),
        }
    }

    /// Creates from point and N direction.
    /// X and Y are computed automatically.
    pub fn new(p: Pnt, n: Dir) -> Self {
        // Choose X direction perpendicular to N
        let vxdir = if n.z_val().abs() > 0.9 {
            // N is close to Z, use X axis
            Dir::from_xyz(n.xyz().crossed(&super::XYZ::from_coords(1.0, 0.0, 0.0)))
                .unwrap_or(Dir::x())
        } else {
            // Use Z axis
            Dir::from_xyz(n.xyz().crossed(&super::XYZ::from_coords(0.0, 0.0, 1.0)))
                .unwrap_or(Dir::x())
        };
        let vydir = Dir::from_xyz(n.xyz().crossed(vxdir.xyz())).unwrap_or(Dir::y());
        
        Self {
            axis: Ax1::new(p, n),
            vxdir,
            vydir,
        }
    }

    /// Creates from point, N direction, and X direction.
    pub fn new_with_x(p: Pnt, n: Dir, vx: Dir) -> Self {
        // Project vx onto plane perpendicular to n
        let dot = n.dot(&vx);
        let mut proj = *vx.xyz();
        proj.subtract(&n.xyz().multiplied(dot));
        let vxdir = Dir::from_xyz(proj).unwrap_or(Dir::x());
        let vydir = Dir::from_xyz(n.xyz().crossed(vxdir.xyz())).unwrap_or(Dir::y());
        
        Self {
            axis: Ax1::new(p, n),
            vxdir,
            vydir,
        }
    }

    /// Sets the axis (main direction).
    pub fn set_axis(&mut self, a1: Ax1) {
        self.axis = a1;
        // Recompute Y
        if let Some(vy) = Dir::from_xyz(self.axis.direction().xyz().crossed(self.vxdir.xyz())) {
            self.vydir = vy;
        }
    }

    /// Sets the N direction.
    pub fn set_direction(&mut self, v: Dir) {
        self.axis.set_direction(v);
        // Recompute Y
        if let Some(vy) = Dir::from_xyz(v.xyz().crossed(self.vxdir.xyz())) {
            self.vydir = vy;
        }
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
            if let Some(vy) = Dir::from_xyz(n.xyz().crossed(self.vxdir.xyz())) {
                self.vydir = vy;
            }
        }
    }

    /// Sets the Y direction.
    pub fn set_ydirection(&mut self, vy: Dir) {
        // Compute X from N × Y (left-handed intermediate)
        let n = self.axis.direction();
        if let Some(vx) = Dir::from_xyz(vy.xyz().crossed(n.xyz())) {
            self.vxdir = vx;
            // Recompute Y to ensure right-handedness
            if let Some(new_vy) = Dir::from_xyz(n.xyz().crossed(self.vxdir.xyz())) {
                self.vydir = new_vy;
            }
        }
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

    /// Returns angle to other Ax2.
    #[inline]
    pub fn angle(&self, other: &Ax2) -> f64 {
        self.axis.direction().angle(other.axis.direction())
    }

    /// Returns true if coplanar with other.
    pub fn is_coplanar(&self, other: &Ax2, linear_tol: f64, angular_tol: f64) -> bool {
        // N directions must be parallel
        if !self.axis.direction().is_parallel(other.axis.direction(), angular_tol) {
            return false;
        }
        // Other location must be in this plane
        let v = super::Vec3::from_points(self.location(), other.location());
        let dist = v.xyz().dot(self.axis.direction().xyz()).abs();
        dist <= linear_tol
    }

    /// Returns true if coplanar with axis.
    pub fn is_coplanar_with_ax1(&self, a1: &Ax1, linear_tol: f64, angular_tol: f64) -> bool {
        // Axis direction must be perpendicular to this N
        if !self.axis.direction().is_normal(a1.direction(), angular_tol) {
            return false;
        }
        // Axis location must be in this plane
        let v = super::Vec3::from_points(self.location(), a1.location());
        let dist = v.xyz().dot(self.axis.direction().xyz()).abs();
        dist <= linear_tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision;

    #[test]
    fn test_ax2_standard() {
        let a = Ax2::standard();
        assert!((a.direction().z_val() - 1.0).abs() < 1e-10);
        assert!((a.xdirection().x_val() - 1.0).abs() < 1e-10);
        assert!((a.ydirection().y_val() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ax2_right_handed() {
        let a = Ax2::standard();
        // X × Y should equal Z
        let cross = a.xdirection().crossed(a.ydirection()).unwrap();
        assert!((cross.z_val() - 1.0).abs() < 1e-10);
    }
}
