//! Unit direction vector.
//!
//! Port of OCCT's gp_Dir class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Dir.hxx

use super::XYZ;
use crate::precision;

/// A unit vector (direction) in 3D space.
/// Always normalized (magnitude = 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dir {
    coord: XYZ,
}

impl Default for Dir {
    fn default() -> Self {
        Self::z()
    }
}

impl Dir {
    /// X direction (1, 0, 0).
    pub const fn x() -> Self {
        Self { coord: XYZ::from_coords(1.0, 0.0, 0.0) }
    }

    /// Y direction (0, 1, 0).
    pub const fn y() -> Self {
        Self { coord: XYZ::from_coords(0.0, 1.0, 0.0) }
    }

    /// Z direction (0, 0, 1).
    pub const fn z() -> Self {
        Self { coord: XYZ::from_coords(0.0, 0.0, 1.0) }
    }

    /// Creates a direction from XYZ (normalizes).
    /// Returns None if vector is too small.
    pub fn from_xyz(xyz: XYZ) -> Option<Self> {
        let mut coord = xyz;
        if coord.normalize() {
            Some(Self { coord })
        } else {
            None
        }
    }

    /// Creates a direction from coordinates (normalizes).
    /// Returns None if vector is too small.
    pub fn from_coords(x: f64, y: f64, z: f64) -> Option<Self> {
        Self::from_xyz(XYZ::from_coords(x, y, z))
    }

    /// Creates a direction from coordinates (normalizes).
    /// Panics if vector is too small.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self::from_coords(x, y, z).expect("Dir::new: vector too small to normalize")
    }

    /// Sets from XYZ (normalizes). Returns false if too small.
    pub fn set_xyz(&mut self, xyz: XYZ) -> bool {
        let mut coord = xyz;
        if coord.normalize() {
            self.coord = coord;
            true
        } else {
            false
        }
    }

    /// Sets from coordinates (normalizes). Returns false if too small.
    pub fn set_coord(&mut self, x: f64, y: f64, z: f64) -> bool {
        self.set_xyz(XYZ::from_coords(x, y, z))
    }

    /// Sets X and renormalizes.
    pub fn set_x(&mut self, x: f64) -> bool {
        let xyz = XYZ::from_coords(x, self.coord.y(), self.coord.z());
        self.set_xyz(xyz)
    }

    /// Sets Y and renormalizes.
    pub fn set_y(&mut self, y: f64) -> bool {
        let xyz = XYZ::from_coords(self.coord.x(), y, self.coord.z());
        self.set_xyz(xyz)
    }

    /// Sets Z and renormalizes.
    pub fn set_z(&mut self, z: f64) -> bool {
        let xyz = XYZ::from_coords(self.coord.x(), self.coord.y(), z);
        self.set_xyz(xyz)
    }

    /// Returns X component.
    #[inline]
    pub const fn x_val(&self) -> f64 {
        self.coord.x()
    }

    /// Returns X component (alias for x_val).
    #[inline]
    pub const fn x(&self) -> f64 {
        self.coord.x()
    }

    /// Returns Y component.
    #[inline]
    pub const fn y_val(&self) -> f64 {
        self.coord.y()
    }

    /// Returns Y component (alias for y_val).
    #[inline]
    pub const fn y(&self) -> f64 {
        self.coord.y()
    }

    /// Returns Z component.
    #[inline]
    pub const fn z_val(&self) -> f64 {
        self.coord.z()
    }

    /// Returns Z component (alias for z_val).
    #[inline]
    pub const fn z(&self) -> f64 {
        self.coord.z()
    }

    /// Returns XYZ coordinates.
    #[inline]
    pub const fn xyz(&self) -> &XYZ {
        &self.coord
    }

    /// Returns coordinate by index (1=X, 2=Y, 3=Z).
    #[inline]
    pub fn coord(&self, index: usize) -> f64 {
        self.coord.coord(index)
    }

    /// Returns all coordinates as tuple.
    #[inline]
    pub const fn coords(&self) -> (f64, f64, f64) {
        self.coord.coords()
    }

    /// Returns true if parallel to other within angular tolerance.
    pub fn is_parallel(&self, other: &Dir, angular_tolerance: f64) -> bool {
        let cross = self.coord.crossed(&other.coord);
        let sin_angle = cross.modulus();
        sin_angle <= angular_tolerance
    }

    /// Returns true if opposite to other within angular tolerance.
    pub fn is_opposite(&self, other: &Dir, angular_tolerance: f64) -> bool {
        let dot = self.coord.dot(&other.coord);
        if dot >= 0.0 {
            return false;
        }
        let cross = self.coord.crossed(&other.coord);
        let sin_angle = cross.modulus();
        sin_angle <= angular_tolerance
    }

    /// Returns true if perpendicular to other within angular tolerance.
    pub fn is_normal(&self, other: &Dir, angular_tolerance: f64) -> bool {
        let dot = self.coord.dot(&other.coord);
        dot.abs() <= angular_tolerance
    }

    /// Returns angle to other direction (radians, 0 to PI).
    pub fn angle(&self, other: &Dir) -> f64 {
        let dot = self.coord.dot(&other.coord).clamp(-1.0, 1.0);
        dot.acos()
    }

    /// Returns angle to other direction with reference direction for sign.
    /// Returns angle in [-PI, PI].
    pub fn angle_with_ref(&self, other: &Dir, ref_dir: &Dir) -> f64 {
        let angle = self.angle(other);
        let cross = self.coord.crossed(&other.coord);
        if cross.dot(&ref_dir.coord) < 0.0 {
            -angle
        } else {
            angle
        }
    }

    /// Computes cross product (returns Dir, normalizes result).
    pub fn crossed(&self, other: &Dir) -> Option<Dir> {
        let cross = self.coord.crossed(&other.coord);
        Dir::from_xyz(cross)
    }

    /// Computes dot product.
    #[inline]
    pub fn dot(&self, other: &Dir) -> f64 {
        self.coord.dot(&other.coord)
    }

    /// Reverses direction in place.
    #[inline]
    pub fn reverse(&mut self) {
        self.coord.reverse();
    }

    /// Returns reversed direction.
    #[inline]
    pub fn reversed(&self) -> Dir {
        Dir { coord: self.coord.reversed() }
    }
}

impl std::ops::Neg for Dir {
    type Output = Dir;
    fn neg(self) -> Dir {
        self.reversed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dir_x() {
        let d = Dir::x();
        assert_eq!(d.x_val(), 1.0);
        assert_eq!(d.y_val(), 0.0);
        assert_eq!(d.z_val(), 0.0);
    }

    #[test]
    fn test_dir_from_coords() {
        let d = Dir::from_coords(3.0, 4.0, 0.0).unwrap();
        assert!((d.x_val() - 0.6).abs() < 1e-10);
        assert!((d.y_val() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_dir_zero_vector() {
        assert!(Dir::from_coords(0.0, 0.0, 0.0).is_none());
    }

    #[test]
    fn test_dir_angle() {
        let x = Dir::x();
        let y = Dir::y();
        let angle = x.angle(&y);
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_dir_is_parallel() {
        let d1 = Dir::x();
        let d2 = Dir::from_coords(2.0, 0.0, 0.0).unwrap();
        assert!(d1.is_parallel(&d2, precision::ANGULAR));
    }

    #[test]
    fn test_dir_is_perpendicular() {
        let x = Dir::x();
        let y = Dir::y();
        assert!(x.is_normal(&y, precision::ANGULAR));
    }

    #[test]
    fn test_dir_crossed() {
        let x = Dir::x();
        let y = Dir::y();
        let z = x.crossed(&y).unwrap();
        assert!((z.z_val() - 1.0).abs() < 1e-10);
    }
}
