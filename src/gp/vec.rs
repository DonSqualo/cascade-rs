//! 3D vector.
//!
//! Port of OCCT's gp_Vec class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Vec.hxx

use super::XYZ;
use crate::precision;

/// A 3D vector in cartesian space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    coord: XYZ,
}

impl Vec3 {
    /// Creates a null vector.
    #[inline]
    pub const fn new() -> Self {
        Self { coord: XYZ::new() }
    }

    /// Creates a vector from XYZ coordinates.
    #[inline]
    pub const fn from_xyz(xyz: XYZ) -> Self {
        Self { coord: xyz }
    }

    /// Creates a vector from coordinates.
    #[inline]
    pub const fn from_coords(x: f64, y: f64, z: f64) -> Self {
        Self {
            coord: XYZ::from_coords(x, y, z),
        }
    }

    /// Creates a vector from two points (P1 -> P2).
    #[inline]
    pub fn from_points(p1: &super::Pnt, p2: &super::Pnt) -> Self {
        Self {
            coord: p2.xyz().subtracted(p1.xyz()),
        }
    }

    /// Sets coordinates.
    #[inline]
    pub fn set_coord(&mut self, x: f64, y: f64, z: f64) {
        self.coord.set_coord(x, y, z);
    }

    /// Sets X coordinate.
    #[inline]
    pub fn set_x(&mut self, x: f64) {
        self.coord.set_x(x);
    }

    /// Sets Y coordinate.
    #[inline]
    pub fn set_y(&mut self, y: f64) {
        self.coord.set_y(y);
    }

    /// Sets Z coordinate.
    #[inline]
    pub fn set_z(&mut self, z: f64) {
        self.coord.set_z(z);
    }

    /// Sets from XYZ.
    #[inline]
    pub fn set_xyz(&mut self, xyz: XYZ) {
        self.coord = xyz;
    }

    /// Returns X coordinate.
    #[inline]
    pub const fn x(&self) -> f64 {
        self.coord.x()
    }

    /// Returns Y coordinate.
    #[inline]
    pub const fn y(&self) -> f64 {
        self.coord.y()
    }

    /// Returns Z coordinate.
    #[inline]
    pub const fn z(&self) -> f64 {
        self.coord.z()
    }

    /// Returns XYZ coordinates.
    #[inline]
    pub const fn xyz(&self) -> &XYZ {
        &self.coord
    }

    /// Returns the magnitude (length).
    #[inline]
    pub fn magnitude(&self) -> f64 {
        self.coord.modulus()
    }

    /// Returns square of magnitude.
    #[inline]
    pub const fn square_magnitude(&self) -> f64 {
        self.coord.square_modulus()
    }

    /// Adds another vector in place.
    #[inline]
    pub fn add(&mut self, other: &Vec3) {
        self.coord.add(&other.coord);
    }

    /// Returns sum of vectors.
    #[inline]
    pub fn added(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            coord: self.coord.added(&other.coord),
        }
    }

    /// Subtracts another vector in place.
    #[inline]
    pub fn subtract(&mut self, other: &Vec3) {
        self.coord.subtract(&other.coord);
    }

    /// Returns difference of vectors.
    #[inline]
    pub fn subtracted(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            coord: self.coord.subtracted(&other.coord),
        }
    }

    /// Multiplies by scalar in place.
    #[inline]
    pub fn multiply(&mut self, scalar: f64) {
        self.coord.multiply(scalar);
    }

    /// Returns scaled vector.
    #[inline]
    pub fn multiplied(&self, scalar: f64) -> Vec3 {
        Vec3 {
            coord: self.coord.multiplied(scalar),
        }
    }

    /// Divides by scalar in place.
    #[inline]
    pub fn divide(&mut self, scalar: f64) {
        self.coord.divide(scalar);
    }

    /// Returns divided vector.
    #[inline]
    pub fn divided(&self, scalar: f64) -> Vec3 {
        Vec3 {
            coord: self.coord.divided(scalar),
        }
    }

    /// Reverses direction in place.
    #[inline]
    pub fn reverse(&mut self) {
        self.coord.reverse();
    }

    /// Returns reversed vector.
    #[inline]
    pub fn reversed(&self) -> Vec3 {
        Vec3 {
            coord: self.coord.reversed(),
        }
    }

    /// Computes dot product.
    #[inline]
    pub const fn dot(&self, other: &Vec3) -> f64 {
        self.coord.dot(&other.coord)
    }

    /// Computes cross product in place.
    #[inline]
    pub fn cross(&mut self, other: &Vec3) {
        self.coord.cross(&other.coord);
    }

    /// Returns cross product.
    #[inline]
    pub fn crossed(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            coord: self.coord.crossed(&other.coord),
        }
    }

    /// Normalizes in place.
    pub fn normalize(&mut self) -> bool {
        self.coord.normalize()
    }

    /// Returns normalized vector.
    pub fn normalized(&self) -> Option<Vec3> {
        self.coord.normalized().map(|c| Vec3 { coord: c })
    }

    /// Returns true if parallel to other vector within angular tolerance.
    pub fn is_parallel(&self, other: &Vec3, angular_tolerance: f64) -> bool {
        let cross = self.crossed(other);
        let sin_angle = cross.magnitude() / (self.magnitude() * other.magnitude());
        sin_angle.abs() <= angular_tolerance
    }

    /// Returns true if perpendicular to other vector within angular tolerance.
    pub fn is_normal(&self, other: &Vec3, angular_tolerance: f64) -> bool {
        let dot = self.dot(other);
        let cos_angle = dot / (self.magnitude() * other.magnitude());
        cos_angle.abs() <= angular_tolerance
    }

    /// Returns angle to other vector (radians, 0 to PI).
    pub fn angle(&self, other: &Vec3) -> f64 {
        let dot = self.dot(other);
        let mags = self.magnitude() * other.magnitude();
        if mags < precision::CONFUSION {
            return 0.0;
        }
        let cos_angle = (dot / mags).clamp(-1.0, 1.0);
        cos_angle.acos()
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        self.added(&other)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        self.subtracted(&other)
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, scalar: f64) -> Vec3 {
        self.multiplied(scalar)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        self.reversed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_magnitude() {
        let v = Vec3::from_coords(3.0, 4.0, 0.0);
        assert!((v.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec_dot() {
        let v1 = Vec3::from_coords(1.0, 0.0, 0.0);
        let v2 = Vec3::from_coords(0.0, 1.0, 0.0);
        assert_eq!(v1.dot(&v2), 0.0);
    }

    #[test]
    fn test_vec_cross() {
        let i = Vec3::from_coords(1.0, 0.0, 0.0);
        let j = Vec3::from_coords(0.0, 1.0, 0.0);
        let k = i.crossed(&j);
        assert!((k.z() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec_angle() {
        let v1 = Vec3::from_coords(1.0, 0.0, 0.0);
        let v2 = Vec3::from_coords(0.0, 1.0, 0.0);
        let angle = v1.angle(&v2);
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }
}
