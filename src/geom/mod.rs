//! Geometry primitives (OpenCASCADE gp_* equivalents)
//!
//! This module provides basic geometric types:
//! - `Pnt` (gp_Pnt) - 3D point
//! - `Vec3` (gp_Vec) - 3D vector  
//! - `Dir` (gp_Dir) - 3D unit vector (direction)
//! - `Pnt2d` (gp_Pnt2d) - 2D point
//! - `Vec2d` (gp_Vec2d) - 2D vector

use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Neg, Sub};

/// Tolerance for geometric comparisons
pub const TOLERANCE: f64 = 1e-10;

/// A 3D point (equivalent to OpenCASCADE gp_Pnt)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Pnt {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Pnt {
    /// Create a new point
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Origin point (0, 0, 0)
    pub fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Create from array
    pub fn from_array(arr: [f64; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }

    /// Convert to array
    pub fn to_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Distance to another point
    pub fn distance(&self, other: &Pnt) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Squared distance to another point (faster, no sqrt)
    pub fn distance_squared(&self, other: &Pnt) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Check if approximately equal to another point
    pub fn is_equal(&self, other: &Pnt, tolerance: f64) -> bool {
        self.distance(other) <= tolerance
    }

    /// Vector from this point to another
    pub fn vec_to(&self, other: &Pnt) -> Vec3 {
        Vec3::new(other.x - self.x, other.y - self.y, other.z - self.z)
    }

    /// Translate point by vector
    pub fn translated(&self, v: &Vec3) -> Pnt {
        Pnt::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }

    /// Midpoint between this and another point
    pub fn midpoint(&self, other: &Pnt) -> Pnt {
        Pnt::new(
            (self.x + other.x) / 2.0,
            (self.y + other.y) / 2.0,
            (self.z + other.z) / 2.0,
        )
    }
}

impl Add<Vec3> for Pnt {
    type Output = Pnt;

    fn add(self, v: Vec3) -> Pnt {
        Pnt::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }
}

impl Sub<Pnt> for Pnt {
    type Output = Vec3;

    fn sub(self, other: Pnt) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

/// A 3D vector (equivalent to OpenCASCADE gp_Vec)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    /// Create a new vector
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Zero vector
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Unit X vector
    pub fn unit_x() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }

    /// Unit Y vector
    pub fn unit_y() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }

    /// Unit Z vector
    pub fn unit_z() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }

    /// Create from array
    pub fn from_array(arr: [f64; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }

    /// Convert to array
    pub fn to_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Magnitude (length) of the vector
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Squared magnitude (faster, no sqrt)
    pub fn magnitude_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Normalize to unit length (returns None if zero vector)
    pub fn normalized(&self) -> Option<Dir> {
        Dir::try_new(self.x, self.y, self.z)
    }

    /// Normalize in place, returns false if zero vector
    pub fn normalize(&mut self) -> bool {
        let mag = self.magnitude();
        if mag > TOLERANCE {
            self.x /= mag;
            self.y /= mag;
            self.z /= mag;
            true
        } else {
            false
        }
    }

    /// Dot product with another vector
    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product with another vector
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Reverse (negate) the vector
    pub fn reverse(&self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }

    /// Scale the vector
    pub fn scaled(&self, factor: f64) -> Vec3 {
        Vec3::new(self.x * factor, self.y * factor, self.z * factor)
    }

    /// Angle with another vector (in radians)
    pub fn angle(&self, other: &Vec3) -> f64 {
        let dot = self.dot(other);
        let mag_product = self.magnitude() * other.magnitude();
        if mag_product < TOLERANCE {
            0.0
        } else {
            (dot / mag_product).clamp(-1.0, 1.0).acos()
        }
    }

    /// Check if approximately parallel to another vector
    pub fn is_parallel(&self, other: &Vec3, tolerance: f64) -> bool {
        let cross_mag = self.cross(other).magnitude();
        let mag_product = self.magnitude() * other.magnitude();
        if mag_product < TOLERANCE {
            true // zero vectors are parallel to everything
        } else {
            cross_mag / mag_product <= tolerance
        }
    }

    /// Check if approximately perpendicular to another vector
    pub fn is_perpendicular(&self, other: &Vec3, tolerance: f64) -> bool {
        let dot = self.dot(other).abs();
        let mag_product = self.magnitude() * other.magnitude();
        if mag_product < TOLERANCE {
            true // zero vectors are perpendicular to everything
        } else {
            dot / mag_product <= tolerance
        }
    }
}

impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, scalar: f64) -> Vec3 {
        Vec3::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, v: Vec3) -> Vec3 {
        Vec3::new(v.x * self, v.y * self, v.z * self)
    }
}

/// A 3D unit vector / direction (equivalent to OpenCASCADE gp_Dir)
///
/// Unlike `Vec3`, a `Dir` is always normalized (magnitude = 1).
/// This is enforced at construction time.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Dir {
    x: f64,
    y: f64,
    z: f64,
}

impl Dir {
    /// Create a new direction by normalizing the input components.
    /// 
    /// # Panics
    /// Panics if the input vector has zero magnitude.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self::try_new(x, y, z).expect("Cannot create Dir from zero vector")
    }

    /// Try to create a new direction. Returns None if the input vector has zero magnitude.
    pub fn try_new(x: f64, y: f64, z: f64) -> Option<Self> {
        let mag = (x * x + y * y + z * z).sqrt();
        if mag > TOLERANCE {
            Some(Self {
                x: x / mag,
                y: y / mag,
                z: z / mag,
            })
        } else {
            None
        }
    }

    /// Create from array (normalizes automatically)
    pub fn from_array(arr: [f64; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }

    /// Try to create from array
    pub fn try_from_array(arr: [f64; 3]) -> Option<Self> {
        Self::try_new(arr[0], arr[1], arr[2])
    }

    /// Convert to array
    pub fn to_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Get X component
    pub fn x(&self) -> f64 {
        self.x
    }

    /// Get Y component
    pub fn y(&self) -> f64 {
        self.y
    }

    /// Get Z component
    pub fn z(&self) -> f64 {
        self.z
    }

    /// Unit X direction (+X axis)
    pub fn x_axis() -> Self {
        Self { x: 1.0, y: 0.0, z: 0.0 }
    }

    /// Unit Y direction (+Y axis)
    pub fn y_axis() -> Self {
        Self { x: 0.0, y: 1.0, z: 0.0 }
    }

    /// Unit Z direction (+Z axis)
    pub fn z_axis() -> Self {
        Self { x: 0.0, y: 0.0, z: 1.0 }
    }

    /// Reverse (negate) the direction
    pub fn reverse(&self) -> Dir {
        Dir {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Alias for reverse (matches OCCT's Reversed())
    pub fn reversed(&self) -> Dir {
        self.reverse()
    }

    /// Angle with another direction (in radians, always positive, 0 to π)
    pub fn angle(&self, other: &Dir) -> f64 {
        let dot = self.dot(other);
        // Clamp to handle floating point errors
        dot.clamp(-1.0, 1.0).acos()
    }

    /// Angle with a vector (in radians)
    pub fn angle_with_vec(&self, v: &Vec3) -> f64 {
        let v_mag = v.magnitude();
        if v_mag < TOLERANCE {
            0.0
        } else {
            let dot = self.x * v.x + self.y * v.y + self.z * v.z;
            (dot / v_mag).clamp(-1.0, 1.0).acos()
        }
    }

    /// Cross product with another direction (returns Vec3, may not be unit length)
    pub fn cross(&self, other: &Dir) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Cross product returning a direction (normalized result)
    /// Returns None if the directions are parallel
    pub fn cross_dir(&self, other: &Dir) -> Option<Dir> {
        let cross = self.cross(other);
        cross.normalized()
    }

    /// Dot product with another direction
    pub fn dot(&self, other: &Dir) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Dot product with a vector
    pub fn dot_vec(&self, v: &Vec3) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    /// Check if approximately parallel to another direction
    pub fn is_parallel(&self, other: &Dir, angular_tolerance: f64) -> bool {
        let cross_mag = self.cross(other).magnitude();
        cross_mag <= angular_tolerance.sin()
    }

    /// Check if approximately opposite to another direction
    pub fn is_opposite(&self, other: &Dir, angular_tolerance: f64) -> bool {
        self.is_parallel(other, angular_tolerance) && self.dot(other) < 0.0
    }

    /// Check if approximately equal to another direction
    pub fn is_equal(&self, other: &Dir, angular_tolerance: f64) -> bool {
        self.is_parallel(other, angular_tolerance) && self.dot(other) > 0.0
    }

    /// Check if approximately perpendicular to another direction
    pub fn is_perpendicular(&self, other: &Dir, angular_tolerance: f64) -> bool {
        self.dot(other).abs() <= angular_tolerance.sin()
    }

    /// Convert to a Vec3 (unit vector)
    pub fn to_vec3(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Scale to a vector of given magnitude
    pub fn scaled(&self, magnitude: f64) -> Vec3 {
        Vec3::new(self.x * magnitude, self.y * magnitude, self.z * magnitude)
    }

    /// Find a direction perpendicular to this one
    /// The choice of perpendicular is arbitrary but deterministic
    pub fn perpendicular(&self) -> Dir {
        // Choose the smallest component to cross with
        let abs_x = self.x.abs();
        let abs_y = self.y.abs();
        let abs_z = self.z.abs();

        let helper = if abs_x <= abs_y && abs_x <= abs_z {
            Vec3::new(1.0, 0.0, 0.0)
        } else if abs_y <= abs_x && abs_y <= abs_z {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };

        let cross = Vec3::new(
            self.y * helper.z - self.z * helper.y,
            self.z * helper.x - self.x * helper.z,
            self.x * helper.y - self.y * helper.x,
        );

        cross.normalized().expect("perpendicular should exist for unit vector")
    }
}

impl Neg for Dir {
    type Output = Dir;

    fn neg(self) -> Dir {
        self.reverse()
    }
}

impl From<Dir> for Vec3 {
    fn from(d: Dir) -> Vec3 {
        d.to_vec3()
    }
}

impl From<Dir> for [f64; 3] {
    fn from(d: Dir) -> [f64; 3] {
        d.to_array()
    }
}

/// A 2D point (equivalent to OpenCASCADE gp_Pnt2d)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Pnt2d {
    pub x: f64,
    pub y: f64,
}

impl Pnt2d {
    /// Create a new 2D point
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Origin point (0, 0)
    pub fn origin() -> Self {
        Self::new(0.0, 0.0)
    }

    /// Create from array
    pub fn from_array(arr: [f64; 2]) -> Self {
        Self::new(arr[0], arr[1])
    }

    /// Convert to array
    pub fn to_array(&self) -> [f64; 2] {
        [self.x, self.y]
    }

    /// Convert to 3D point (with z=0)
    pub fn to_3d(&self) -> Pnt {
        Pnt::new(self.x, self.y, 0.0)
    }

    /// Create from 3D point (drops z coordinate)
    pub fn from_3d(p: &Pnt) -> Self {
        Self::new(p.x, p.y)
    }

    /// Distance to another point
    pub fn distance(&self, other: &Pnt2d) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Squared distance to another point (faster, no sqrt)
    pub fn distance_squared(&self, other: &Pnt2d) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Check if approximately equal to another point
    pub fn is_equal(&self, other: &Pnt2d, tolerance: f64) -> bool {
        self.distance(other) <= tolerance
    }

    /// Vector from this point to another
    pub fn vec_to(&self, other: &Pnt2d) -> Vec2d {
        Vec2d::new(other.x - self.x, other.y - self.y)
    }

    /// Translate point by vector
    pub fn translated(&self, v: &Vec2d) -> Pnt2d {
        Pnt2d::new(self.x + v.x, self.y + v.y)
    }

    /// Midpoint between this and another point
    pub fn midpoint(&self, other: &Pnt2d) -> Pnt2d {
        Pnt2d::new(
            (self.x + other.x) / 2.0,
            (self.y + other.y) / 2.0,
        )
    }
}

impl Add<Vec2d> for Pnt2d {
    type Output = Pnt2d;

    fn add(self, v: Vec2d) -> Pnt2d {
        Pnt2d::new(self.x + v.x, self.y + v.y)
    }
}

impl Sub<Pnt2d> for Pnt2d {
    type Output = Vec2d;

    fn sub(self, other: Pnt2d) -> Vec2d {
        Vec2d::new(self.x - other.x, self.y - other.y)
    }
}

/// A 2D vector (equivalent to OpenCASCADE gp_Vec2d)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec2d {
    pub x: f64,
    pub y: f64,
}

impl Vec2d {
    /// Create a new vector
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Zero vector
    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    /// Unit X vector
    pub fn unit_x() -> Self {
        Self::new(1.0, 0.0)
    }

    /// Unit Y vector
    pub fn unit_y() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Create from array
    pub fn from_array(arr: [f64; 2]) -> Self {
        Self::new(arr[0], arr[1])
    }

    /// Convert to array
    pub fn to_array(&self) -> [f64; 2] {
        [self.x, self.y]
    }

    /// Convert to 3D vector (with z=0)
    pub fn to_3d(&self) -> Vec3 {
        Vec3::new(self.x, self.y, 0.0)
    }

    /// Create from 3D vector (drops z coordinate)
    pub fn from_3d(v: &Vec3) -> Self {
        Self::new(v.x, v.y)
    }

    /// Magnitude (length) of the vector
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Squared magnitude (faster, no sqrt)
    pub fn magnitude_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    /// Normalize to unit length (returns new normalized vector)
    /// Returns None if the vector is zero
    pub fn normalized(&self) -> Option<Vec2d> {
        let mag = self.magnitude();
        if mag > TOLERANCE {
            Some(Vec2d::new(self.x / mag, self.y / mag))
        } else {
            None
        }
    }

    /// Normalize in place, returns false if zero vector
    pub fn normalize(&mut self) -> bool {
        let mag = self.magnitude();
        if mag > TOLERANCE {
            self.x /= mag;
            self.y /= mag;
            true
        } else {
            false
        }
    }

    /// Dot product with another vector
    pub fn dot(&self, other: &Vec2d) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Cross product with another vector (returns scalar in 2D)
    /// The result is the z-component of the 3D cross product
    pub fn cross(&self, other: &Vec2d) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// Reverse (negate) the vector
    pub fn reverse(&self) -> Vec2d {
        Vec2d::new(-self.x, -self.y)
    }

    /// Scale the vector
    pub fn scaled(&self, factor: f64) -> Vec2d {
        Vec2d::new(self.x * factor, self.y * factor)
    }

    /// Angle with another vector (in radians)
    pub fn angle(&self, other: &Vec2d) -> f64 {
        let dot = self.dot(other);
        let mag_product = self.magnitude() * other.magnitude();
        if mag_product < TOLERANCE {
            0.0
        } else {
            (dot / mag_product).clamp(-1.0, 1.0).acos()
        }
    }

    /// Check if approximately parallel to another vector
    pub fn is_parallel(&self, other: &Vec2d, tolerance: f64) -> bool {
        let cross_mag = self.cross(other).abs();
        let mag_product = self.magnitude() * other.magnitude();
        if mag_product < TOLERANCE {
            true // zero vectors are parallel to everything
        } else {
            cross_mag / mag_product <= tolerance
        }
    }

    /// Check if approximately perpendicular to another vector
    pub fn is_perpendicular(&self, other: &Vec2d, tolerance: f64) -> bool {
        let dot = self.dot(other).abs();
        let mag_product = self.magnitude() * other.magnitude();
        if mag_product < TOLERANCE {
            true // zero vectors are perpendicular to everything
        } else {
            dot / mag_product <= tolerance
        }
    }
}

impl Add for Vec2d {
    type Output = Vec2d;

    fn add(self, other: Vec2d) -> Vec2d {
        Vec2d::new(self.x + other.x, self.y + other.y)
    }
}

impl Sub for Vec2d {
    type Output = Vec2d;

    fn sub(self, other: Vec2d) -> Vec2d {
        Vec2d::new(self.x - other.x, self.y - other.y)
    }
}

impl Neg for Vec2d {
    type Output = Vec2d;

    fn neg(self) -> Vec2d {
        Vec2d::new(-self.x, -self.y)
    }
}

impl Mul<f64> for Vec2d {
    type Output = Vec2d;

    fn mul(self, scalar: f64) -> Vec2d {
        Vec2d::new(self.x * scalar, self.y * scalar)
    }
}

impl Mul<Vec2d> for f64 {
    type Output = Vec2d;

    fn mul(self, v: Vec2d) -> Vec2d {
        Vec2d::new(v.x * self, v.y * self)
    }
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TEST_TOL: f64 = 1e-9;

    // ===== Pnt Tests =====

    #[test]
    fn test_pnt_creation() {
        let p = Pnt::new(1.0, 2.0, 3.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
    }

    #[test]
    fn test_pnt_origin() {
        let p = Pnt::origin();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
        assert_eq!(p.z, 0.0);
    }

    #[test]
    fn test_pnt_distance() {
        let p1 = Pnt::new(0.0, 0.0, 0.0);
        let p2 = Pnt::new(3.0, 4.0, 0.0);
        assert!((p1.distance(&p2) - 5.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_pnt_vec_to() {
        let p1 = Pnt::new(1.0, 2.0, 3.0);
        let p2 = Pnt::new(4.0, 6.0, 8.0);
        let v = p1.vec_to(&p2);
        assert!((v.x - 3.0).abs() < TEST_TOL);
        assert!((v.y - 4.0).abs() < TEST_TOL);
        assert!((v.z - 5.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_pnt_midpoint() {
        let p1 = Pnt::new(0.0, 0.0, 0.0);
        let p2 = Pnt::new(2.0, 4.0, 6.0);
        let mid = p1.midpoint(&p2);
        assert!((mid.x - 1.0).abs() < TEST_TOL);
        assert!((mid.y - 2.0).abs() < TEST_TOL);
        assert!((mid.z - 3.0).abs() < TEST_TOL);
    }

    // ===== Vec3 Tests =====

    #[test]
    fn test_vec3_magnitude() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.magnitude() - 5.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_vec3_normalized() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let d = v.normalized().unwrap();
        assert!((d.x() - 0.6).abs() < TEST_TOL);
        assert!((d.y() - 0.8).abs() < TEST_TOL);
        assert!((d.z() - 0.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_vec3_zero_normalized() {
        let v = Vec3::zero();
        assert!(v.normalized().is_none());
    }

    #[test]
    fn test_vec3_dot() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        assert!((v1.dot(&v2) - 32.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_vec3_cross() {
        let v1 = Vec3::unit_x();
        let v2 = Vec3::unit_y();
        let cross = v1.cross(&v2);
        assert!((cross.x - 0.0).abs() < TEST_TOL);
        assert!((cross.y - 0.0).abs() < TEST_TOL);
        assert!((cross.z - 1.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_vec3_angle() {
        let v1 = Vec3::unit_x();
        let v2 = Vec3::unit_y();
        assert!((v1.angle(&v2) - PI / 2.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_vec3_is_parallel() {
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(2.0, 0.0, 0.0);
        let v3 = Vec3::new(0.0, 1.0, 0.0);
        assert!(v1.is_parallel(&v2, TEST_TOL));
        assert!(!v1.is_parallel(&v3, TEST_TOL));
    }

    // ===== Dir Tests =====

    #[test]
    fn test_dir_creation_normalizes() {
        let d = Dir::new(3.0, 4.0, 0.0);
        let mag = (d.x() * d.x() + d.y() * d.y() + d.z() * d.z()).sqrt();
        assert!((mag - 1.0).abs() < TEST_TOL);
        assert!((d.x() - 0.6).abs() < TEST_TOL);
        assert!((d.y() - 0.8).abs() < TEST_TOL);
    }

    #[test]
    #[should_panic(expected = "Cannot create Dir from zero vector")]
    fn test_dir_zero_panics() {
        let _d = Dir::new(0.0, 0.0, 0.0);
    }

    #[test]
    fn test_dir_try_new_zero() {
        assert!(Dir::try_new(0.0, 0.0, 0.0).is_none());
    }

    #[test]
    fn test_dir_reverse() {
        let d = Dir::new(1.0, 0.0, 0.0);
        let r = d.reverse();
        assert!((r.x() - (-1.0)).abs() < TEST_TOL);
        assert!((r.y() - 0.0).abs() < TEST_TOL);
        assert!((r.z() - 0.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_dir_angle() {
        let d1 = Dir::x_axis();
        let d2 = Dir::y_axis();
        assert!((d1.angle(&d2) - PI / 2.0).abs() < TEST_TOL);

        let d3 = Dir::new(1.0, 1.0, 0.0);
        assert!((d1.angle(&d3) - PI / 4.0).abs() < TEST_TOL);

        // Parallel (same direction)
        let d4 = Dir::x_axis();
        assert!(d1.angle(&d4).abs() < TEST_TOL);

        // Antiparallel (opposite direction)
        let d5 = d1.reverse();
        assert!((d1.angle(&d5) - PI).abs() < TEST_TOL);
    }

    #[test]
    fn test_dir_cross() {
        let d1 = Dir::x_axis();
        let d2 = Dir::y_axis();
        let cross = d1.cross(&d2);
        assert!((cross.x - 0.0).abs() < TEST_TOL);
        assert!((cross.y - 0.0).abs() < TEST_TOL);
        assert!((cross.z - 1.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_dir_cross_dir() {
        let d1 = Dir::x_axis();
        let d2 = Dir::y_axis();
        let cross = d1.cross_dir(&d2).unwrap();
        assert!((cross.x() - 0.0).abs() < TEST_TOL);
        assert!((cross.y() - 0.0).abs() < TEST_TOL);
        assert!((cross.z() - 1.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_dir_cross_dir_parallel_returns_none() {
        let d1 = Dir::x_axis();
        let d2 = Dir::x_axis();
        assert!(d1.cross_dir(&d2).is_none());
    }

    #[test]
    fn test_dir_dot() {
        let d1 = Dir::x_axis();
        let d2 = Dir::y_axis();
        assert!(d1.dot(&d2).abs() < TEST_TOL);

        let d3 = Dir::x_axis();
        assert!((d1.dot(&d3) - 1.0).abs() < TEST_TOL);

        let d4 = d1.reverse();
        assert!((d1.dot(&d4) - (-1.0)).abs() < TEST_TOL);
    }

    #[test]
    fn test_dir_is_parallel() {
        let d1 = Dir::x_axis();
        let d2 = Dir::x_axis();
        let d3 = Dir::new(-1.0, 0.0, 0.0);
        let d4 = Dir::y_axis();

        assert!(d1.is_parallel(&d2, 1e-6));
        assert!(d1.is_parallel(&d3, 1e-6)); // opposite but parallel
        assert!(!d1.is_parallel(&d4, 1e-6));
    }

    #[test]
    fn test_dir_is_opposite() {
        let d1 = Dir::x_axis();
        let d2 = Dir::new(-1.0, 0.0, 0.0);
        let d3 = Dir::x_axis();

        assert!(d1.is_opposite(&d2, 1e-6));
        assert!(!d1.is_opposite(&d3, 1e-6));
    }

    #[test]
    fn test_dir_is_perpendicular() {
        let d1 = Dir::x_axis();
        let d2 = Dir::y_axis();
        let d3 = Dir::z_axis();
        let d4 = Dir::x_axis();

        assert!(d1.is_perpendicular(&d2, 1e-6));
        assert!(d1.is_perpendicular(&d3, 1e-6));
        assert!(!d1.is_perpendicular(&d4, 1e-6));
    }

    #[test]
    fn test_dir_to_vec3() {
        let d = Dir::new(1.0, 0.0, 0.0);
        let v = d.to_vec3();
        assert!((v.x - 1.0).abs() < TEST_TOL);
        assert!((v.y - 0.0).abs() < TEST_TOL);
        assert!((v.z - 0.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_dir_scaled() {
        let d = Dir::new(1.0, 0.0, 0.0);
        let v = d.scaled(5.0);
        assert!((v.x - 5.0).abs() < TEST_TOL);
        assert!((v.y - 0.0).abs() < TEST_TOL);
        assert!((v.z - 0.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_dir_perpendicular() {
        let d = Dir::z_axis();
        let perp = d.perpendicular();
        
        // Perpendicular should be unit length
        let mag = (perp.x() * perp.x() + perp.y() * perp.y() + perp.z() * perp.z()).sqrt();
        assert!((mag - 1.0).abs() < TEST_TOL);
        
        // Perpendicular should be perpendicular
        assert!(perp.is_perpendicular(&d, 1e-6));
    }

    #[test]
    fn test_dir_neg() {
        let d = Dir::x_axis();
        let neg_d = -d;
        assert!((neg_d.x() - (-1.0)).abs() < TEST_TOL);
    }

    #[test]
    fn test_dir_from_into() {
        let d = Dir::x_axis();
        let v: Vec3 = d.into();
        assert!((v.x - 1.0).abs() < TEST_TOL);
        
        let arr: [f64; 3] = d.into();
        assert!((arr[0] - 1.0).abs() < TEST_TOL);
    }

    // ===== Pnt2d Tests =====

    #[test]
    fn test_pnt2d_creation() {
        let p = Pnt2d::new(1.0, 2.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
    }

    #[test]
    fn test_pnt2d_origin() {
        let p = Pnt2d::origin();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
    }

    #[test]
    fn test_pnt2d_distance() {
        let p1 = Pnt2d::new(0.0, 0.0);
        let p2 = Pnt2d::new(3.0, 4.0);
        assert!((p1.distance(&p2) - 5.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_pnt2d_distance_squared() {
        let p1 = Pnt2d::new(0.0, 0.0);
        let p2 = Pnt2d::new(3.0, 4.0);
        assert!((p1.distance_squared(&p2) - 25.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_pnt2d_vec_to() {
        let p1 = Pnt2d::new(1.0, 2.0);
        let p2 = Pnt2d::new(4.0, 6.0);
        let v = p1.vec_to(&p2);
        assert!((v.x - 3.0).abs() < TEST_TOL);
        assert!((v.y - 4.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_pnt2d_midpoint() {
        let p1 = Pnt2d::new(0.0, 0.0);
        let p2 = Pnt2d::new(2.0, 4.0);
        let mid = p1.midpoint(&p2);
        assert!((mid.x - 1.0).abs() < TEST_TOL);
        assert!((mid.y - 2.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_pnt2d_to_3d_and_back() {
        let p2d = Pnt2d::new(1.5, 2.5);
        let p3d = p2d.to_3d();
        assert!((p3d.x - 1.5).abs() < TEST_TOL);
        assert!((p3d.y - 2.5).abs() < TEST_TOL);
        assert!((p3d.z - 0.0).abs() < TEST_TOL);

        let p2d_back = Pnt2d::from_3d(&p3d);
        assert!((p2d_back.x - 1.5).abs() < TEST_TOL);
        assert!((p2d_back.y - 2.5).abs() < TEST_TOL);
    }

    #[test]
    fn test_pnt2d_is_equal() {
        let p1 = Pnt2d::new(1.0, 1.0);
        let p2 = Pnt2d::new(1.0 + 1e-12, 1.0);
        assert!(p1.is_equal(&p2, 1e-9));
    }

    // ===== Vec2d Tests =====

    #[test]
    fn test_vec2d_creation() {
        let v = Vec2d::new(3.0, 4.0);
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 4.0);
    }

    #[test]
    fn test_vec2d_zero() {
        let v = Vec2d::zero();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
    }

    #[test]
    fn test_vec2d_unit_vectors() {
        let ux = Vec2d::unit_x();
        assert_eq!(ux.x, 1.0);
        assert_eq!(ux.y, 0.0);

        let uy = Vec2d::unit_y();
        assert_eq!(uy.x, 0.0);
        assert_eq!(uy.y, 1.0);
    }

    #[test]
    fn test_vec2d_magnitude() {
        let v = Vec2d::new(3.0, 4.0);
        assert!((v.magnitude() - 5.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_vec2d_magnitude_squared() {
        let v = Vec2d::new(3.0, 4.0);
        assert!((v.magnitude_squared() - 25.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_vec2d_normalized() {
        let v = Vec2d::new(3.0, 4.0);
        let n = v.normalized().unwrap();
        assert!((n.x - 0.6).abs() < TEST_TOL);
        assert!((n.y - 0.8).abs() < TEST_TOL);
    }

    #[test]
    fn test_vec2d_zero_normalized() {
        let v = Vec2d::zero();
        assert!(v.normalized().is_none());
    }

    #[test]
    fn test_vec2d_normalize_in_place() {
        let mut v = Vec2d::new(3.0, 4.0);
        assert!(v.normalize());
        assert!((v.x - 0.6).abs() < TEST_TOL);
        assert!((v.y - 0.8).abs() < TEST_TOL);

        let mut v_zero = Vec2d::zero();
        assert!(!v_zero.normalize());
    }

    #[test]
    fn test_vec2d_dot() {
        let v1 = Vec2d::new(1.0, 2.0);
        let v2 = Vec2d::new(3.0, 4.0);
        assert!((v1.dot(&v2) - 11.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_vec2d_cross() {
        let v1 = Vec2d::unit_x();
        let v2 = Vec2d::unit_y();
        let cross = v1.cross(&v2);
        assert!((cross - 1.0).abs() < TEST_TOL);

        let v3 = Vec2d::unit_y();
        let v4 = Vec2d::unit_x();
        let cross2 = v3.cross(&v4);
        assert!((cross2 - (-1.0)).abs() < TEST_TOL);
    }

    #[test]
    fn test_vec2d_angle() {
        let v1 = Vec2d::unit_x();
        let v2 = Vec2d::unit_y();
        assert!((v1.angle(&v2) - PI / 2.0).abs() < TEST_TOL);

        // Parallel vectors
        let v3 = Vec2d::new(2.0, 0.0);
        assert!(v1.angle(&v3).abs() < TEST_TOL);

        // Opposite vectors
        let v4 = Vec2d::new(-1.0, 0.0);
        assert!((v1.angle(&v4) - PI).abs() < TEST_TOL);
    }

    #[test]
    fn test_vec2d_is_parallel() {
        let v1 = Vec2d::new(1.0, 0.0);
        let v2 = Vec2d::new(2.0, 0.0);
        let v3 = Vec2d::new(0.0, 1.0);
        assert!(v1.is_parallel(&v2, TEST_TOL));
        assert!(!v1.is_parallel(&v3, TEST_TOL));
    }

    #[test]
    fn test_vec2d_is_perpendicular() {
        let v1 = Vec2d::unit_x();
        let v2 = Vec2d::unit_y();
        let v3 = Vec2d::unit_x();
        assert!(v1.is_perpendicular(&v2, TEST_TOL));
        assert!(!v1.is_perpendicular(&v3, TEST_TOL));
    }

    #[test]
    fn test_vec2d_reverse() {
        let v = Vec2d::new(3.0, 4.0);
        let rev = v.reverse();
        assert!((rev.x - (-3.0)).abs() < TEST_TOL);
        assert!((rev.y - (-4.0)).abs() < TEST_TOL);
    }

    #[test]
    fn test_vec2d_scaled() {
        let v = Vec2d::new(2.0, 3.0);
        let scaled = v.scaled(2.0);
        assert!((scaled.x - 4.0).abs() < TEST_TOL);
        assert!((scaled.y - 6.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_vec2d_to_3d_and_back() {
        let v2d = Vec2d::new(1.5, 2.5);
        let v3d = v2d.to_3d();
        assert!((v3d.x - 1.5).abs() < TEST_TOL);
        assert!((v3d.y - 2.5).abs() < TEST_TOL);
        assert!((v3d.z - 0.0).abs() < TEST_TOL);

        let v2d_back = Vec2d::from_3d(&v3d);
        assert!((v2d_back.x - 1.5).abs() < TEST_TOL);
        assert!((v2d_back.y - 2.5).abs() < TEST_TOL);
    }

    #[test]
    fn test_vec2d_operators() {
        let v1 = Vec2d::new(1.0, 2.0);
        let v2 = Vec2d::new(3.0, 4.0);

        // Add
        let sum = v1 + v2;
        assert!((sum.x - 4.0).abs() < TEST_TOL);
        assert!((sum.y - 6.0).abs() < TEST_TOL);

        // Sub
        let diff = v1 - v2;
        assert!((diff.x - (-2.0)).abs() < TEST_TOL);
        assert!((diff.y - (-2.0)).abs() < TEST_TOL);

        // Neg
        let neg = -v1;
        assert!((neg.x - (-1.0)).abs() < TEST_TOL);
        assert!((neg.y - (-2.0)).abs() < TEST_TOL);

        // Mul with scalar
        let scaled = v1 * 2.0;
        assert!((scaled.x - 2.0).abs() < TEST_TOL);
        assert!((scaled.y - 4.0).abs() < TEST_TOL);

        // Scalar mul with vector
        let scaled2 = 3.0 * v2;
        assert!((scaled2.x - 9.0).abs() < TEST_TOL);
        assert!((scaled2.y - 12.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_pnt2d_vec2d_operators() {
        let p1 = Pnt2d::new(1.0, 2.0);
        let p2 = Pnt2d::new(4.0, 6.0);
        let v = Vec2d::new(2.0, 3.0);

        // Point + Vector
        let p_trans = p1 + v;
        assert!((p_trans.x - 3.0).abs() < TEST_TOL);
        assert!((p_trans.y - 5.0).abs() < TEST_TOL);

        // Point - Point = Vector
        let vec = p2 - p1;
        assert!((vec.x - 3.0).abs() < TEST_TOL);
        assert!((vec.y - 4.0).abs() < TEST_TOL);
    }
}
