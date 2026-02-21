//! 3D coordinate triplet.
//!
//! Port of OCCT's gp_XYZ class.
//! Source: src/FoundationClasses/TKMath/gp/gp_XYZ.hxx
//!
//! This is the most fundamental type in OCCT. Every other geometric
//! primitive (Pnt, Vec, Dir) is built on XYZ.

use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg, Index, IndexMut};
use crate::precision;

/// 3D cartesian coordinate entity {X, Y, Z}.
/// Used for algebraic calculations and as storage for geometric types.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct XYZ {
    x: f64,
    y: f64,
    z: f64,
}

impl XYZ {
    /// Creates an XYZ with zero coordinates (0, 0, 0).
    #[inline]
    pub const fn new() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Creates an XYZ with given coordinates.
    #[inline]
    pub const fn from_coords(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Sets all three coordinates.
    #[inline]
    pub fn set_coord(&mut self, x: f64, y: f64, z: f64) {
        self.x = x;
        self.y = y;
        self.z = z;
    }

    /// Sets coordinate by index (1=X, 2=Y, 3=Z).
    /// Panics if index not in {1, 2, 3}.
    #[inline]
    pub fn set_coord_index(&mut self, index: usize, value: f64) {
        match index {
            1 => self.x = value,
            2 => self.y = value,
            3 => self.z = value,
            _ => panic!("XYZ::set_coord_index: index {} out of range [1,3]", index),
        }
    }

    /// Sets the X coordinate.
    #[inline]
    pub fn set_x(&mut self, x: f64) {
        self.x = x;
    }

    /// Sets the Y coordinate.
    #[inline]
    pub fn set_y(&mut self, y: f64) {
        self.y = y;
    }

    /// Sets the Z coordinate.
    #[inline]
    pub fn set_z(&mut self, z: f64) {
        self.z = z;
    }

    /// Returns coordinate by index (1=X, 2=Y, 3=Z).
    /// Panics if index not in {1, 2, 3}.
    #[inline]
    pub fn coord(&self, index: usize) -> f64 {
        match index {
            1 => self.x,
            2 => self.y,
            3 => self.z,
            _ => panic!("XYZ::coord: index {} out of range [1,3]", index),
        }
    }

    /// Returns all three coordinates as a tuple.
    #[inline]
    pub const fn coords(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }

    /// Returns the X coordinate.
    #[inline]
    pub const fn x(&self) -> f64 {
        self.x
    }

    /// Returns the Y coordinate.
    #[inline]
    pub const fn y(&self) -> f64 {
        self.y
    }

    /// Returns the Z coordinate.
    #[inline]
    pub const fn z(&self) -> f64 {
        self.z
    }

    /// Returns a reference to coordinates as array.
    /// OCCT equivalent: GetData()
    #[inline]
    pub fn as_array(&self) -> &[f64; 3] {
        // Safety: XYZ has same layout as [f64; 3]
        unsafe { &*(self as *const XYZ as *const [f64; 3]) }
    }

    /// Returns a mutable reference to coordinates as array.
    /// OCCT equivalent: ChangeData()
    #[inline]
    pub fn as_array_mut(&mut self) -> &mut [f64; 3] {
        // Safety: XYZ has same layout as [f64; 3]
        unsafe { &mut *(self as *mut XYZ as *mut [f64; 3]) }
    }

    /// Computes the modulus (length) of the vector.
    #[inline]
    pub fn modulus(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Computes the square of the modulus.
    #[inline]
    pub const fn square_modulus(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Returns true if modulus <= linear_tolerance.
    #[inline]
    pub fn is_equal(&self, other: &XYZ, tolerance: f64) -> bool {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt() <= tolerance
    }

    /// Adds another XYZ in place.
    #[inline]
    pub fn add(&mut self, other: &XYZ) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }

    /// Returns sum of this and other XYZ.
    #[inline]
    pub fn added(&self, other: &XYZ) -> XYZ {
        XYZ {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Subtracts another XYZ in place.
    #[inline]
    pub fn subtract(&mut self, other: &XYZ) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }

    /// Returns difference of this and other XYZ.
    #[inline]
    pub fn subtracted(&self, other: &XYZ) -> XYZ {
        XYZ {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    /// Multiplies by scalar in place.
    #[inline]
    pub fn multiply(&mut self, scalar: f64) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
    }

    /// Returns this multiplied by scalar.
    #[inline]
    pub fn multiplied(&self, scalar: f64) -> XYZ {
        XYZ {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }

    /// Divides by scalar in place.
    #[inline]
    pub fn divide(&mut self, scalar: f64) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
    }

    /// Returns this divided by scalar.
    #[inline]
    pub fn divided(&self, scalar: f64) -> XYZ {
        XYZ {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }

    /// Reverses direction in place.
    #[inline]
    pub fn reverse(&mut self) {
        self.x = -self.x;
        self.y = -self.y;
        self.z = -self.z;
    }

    /// Returns reversed XYZ.
    #[inline]
    pub fn reversed(&self) -> XYZ {
        XYZ {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Computes dot product.
    #[inline]
    pub const fn dot(&self, other: &XYZ) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Computes cross product in place.
    #[inline]
    pub fn cross(&mut self, other: &XYZ) {
        let x = self.y * other.z - self.z * other.y;
        let y = self.z * other.x - self.x * other.z;
        let z = self.x * other.y - self.y * other.x;
        self.x = x;
        self.y = y;
        self.z = z;
    }

    /// Returns cross product.
    #[inline]
    pub fn crossed(&self, other: &XYZ) -> XYZ {
        XYZ {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Computes triple scalar product (self · (v1 × v2)).
    #[inline]
    pub fn dot_cross(&self, v1: &XYZ, v2: &XYZ) -> f64 {
        let crossed = v1.crossed(v2);
        self.dot(&crossed)
    }

    /// Computes self × (v1 × v2) in place.
    #[inline]
    pub fn cross_cross(&mut self, v1: &XYZ, v2: &XYZ) {
        let inner = v1.crossed(v2);
        self.cross(&inner);
    }

    /// Returns self × (v1 × v2).
    #[inline]
    pub fn cross_crossed(&self, v1: &XYZ, v2: &XYZ) -> XYZ {
        let inner = v1.crossed(v2);
        self.crossed(&inner)
    }

    /// Computes magnitude of cross product: ||self × other||.
    #[inline]
    pub fn cross_magnitude(&self, other: &XYZ) -> f64 {
        self.cross_square_magnitude(other).sqrt()
    }

    /// Computes square magnitude of cross product: ||self × other||².
    #[inline]
    pub const fn cross_square_magnitude(&self, other: &XYZ) -> f64 {
        let x = self.y * other.z - self.z * other.y;
        let y = self.z * other.x - self.x * other.z;
        let z = self.x * other.y - self.y * other.x;
        x * x + y * y + z * z
    }

    /// Component-wise multiplication in place.
    #[inline]
    pub fn multiply_xyz(&mut self, other: &XYZ) {
        self.x *= other.x;
        self.y *= other.y;
        self.z *= other.z;
    }

    /// Returns component-wise multiplication.
    #[inline]
    pub fn multiplied_xyz(&self, other: &XYZ) -> XYZ {
        XYZ {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }

    /// Normalizes in place. Returns false if modulus is too small.
    pub fn normalize(&mut self) -> bool {
        let d = self.modulus();
        if d <= precision::CONFUSION {
            return false;
        }
        self.x /= d;
        self.y /= d;
        self.z /= d;
        true
    }

    /// Returns normalized XYZ. Returns None if modulus is too small.
    pub fn normalized(&self) -> Option<XYZ> {
        let d = self.modulus();
        if d <= precision::CONFUSION {
            return None;
        }
        Some(XYZ {
            x: self.x / d,
            y: self.y / d,
            z: self.z / d,
        })
    }

    // =========================================================================
    // SetLinearForm variants (all 6 from OCCT)
    // =========================================================================

    /// Sets to linear form: a1*xyz1 + a2*xyz2 + a3*xyz3 + xyz4.
    #[inline]
    pub fn set_linear_form_4(
        &mut self,
        a1: f64, xyz1: &XYZ,
        a2: f64, xyz2: &XYZ,
        a3: f64, xyz3: &XYZ,
        xyz4: &XYZ
    ) {
        self.x = a1 * xyz1.x + a2 * xyz2.x + a3 * xyz3.x + xyz4.x;
        self.y = a1 * xyz1.y + a2 * xyz2.y + a3 * xyz3.y + xyz4.y;
        self.z = a1 * xyz1.z + a2 * xyz2.z + a3 * xyz3.z + xyz4.z;
    }

    /// Sets to linear form: a1*xyz1 + a2*xyz2 + a3*xyz3.
    #[inline]
    pub fn set_linear_form_3w(
        &mut self,
        a1: f64, xyz1: &XYZ,
        a2: f64, xyz2: &XYZ,
        a3: f64, xyz3: &XYZ
    ) {
        self.x = a1 * xyz1.x + a2 * xyz2.x + a3 * xyz3.x;
        self.y = a1 * xyz1.y + a2 * xyz2.y + a3 * xyz3.y;
        self.z = a1 * xyz1.z + a2 * xyz2.z + a3 * xyz3.z;
    }

    /// Sets to linear form: a1*xyz1 + a2*xyz2 + xyz3.
    #[inline]
    pub fn set_linear_form_3(
        &mut self,
        a1: f64, xyz1: &XYZ,
        a2: f64, xyz2: &XYZ,
        xyz3: &XYZ
    ) {
        self.x = a1 * xyz1.x + a2 * xyz2.x + xyz3.x;
        self.y = a1 * xyz1.y + a2 * xyz2.y + xyz3.y;
        self.z = a1 * xyz1.z + a2 * xyz2.z + xyz3.z;
    }

    /// Sets to linear form: a1*xyz1 + a2*xyz2.
    #[inline]
    pub fn set_linear_form_2w(&mut self, a1: f64, xyz1: &XYZ, a2: f64, xyz2: &XYZ) {
        self.x = a1 * xyz1.x + a2 * xyz2.x;
        self.y = a1 * xyz1.y + a2 * xyz2.y;
        self.z = a1 * xyz1.z + a2 * xyz2.z;
    }

    /// Sets to linear form: a1*xyz1 + xyz2.
    #[inline]
    pub fn set_linear_form_2(&mut self, a1: f64, xyz1: &XYZ, xyz2: &XYZ) {
        self.x = a1 * xyz1.x + xyz2.x;
        self.y = a1 * xyz1.y + xyz2.y;
        self.z = a1 * xyz1.z + xyz2.z;
    }

    /// Sets to linear form: xyz1 + xyz2.
    #[inline]
    pub fn set_linear_form(&mut self, xyz1: &XYZ, xyz2: &XYZ) {
        self.x = xyz1.x + xyz2.x;
        self.y = xyz1.y + xyz2.y;
        self.z = xyz1.z + xyz2.z;
    }
}

// Operator implementations (matching OCCT's operator overloads)

impl Add for XYZ {
    type Output = XYZ;
    #[inline]
    fn add(self, other: XYZ) -> XYZ {
        XYZ {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl AddAssign for XYZ {
    #[inline]
    fn add_assign(&mut self, other: XYZ) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl Sub for XYZ {
    type Output = XYZ;
    #[inline]
    fn sub(self, other: XYZ) -> XYZ {
        XYZ {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl SubAssign for XYZ {
    #[inline]
    fn sub_assign(&mut self, other: XYZ) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl Mul<f64> for XYZ {
    type Output = XYZ;
    #[inline]
    fn mul(self, scalar: f64) -> XYZ {
        XYZ {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Mul<XYZ> for f64 {
    type Output = XYZ;
    #[inline]
    fn mul(self, xyz: XYZ) -> XYZ {
        XYZ {
            x: xyz.x * self,
            y: xyz.y * self,
            z: xyz.z * self,
        }
    }
}

impl MulAssign<f64> for XYZ {
    #[inline]
    fn mul_assign(&mut self, scalar: f64) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
    }
}

impl Div<f64> for XYZ {
    type Output = XYZ;
    #[inline]
    fn div(self, scalar: f64) -> XYZ {
        XYZ {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

impl DivAssign<f64> for XYZ {
    #[inline]
    fn div_assign(&mut self, scalar: f64) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
    }
}

impl Neg for XYZ {
    type Output = XYZ;
    #[inline]
    fn neg(self) -> XYZ {
        XYZ {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Index<usize> for XYZ {
    type Output = f64;
    #[inline]
    fn index(&self, index: usize) -> &f64 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("XYZ index out of bounds: {}", index),
        }
    }
}

impl IndexMut<usize> for XYZ {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("XYZ index out of bounds: {}", index),
        }
    }
}

impl From<[f64; 3]> for XYZ {
    #[inline]
    fn from(arr: [f64; 3]) -> Self {
        XYZ { x: arr[0], y: arr[1], z: arr[2] }
    }
}

impl From<(f64, f64, f64)> for XYZ {
    #[inline]
    fn from(tuple: (f64, f64, f64)) -> Self {
        XYZ { x: tuple.0, y: tuple.1, z: tuple.2 }
    }
}

impl From<XYZ> for [f64; 3] {
    #[inline]
    fn from(xyz: XYZ) -> Self {
        [xyz.x, xyz.y, xyz.z]
    }
}

// Hash implementation matching OCCT's std::hash<gp_XYZ>
impl std::hash::Hash for XYZ {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // OCCT uses a specific hash combining integer casts
        // We'll use a simpler but consistent approach
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
        self.z.to_bits().hash(state);
    }
}

impl Eq for XYZ {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xyz_default() {
        let xyz = XYZ::new();
        assert_eq!(xyz.x(), 0.0);
        assert_eq!(xyz.y(), 0.0);
        assert_eq!(xyz.z(), 0.0);
    }

    #[test]
    fn test_xyz_from_coords() {
        let xyz = XYZ::from_coords(1.0, 2.0, 3.0);
        assert_eq!(xyz.x(), 1.0);
        assert_eq!(xyz.y(), 2.0);
        assert_eq!(xyz.z(), 3.0);
    }

    #[test]
    fn test_xyz_coord_index() {
        let xyz = XYZ::from_coords(1.0, 2.0, 3.0);
        assert_eq!(xyz.coord(1), 1.0);
        assert_eq!(xyz.coord(2), 2.0);
        assert_eq!(xyz.coord(3), 3.0);
    }

    #[test]
    #[should_panic]
    fn test_xyz_coord_index_out_of_range() {
        let xyz = XYZ::new();
        xyz.coord(4);
    }

    #[test]
    fn test_xyz_set_coord_index() {
        let mut xyz = XYZ::new();
        xyz.set_coord_index(1, 1.5);
        xyz.set_coord_index(2, 2.5);
        xyz.set_coord_index(3, 3.5);
        assert_eq!(xyz.x(), 1.5);
        assert_eq!(xyz.y(), 2.5);
        assert_eq!(xyz.z(), 3.5);
    }

    #[test]
    fn test_xyz_modulus() {
        let xyz = XYZ::from_coords(3.0, 4.0, 0.0);
        assert!((xyz.modulus() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_xyz_square_modulus() {
        let xyz = XYZ::from_coords(1.0, 2.0, 3.0);
        assert_eq!(xyz.square_modulus(), 14.0);
    }

    #[test]
    fn test_xyz_add() {
        let a = XYZ::from_coords(1.0, 2.0, 3.0);
        let b = XYZ::from_coords(4.0, 5.0, 6.0);
        let c = a + b;
        assert_eq!(c.x(), 5.0);
        assert_eq!(c.y(), 7.0);
        assert_eq!(c.z(), 9.0);
    }

    #[test]
    fn test_xyz_subtract() {
        let a = XYZ::from_coords(4.0, 5.0, 6.0);
        let b = XYZ::from_coords(1.0, 2.0, 3.0);
        let c = a - b;
        assert_eq!(c.x(), 3.0);
        assert_eq!(c.y(), 3.0);
        assert_eq!(c.z(), 3.0);
    }

    #[test]
    fn test_xyz_multiply() {
        let xyz = XYZ::from_coords(1.0, 2.0, 3.0);
        let scaled = xyz * 2.0;
        assert_eq!(scaled.x(), 2.0);
        assert_eq!(scaled.y(), 4.0);
        assert_eq!(scaled.z(), 6.0);
    }

    #[test]
    fn test_xyz_dot() {
        let a = XYZ::from_coords(1.0, 0.0, 0.0);
        let b = XYZ::from_coords(0.0, 1.0, 0.0);
        assert_eq!(a.dot(&b), 0.0);

        let c = XYZ::from_coords(1.0, 2.0, 3.0);
        let d = XYZ::from_coords(4.0, 5.0, 6.0);
        assert_eq!(c.dot(&d), 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn test_xyz_cross() {
        let i = XYZ::from_coords(1.0, 0.0, 0.0);
        let j = XYZ::from_coords(0.0, 1.0, 0.0);
        let k = i.crossed(&j);
        assert!((k.x() - 0.0).abs() < 1e-10);
        assert!((k.y() - 0.0).abs() < 1e-10);
        assert!((k.z() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_xyz_cross_magnitude() {
        // |i × j| = 1
        let i = XYZ::from_coords(1.0, 0.0, 0.0);
        let j = XYZ::from_coords(0.0, 1.0, 0.0);
        assert!((i.cross_magnitude(&j) - 1.0).abs() < 1e-10);

        // Parallel vectors: cross magnitude = 0
        let a = XYZ::from_coords(1.0, 2.0, 3.0);
        let b = XYZ::from_coords(2.0, 4.0, 6.0);
        assert!(a.cross_magnitude(&b) < 1e-10);
    }

    #[test]
    fn test_xyz_cross_square_magnitude() {
        let i = XYZ::from_coords(1.0, 0.0, 0.0);
        let j = XYZ::from_coords(0.0, 1.0, 0.0);
        assert!((i.cross_square_magnitude(&j) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_xyz_dot_cross() {
        // Triple scalar product: a · (b × c) = det([a, b, c])
        let a = XYZ::from_coords(1.0, 0.0, 0.0);
        let b = XYZ::from_coords(0.0, 1.0, 0.0);
        let c = XYZ::from_coords(0.0, 0.0, 1.0);
        // det of identity = 1
        assert!((a.dot_cross(&b, &c) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_xyz_cross_cross() {
        // a × (b × c) = b(a·c) - c(a·b)  (BAC-CAB rule)
        let a = XYZ::from_coords(1.0, 2.0, 3.0);
        let b = XYZ::from_coords(4.0, 5.0, 6.0);
        let c = XYZ::from_coords(7.0, 8.0, 9.0);
        
        let result = a.cross_crossed(&b, &c);
        
        // Verify using BAC-CAB
        let a_dot_c = a.dot(&c);
        let a_dot_b = a.dot(&b);
        let expected = b.multiplied(a_dot_c).subtracted(&c.multiplied(a_dot_b));
        
        assert!((result.x() - expected.x()).abs() < 1e-10);
        assert!((result.y() - expected.y()).abs() < 1e-10);
        assert!((result.z() - expected.z()).abs() < 1e-10);
    }

    #[test]
    fn test_xyz_multiply_xyz() {
        // Component-wise multiplication
        let mut a = XYZ::from_coords(1.0, 2.0, 3.0);
        let b = XYZ::from_coords(4.0, 5.0, 6.0);
        a.multiply_xyz(&b);
        assert_eq!(a.x(), 4.0);  // 1*4
        assert_eq!(a.y(), 10.0); // 2*5
        assert_eq!(a.z(), 18.0); // 3*6
    }

    #[test]
    fn test_xyz_multiplied_xyz() {
        let a = XYZ::from_coords(2.0, 3.0, 4.0);
        let b = XYZ::from_coords(5.0, 6.0, 7.0);
        let c = a.multiplied_xyz(&b);
        assert_eq!(c.x(), 10.0);
        assert_eq!(c.y(), 18.0);
        assert_eq!(c.z(), 28.0);
    }

    #[test]
    fn test_xyz_normalize() {
        let mut xyz = XYZ::from_coords(3.0, 4.0, 0.0);
        assert!(xyz.normalize());
        assert!((xyz.modulus() - 1.0).abs() < 1e-10);
        assert!((xyz.x() - 0.6).abs() < 1e-10);
        assert!((xyz.y() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_xyz_normalize_zero_vector() {
        let mut xyz = XYZ::new();
        assert!(!xyz.normalize());
    }

    #[test]
    fn test_xyz_reverse() {
        let xyz = XYZ::from_coords(1.0, -2.0, 3.0);
        let rev = xyz.reversed();
        assert_eq!(rev.x(), -1.0);
        assert_eq!(rev.y(), 2.0);
        assert_eq!(rev.z(), -3.0);
    }

    #[test]
    fn test_xyz_is_equal() {
        let a = XYZ::from_coords(1.0, 2.0, 3.0);
        let b = XYZ::from_coords(1.0 + 1e-8, 2.0, 3.0);
        assert!(a.is_equal(&b, 1e-7));
        assert!(!a.is_equal(&b, 1e-9));
    }

    #[test]
    fn test_xyz_set_linear_form_all_variants() {
        let xyz1 = XYZ::from_coords(1.0, 0.0, 0.0);
        let xyz2 = XYZ::from_coords(0.0, 1.0, 0.0);
        let xyz3 = XYZ::from_coords(0.0, 0.0, 1.0);
        let xyz4 = XYZ::from_coords(1.0, 1.0, 1.0);

        // set_linear_form (xyz1 + xyz2)
        let mut r = XYZ::new();
        r.set_linear_form(&xyz1, &xyz2);
        assert_eq!(r, XYZ::from_coords(1.0, 1.0, 0.0));

        // set_linear_form_2 (a1*xyz1 + xyz2)
        let mut r = XYZ::new();
        r.set_linear_form_2(2.0, &xyz1, &xyz2);
        assert_eq!(r, XYZ::from_coords(2.0, 1.0, 0.0));

        // set_linear_form_2w (a1*xyz1 + a2*xyz2)
        let mut r = XYZ::new();
        r.set_linear_form_2w(2.0, &xyz1, 3.0, &xyz2);
        assert_eq!(r, XYZ::from_coords(2.0, 3.0, 0.0));

        // set_linear_form_3 (a1*xyz1 + a2*xyz2 + xyz3)
        let mut r = XYZ::new();
        r.set_linear_form_3(2.0, &xyz1, 3.0, &xyz2, &xyz3);
        assert_eq!(r, XYZ::from_coords(2.0, 3.0, 1.0));

        // set_linear_form_3w (a1*xyz1 + a2*xyz2 + a3*xyz3)
        let mut r = XYZ::new();
        r.set_linear_form_3w(2.0, &xyz1, 3.0, &xyz2, 4.0, &xyz3);
        assert_eq!(r, XYZ::from_coords(2.0, 3.0, 4.0));

        // set_linear_form_4 (a1*xyz1 + a2*xyz2 + a3*xyz3 + xyz4)
        let mut r = XYZ::new();
        r.set_linear_form_4(2.0, &xyz1, 3.0, &xyz2, 4.0, &xyz3, &xyz4);
        assert_eq!(r, XYZ::from_coords(3.0, 4.0, 5.0));
    }

    #[test]
    fn test_xyz_divide() {
        let xyz = XYZ::from_coords(4.0, 6.0, 8.0);
        let divided = xyz.divided(2.0);
        assert_eq!(divided.x(), 2.0);
        assert_eq!(divided.y(), 3.0);
        assert_eq!(divided.z(), 4.0);
    }

    #[test]
    fn test_xyz_operators() {
        let a = XYZ::from_coords(1.0, 2.0, 3.0);
        
        // Negation
        let neg = -a;
        assert_eq!(neg, XYZ::from_coords(-1.0, -2.0, -3.0));
        
        // Scalar multiplication (both directions)
        let scaled = a * 2.0;
        assert_eq!(scaled, XYZ::from_coords(2.0, 4.0, 6.0));
        let scaled2 = 2.0 * a;
        assert_eq!(scaled2, XYZ::from_coords(2.0, 4.0, 6.0));
        
        // Division
        let div = a / 2.0;
        assert_eq!(div, XYZ::from_coords(0.5, 1.0, 1.5));
    }

    #[test]
    fn test_xyz_as_array() {
        let xyz = XYZ::from_coords(1.0, 2.0, 3.0);
        let arr = xyz.as_array();
        assert_eq!(arr[0], 1.0);
        assert_eq!(arr[1], 2.0);
        assert_eq!(arr[2], 3.0);
    }

    #[test]
    fn test_xyz_index() {
        let xyz = XYZ::from_coords(1.0, 2.0, 3.0);
        assert_eq!(xyz[0], 1.0);
        assert_eq!(xyz[1], 2.0);
        assert_eq!(xyz[2], 3.0);
    }

    #[test]
    fn test_xyz_from_conversions() {
        let xyz1: XYZ = [1.0, 2.0, 3.0].into();
        assert_eq!(xyz1, XYZ::from_coords(1.0, 2.0, 3.0));

        let xyz2: XYZ = (4.0, 5.0, 6.0).into();
        assert_eq!(xyz2, XYZ::from_coords(4.0, 5.0, 6.0));

        let arr: [f64; 3] = xyz1.into();
        assert_eq!(arr, [1.0, 2.0, 3.0]);
    }
}
