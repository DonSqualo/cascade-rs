//! 2D coordinate pair.
//!
//! Port of OCCT's gp_XY class.
//! Source: src/FoundationClasses/TKMath/gp/gp_XY.hxx
//!
//! This is the foundation for 2D geometric types.

use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg, Index, IndexMut};
use crate::precision;

/// 2D cartesian coordinate entity {X, Y}.
/// Used for algebraic calculations and as storage for 2D geometric types.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct XY {
    x: f64,
    y: f64,
}

impl XY {
    /// Creates an XY with zero coordinates (0, 0).
    #[inline]
    pub const fn new() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Creates an XY with given coordinates.
    #[inline]
    pub const fn from_coords(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Sets both coordinates.
    #[inline]
    pub fn set_coord(&mut self, x: f64, y: f64) {
        self.x = x;
        self.y = y;
    }

    /// Sets coordinate by index (1=X, 2=Y).
    /// Panics if index not in {1, 2}.
    #[inline]
    pub fn set_coord_index(&mut self, index: usize, value: f64) {
        match index {
            1 => self.x = value,
            2 => self.y = value,
            _ => panic!("XY::set_coord_index: index {} out of range [1,2]", index),
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

    /// Returns coordinate by index (1=X, 2=Y).
    /// Panics if index not in {1, 2}.
    #[inline]
    pub fn coord(&self, index: usize) -> f64 {
        match index {
            1 => self.x,
            2 => self.y,
            _ => panic!("XY::coord: index {} out of range [1,2]", index),
        }
    }

    /// Returns both coordinates as a tuple.
    #[inline]
    pub const fn coords(&self) -> (f64, f64) {
        (self.x, self.y)
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

    /// Computes the modulus (length) of the vector.
    #[inline]
    pub fn modulus(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Computes the square of the modulus.
    #[inline]
    pub const fn square_modulus(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    /// Returns true if coordinates are equal within tolerance.
    #[inline]
    pub fn is_equal(&self, other: &XY, tolerance: f64) -> bool {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt() <= tolerance
    }

    /// Adds another XY in place.
    #[inline]
    pub fn add(&mut self, other: &XY) {
        self.x += other.x;
        self.y += other.y;
    }

    /// Returns sum of this and other XY.
    #[inline]
    pub fn added(&self, other: &XY) -> XY {
        XY {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    /// Subtracts another XY in place.
    #[inline]
    pub fn subtract(&mut self, other: &XY) {
        self.x -= other.x;
        self.y -= other.y;
    }

    /// Returns difference of this and other XY.
    #[inline]
    pub fn subtracted(&self, other: &XY) -> XY {
        XY {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    /// Multiplies by scalar in place.
    #[inline]
    pub fn multiply(&mut self, scalar: f64) {
        self.x *= scalar;
        self.y *= scalar;
    }

    /// Returns this multiplied by scalar.
    #[inline]
    pub fn multiplied(&self, scalar: f64) -> XY {
        XY {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }

    /// Divides by scalar in place.
    #[inline]
    pub fn divide(&mut self, scalar: f64) {
        self.x /= scalar;
        self.y /= scalar;
    }

    /// Returns this divided by scalar.
    #[inline]
    pub fn divided(&self, scalar: f64) -> XY {
        XY {
            x: self.x / scalar,
            y: self.y / scalar,
        }
    }

    /// Reverses direction in place.
    #[inline]
    pub fn reverse(&mut self) {
        self.x = -self.x;
        self.y = -self.y;
    }

    /// Returns reversed XY.
    #[inline]
    pub fn reversed(&self) -> XY {
        XY {
            x: -self.x,
            y: -self.y,
        }
    }

    /// Computes dot product.
    #[inline]
    pub const fn dot(&self, other: &XY) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Computes 2D cross product (returns scalar, not vector).
    /// Result: self.x * other.y - self.y * other.x
    #[inline]
    pub const fn crossed(&self, other: &XY) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// Computes magnitude of cross product: |self × other|.
    #[inline]
    pub fn cross_magnitude(&self, other: &XY) -> f64 {
        (self.x * other.y - self.y * other.x).abs()
    }

    /// Computes square magnitude of cross product.
    #[inline]
    pub const fn cross_square_magnitude(&self, other: &XY) -> f64 {
        let z = self.x * other.y - self.y * other.x;
        z * z
    }

    /// Component-wise multiplication in place.
    #[inline]
    pub fn multiply_xy(&mut self, other: &XY) {
        self.x *= other.x;
        self.y *= other.y;
    }

    /// Returns component-wise multiplication.
    #[inline]
    pub fn multiplied_xy(&self, other: &XY) -> XY {
        XY {
            x: self.x * other.x,
            y: self.y * other.y,
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
        true
    }

    /// Returns normalized XY. Returns None if modulus is too small.
    pub fn normalized(&self) -> Option<XY> {
        let d = self.modulus();
        if d <= precision::CONFUSION {
            return None;
        }
        Some(XY {
            x: self.x / d,
            y: self.y / d,
        })
    }

    // =========================================================================
    // SetLinearForm variants
    // =========================================================================

    /// Sets to linear form: a1*xy1 + a2*xy2 + xy3.
    #[inline]
    pub fn set_linear_form_3(
        &mut self,
        a1: f64, xy1: &XY,
        a2: f64, xy2: &XY,
        xy3: &XY
    ) {
        self.x = a1 * xy1.x + a2 * xy2.x + xy3.x;
        self.y = a1 * xy1.y + a2 * xy2.y + xy3.y;
    }

    /// Sets to linear form: a1*xy1 + a2*xy2.
    #[inline]
    pub fn set_linear_form_2w(&mut self, a1: f64, xy1: &XY, a2: f64, xy2: &XY) {
        self.x = a1 * xy1.x + a2 * xy2.x;
        self.y = a1 * xy1.y + a2 * xy2.y;
    }

    /// Sets to linear form: a1*xy1 + xy2.
    #[inline]
    pub fn set_linear_form_2(&mut self, a1: f64, xy1: &XY, xy2: &XY) {
        self.x = a1 * xy1.x + xy2.x;
        self.y = a1 * xy1.y + xy2.y;
    }

    /// Sets to linear form: xy1 + xy2.
    #[inline]
    pub fn set_linear_form(&mut self, xy1: &XY, xy2: &XY) {
        self.x = xy1.x + xy2.x;
        self.y = xy1.y + xy2.y;
    }
}

// Operator implementations

impl Add for XY {
    type Output = XY;
    #[inline]
    fn add(self, other: XY) -> XY {
        XY {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl AddAssign for XY {
    #[inline]
    fn add_assign(&mut self, other: XY) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl Sub for XY {
    type Output = XY;
    #[inline]
    fn sub(self, other: XY) -> XY {
        XY {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl SubAssign for XY {
    #[inline]
    fn sub_assign(&mut self, other: XY) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl Mul<f64> for XY {
    type Output = XY;
    #[inline]
    fn mul(self, scalar: f64) -> XY {
        XY {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl Mul<XY> for f64 {
    type Output = XY;
    #[inline]
    fn mul(self, xy: XY) -> XY {
        XY {
            x: xy.x * self,
            y: xy.y * self,
        }
    }
}

impl MulAssign<f64> for XY {
    #[inline]
    fn mul_assign(&mut self, scalar: f64) {
        self.x *= scalar;
        self.y *= scalar;
    }
}

impl Div<f64> for XY {
    type Output = XY;
    #[inline]
    fn div(self, scalar: f64) -> XY {
        XY {
            x: self.x / scalar,
            y: self.y / scalar,
        }
    }
}

impl DivAssign<f64> for XY {
    #[inline]
    fn div_assign(&mut self, scalar: f64) {
        self.x /= scalar;
        self.y /= scalar;
    }
}

impl Neg for XY {
    type Output = XY;
    #[inline]
    fn neg(self) -> XY {
        XY {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl Index<usize> for XY {
    type Output = f64;
    #[inline]
    fn index(&self, index: usize) -> &f64 {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("XY index out of bounds: {}", index),
        }
    }
}

impl IndexMut<usize> for XY {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("XY index out of bounds: {}", index),
        }
    }
}

impl From<[f64; 2]> for XY {
    #[inline]
    fn from(arr: [f64; 2]) -> Self {
        XY { x: arr[0], y: arr[1] }
    }
}

impl From<(f64, f64)> for XY {
    #[inline]
    fn from(tuple: (f64, f64)) -> Self {
        XY { x: tuple.0, y: tuple.1 }
    }
}

impl From<XY> for [f64; 2] {
    #[inline]
    fn from(xy: XY) -> Self {
        [xy.x, xy.y]
    }
}

impl std::hash::Hash for XY {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
    }
}

impl Eq for XY {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xy_new() {
        let xy = XY::new();
        assert_eq!(xy.x(), 0.0);
        assert_eq!(xy.y(), 0.0);
    }

    #[test]
    fn test_xy_from_coords() {
        let xy = XY::from_coords(1.0, 2.0);
        assert_eq!(xy.x(), 1.0);
        assert_eq!(xy.y(), 2.0);
    }

    #[test]
    fn test_xy_set_coord() {
        let mut xy = XY::new();
        xy.set_coord(3.0, 4.0);
        assert_eq!(xy.x(), 3.0);
        assert_eq!(xy.y(), 4.0);
    }

    #[test]
    fn test_xy_set_coord_index() {
        let mut xy = XY::new();
        xy.set_coord_index(1, 5.0);
        xy.set_coord_index(2, 6.0);
        assert_eq!(xy.x(), 5.0);
        assert_eq!(xy.y(), 6.0);
    }

    #[test]
    #[should_panic]
    fn test_xy_set_coord_index_out_of_range() {
        let mut xy = XY::new();
        xy.set_coord_index(3, 7.0);
    }

    #[test]
    fn test_xy_coord_index() {
        let xy = XY::from_coords(1.0, 2.0);
        assert_eq!(xy.coord(1), 1.0);
        assert_eq!(xy.coord(2), 2.0);
    }

    #[test]
    #[should_panic]
    fn test_xy_coord_index_out_of_range() {
        let xy = XY::new();
        xy.coord(3);
    }

    #[test]
    fn test_xy_modulus() {
        let xy = XY::from_coords(3.0, 4.0);
        assert!((xy.modulus() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_xy_square_modulus() {
        let xy = XY::from_coords(3.0, 4.0);
        assert_eq!(xy.square_modulus(), 25.0);
    }

    #[test]
    fn test_xy_add() {
        let a = XY::from_coords(1.0, 2.0);
        let b = XY::from_coords(3.0, 4.0);
        let c = a + b;
        assert_eq!(c.x(), 4.0);
        assert_eq!(c.y(), 6.0);
    }

    #[test]
    fn test_xy_subtract() {
        let a = XY::from_coords(5.0, 7.0);
        let b = XY::from_coords(1.0, 2.0);
        let c = a - b;
        assert_eq!(c.x(), 4.0);
        assert_eq!(c.y(), 5.0);
    }

    #[test]
    fn test_xy_multiply() {
        let xy = XY::from_coords(1.0, 2.0);
        let scaled = xy * 3.0;
        assert_eq!(scaled.x(), 3.0);
        assert_eq!(scaled.y(), 6.0);
    }

    #[test]
    fn test_xy_multiply_scalar_left() {
        let xy = XY::from_coords(1.0, 2.0);
        let scaled = 3.0 * xy;
        assert_eq!(scaled.x(), 3.0);
        assert_eq!(scaled.y(), 6.0);
    }

    #[test]
    fn test_xy_divide() {
        let xy = XY::from_coords(4.0, 6.0);
        let divided = xy / 2.0;
        assert_eq!(divided.x(), 2.0);
        assert_eq!(divided.y(), 3.0);
    }

    #[test]
    fn test_xy_dot() {
        let a = XY::from_coords(1.0, 0.0);
        let b = XY::from_coords(0.0, 1.0);
        assert_eq!(a.dot(&b), 0.0);

        let c = XY::from_coords(1.0, 2.0);
        let d = XY::from_coords(3.0, 4.0);
        assert_eq!(c.dot(&d), 11.0); // 1*3 + 2*4
    }

    #[test]
    fn test_xy_crossed() {
        let a = XY::from_coords(1.0, 0.0);
        let b = XY::from_coords(0.0, 1.0);
        assert_eq!(a.crossed(&b), 1.0);

        let c = XY::from_coords(2.0, 3.0);
        let d = XY::from_coords(4.0, 5.0);
        assert_eq!(c.crossed(&d), 10.0 - 12.0); // 2*5 - 3*4 = -2
    }

    #[test]
    fn test_xy_cross_magnitude() {
        let a = XY::from_coords(1.0, 0.0);
        let b = XY::from_coords(0.0, 1.0);
        assert_eq!(a.cross_magnitude(&b), 1.0);

        // Parallel vectors: cross magnitude = 0
        let c = XY::from_coords(1.0, 2.0);
        let d = XY::from_coords(2.0, 4.0);
        assert!(c.cross_magnitude(&d) < 1e-10);
    }

    #[test]
    fn test_xy_cross_square_magnitude() {
        let a = XY::from_coords(1.0, 0.0);
        let b = XY::from_coords(0.0, 1.0);
        assert_eq!(a.cross_square_magnitude(&b), 1.0);
    }

    #[test]
    fn test_xy_normalize() {
        let mut xy = XY::from_coords(3.0, 4.0);
        assert!(xy.normalize());
        assert!((xy.modulus() - 1.0).abs() < 1e-10);
        assert!((xy.x() - 0.6).abs() < 1e-10);
        assert!((xy.y() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_xy_normalize_zero_vector() {
        let mut xy = XY::new();
        assert!(!xy.normalize());
    }

    #[test]
    fn test_xy_reverse() {
        let xy = XY::from_coords(1.0, -2.0);
        let rev = xy.reversed();
        assert_eq!(rev.x(), -1.0);
        assert_eq!(rev.y(), 2.0);
    }

    #[test]
    fn test_xy_is_equal() {
        let a = XY::from_coords(1.0, 2.0);
        let b = XY::from_coords(1.0 + 1e-8, 2.0);
        assert!(a.is_equal(&b, 1e-7));
        assert!(!a.is_equal(&b, 1e-9));
    }

    #[test]
    fn test_xy_set_linear_form() {
        let xy1 = XY::from_coords(1.0, 2.0);
        let xy2 = XY::from_coords(3.0, 4.0);
        let mut result = XY::new();
        result.set_linear_form(&xy1, &xy2);
        assert_eq!(result.x(), 4.0);
        assert_eq!(result.y(), 6.0);
    }

    #[test]
    fn test_xy_set_linear_form_2() {
        let xy1 = XY::from_coords(1.0, 2.0);
        let xy2 = XY::from_coords(3.0, 4.0);
        let mut result = XY::new();
        result.set_linear_form_2(2.0, &xy1, &xy2);
        assert_eq!(result.x(), 5.0);
        assert_eq!(result.y(), 8.0);
    }

    #[test]
    fn test_xy_set_linear_form_2w() {
        let xy1 = XY::from_coords(1.0, 2.0);
        let xy2 = XY::from_coords(3.0, 4.0);
        let mut result = XY::new();
        result.set_linear_form_2w(2.0, &xy1, 3.0, &xy2);
        assert_eq!(result.x(), 11.0);
        assert_eq!(result.y(), 16.0);
    }

    #[test]
    fn test_xy_set_linear_form_3() {
        let xy1 = XY::from_coords(1.0, 2.0);
        let xy2 = XY::from_coords(3.0, 4.0);
        let xy3 = XY::from_coords(5.0, 6.0);
        let mut result = XY::new();
        result.set_linear_form_3(2.0, &xy1, 3.0, &xy2, &xy3);
        assert_eq!(result.x(), 16.0);
        assert_eq!(result.y(), 22.0);
    }

    #[test]
    fn test_xy_operators() {
        let a = XY::from_coords(1.0, 2.0);
        
        // Negation
        let neg = -a;
        assert_eq!(neg, XY::from_coords(-1.0, -2.0));
        
        // Add assign
        let mut c = XY::from_coords(1.0, 1.0);
        c += XY::from_coords(2.0, 3.0);
        assert_eq!(c, XY::from_coords(3.0, 4.0));
    }

    #[test]
    fn test_xy_index() {
        let xy = XY::from_coords(1.0, 2.0);
        assert_eq!(xy[0], 1.0);
        assert_eq!(xy[1], 2.0);
    }

    #[test]
    fn test_xy_from_conversions() {
        let xy1: XY = [1.0, 2.0].into();
        assert_eq!(xy1, XY::from_coords(1.0, 2.0));

        let xy2: XY = (3.0, 4.0).into();
        assert_eq!(xy2, XY::from_coords(3.0, 4.0));

        let arr: [f64; 2] = xy1.into();
        assert_eq!(arr, [1.0, 2.0]);
    }

    #[test]
    fn test_xy_multiply_xy() {
        let a = XY::from_coords(2.0, 3.0);
        let b = XY::from_coords(4.0, 5.0);
        let c = a.multiplied_xy(&b);
        assert_eq!(c.x(), 8.0);
        assert_eq!(c.y(), 15.0);
    }
}
