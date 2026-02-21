//! 2D vector.
//!
//! Port of OCCT's gp_Vec2d class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Vec2d.hxx

use crate::gp::XY;
use crate::precision;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg};

/// A 2D vector in cartesian coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2d {
    xy: XY,
}

impl Vec2d {
    /// Creates a zero vector.
    #[inline]
    pub const fn new() -> Self {
        Self { xy: XY::new() }
    }

    /// Creates a vector with given components.
    #[inline]
    pub const fn from_coords(x: f64, y: f64) -> Self {
        Self { xy: XY::from_coords(x, y) }
    }

    /// Creates a vector from an XY.
    #[inline]
    pub const fn from_xy(xy: XY) -> Self {
        Self { xy }
    }

    /// Sets the X component.
    #[inline]
    pub fn set_x(&mut self, x: f64) {
        self.xy.set_x(x);
    }

    /// Sets the Y component.
    #[inline]
    pub fn set_y(&mut self, y: f64) {
        self.xy.set_y(y);
    }

    /// Sets both components.
    #[inline]
    pub fn set_coord(&mut self, x: f64, y: f64) {
        self.xy.set_coord(x, y);
    }

    /// Returns the X component.
    #[inline]
    pub const fn x(&self) -> f64 {
        self.xy.x()
    }

    /// Returns the Y component.
    #[inline]
    pub const fn y(&self) -> f64 {
        self.xy.y()
    }

    /// Returns components as tuple.
    #[inline]
    pub const fn coords(&self) -> (f64, f64) {
        self.xy.coords()
    }

    /// Returns the underlying XY.
    #[inline]
    pub const fn xy(&self) -> XY {
        self.xy
    }

    /// Returns the magnitude (length) of the vector.
    #[inline]
    pub fn magnitude(&self) -> f64 {
        self.xy.modulus()
    }

    /// Returns the squared magnitude of the vector.
    #[inline]
    pub const fn square_magnitude(&self) -> f64 {
        self.xy.square_modulus()
    }

    /// Computes the dot product.
    #[inline]
    pub const fn dot(&self, other: &Vec2d) -> f64 {
        self.xy.dot(&other.xy)
    }

    /// Computes the 2D cross product (scalar result).
    #[inline]
    pub const fn crossed(&self, other: &Vec2d) -> f64 {
        self.xy.crossed(&other.xy)
    }

    /// Returns the magnitude of the cross product: |self × other|.
    #[inline]
    pub fn cross_magnitude(&self, other: &Vec2d) -> f64 {
        self.xy.cross_magnitude(&other.xy)
    }

    /// Normalizes the vector in place. Returns false if zero-length.
    pub fn normalize(&mut self) -> bool {
        self.xy.normalize()
    }

    /// Returns normalized copy. Returns None if zero-length.
    pub fn normalized(&self) -> Option<Vec2d> {
        self.xy.normalized().map(Vec2d::from_xy)
    }

    /// Returns the angle (in radians) from positive X-axis.
    #[inline]
    pub fn angle(&self) -> f64 {
        self.xy.y().atan2(self.xy.x())
    }

    /// Reverses direction in place.
    #[inline]
    pub fn reverse(&mut self) {
        self.xy.reverse();
    }

    /// Returns reversed copy.
    #[inline]
    pub fn reversed(&self) -> Vec2d {
        Vec2d::from_xy(self.xy.reversed())
    }

    /// Adds another vector in place.
    #[inline]
    pub fn add(&mut self, other: &Vec2d) {
        self.xy += other.xy;
    }

    /// Returns sum of vectors.
    #[inline]
    pub fn added(&self, other: &Vec2d) -> Vec2d {
        Vec2d::from_xy(self.xy.added(&other.xy))
    }

    /// Subtracts another vector in place.
    #[inline]
    pub fn subtract(&mut self, other: &Vec2d) {
        self.xy.subtract(&other.xy);
    }

    /// Returns difference of vectors.
    #[inline]
    pub fn subtracted(&self, other: &Vec2d) -> Vec2d {
        Vec2d::from_xy(self.xy.subtracted(&other.xy))
    }

    /// Multiplies by scalar in place.
    #[inline]
    pub fn multiply(&mut self, scalar: f64) {
        self.xy.multiply(scalar);
    }

    /// Returns vector multiplied by scalar.
    #[inline]
    pub fn multiplied(&self, scalar: f64) -> Vec2d {
        Vec2d::from_xy(self.xy.multiplied(scalar))
    }

    /// Divides by scalar in place.
    #[inline]
    pub fn divide(&mut self, scalar: f64) {
        self.xy.divide(scalar);
    }

    /// Returns vector divided by scalar.
    #[inline]
    pub fn divided(&self, scalar: f64) -> Vec2d {
        Vec2d::from_xy(self.xy.divided(scalar))
    }

    /// Component-wise multiplication in place.
    #[inline]
    pub fn multiply_vec(&mut self, other: &Vec2d) {
        self.xy.multiply_xy(&other.xy);
    }

    /// Returns component-wise multiplication.
    #[inline]
    pub fn multiplied_vec(&self, other: &Vec2d) -> Vec2d {
        Vec2d::from_xy(self.xy.multiplied_xy(&other.xy))
    }
}

// Operator implementations

impl Add for Vec2d {
    type Output = Vec2d;
    #[inline]
    fn add(self, other: Vec2d) -> Vec2d {
        self.added(&other)
    }
}

impl AddAssign for Vec2d {
    #[inline]
    fn add_assign(&mut self, other: Vec2d) {
        self.add(&other);
    }
}

impl Sub for Vec2d {
    type Output = Vec2d;
    #[inline]
    fn sub(self, other: Vec2d) -> Vec2d {
        self.subtracted(&other)
    }
}

impl SubAssign for Vec2d {
    #[inline]
    fn sub_assign(&mut self, other: Vec2d) {
        self.subtract(&other);
    }
}

impl Mul<f64> for Vec2d {
    type Output = Vec2d;
    #[inline]
    fn mul(self, scalar: f64) -> Vec2d {
        self.multiplied(scalar)
    }
}

impl Mul<Vec2d> for f64 {
    type Output = Vec2d;
    #[inline]
    fn mul(self, vec: Vec2d) -> Vec2d {
        vec.multiplied(self)
    }
}

impl MulAssign<f64> for Vec2d {
    #[inline]
    fn mul_assign(&mut self, scalar: f64) {
        self.multiply(scalar);
    }
}

impl Div<f64> for Vec2d {
    type Output = Vec2d;
    #[inline]
    fn div(self, scalar: f64) -> Vec2d {
        self.divided(scalar)
    }
}

impl DivAssign<f64> for Vec2d {
    #[inline]
    fn div_assign(&mut self, scalar: f64) {
        self.divide(scalar);
    }
}

impl Neg for Vec2d {
    type Output = Vec2d;
    #[inline]
    fn neg(self) -> Vec2d {
        self.reversed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2d_new() {
        let v = Vec2d::new();
        assert_eq!(v.x(), 0.0);
        assert_eq!(v.y(), 0.0);
    }

    #[test]
    fn test_vec2d_from_coords() {
        let v = Vec2d::from_coords(3.0, 4.0);
        assert_eq!(v.x(), 3.0);
        assert_eq!(v.y(), 4.0);
    }

    #[test]
    fn test_vec2d_magnitude() {
        let v = Vec2d::from_coords(3.0, 4.0);
        assert!((v.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2d_square_magnitude() {
        let v = Vec2d::from_coords(3.0, 4.0);
        assert_eq!(v.square_magnitude(), 25.0);
    }

    #[test]
    fn test_vec2d_dot() {
        let a = Vec2d::from_coords(1.0, 0.0);
        let b = Vec2d::from_coords(0.0, 1.0);
        assert_eq!(a.dot(&b), 0.0);

        let c = Vec2d::from_coords(1.0, 2.0);
        let d = Vec2d::from_coords(3.0, 4.0);
        assert_eq!(c.dot(&d), 11.0); // 1*3 + 2*4
    }

    #[test]
    fn test_vec2d_crossed() {
        let a = Vec2d::from_coords(1.0, 0.0);
        let b = Vec2d::from_coords(0.0, 1.0);
        assert_eq!(a.crossed(&b), 1.0);
    }

    #[test]
    fn test_vec2d_cross_magnitude() {
        let a = Vec2d::from_coords(1.0, 0.0);
        let b = Vec2d::from_coords(0.0, 1.0);
        assert_eq!(a.cross_magnitude(&b), 1.0);
    }

    #[test]
    fn test_vec2d_normalize() {
        let mut v = Vec2d::from_coords(3.0, 4.0);
        assert!(v.normalize());
        assert!((v.magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2d_normalize_zero() {
        let mut v = Vec2d::new();
        assert!(!v.normalize());
    }

    #[test]
    fn test_vec2d_add() {
        let a = Vec2d::from_coords(1.0, 2.0);
        let b = Vec2d::from_coords(3.0, 4.0);
        let c = a + b;
        assert_eq!(c.x(), 4.0);
        assert_eq!(c.y(), 6.0);
    }

    #[test]
    fn test_vec2d_subtract() {
        let a = Vec2d::from_coords(5.0, 6.0);
        let b = Vec2d::from_coords(1.0, 2.0);
        let c = a - b;
        assert_eq!(c.x(), 4.0);
        assert_eq!(c.y(), 4.0);
    }

    #[test]
    fn test_vec2d_multiply() {
        let v = Vec2d::from_coords(1.0, 2.0);
        let scaled = v * 3.0;
        assert_eq!(scaled.x(), 3.0);
        assert_eq!(scaled.y(), 6.0);
    }

    #[test]
    fn test_vec2d_angle() {
        let v = Vec2d::from_coords(1.0, 0.0);
        assert!(v.angle().abs() < 1e-10);

        let v90 = Vec2d::from_coords(0.0, 1.0);
        assert!((v90.angle() - std::f64::consts::PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2d_reverse() {
        let v = Vec2d::from_coords(1.0, 2.0);
        let rev = v.reversed();
        assert_eq!(rev.x(), -1.0);
        assert_eq!(rev.y(), -2.0);
    }
}
