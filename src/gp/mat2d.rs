//! 2x2 matrix.
//!
//! Port of OCCT's gp_Mat2d class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Mat2d.hxx

use crate::gp::XY;
use crate::precision;

/// A 2x2 matrix for 2D transformations and computations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mat2d {
    // Stored as column-major: [col0, col1] where col = [row0, row1]
    mat: [[f64; 2]; 2],
}

impl Mat2d {
    /// Creates a zero matrix.
    #[inline]
    pub const fn new() -> Self {
        Self { mat: [[0.0, 0.0], [0.0, 0.0]] }
    }

    /// Creates a matrix from two column vectors.
    #[inline]
    pub fn from_cols(col1: XY, col2: XY) -> Self {
        let (x1, y1) = col1.coords();
        let (x2, y2) = col2.coords();
        Self { mat: [[x1, y1], [x2, y2]] }
    }

    /// Creates an identity matrix.
    #[inline]
    pub const fn identity() -> Self {
        Self { mat: [[1.0, 0.0], [0.0, 1.0]] }
    }

    /// Creates a rotation matrix from angle in radians.
    #[inline]
    pub fn from_rotation(angle: f64) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        // [[cos, -sin], [sin, cos]] (row-major storage)
        Self { mat: [[cos, -sin], [sin, cos]] }
    }

    /// Creates a scaling matrix.
    #[inline]
    pub fn from_scale(sx: f64, sy: f64) -> Self {
        Self { mat: [[sx, 0.0], [0.0, sy]] }
    }

    /// Creates a uniform scaling matrix.
    #[inline]
    pub const fn from_uniform_scale(scale: f64) -> Self {
        Self { mat: [[scale, 0.0], [0.0, scale]] }
    }

    /// Sets this matrix to zero.
    #[inline]
    pub fn set_zero(&mut self) {
        self.mat = [[0.0, 0.0], [0.0, 0.0]];
    }

    /// Sets this matrix to identity.
    #[inline]
    pub fn set_identity(&mut self) {
        self.mat = [[1.0, 0.0], [0.0, 1.0]];
    }

    /// Sets a diagonal matrix.
    #[inline]
    pub fn set_diagonal(&mut self, x1: f64, x2: f64) {
        self.mat[0][0] = x1;
        self.mat[1][1] = x2;
    }

    /// Sets a rotation matrix from angle in radians.
    #[inline]
    pub fn set_rotation(&mut self, angle: f64) {
        let cos = angle.cos();
        let sin = angle.sin();
        self.mat[0][0] = cos;
        self.mat[0][1] = sin;
        self.mat[1][0] = -sin;
        self.mat[1][1] = cos;
    }

    /// Sets a scaling matrix.
    #[inline]
    pub fn set_scale(&mut self, scale: f64) {
        self.mat[0][0] = scale;
        self.mat[1][1] = scale;
        self.mat[0][1] = 0.0;
        self.mat[1][0] = 0.0;
    }

    /// Gets a matrix element by row and column (1-indexed).
    #[inline]
    pub fn value(&self, row: usize, col: usize) -> f64 {
        assert!((1..=2).contains(&row), "Row {} out of range [1,2]", row);
        assert!((1..=2).contains(&col), "Col {} out of range [1,2]", col);
        self.mat[col - 1][row - 1]
    }

    /// Sets a matrix element by row and column (1-indexed).
    #[inline]
    pub fn set_value(&mut self, row: usize, col: usize, value: f64) {
        assert!((1..=2).contains(&row), "Row {} out of range [1,2]", row);
        assert!((1..=2).contains(&col), "Col {} out of range [1,2]", col);
        self.mat[col - 1][row - 1] = value;
    }

    /// Gets column by index (1-indexed).
    #[inline]
    pub fn column(&self, col: usize) -> XY {
        assert!((1..=2).contains(&col), "Column {} out of range [1,2]", col);
        XY::from_coords(self.mat[col - 1][0], self.mat[col - 1][1])
    }

    /// Sets column by index (1-indexed).
    #[inline]
    pub fn set_column(&mut self, col: usize, value: XY) {
        assert!((1..=2).contains(&col), "Column {} out of range [1,2]", col);
        let (x, y) = value.coords();
        self.mat[col - 1][0] = x;
        self.mat[col - 1][1] = y;
    }

    /// Sets both columns.
    #[inline]
    pub fn set_columns(&mut self, col1: XY, col2: XY) {
        let (x1, y1) = col1.coords();
        let (x2, y2) = col2.coords();
        self.mat[0][0] = x1;
        self.mat[0][1] = y1;
        self.mat[1][0] = x2;
        self.mat[1][1] = y2;
    }

    /// Gets row by index (1-indexed).
    #[inline]
    pub fn row(&self, row: usize) -> XY {
        assert!((1..=2).contains(&row), "Row {} out of range [1,2]", row);
        let r = row - 1;
        XY::from_coords(self.mat[0][r], self.mat[1][r])
    }

    /// Sets row by index (1-indexed).
    #[inline]
    pub fn set_row(&mut self, row: usize, value: XY) {
        assert!((1..=2).contains(&row), "Row {} out of range [1,2]", row);
        let r = row - 1;
        let (x, y) = value.coords();
        self.mat[0][r] = x;
        self.mat[1][r] = y;
    }

    /// Sets both rows.
    #[inline]
    pub fn set_rows(&mut self, row1: XY, row2: XY) {
        let (x1, y1) = row1.coords();
        let (x2, y2) = row2.coords();
        self.mat[0][0] = x1;
        self.mat[1][0] = y1;
        self.mat[0][1] = x2;
        self.mat[1][1] = y2;
    }

    /// Gets diagonal.
    #[inline]
    pub fn diagonal(&self) -> XY {
        XY::from_coords(self.mat[0][0], self.mat[1][1])
    }

    /// Computes the determinant.
    #[inline]
    pub const fn determinant(&self) -> f64 {
        self.mat[0][0] * self.mat[1][1] - self.mat[1][0] * self.mat[0][1]
    }

    /// Checks if the matrix is singular.
    #[inline]
    pub fn is_singular(&self) -> bool {
        let det = self.determinant().abs();
        det <= precision::CONFUSION
    }

    /// Adds another matrix in place.
    #[inline]
    pub fn add(&mut self, other: &Mat2d) {
        self.mat[0][0] += other.mat[0][0];
        self.mat[0][1] += other.mat[0][1];
        self.mat[1][0] += other.mat[1][0];
        self.mat[1][1] += other.mat[1][1];
    }

    /// Returns the sum of this and another matrix.
    #[inline]
    pub fn added(&self, other: &Mat2d) -> Mat2d {
        Mat2d {
            mat: [
                [self.mat[0][0] + other.mat[0][0], self.mat[0][1] + other.mat[0][1]],
                [self.mat[1][0] + other.mat[1][0], self.mat[1][1] + other.mat[1][1]],
            ],
        }
    }

    /// Subtracts another matrix in place.
    #[inline]
    pub fn subtract(&mut self, other: &Mat2d) {
        self.mat[0][0] -= other.mat[0][0];
        self.mat[0][1] -= other.mat[0][1];
        self.mat[1][0] -= other.mat[1][0];
        self.mat[1][1] -= other.mat[1][1];
    }

    /// Returns the difference of this and another matrix.
    #[inline]
    pub fn subtracted(&self, other: &Mat2d) -> Mat2d {
        Mat2d {
            mat: [
                [self.mat[0][0] - other.mat[0][0], self.mat[0][1] - other.mat[0][1]],
                [self.mat[1][0] - other.mat[1][0], self.mat[1][1] - other.mat[1][1]],
            ],
        }
    }

    /// Multiplies by a scalar in place.
    #[inline]
    pub fn multiply_scalar(&mut self, scalar: f64) {
        self.mat[0][0] *= scalar;
        self.mat[0][1] *= scalar;
        self.mat[1][0] *= scalar;
        self.mat[1][1] *= scalar;
    }

    /// Returns this multiplied by a scalar.
    #[inline]
    pub fn multiplied_scalar(&self, scalar: f64) -> Mat2d {
        Mat2d {
            mat: [
                [self.mat[0][0] * scalar, self.mat[0][1] * scalar],
                [self.mat[1][0] * scalar, self.mat[1][1] * scalar],
            ],
        }
    }

    /// Divides by a scalar in place.
    #[inline]
    pub fn divide(&mut self, scalar: f64) {
        self.mat[0][0] /= scalar;
        self.mat[0][1] /= scalar;
        self.mat[1][0] /= scalar;
        self.mat[1][1] /= scalar;
    }

    /// Returns this divided by a scalar.
    #[inline]
    pub fn divided(&self, scalar: f64) -> Mat2d {
        Mat2d {
            mat: [
                [self.mat[0][0] / scalar, self.mat[0][1] / scalar],
                [self.mat[1][0] / scalar, self.mat[1][1] / scalar],
            ],
        }
    }

    /// Multiplies by another matrix in place (this = this * other).
    #[inline]
    pub fn multiply(&mut self, other: &Mat2d) {
        let m00 = self.mat[0][0] * other.mat[0][0] + self.mat[1][0] * other.mat[0][1];
        let m01 = self.mat[0][1] * other.mat[0][0] + self.mat[1][1] * other.mat[0][1];
        let m10 = self.mat[0][0] * other.mat[1][0] + self.mat[1][0] * other.mat[1][1];
        let m11 = self.mat[0][1] * other.mat[1][0] + self.mat[1][1] * other.mat[1][1];
        self.mat[0] = [m00, m01];
        self.mat[1] = [m10, m11];
    }

    /// Returns the product of this and another matrix.
    #[inline]
    pub fn multiplied(&self, other: &Mat2d) -> Mat2d {
        let m00 = self.mat[0][0] * other.mat[0][0] + self.mat[1][0] * other.mat[0][1];
        let m01 = self.mat[0][1] * other.mat[0][0] + self.mat[1][1] * other.mat[0][1];
        let m10 = self.mat[0][0] * other.mat[1][0] + self.mat[1][0] * other.mat[1][1];
        let m11 = self.mat[0][1] * other.mat[1][0] + self.mat[1][1] * other.mat[1][1];
        Mat2d {
            mat: [[m00, m01], [m10, m11]],
        }
    }

    /// Transposes in place.
    #[inline]
    pub fn transpose(&mut self) {
        let temp = self.mat[0][1];
        self.mat[0][1] = self.mat[1][0];
        self.mat[1][0] = temp;
    }

    /// Returns the transpose.
    #[inline]
    pub fn transposed(&self) -> Mat2d {
        Mat2d {
            mat: [
                [self.mat[0][0], self.mat[1][0]],
                [self.mat[0][1], self.mat[1][1]],
            ],
        }
    }

    /// Inverts in place. Returns true if successful, false if singular.
    pub fn invert(&mut self) -> bool {
        let det = self.determinant();
        if det.abs() <= precision::CONFUSION {
            return false;
        }
        let inv_det = 1.0 / det;
        let m00 = self.mat[1][1] * inv_det;
        let m01 = -self.mat[0][1] * inv_det;
        let m10 = -self.mat[1][0] * inv_det;
        let m11 = self.mat[0][0] * inv_det;
        self.mat[0] = [m00, m01];
        self.mat[1] = [m10, m11];
        true
    }

    /// Returns the inverse. Returns None if singular.
    pub fn inverted(&self) -> Option<Mat2d> {
        let mut result = *self;
        if result.invert() {
            Some(result)
        } else {
            None
        }
    }

    /// Computes the product of this matrix with a 2D vector.
    #[inline]
    pub fn multiply_xy(&self, xy: XY) -> XY {
        let (x, y) = xy.coords();
        // Matrix-vector multiplication: [row0 · v, row1 · v]
        // mat[0] is row 0, mat[1] is row 1
        XY::from_coords(
            self.mat[0][0] * x + self.mat[0][1] * y,
            self.mat[1][0] * x + self.mat[1][1] * y,
        )
    }
}

// Operator implementations
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};

impl Add for Mat2d {
    type Output = Mat2d;
    #[inline]
    fn add(self, other: Mat2d) -> Mat2d {
        self.added(&other)
    }
}

impl AddAssign for Mat2d {
    #[inline]
    fn add_assign(&mut self, other: Mat2d) {
        self.add(&other);
    }
}

impl Sub for Mat2d {
    type Output = Mat2d;
    #[inline]
    fn sub(self, other: Mat2d) -> Mat2d {
        self.subtracted(&other)
    }
}

impl SubAssign for Mat2d {
    #[inline]
    fn sub_assign(&mut self, other: Mat2d) {
        self.subtract(&other);
    }
}

impl Mul<Mat2d> for Mat2d {
    type Output = Mat2d;
    #[inline]
    fn mul(self, other: Mat2d) -> Mat2d {
        self.multiplied(&other)
    }
}

impl MulAssign<Mat2d> for Mat2d {
    #[inline]
    fn mul_assign(&mut self, other: Mat2d) {
        self.multiply(&other);
    }
}

impl Mul<f64> for Mat2d {
    type Output = Mat2d;
    #[inline]
    fn mul(self, scalar: f64) -> Mat2d {
        self.multiplied_scalar(scalar)
    }
}

impl Mul<Mat2d> for f64 {
    type Output = Mat2d;
    #[inline]
    fn mul(self, mat: Mat2d) -> Mat2d {
        mat.multiplied_scalar(self)
    }
}

impl MulAssign<f64> for Mat2d {
    #[inline]
    fn mul_assign(&mut self, scalar: f64) {
        self.multiply_scalar(scalar);
    }
}

impl Div<f64> for Mat2d {
    type Output = Mat2d;
    #[inline]
    fn div(self, scalar: f64) -> Mat2d {
        self.divided(scalar)
    }
}

impl DivAssign<f64> for Mat2d {
    #[inline]
    fn div_assign(&mut self, scalar: f64) {
        self.divide(scalar);
    }
}

impl Mul<XY> for Mat2d {
    type Output = XY;
    #[inline]
    fn mul(self, xy: XY) -> XY {
        self.multiply_xy(xy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat2d_new() {
        let mat = Mat2d::new();
        assert_eq!(mat.determinant(), 0.0);
    }

    #[test]
    fn test_mat2d_identity() {
        let mat = Mat2d::identity();
        assert_eq!(mat.value(1, 1), 1.0);
        assert_eq!(mat.value(1, 2), 0.0);
        assert_eq!(mat.value(2, 1), 0.0);
        assert_eq!(mat.value(2, 2), 1.0);
        assert!((mat.determinant() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat2d_rotation() {
        let angle = std::f64::consts::PI / 4.0; // 45 degrees
        let mat = Mat2d::from_rotation(angle);
        let cos45 = angle.cos();
        let sin45 = angle.sin();
        // Storage: [[cos, -sin], [sin, cos]]
        // With value(r,c) = mat[c-1][r-1]:
        // value(1,1) = mat[0][0] = cos
        // value(1,2) = mat[1][0] = sin
        // value(2,1) = mat[0][1] = -sin
        // value(2,2) = mat[1][1] = cos
        assert!((mat.value(1, 1) - cos45).abs() < 1e-10);
        assert!((mat.value(1, 2) - sin45).abs() < 1e-10);  // +sin
        assert!((mat.value(2, 1) + sin45).abs() < 1e-10);  // -sin
        assert!((mat.value(2, 2) - cos45).abs() < 1e-10);
    }

    #[test]
    fn test_mat2d_scale() {
        let mat = Mat2d::from_scale(2.0, 3.0);
        assert_eq!(mat.value(1, 1), 2.0);
        assert_eq!(mat.value(2, 2), 3.0);
        assert_eq!(mat.value(1, 2), 0.0);
        assert_eq!(mat.value(2, 1), 0.0);
    }

    #[test]
    fn test_mat2d_determinant() {
        let mat = Mat2d::from_cols(
            XY::from_coords(1.0, 0.0),
            XY::from_coords(0.0, 1.0),
        );
        assert_eq!(mat.determinant(), 1.0);

        let mat2 = Mat2d::from_cols(
            XY::from_coords(2.0, 0.0),
            XY::from_coords(0.0, 3.0),
        );
        assert_eq!(mat2.determinant(), 6.0);
    }

    #[test]
    fn test_mat2d_transpose() {
        let mat = Mat2d::from_cols(
            XY::from_coords(1.0, 2.0),
            XY::from_coords(3.0, 4.0),
        );
        let t = mat.transposed();
        assert_eq!(t.value(1, 1), 1.0);
        assert_eq!(t.value(1, 2), 2.0);
        assert_eq!(t.value(2, 1), 3.0);
        assert_eq!(t.value(2, 2), 4.0);
    }

    #[test]
    fn test_mat2d_multiply_matrices() {
        let a = Mat2d::identity();
        let b = Mat2d::from_scale(2.0, 3.0);
        let c = a.multiplied(&b);
        assert_eq!(c.value(1, 1), 2.0);
        assert_eq!(c.value(2, 2), 3.0);
    }

    #[test]
    fn test_mat2d_multiply_xy() {
        let mat = Mat2d::from_scale(2.0, 3.0);
        let xy = XY::from_coords(1.0, 1.0);
        let result = mat.multiply_xy(xy);
        assert_eq!(result.x(), 2.0);
        assert_eq!(result.y(), 3.0);
    }

    #[test]
    fn test_mat2d_invert() {
        let mat = Mat2d::from_cols(
            XY::from_coords(1.0, 0.0),
            XY::from_coords(0.0, 2.0),
        );
        let inv = mat.inverted().unwrap();
        assert_eq!(inv.value(1, 1), 1.0);
        assert_eq!(inv.value(2, 2), 0.5);

        // Verify M * M^-1 = I
        let product = mat.multiplied(&inv);
        assert!((product.value(1, 1) - 1.0).abs() < 1e-10);
        assert!((product.value(2, 2) - 1.0).abs() < 1e-10);
        assert!(product.value(1, 2).abs() < 1e-10);
        assert!(product.value(2, 1).abs() < 1e-10);
    }

    #[test]
    fn test_mat2d_singular() {
        let mat = Mat2d::from_cols(
            XY::from_coords(1.0, 2.0),
            XY::from_coords(2.0, 4.0),
        );
        assert!(mat.is_singular());
        assert!(mat.inverted().is_none());
    }
}
