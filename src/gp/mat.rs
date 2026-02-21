//! 3x3 matrix.
//!
//! Port of OCCT's gp_Mat class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Mat.hxx

use super::XYZ;

/// 3x3 matrix for transformations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat {
    // Column-major storage (matching OCCT)
    data: [[f64; 3]; 3],
}

impl Default for Mat {
    fn default() -> Self {
        Self::identity()
    }
}

impl Mat {
    /// Creates identity matrix.
    pub const fn identity() -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
    }

    /// Creates zero matrix.
    pub const fn zero() -> Self {
        Self {
            data: [[0.0; 3]; 3],
        }
    }

    /// Gets value at (row, col). 1-indexed like OCCT.
    pub fn value(&self, row: usize, col: usize) -> f64 {
        self.data[col - 1][row - 1]
    }

    /// Sets value at (row, col). 1-indexed like OCCT.
    pub fn set_value(&mut self, row: usize, col: usize, value: f64) {
        self.data[col - 1][row - 1] = value;
    }

    /// Multiplies matrix by XYZ (column vector).
    pub fn multiply_xyz(&self, xyz: &XYZ) -> XYZ {
        XYZ::from_coords(
            self.data[0][0] * xyz.x() + self.data[1][0] * xyz.y() + self.data[2][0] * xyz.z(),
            self.data[0][1] * xyz.x() + self.data[1][1] * xyz.y() + self.data[2][1] * xyz.z(),
            self.data[0][2] * xyz.x() + self.data[1][2] * xyz.y() + self.data[2][2] * xyz.z(),
        )
    }

    /// Computes determinant.
    pub fn determinant(&self) -> f64 {
        let m = &self.data;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1])
            + m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1])
    }

    /// Transposes in place.
    pub fn transpose(&mut self) {
        let tmp01 = self.data[0][1];
        let tmp02 = self.data[0][2];
        let tmp12 = self.data[1][2];
        self.data[0][1] = self.data[1][0];
        self.data[0][2] = self.data[2][0];
        self.data[1][2] = self.data[2][1];
        self.data[1][0] = tmp01;
        self.data[2][0] = tmp02;
        self.data[2][1] = tmp12;
    }

    /// Returns transposed matrix.
    pub fn transposed(&self) -> Mat {
        let mut result = *self;
        result.transpose();
        result
    }

    /// Sets rotation around X axis.
    pub fn set_rotation_x(&mut self, angle: f64) {
        let c = angle.cos();
        let s = angle.sin();
        self.data = [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ];
    }

    /// Sets rotation around Y axis.
    pub fn set_rotation_y(&mut self, angle: f64) {
        let c = angle.cos();
        let s = angle.sin();
        self.data = [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ];
    }

    /// Sets rotation around Z axis.
    pub fn set_rotation_z(&mut self, angle: f64) {
        let c = angle.cos();
        let s = angle.sin();
        self.data = [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ];
    }

    /// Matrix multiplication.
    pub fn multiplied(&self, other: &Mat) -> Mat {
        let mut result = Mat::zero();
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += self.data[k][i] * other.data[j][k];
                }
                result.data[j][i] = sum;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_identity() {
        let m = Mat::identity();
        assert_eq!(m.value(1, 1), 1.0);
        assert_eq!(m.value(2, 2), 1.0);
        assert_eq!(m.value(3, 3), 1.0);
        assert_eq!(m.value(1, 2), 0.0);
    }

    #[test]
    fn test_mat_determinant_identity() {
        let m = Mat::identity();
        assert!((m.determinant() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat_multiply_xyz() {
        let m = Mat::identity();
        let xyz = XYZ::from_coords(1.0, 2.0, 3.0);
        let result = m.multiply_xyz(&xyz);
        assert_eq!(result.x(), 1.0);
        assert_eq!(result.y(), 2.0);
        assert_eq!(result.z(), 3.0);
    }
}
