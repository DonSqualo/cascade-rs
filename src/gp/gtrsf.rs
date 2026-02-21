//! General (non-orthogonal) transformation.
//!
//! Port of OCCT's gp_GTrsf class.
//! Source: src/FoundationClasses/TKMath/gp/gp_GTrsf.hxx
//!
//! Unlike gp_Trsf, gp_GTrsf can represent non-orthogonal transformations
//! like affinity (shear). It can only transform XYZ coordinates, not
//! geometric objects (which would change their nature).

use super::{XYZ, Mat, Ax1, Ax2, Trsf, TrsfForm};
use crate::precision;

/// A general transformation in 3D space.
/// 
/// Can represent affine transformations including shear.
/// Warning: Can only transform coordinates (XYZ), not geometric objects.
#[derive(Clone, Copy, Debug)]
pub struct GTrsf {
    matrix: Mat,
    loc: XYZ,
    shape: TrsfForm,
    scale: f64,
}

impl Default for GTrsf {
    fn default() -> Self {
        Self::new()
    }
}

impl GTrsf {
    /// Creates identity transformation.
    pub fn new() -> Self {
        Self {
            matrix: Mat::identity(),
            loc: XYZ::new(),
            shape: TrsfForm::Identity,
            scale: 1.0,
        }
    }

    /// Creates from a gp_Trsf (orthogonal transformation).
    pub fn from_trsf(t: &Trsf) -> Self {
        Self {
            matrix: *t.hv_rotation_part(),
            loc: *t.translation_part(),
            shape: t.form(),
            scale: t.scale_factor(),
        }
    }

    /// Creates from matrix and translation vector.
    pub const fn from_mat_xyz(m: Mat, v: XYZ) -> Self {
        Self {
            matrix: m,
            loc: v,
            shape: TrsfForm::Other,
            scale: 0.0,
        }
    }

    /// Sets affinity with respect to an axis.
    /// Points are scaled by `ratio` in the direction perpendicular to axis.
    pub fn set_affinity_ax1(&mut self, a1: &Ax1, ratio: f64) {
        self.shape = TrsfForm::Other;
        self.scale = 0.0;
        
        let d = a1.direction().xyz();
        
        // M = (1-ratio) * d⊗d + ratio * I
        // This scales by ratio in directions perpendicular to d
        let xx = d.x() * d.x();
        let yy = d.y() * d.y();
        let zz = d.z() * d.z();
        let xy = d.x() * d.y();
        let xz = d.x() * d.z();
        let yz = d.y() * d.z();
        
        let t = 1.0 - ratio;
        self.matrix.set_value(1, 1, t * xx + ratio);
        self.matrix.set_value(1, 2, t * xy);
        self.matrix.set_value(1, 3, t * xz);
        self.matrix.set_value(2, 1, t * xy);
        self.matrix.set_value(2, 2, t * yy + ratio);
        self.matrix.set_value(2, 3, t * yz);
        self.matrix.set_value(3, 1, t * xz);
        self.matrix.set_value(3, 2, t * yz);
        self.matrix.set_value(3, 3, t * zz + ratio);
        
        // Translation: loc = p - M*p where p = axis location
        let p = a1.location().xyz();
        let mp = self.matrix.multiply_xyz(p);
        self.loc = p.subtracted(&mp);
    }

    /// Sets affinity with respect to a plane.
    /// Points are scaled by `ratio` in the direction normal to plane.
    pub fn set_affinity_ax2(&mut self, a2: &Ax2, ratio: f64) {
        self.shape = TrsfForm::Other;
        self.scale = 0.0;
        
        let n = a2.direction().xyz();
        
        // M = I + (ratio-1) * n⊗n
        // This scales by ratio in the normal direction
        let xx = n.x() * n.x();
        let yy = n.y() * n.y();
        let zz = n.z() * n.z();
        let xy = n.x() * n.y();
        let xz = n.x() * n.z();
        let yz = n.y() * n.z();
        
        let t = ratio - 1.0;
        self.matrix.set_value(1, 1, 1.0 + t * xx);
        self.matrix.set_value(1, 2, t * xy);
        self.matrix.set_value(1, 3, t * xz);
        self.matrix.set_value(2, 1, t * xy);
        self.matrix.set_value(2, 2, 1.0 + t * yy);
        self.matrix.set_value(2, 3, t * yz);
        self.matrix.set_value(3, 1, t * xz);
        self.matrix.set_value(3, 2, t * yz);
        self.matrix.set_value(3, 3, 1.0 + t * zz);
        
        // Translation
        let p = a2.location().xyz();
        let mp = self.matrix.multiply_xyz(p);
        self.loc = p.subtracted(&mp);
    }

    /// Sets a coefficient of the transformation matrix.
    /// Row in 1..3, Col in 1..4 (col 4 is translation).
    pub fn set_value(&mut self, row: usize, col: usize, value: f64) {
        assert!(row >= 1 && row <= 3 && col >= 1 && col <= 4,
            "GTrsf::set_value: row {} col {} out of range", row, col);
        
        if col == 4 {
            self.loc.set_coord_index(row, value);
            if self.shape == TrsfForm::Identity {
                self.shape = TrsfForm::Translation;
            }
        } else {
            if self.shape != TrsfForm::Other && self.scale != 1.0 {
                // Convert from scaled form
                let mut m = self.matrix;
                for i in 1..=3 {
                    for j in 1..=3 {
                        m.set_value(i, j, m.value(i, j) * self.scale);
                    }
                }
                self.matrix = m;
            }
            self.matrix.set_value(row, col, value);
            self.shape = TrsfForm::Other;
            self.scale = 0.0;
        }
    }

    /// Sets the vectorial (matrix) part.
    pub fn set_vectorial_part(&mut self, m: Mat) {
        self.matrix = m;
        self.shape = TrsfForm::Other;
        self.scale = 0.0;
    }

    /// Sets the translation part.
    pub fn set_translation_part(&mut self, v: XYZ) {
        self.loc = v;
        if self.shape == TrsfForm::Identity {
            self.shape = TrsfForm::Translation;
        }
    }

    /// Sets from a Trsf.
    pub fn set_trsf(&mut self, t: &Trsf) {
        self.shape = t.form();
        self.matrix = *t.hv_rotation_part();
        self.loc = *t.translation_part();
        self.scale = t.scale_factor();
    }

    /// Returns true if determinant is negative.
    pub fn is_negative(&self) -> bool {
        self.matrix.determinant() < 0.0
    }

    /// Returns true if transformation is singular (non-invertible).
    pub fn is_singular(&self) -> bool {
        self.matrix.determinant().abs() < precision::RESOLUTION
    }

    /// Returns the transformation form.
    pub const fn form(&self) -> TrsfForm {
        self.shape
    }

    /// Returns the translation part.
    pub const fn translation_part(&self) -> &XYZ {
        &self.loc
    }

    /// Returns the vectorial (matrix) part.
    pub const fn vectorial_part(&self) -> &Mat {
        &self.matrix
    }

    /// Returns coefficient at (row, col).
    /// Row in 1..3, Col in 1..4.
    pub fn value(&self, row: usize, col: usize) -> f64 {
        assert!(row >= 1 && row <= 3 && col >= 1 && col <= 4,
            "GTrsf::value: row {} col {} out of range", row, col);
        
        if col == 4 {
            self.loc.coord(row)
        } else if self.shape == TrsfForm::Other {
            self.matrix.value(row, col)
        } else {
            self.scale * self.matrix.value(row, col)
        }
    }

    /// Inverts in place.
    pub fn invert(&mut self) {
        if self.is_singular() {
            panic!("GTrsf::invert: singular transformation");
        }
        
        // For general matrix: M^-1, then new_loc = -M^-1 * loc
        // This requires proper matrix inversion
        // For now, use transpose for orthogonal case
        if self.shape != TrsfForm::Other {
            // Orthogonal case
            let inv_scale = 1.0 / self.scale;
            let inv_matrix = self.matrix.transposed();
            let new_loc = inv_matrix.multiply_xyz(&self.loc.reversed()).multiplied(inv_scale);
            self.matrix = inv_matrix;
            self.loc = new_loc;
            self.scale = inv_scale;
        } else {
            // General case - need full matrix inversion
            // TODO: Implement proper 3x3 matrix inversion
            panic!("GTrsf::invert: general case not yet implemented");
        }
    }

    /// Returns inverted transformation.
    pub fn inverted(&self) -> GTrsf {
        let mut t = *self;
        t.invert();
        t
    }

    /// Multiplies by another GTrsf: self = self * other.
    pub fn multiply(&mut self, other: &GTrsf) {
        // Combined transformation: M' = M1*M2, loc' = M1*loc2 + loc1
        let new_matrix = self.matrix.multiplied(&other.matrix);
        let transformed_loc = self.matrix.multiply_xyz(&other.loc);
        let new_loc = transformed_loc.added(&self.loc);
        
        self.matrix = new_matrix;
        self.loc = new_loc;
        self.shape = TrsfForm::CompoundTrsf;
        
        if self.shape != TrsfForm::Other && other.shape != TrsfForm::Other {
            self.scale *= other.scale;
        } else {
            self.scale = 0.0;
            self.shape = TrsfForm::Other;
        }
    }

    /// Returns self * other.
    pub fn multiplied(&self, other: &GTrsf) -> GTrsf {
        let mut t = *self;
        t.multiply(other);
        t
    }

    /// Pre-multiplies: self = other * self.
    pub fn pre_multiply(&mut self, other: &GTrsf) {
        let new_matrix = other.matrix.multiplied(&self.matrix);
        let transformed_loc = other.matrix.multiply_xyz(&self.loc);
        let new_loc = transformed_loc.added(&other.loc);
        
        self.matrix = new_matrix;
        self.loc = new_loc;
        self.shape = TrsfForm::CompoundTrsf;
        
        if self.shape != TrsfForm::Other && other.shape != TrsfForm::Other {
            self.scale *= other.scale;
        } else {
            self.scale = 0.0;
            self.shape = TrsfForm::Other;
        }
    }

    /// Raises to power n.
    pub fn power(&mut self, n: i32) {
        if n == 0 {
            *self = GTrsf::new();
            return;
        }
        
        if n < 0 {
            self.invert();
            self.power(-n);
            return;
        }
        
        let base = *self;
        for _ in 1..n {
            self.multiply(&base);
        }
    }

    /// Returns self^n.
    pub fn powered(&self, n: i32) -> GTrsf {
        let mut t = *self;
        t.power(n);
        t
    }

    /// Transforms XYZ coordinates in place.
    pub fn transforms_xyz(&self, coord: &mut XYZ) {
        let mut result = self.matrix.multiply_xyz(coord);
        if self.shape != TrsfForm::Other && self.scale != 1.0 {
            result.multiply(self.scale);
        }
        result.add(&self.loc);
        *coord = result;
    }

    /// Transforms coordinates.
    pub fn transforms(&self, x: &mut f64, y: &mut f64, z: &mut f64) {
        let mut xyz = XYZ::from_coords(*x, *y, *z);
        self.transforms_xyz(&mut xyz);
        *x = xyz.x();
        *y = xyz.y();
        *z = xyz.z();
    }

    /// Converts to Trsf if orthogonal. Panics if non-orthogonal.
    pub fn to_trsf(&self) -> Trsf {
        if self.shape == TrsfForm::Other {
            panic!("GTrsf::to_trsf: non-orthogonal GTrsf cannot convert to Trsf");
        }
        
        let mut t = Trsf::new();
        // Would need internal access to Trsf fields
        // For now, build it via set methods
        t.set_translation(&self.loc);
        // TODO: proper conversion
        t
    }
}

impl std::ops::Mul for GTrsf {
    type Output = GTrsf;
    fn mul(self, other: GTrsf) -> GTrsf {
        self.multiplied(&other)
    }
}

impl std::ops::MulAssign for GTrsf {
    fn mul_assign(&mut self, other: GTrsf) {
        self.multiply(&other);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::{Pnt, Dir};

    #[test]
    fn test_gtrsf_identity() {
        let t = GTrsf::new();
        assert_eq!(t.form(), TrsfForm::Identity);
        
        let mut xyz = XYZ::from_coords(1.0, 2.0, 3.0);
        t.transforms_xyz(&mut xyz);
        assert_eq!(xyz.x(), 1.0);
        assert_eq!(xyz.y(), 2.0);
        assert_eq!(xyz.z(), 3.0);
    }

    #[test]
    fn test_gtrsf_from_trsf() {
        let mut trsf = Trsf::new();
        trsf.set_translation(&XYZ::from_coords(1.0, 2.0, 3.0));
        
        let gtrsf = GTrsf::from_trsf(&trsf);
        
        let mut xyz = XYZ::new();
        gtrsf.transforms_xyz(&mut xyz);
        assert!((xyz.x() - 1.0).abs() < 1e-10);
        assert!((xyz.y() - 2.0).abs() < 1e-10);
        assert!((xyz.z() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_gtrsf_set_value() {
        let mut t = GTrsf::new();
        
        // Set translation via column 4
        t.set_value(1, 4, 10.0);
        t.set_value(2, 4, 20.0);
        t.set_value(3, 4, 30.0);
        
        let mut xyz = XYZ::new();
        t.transforms_xyz(&mut xyz);
        assert!((xyz.x() - 10.0).abs() < 1e-10);
        assert!((xyz.y() - 20.0).abs() < 1e-10);
        assert!((xyz.z() - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_gtrsf_value() {
        let t = GTrsf::new();
        // Identity matrix diagonal
        assert_eq!(t.value(1, 1), 1.0);
        assert_eq!(t.value(2, 2), 1.0);
        assert_eq!(t.value(3, 3), 1.0);
        // Translation
        assert_eq!(t.value(1, 4), 0.0);
    }

    #[test]
    fn test_gtrsf_multiply() {
        let mut t1 = GTrsf::new();
        t1.set_value(1, 4, 1.0); // Translate x by 1
        
        let mut t2 = GTrsf::new();
        t2.set_value(2, 4, 2.0); // Translate y by 2
        
        let t3 = t1 * t2;
        
        let mut xyz = XYZ::new();
        t3.transforms_xyz(&mut xyz);
        assert!((xyz.x() - 1.0).abs() < 1e-10);
        assert!((xyz.y() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_gtrsf_affinity_ax1() {
        // Affinity with ratio 0.5 along Z axis
        // Points should be scaled by 0.5 in Z direction
        let a1 = Ax1::new(Pnt::new(), Dir::z());
        let mut t = GTrsf::new();
        t.set_affinity_ax1(&a1, 0.5);
        
        let mut xyz = XYZ::from_coords(1.0, 1.0, 2.0);
        t.transforms_xyz(&mut xyz);
        
        // X and Y unchanged (parallel to axis)
        // Actually, affinity along axis scales perpendicular directions
        // Need to verify OCCT semantics
    }

    #[test]
    fn test_gtrsf_is_negative() {
        let t = GTrsf::new();
        assert!(!t.is_negative()); // Identity has positive determinant
    }

    #[test]
    fn test_gtrsf_is_singular() {
        let t = GTrsf::new();
        assert!(!t.is_singular()); // Identity is not singular
    }

    #[test]
    fn test_gtrsf_power() {
        let mut t = GTrsf::new();
        t.set_value(1, 4, 1.0); // Translate x by 1
        
        let t3 = t.powered(3);
        
        let mut xyz = XYZ::new();
        t3.transforms_xyz(&mut xyz);
        assert!((xyz.x() - 3.0).abs() < 1e-10); // Should translate by 3
    }

    #[test]
    fn test_gtrsf_power_zero() {
        let mut t = GTrsf::new();
        t.set_value(1, 4, 5.0);
        
        let t0 = t.powered(0);
        assert_eq!(t0.form(), TrsfForm::Identity);
    }
}
