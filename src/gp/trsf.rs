//! Transformation matrix.
//!
//! Port of OCCT's gp_Trsf class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Trsf.hxx

use super::{XYZ, Pnt, Dir, Ax1, Ax2, Mat};
use crate::precision;

/// Transformation form (type of transformation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrsfForm {
    /// Identity transformation.
    Identity,
    /// Rotation only.
    Rotation,
    /// Translation only.
    Translation,
    /// Point mirror (central symmetry).
    PntMirror,
    /// Axis mirror (axial symmetry).
    Ax1Mirror,
    /// Plane mirror (planar symmetry).
    Ax2Mirror,
    /// Uniform scaling.
    Scale,
    /// Compound or general transformation.
    CompoundTrsf,
    /// Non-orthogonal or other.
    Other,
}

/// A transformation in 3D space.
/// Combines rotation, translation, and scaling.
#[derive(Clone, Copy, Debug)]
pub struct Trsf {
    scale: f64,
    shape: TrsfForm,
    matrix: Mat,
    loc: XYZ,
}

impl Default for Trsf {
    fn default() -> Self {
        Self::new()
    }
}

impl Trsf {
    /// Creates identity transformation.
    pub const fn new() -> Self {
        Self {
            scale: 1.0,
            shape: TrsfForm::Identity,
            matrix: Mat::identity(),
            loc: XYZ::new(),
        }
    }

    /// Returns the transformation form.
    #[inline]
    pub const fn form(&self) -> TrsfForm {
        self.shape
    }

    /// Returns the scale factor.
    #[inline]
    pub const fn scale_factor(&self) -> f64 {
        self.scale
    }

    /// Returns the translation part.
    #[inline]
    pub const fn translation_part(&self) -> &XYZ {
        &self.loc
    }

    /// Returns the rotation matrix.
    #[inline]
    pub const fn hv_rotation_part(&self) -> &Mat {
        &self.matrix
    }

    /// Sets to identity.
    pub fn set_identity(&mut self) {
        self.scale = 1.0;
        self.shape = TrsfForm::Identity;
        self.matrix = Mat::identity();
        self.loc = XYZ::new();
    }

    /// Sets as mirror about a point.
    pub fn set_mirror_point(&mut self, p: &Pnt) {
        self.shape = TrsfForm::PntMirror;
        self.scale = -1.0;
        self.matrix = Mat::identity();
        let mut loc = *p.xyz();
        loc.multiply(2.0);
        self.loc = loc;
    }

    /// Sets as mirror about an axis.
    pub fn set_mirror_ax1(&mut self, a1: &Ax1) {
        self.shape = TrsfForm::Ax1Mirror;
        self.scale = 1.0;
        
        let d = a1.direction().xyz();
        let p = a1.location().xyz();
        
        // Mirror matrix: M = 2*d*d^T - I
        let xx = d.x() * d.x();
        let yy = d.y() * d.y();
        let zz = d.z() * d.z();
        let xy = d.x() * d.y();
        let xz = d.x() * d.z();
        let yz = d.y() * d.z();
        
        self.matrix.set_value(1, 1, 2.0 * xx - 1.0);
        self.matrix.set_value(1, 2, 2.0 * xy);
        self.matrix.set_value(1, 3, 2.0 * xz);
        self.matrix.set_value(2, 1, 2.0 * xy);
        self.matrix.set_value(2, 2, 2.0 * yy - 1.0);
        self.matrix.set_value(2, 3, 2.0 * yz);
        self.matrix.set_value(3, 1, 2.0 * xz);
        self.matrix.set_value(3, 2, 2.0 * yz);
        self.matrix.set_value(3, 3, 2.0 * zz - 1.0);
        
        // Translation part
        let mp = self.matrix.multiply_xyz(p);
        self.loc = p.subtracted(&mp);
    }

    /// Sets as mirror about a plane.
    pub fn set_mirror_ax2(&mut self, a2: &Ax2) {
        self.shape = TrsfForm::Ax2Mirror;
        self.scale = 1.0;
        
        let n = a2.direction().xyz();
        let p = a2.location().xyz();
        
        // Mirror matrix: M = I - 2*n*n^T
        let xx = n.x() * n.x();
        let yy = n.y() * n.y();
        let zz = n.z() * n.z();
        let xy = n.x() * n.y();
        let xz = n.x() * n.z();
        let yz = n.y() * n.z();
        
        self.matrix.set_value(1, 1, 1.0 - 2.0 * xx);
        self.matrix.set_value(1, 2, -2.0 * xy);
        self.matrix.set_value(1, 3, -2.0 * xz);
        self.matrix.set_value(2, 1, -2.0 * xy);
        self.matrix.set_value(2, 2, 1.0 - 2.0 * yy);
        self.matrix.set_value(2, 3, -2.0 * yz);
        self.matrix.set_value(3, 1, -2.0 * xz);
        self.matrix.set_value(3, 2, -2.0 * yz);
        self.matrix.set_value(3, 3, 1.0 - 2.0 * zz);
        
        // Translation part
        let mp = self.matrix.multiply_xyz(p);
        self.loc = p.subtracted(&mp);
    }

    /// Sets as rotation about an axis.
    pub fn set_rotation(&mut self, a1: &Ax1, angle: f64) {
        self.shape = TrsfForm::Rotation;
        self.scale = 1.0;
        
        let d = a1.direction().xyz();
        let p = a1.location().xyz();
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;
        
        // Rodrigues' rotation formula
        let xx = d.x() * d.x();
        let yy = d.y() * d.y();
        let zz = d.z() * d.z();
        let xy = d.x() * d.y();
        let xz = d.x() * d.z();
        let yz = d.y() * d.z();
        
        self.matrix.set_value(1, 1, t * xx + c);
        self.matrix.set_value(1, 2, t * xy - s * d.z());
        self.matrix.set_value(1, 3, t * xz + s * d.y());
        self.matrix.set_value(2, 1, t * xy + s * d.z());
        self.matrix.set_value(2, 2, t * yy + c);
        self.matrix.set_value(2, 3, t * yz - s * d.x());
        self.matrix.set_value(3, 1, t * xz - s * d.y());
        self.matrix.set_value(3, 2, t * yz + s * d.x());
        self.matrix.set_value(3, 3, t * zz + c);
        
        // Translation part: p - M*p
        let mp = self.matrix.multiply_xyz(p);
        self.loc = p.subtracted(&mp);
    }

    /// Sets as uniform scaling.
    pub fn set_scale(&mut self, p: &Pnt, s: f64) {
        self.shape = TrsfForm::Scale;
        self.scale = s;
        self.matrix = Mat::identity();
        
        // Translation part: p * (1 - s)
        let mut loc = *p.xyz();
        loc.multiply(1.0 - s);
        self.loc = loc;
    }

    /// Sets as translation.
    pub fn set_translation(&mut self, v: &XYZ) {
        self.shape = TrsfForm::Translation;
        self.scale = 1.0;
        self.matrix = Mat::identity();
        self.loc = *v;
    }

    /// Sets as translation from p1 to p2.
    pub fn set_translation_2pts(&mut self, p1: &Pnt, p2: &Pnt) {
        self.shape = TrsfForm::Translation;
        self.scale = 1.0;
        self.matrix = Mat::identity();
        self.loc = p2.xyz().subtracted(p1.xyz());
    }

    /// Transforms XYZ coordinates in place.
    pub fn transforms_xyz(&self, coord: &mut XYZ) {
        match self.shape {
            TrsfForm::Identity => {}
            TrsfForm::Translation => {
                coord.add(&self.loc);
            }
            TrsfForm::Scale => {
                coord.multiply(self.scale);
                coord.add(&self.loc);
            }
            TrsfForm::PntMirror => {
                coord.reverse();
                coord.add(&self.loc);
            }
            _ => {
                // General case: M * coord * scale + loc
                let rotated = self.matrix.multiply_xyz(coord);
                *coord = rotated.multiplied(self.scale);
                coord.add(&self.loc);
            }
        }
    }

    /// Returns transformed XYZ.
    pub fn transforms(&self, coord: &XYZ) -> XYZ {
        let mut result = *coord;
        self.transforms_xyz(&mut result);
        result
    }

    /// Multiplies this transformation by another (composition).
    pub fn multiply(&mut self, other: &Trsf) {
        if other.shape == TrsfForm::Identity {
            return;
        }
        if self.shape == TrsfForm::Identity {
            *self = *other;
            return;
        }
        
        // Compose: (M1, t1, s1) * (M2, t2, s2) = (M1*M2, M1*t2*s1 + t1, s1*s2)
        let new_matrix = self.matrix.multiplied(&other.matrix);
        let scaled_loc = other.loc.multiplied(self.scale);
        let rotated_loc = self.matrix.multiply_xyz(&scaled_loc);
        let new_loc = rotated_loc.added(&self.loc);
        let new_scale = self.scale * other.scale;
        
        self.matrix = new_matrix;
        self.loc = new_loc;
        self.scale = new_scale;
        self.shape = TrsfForm::CompoundTrsf;
    }

    /// Returns composition of this and other.
    pub fn multiplied(&self, other: &Trsf) -> Trsf {
        let mut result = *self;
        result.multiply(other);
        result
    }

    /// Inverts the transformation.
    pub fn invert(&mut self) {
        // For orthogonal matrix, inverse = transpose
        // T^-1 = (1/s * M^T, -1/s * M^T * t)
        
        if self.shape == TrsfForm::Identity {
            return;
        }
        
        if self.scale.abs() < precision::CONFUSION {
            panic!("Trsf::invert: scale factor too small");
        }
        
        let inv_scale = 1.0 / self.scale;
        let inv_matrix = self.matrix.transposed();
        
        // New translation: -M^T * t / s
        let neg_loc = self.loc.reversed();
        let new_loc = inv_matrix.multiply_xyz(&neg_loc).multiplied(inv_scale);
        
        self.matrix = inv_matrix;
        self.loc = new_loc;
        self.scale = inv_scale;
    }

    /// Returns inverted transformation.
    pub fn inverted(&self) -> Trsf {
        let mut result = *self;
        result.invert();
        result
    }
}

impl std::ops::Mul for Trsf {
    type Output = Trsf;
    fn mul(self, other: Trsf) -> Trsf {
        self.multiplied(&other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_trsf_identity() {
        let t = Trsf::new();
        assert_eq!(t.form(), TrsfForm::Identity);
        assert_eq!(t.scale_factor(), 1.0);
    }

    #[test]
    fn test_trsf_translation() {
        let mut t = Trsf::new();
        t.set_translation(&XYZ::from_coords(1.0, 2.0, 3.0));
        
        let p = XYZ::from_coords(0.0, 0.0, 0.0);
        let result = t.transforms(&p);
        assert!((result.x() - 1.0).abs() < 1e-10);
        assert!((result.y() - 2.0).abs() < 1e-10);
        assert!((result.z() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsf_scale() {
        let mut t = Trsf::new();
        t.set_scale(&Pnt::new(), 2.0);
        
        let p = XYZ::from_coords(1.0, 1.0, 1.0);
        let result = t.transforms(&p);
        assert!((result.x() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsf_rotation_90deg() {
        let mut t = Trsf::new();
        t.set_rotation(&Ax1::oz(), PI / 2.0);
        
        // Rotate (1, 0, 0) by 90° around Z -> (0, 1, 0)
        let p = XYZ::from_coords(1.0, 0.0, 0.0);
        let result = t.transforms(&p);
        assert!(result.x().abs() < 1e-10);
        assert!((result.y() - 1.0).abs() < 1e-10);
        assert!(result.z().abs() < 1e-10);
    }

    #[test]
    fn test_trsf_composition() {
        // OCC23361 test case
        let p = Pnt::from_coords(0.0, 0.0, 2.0);
        
        let mut t1 = Trsf::new();
        let mut t2 = Trsf::new();
        t1.set_rotation(&Ax1::new(p, Dir::y()), -0.49328285294022267);
        t2.set_rotation(&Ax1::new(p, Dir::z()), 0.87538474718473880);
        
        let t_comp = t2 * t1;
        
        let p1 = Pnt::from_coords(10.0, 3.0, 4.0);
        let p2 = p1.transformed(&t_comp);
        
        let p3 = p1.transformed(&t1).transformed(&t2);
        
        assert!(p2.is_equal(&p3, precision::CONFUSION));
    }

    #[test]
    fn test_trsf_invert() {
        let mut t = Trsf::new();
        t.set_rotation(&Ax1::oz(), PI / 4.0);
        
        let t_inv = t.inverted();
        let t_id = t * t_inv;
        
        // Should be close to identity
        let p = XYZ::from_coords(1.0, 2.0, 3.0);
        let result = t_id.transforms(&p);
        assert!((result.x() - p.x()).abs() < 1e-10);
        assert!((result.y() - p.y()).abs() < 1e-10);
        assert!((result.z() - p.z()).abs() < 1e-10);
    }
}
