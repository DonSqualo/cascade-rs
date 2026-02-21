//! 2D affine transformation.
//!
//! Port of OCCT's gp_Trsf2d class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Trsf2d.hxx

use crate::gp::{Mat2d, XY, Pnt2d, Vec2d, Dir2d, Ax2d, Ax22d};

/// Represents a 2D affine transformation: P' = M * P + T
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trsf2d {
    scale: f64,
    matrix: Mat2d,
    translation: XY,
}

impl Trsf2d {
    /// Creates the identity transformation.
    #[inline]
    pub fn identity() -> Self {
        Self {
            scale: 1.0,
            matrix: Mat2d::identity(),
            translation: XY::new(),
        }
    }

    /// Creates a translation.
    #[inline]
    pub fn from_translation(vec: Vec2d) -> Self {
        Self {
            scale: 1.0,
            matrix: Mat2d::identity(),
            translation: vec.xy(),
        }
    }

    /// Creates a rotation about the origin.
    #[inline]
    pub fn from_rotation(angle: f64) -> Self {
        Self {
            scale: 1.0,
            matrix: Mat2d::from_rotation(angle),
            translation: XY::new(),
        }
    }

    /// Creates a rotation about a point.
    #[inline]
    pub fn from_rotation_point(center: Pnt2d, angle: f64) -> Self {
        let mut trsf = Self::from_rotation(angle);
        // Translate point to origin, rotate, translate back
        let c = center.xy();
        let neg_c = c.reversed();
        let rotated = trsf.matrix.multiply_xy(neg_c);
        trsf.translation = rotated.added(&c);
        trsf
    }

    /// Creates a scaling about the origin.
    #[inline]
    pub fn from_scale(factor: f64) -> Self {
        Self {
            scale: factor,
            matrix: Mat2d::from_uniform_scale(factor),
            translation: XY::new(),
        }
    }

    /// Creates a scaling about a point.
    #[inline]
    pub fn from_scale_point(center: Pnt2d, factor: f64) -> Self {
        let mut trsf = Self::from_scale(factor);
        let c = center.xy();
        let neg_c = c.reversed().multiplied(factor - 1.0);
        trsf.translation = neg_c;
        trsf
    }

    /// Creates a transformation from an axis and an angle.
    #[inline]
    pub fn from_axis(axis: Ax2d, angle: f64) -> Self {
        let origin = axis.origin();
        let dir = axis.direction();
        
        // Build transformation: translate to origin, rotate, translate back
        let mut trsf = Self::from_rotation(angle);
        let neg_origin = origin.xy().reversed();
        let rotated = trsf.matrix.multiply_xy(neg_origin);
        trsf.translation = rotated.added(&origin.xy());
        trsf
    }

    /// Creates a transformation from a coordinate system.
    #[inline]
    pub fn from_ax22d(ax: Ax22d) -> Self {
        let origin = ax.origin();
        let x_dir = ax.x_direction();
        let y_dir = ax.y_direction();
        
        let mat = Mat2d::from_cols(x_dir.xy(), y_dir.xy());
        Self {
            scale: 1.0,
            matrix: mat,
            translation: origin.xy(),
        }
    }

    /// Returns the scale factor.
    #[inline]
    pub fn scale_factor(&self) -> f64 {
        self.scale
    }

    /// Transforms a point.
    #[inline]
    pub fn transform_point(&self, point: Pnt2d) -> Pnt2d {
        let transformed = self.matrix.multiply_xy(point.xy()).added(&self.translation);
        Pnt2d::from_xy(transformed)
    }

    /// Transforms a vector.
    #[inline]
    pub fn transform_vector(&self, vec: Vec2d) -> Vec2d {
        Vec2d::from_xy(self.matrix.multiply_xy(vec.xy()))
    }

    /// Transforms a direction.
    #[inline]
    pub fn transform_direction(&self, dir: Dir2d) -> Dir2d {
        Dir2d::from_xy(self.matrix.multiply_xy(dir.xy()))
    }

    /// Inverts this transformation in place. Returns false if singular.
    pub fn invert(&mut self) -> bool {
        if let Some(inv_mat) = self.matrix.inverted() {
            self.matrix = inv_mat;
            self.translation = self.matrix.multiply_xy(self.translation.reversed());
            if (self.scale - 0.0).abs() > 1e-10 {
                self.translation = self.translation.multiplied(1.0 / (self.scale * self.scale));
                self.scale = 1.0 / self.scale;
            }
            true
        } else {
            false
        }
    }

    /// Returns the inverse. Returns None if singular.
    #[inline]
    pub fn inverted(&self) -> Option<Trsf2d> {
        let mut result = *self;
        if result.invert() {
            Some(result)
        } else {
            None
        }
    }

    /// Composes two transformations: self = self ∘ other.
    pub fn multiply(&mut self, other: &Trsf2d) {
        let new_mat = self.matrix.multiplied(&other.matrix);
        let new_trans = self.matrix.multiply_xy(other.translation).added(&self.translation);
        self.matrix = new_mat;
        self.translation = new_trans;
        self.scale *= other.scale;
    }

    /// Returns the composition of two transformations.
    #[inline]
    pub fn multiplied(&self, other: &Trsf2d) -> Trsf2d {
        let mut result = *self;
        result.multiply(other);
        result
    }

    /// Computes the matrix.
    #[inline]
    pub fn matrix(&self) -> Mat2d {
        self.matrix
    }

    /// Computes the translation vector.
    #[inline]
    pub fn translation(&self) -> Vec2d {
        Vec2d::from_xy(self.translation)
    }
}

impl Default for Trsf2d {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trsf2d_identity() {
        let t = Trsf2d::identity();
        let p = Pnt2d::from_coords(1.0, 2.0);
        let p2 = t.transform_point(p);
        assert_eq!(p2.x(), 1.0);
        assert_eq!(p2.y(), 2.0);
    }

    #[test]
    fn test_trsf2d_translation() {
        let vec = Vec2d::from_coords(3.0, 4.0);
        let t = Trsf2d::from_translation(vec);
        let p = Pnt2d::from_coords(1.0, 2.0);
        let p2 = t.transform_point(p);
        assert_eq!(p2.x(), 4.0);
        assert_eq!(p2.y(), 6.0);
    }

    #[test]
    fn test_trsf2d_rotation() {
        let t = Trsf2d::from_rotation(std::f64::consts::PI / 2.0); // 90 degrees
        let p = Pnt2d::from_coords(1.0, 0.0);
        let p2 = t.transform_point(p);
        assert!(p2.x().abs() < 1e-10);
        assert!((p2.y() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsf2d_scale() {
        let t = Trsf2d::from_scale(2.0);
        let p = Pnt2d::from_coords(1.0, 2.0);
        let p2 = t.transform_point(p);
        assert_eq!(p2.x(), 2.0);
        assert_eq!(p2.y(), 4.0);
    }

    #[test]
    fn test_trsf2d_transform_vector() {
        let t = Trsf2d::from_rotation(std::f64::consts::PI / 2.0);
        let v = Vec2d::from_coords(1.0, 0.0);
        let v2 = t.transform_vector(v);
        assert!(v2.x().abs() < 1e-10);
        assert!((v2.y() - 1.0).abs() < 1e-10);
    }
}
