//! 3D point.
//!
//! Port of OCCT's gp_Pnt class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Pnt.hxx

use super::{XYZ, Vec3, Ax1, Ax2, Trsf};
use crate::precision;

/// A 3D cartesian point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pnt {
    coord: XYZ,
}

impl Pnt {
    /// Creates a point at origin (0, 0, 0).
    #[inline]
    pub const fn new() -> Self {
        Self { coord: XYZ::new() }
    }

    /// Creates a point from XYZ coordinates.
    #[inline]
    pub const fn from_xyz(xyz: XYZ) -> Self {
        Self { coord: xyz }
    }

    /// Creates a point from coordinates.
    #[inline]
    pub const fn from_coords(x: f64, y: f64, z: f64) -> Self {
        Self {
            coord: XYZ::from_coords(x, y, z),
        }
    }

    /// Sets coordinate by index (1=X, 2=Y, 3=Z).
    #[inline]
    pub fn set_coord_index(&mut self, index: usize, value: f64) {
        self.coord.set_coord_index(index, value);
    }

    /// Sets all coordinates.
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

    /// Returns coordinate by index (1=X, 2=Y, 3=Z).
    #[inline]
    pub fn coord(&self, index: usize) -> f64 {
        self.coord.coord(index)
    }

    /// Returns all coordinates as tuple.
    #[inline]
    pub const fn coords(&self) -> (f64, f64, f64) {
        self.coord.coords()
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

    /// Returns mutable XYZ coordinates.
    #[inline]
    pub fn change_coord(&mut self) -> &mut XYZ {
        &mut self.coord
    }

    /// Barycenter: sets this to (alpha*this + beta*p) / (alpha + beta).
    #[inline]
    pub fn bary_center(&mut self, alpha: f64, p: &Pnt, beta: f64) {
        self.coord.set_linear_form_2w(alpha, &self.coord.clone(), beta, &p.coord);
        self.coord.divide(alpha + beta);
    }

    /// Returns true if distance to other <= linear_tolerance.
    #[inline]
    pub fn is_equal(&self, other: &Pnt, linear_tolerance: f64) -> bool {
        self.distance(other) <= linear_tolerance
    }

    /// Computes distance to another point.
    #[inline]
    pub fn distance(&self, other: &Pnt) -> f64 {
        self.square_distance(other).sqrt()
    }

    /// Computes square distance to another point.
    #[inline]
    pub fn square_distance(&self, other: &Pnt) -> f64 {
        let dx = self.coord.x() - other.coord.x();
        let dy = self.coord.y() - other.coord.y();
        let dz = self.coord.z() - other.coord.z();
        dx * dx + dy * dy + dz * dz
    }

    /// Mirrors about a point (symmetry center).
    pub fn mirror_point(&mut self, p: &Pnt) {
        self.coord.reverse();
        let mut xyz = p.coord;
        xyz.multiply(2.0);
        self.coord.add(&xyz);
    }

    /// Returns point mirrored about another point.
    pub fn mirrored_point(&self, p: &Pnt) -> Pnt {
        let mut result = *self;
        result.mirror_point(p);
        result
    }

    /// Mirrors about an axis.
    pub fn mirror_ax1(&mut self, a1: &Ax1) {
        let mut t = Trsf::new();
        t.set_mirror_ax1(a1);
        t.transforms_xyz(&mut self.coord);
    }

    /// Returns point mirrored about axis.
    pub fn mirrored_ax1(&self, a1: &Ax1) -> Pnt {
        let mut result = *self;
        result.mirror_ax1(a1);
        result
    }

    /// Mirrors about a plane.
    pub fn mirror_ax2(&mut self, a2: &Ax2) {
        let mut t = Trsf::new();
        t.set_mirror_ax2(a2);
        t.transforms_xyz(&mut self.coord);
    }

    /// Returns point mirrored about plane.
    pub fn mirrored_ax2(&self, a2: &Ax2) -> Pnt {
        let mut result = *self;
        result.mirror_ax2(a2);
        result
    }

    /// Rotates about axis by angle (radians).
    pub fn rotate(&mut self, a1: &Ax1, angle: f64) {
        let mut t = Trsf::new();
        t.set_rotation(a1, angle);
        t.transforms_xyz(&mut self.coord);
    }

    /// Returns point rotated about axis.
    pub fn rotated(&self, a1: &Ax1, angle: f64) -> Pnt {
        let mut result = *self;
        result.rotate(a1, angle);
        result
    }

    /// Scales about a point.
    pub fn scale(&mut self, p: &Pnt, s: f64) {
        let mut xyz = p.coord;
        xyz.multiply(1.0 - s);
        self.coord.multiply(s);
        self.coord.add(&xyz);
    }

    /// Returns scaled point.
    pub fn scaled(&self, p: &Pnt, s: f64) -> Pnt {
        let mut result = *self;
        result.scale(p, s);
        result
    }

    /// Applies transformation.
    pub fn transform(&mut self, t: &Trsf) {
        t.transforms_xyz(&mut self.coord);
    }

    /// Returns transformed point.
    pub fn transformed(&self, t: &Trsf) -> Pnt {
        let mut result = *self;
        result.transform(t);
        result
    }

    /// Translates by vector.
    pub fn translate(&mut self, v: &Vec3) {
        self.coord.add(v.xyz());
    }

    /// Returns translated point.
    pub fn translated(&self, v: &Vec3) -> Pnt {
        let mut result = *self;
        result.translate(v);
        result
    }

    /// Translates from p1 to p2.
    pub fn translate_2pts(&mut self, p1: &Pnt, p2: &Pnt) {
        self.coord.add(&p2.coord);
        self.coord.subtract(&p1.coord);
    }

    /// Returns point translated from p1 to p2.
    pub fn translated_2pts(&self, p1: &Pnt, p2: &Pnt) -> Pnt {
        let mut result = *self;
        result.translate_2pts(p1, p2);
        result
    }
}

impl From<[f64; 3]> for Pnt {
    fn from(arr: [f64; 3]) -> Self {
        Pnt::from_coords(arr[0], arr[1], arr[2])
    }
}

impl From<(f64, f64, f64)> for Pnt {
    fn from(tuple: (f64, f64, f64)) -> Self {
        Pnt::from_coords(tuple.0, tuple.1, tuple.2)
    }
}

impl From<Pnt> for [f64; 3] {
    fn from(pnt: Pnt) -> Self {
        [pnt.x(), pnt.y(), pnt.z()]
    }
}

// Hash matching OCCT
impl std::hash::Hash for Pnt {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coord.hash(state);
    }
}

impl Eq for Pnt {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pnt_default() {
        let p = Pnt::new();
        assert_eq!(p.x(), 0.0);
        assert_eq!(p.y(), 0.0);
        assert_eq!(p.z(), 0.0);
    }

    #[test]
    fn test_pnt_from_coords() {
        let p = Pnt::from_coords(1.0, 2.0, 3.0);
        assert_eq!(p.x(), 1.0);
        assert_eq!(p.y(), 2.0);
        assert_eq!(p.z(), 3.0);
    }

    #[test]
    fn test_pnt_distance() {
        let p1 = Pnt::from_coords(0.0, 0.0, 0.0);
        let p2 = Pnt::from_coords(3.0, 4.0, 0.0);
        assert!((p1.distance(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pnt_is_equal() {
        let p1 = Pnt::from_coords(1.0, 2.0, 3.0);
        let p2 = Pnt::from_coords(1.0 + 1e-8, 2.0, 3.0);
        assert!(p1.is_equal(&p2, precision::CONFUSION));
    }

    #[test]
    fn test_pnt_scale() {
        let mut p = Pnt::from_coords(2.0, 0.0, 0.0);
        let origin = Pnt::new();
        p.scale(&origin, 2.0);
        assert!((p.x() - 4.0).abs() < 1e-10);
    }
}
