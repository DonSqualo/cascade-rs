//! Line in 3D space.
//!
//! Port of OCCT's gp_Lin class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Lin.hxx

use super::{Ax1, Ax2, Dir, Pnt, Vec3, Trsf};
use std::f64::consts::PI;

/// A line in 3D space.
/// Defined by a position (point) and direction.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lin {
    pos: Ax1,
}

impl Lin {
    /// Creates a line corresponding to the Z axis of the reference coordinate system.
    #[inline]
    pub const fn new() -> Self {
        Self {
            pos: Ax1::from_pnt_dir(Pnt::new(), Dir::z()),
        }
    }

    /// Creates a line defined by an axis.
    #[inline]
    pub const fn from_ax1(ax1: Ax1) -> Self {
        Self { pos: ax1 }
    }

    /// Creates a line passing through a point and parallel to a direction.
    #[inline]
    pub const fn from_pnt_dir(p: Pnt, v: Dir) -> Self {
        Self {
            pos: Ax1::from_pnt_dir(p, v),
        }
    }

    /// Returns the direction of the line.
    #[inline]
    pub const fn direction(&self) -> Dir {
        self.pos.direction()
    }

    /// Returns the location (origin) point of the line.
    #[inline]
    pub const fn location(&self) -> Pnt {
        self.pos.location()
    }

    /// Returns the axis placement (same location and direction).
    #[inline]
    pub const fn position(&self) -> Ax1 {
        self.pos
    }

    /// Sets the direction of the line.
    #[inline]
    pub fn set_direction(&mut self, v: Dir) {
        self.pos.set_direction(v);
    }

    /// Sets the location point (origin) of the line.
    #[inline]
    pub fn set_location(&mut self, p: Pnt) {
        self.pos.set_location(p);
    }

    /// Sets the position (axis placement) of the line.
    #[inline]
    pub fn set_position(&mut self, ax1: Ax1) {
        self.pos = ax1;
    }

    /// Reverses the direction of the line (in-place).
    #[inline]
    pub fn reverse(&mut self) {
        self.pos.reverse();
    }

    /// Returns a line with reversed direction.
    #[inline]
    pub fn reversed(&self) -> Lin {
        let mut result = *self;
        result.reverse();
        result
    }

    /// Computes the angle between two lines in radians.
    #[inline]
    pub fn angle(&self, other: &Lin) -> f64 {
        self.pos.direction().angle(&other.pos.direction())
    }

    /// Computes the distance from the line to a point.
    #[inline]
    pub fn distance(&self, p: &Pnt) -> f64 {
        self.square_distance(p).sqrt()
    }

    /// Computes the square distance from the line to a point.
    #[inline]
    pub fn square_distance(&self, p: &Pnt) -> f64 {
        let loc = self.location();
        let mut v = Vec3::from_points(&loc, p);
        v.cross(&self.direction().to_vec());
        v.square_magnitude()
    }

    /// Computes the distance between two lines.
    pub fn distance_to_line(&self, other: &Lin) -> f64 {
        self.square_distance_to_line(other).sqrt()
    }

    /// Computes the square distance between two lines.
    pub fn square_distance_to_line(&self, other: &Lin) -> f64 {
        let d = self.distance_to_line_checked(other);
        d * d
    }

    /// Computes the distance between two lines (helper).
    fn distance_to_line_checked(&self, other: &Lin) -> f64 {
        // Vector from one line to another
        let v1 = self.direction().to_vec();
        let v2 = other.direction().to_vec();
        let w0 = Vec3::from_points(&self.location(), &other.location());

        let a = v1.dot(&v1);
        let b = v1.dot(&v2);
        let c = v2.dot(&v2);
        let d = v1.dot(&w0);
        let e = v2.dot(&w0);

        let denom = a * c - b * b;

        // Lines are parallel
        if denom.abs() < 1e-15 {
            return self.distance(&other.location());
        }

        // Lines are not parallel - compute closest distance
        let mut w = w0.clone();
        w.cross(&v2);
        return w.magnitude() / denom.abs().sqrt();
    }

    /// Returns true if the line contains the point (within tolerance).
    #[inline]
    pub fn contains(&self, p: &Pnt, tol: f64) -> bool {
        self.distance(p) <= tol
    }

    /// Returns a line normal to this line passing through the point.
    /// Raises if the distance is nearly zero (ambiguous case).
    pub fn normal(&self, p: &Pnt) -> Lin {
        let loc = self.location();
        let v = Vec3::from_points(&loc, p);
        let dir = self.direction();

        // Normal is perpendicular in the plane defined by the line and point
        let cross_cross = dir.cross_crossed(&v.normalize(), &dir);
        
        Lin::from_pnt_dir(*p, cross_cross)
    }

    /// Mirror (reflect) the line through a point (in-place).
    pub fn mirror_pnt(&mut self, p: &Pnt) {
        self.pos.mirror_pnt(p);
    }

    /// Returns a mirrored line through a point.
    pub fn mirrored_pnt(&self, p: &Pnt) -> Lin {
        let mut result = *self;
        result.mirror_pnt(p);
        result
    }

    /// Mirror (reflect) the line through an axis (in-place).
    pub fn mirror_ax1(&mut self, ax: &Ax1) {
        self.pos.mirror_ax1(ax);
    }

    /// Returns a mirrored line through an axis.
    pub fn mirrored_ax1(&self, ax: &Ax1) -> Lin {
        let mut result = *self;
        result.mirror_ax1(ax);
        result
    }

    /// Mirror (reflect) the line through a plane (in-place).
    pub fn mirror_ax2(&mut self, ax: &Ax2) {
        self.pos.mirror_ax2(ax);
    }

    /// Returns a mirrored line through a plane.
    pub fn mirrored_ax2(&self, ax: &Ax2) -> Lin {
        let mut result = *self;
        result.mirror_ax2(ax);
        result
    }

    /// Rotate the line (in-place).
    pub fn rotate(&mut self, ax: &Ax1, ang: f64) {
        self.pos.rotate(ax, ang);
    }

    /// Returns a rotated line.
    pub fn rotated(&self, ax: &Ax1, ang: f64) -> Lin {
        let mut result = *self;
        result.rotate(ax, ang);
        result
    }

    /// Scale the line (in-place).
    pub fn scale(&mut self, p: &Pnt, s: f64) {
        self.pos.scale(p, s);
    }

    /// Returns a scaled line.
    pub fn scaled(&self, p: &Pnt, s: f64) -> Lin {
        let mut result = *self;
        result.scale(p, s);
        result
    }

    /// Transform the line (in-place).
    pub fn transform(&mut self, t: &Trsf) {
        self.pos.transform(t);
    }

    /// Returns a transformed line.
    pub fn transformed(&self, t: &Trsf) -> Lin {
        let mut result = *self;
        result.transform(t);
        result
    }

    /// Translate the line (in-place).
    pub fn translate(&mut self, v: &Vec3) {
        self.pos.translate(v);
    }

    /// Returns a translated line.
    pub fn translated(&self, v: &Vec3) -> Lin {
        let mut result = *self;
        result.translate(v);
        result
    }

    /// Translate the line between two points (in-place).
    pub fn translate_2pts(&mut self, p1: &Pnt, p2: &Pnt) {
        self.pos.translate_2pts(p1, p2);
    }

    /// Returns a line translated between two points.
    pub fn translated_2pts(&self, p1: &Pnt, p2: &Pnt) -> Lin {
        let mut result = *self;
        result.translate_2pts(p1, p2);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision;

    #[test]
    fn test_lin_default() {
        let lin = Lin::new();
        assert_eq!(lin.location().x(), 0.0);
        assert_eq!(lin.direction().x_val(), 0.0);
    }

    #[test]
    fn test_lin_from_pnt_dir() {
        let p = Pnt::from_coords(1.0, 2.0, 3.0);
        let d = Dir::from_xyz(0.0, 0.0, 1.0).unwrap();
        let lin = Lin::from_pnt_dir(p, d);
        
        assert!((lin.location().x() - 1.0).abs() < 1e-10);
        assert!((lin.location().y() - 2.0).abs() < 1e-10);
        assert!((lin.location().z() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_lin_direction() {
        let d = Dir::from_xyz(1.0, 0.0, 0.0).unwrap();
        let lin = Lin::from_pnt_dir(Pnt::new(), d);
        assert!((lin.direction().x_val() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lin_set_location() {
        let mut lin = Lin::new();
        let p = Pnt::from_coords(5.0, 6.0, 7.0);
        lin.set_location(p);
        
        assert!((lin.location().x() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_lin_distance_to_point() {
        let lin = Lin::from_pnt_dir(
            Pnt::from_coords(0.0, 0.0, 0.0),
            Dir::from_xyz(1.0, 0.0, 0.0).unwrap(),
        );
        
        let p = Pnt::from_coords(0.0, 3.0, 4.0);
        let dist = lin.distance(&p);
        
        // Distance should be 5 (3-4-5 triangle)
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_lin_distance_zero_on_line() {
        let lin = Lin::from_pnt_dir(
            Pnt::from_coords(1.0, 2.0, 3.0),
            Dir::from_xyz(1.0, 0.0, 0.0).unwrap(),
        );
        
        let p = Pnt::from_coords(5.0, 2.0, 3.0);
        let dist = lin.distance(&p);
        
        assert!(dist < 1e-10);
    }

    #[test]
    fn test_lin_angle() {
        let lin1 = Lin::from_pnt_dir(
            Pnt::new(),
            Dir::from_xyz(1.0, 0.0, 0.0).unwrap(),
        );
        let lin2 = Lin::from_pnt_dir(
            Pnt::new(),
            Dir::from_xyz(0.0, 1.0, 0.0).unwrap(),
        );
        
        let angle = lin1.angle(&lin2);
        assert!((angle - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_lin_contains() {
        let lin = Lin::from_pnt_dir(
            Pnt::from_coords(0.0, 0.0, 0.0),
            Dir::from_xyz(1.0, 0.0, 0.0).unwrap(),
        );
        
        let p = Pnt::from_coords(5.0, 0.0, 0.0);
        assert!(lin.contains(&p, precision::CONFUSION));
    }

    #[test]
    fn test_lin_reverse() {
        let d = Dir::from_xyz(1.0, 0.0, 0.0).unwrap();
        let lin = Lin::from_pnt_dir(Pnt::new(), d);
        
        let mut lin_rev = lin;
        lin_rev.reverse();
        
        assert!((lin_rev.direction().x_val() + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lin_translate() {
        let lin = Lin::from_pnt_dir(
            Pnt::from_coords(1.0, 2.0, 3.0),
            Dir::from_xyz(1.0, 0.0, 0.0).unwrap(),
        );
        
        let v = Vec3::from_coords(1.0, 1.0, 1.0);
        let lin_trans = lin.translated(&v);
        
        assert!((lin_trans.location().x() - 2.0).abs() < 1e-10);
        assert!((lin_trans.location().y() - 3.0).abs() < 1e-10);
        assert!((lin_trans.location().z() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_lin_parallel_lines_distance() {
        let lin1 = Lin::from_pnt_dir(
            Pnt::from_coords(0.0, 0.0, 0.0),
            Dir::from_xyz(1.0, 0.0, 0.0).unwrap(),
        );
        let lin2 = Lin::from_pnt_dir(
            Pnt::from_coords(0.0, 1.0, 0.0),
            Dir::from_xyz(1.0, 0.0, 0.0).unwrap(),
        );
        
        let dist = lin1.distance_to_line(&lin2);
        assert!((dist - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lin_intersecting_lines_distance() {
        // Two lines that intersect at origin
        let lin1 = Lin::from_pnt_dir(
            Pnt::new(),
            Dir::from_xyz(1.0, 0.0, 0.0).unwrap(),
        );
        let lin2 = Lin::from_pnt_dir(
            Pnt::new(),
            Dir::from_xyz(0.0, 1.0, 0.0).unwrap(),
        );
        
        let dist = lin1.distance_to_line(&lin2);
        assert!(dist < 1e-10);
    }

    #[test]
    fn test_lin_skew_lines_distance() {
        // Two skew lines
        let lin1 = Lin::from_pnt_dir(
            Pnt::from_coords(0.0, 0.0, 0.0),
            Dir::from_xyz(1.0, 0.0, 0.0).unwrap(),
        );
        let lin2 = Lin::from_pnt_dir(
            Pnt::from_coords(0.0, 1.0, 1.0),
            Dir::from_xyz(0.0, 1.0, 0.0).unwrap(),
        );
        
        let dist = lin1.distance_to_line(&lin2);
        assert!((dist - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lin_rotated() {
        let lin = Lin::from_pnt_dir(
            Pnt::new(),
            Dir::from_xyz(1.0, 0.0, 0.0).unwrap(),
        );
        
        let ax = Ax1::from_pnt_dir(Pnt::new(), Dir::from_xyz(0.0, 0.0, 1.0).unwrap());
        let lin_rot = lin.rotated(&ax, PI / 2.0);
        
        // After 90 degree rotation around Z, X direction becomes Y direction
        assert!((lin_rot.direction().y_val() - 1.0).abs() < 1e-10);
        assert!(lin_rot.direction().x_val().abs() < 1e-10);
    }

    #[test]
    fn test_lin_scaled() {
        let lin = Lin::from_pnt_dir(
            Pnt::from_coords(1.0, 0.0, 0.0),
            Dir::from_xyz(1.0, 0.0, 0.0).unwrap(),
        );
        
        let lin_scaled = lin.scaled(&Pnt::new(), 2.0);
        
        assert!((lin_scaled.location().x() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_lin_normal() {
        let lin = Lin::from_pnt_dir(
            Pnt::new(),
            Dir::from_xyz(1.0, 0.0, 0.0).unwrap(),
        );
        
        let p = Pnt::from_coords(0.0, 1.0, 0.0);
        let norm = lin.normal(&p);
        
        // Normal should pass through the point
        assert!(norm.contains(&p, 1e-10));
    }
}
