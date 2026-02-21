//! 2D point.
//!
//! Port of OCCT's gp_Pnt2d class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Pnt2d.hxx

use crate::gp::{XY, Vec2d, Ax2d};
use std::ops::{Add, Sub};

/// A 2D point in cartesian coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pnt2d {
    coord: XY,
}

impl Pnt2d {
    /// Creates a point at the origin (0, 0).
    #[inline]
    pub const fn new() -> Self {
        Self { coord: XY::new() }
    }

    /// Creates a point with given coordinates.
    #[inline]
    pub const fn from_coords(x: f64, y: f64) -> Self {
        Self { coord: XY::from_coords(x, y) }
    }

    /// Creates a point from an XY.
    #[inline]
    pub const fn from_xy(xy: XY) -> Self {
        Self { coord: xy }
    }

    /// Sets the X coordinate.
    #[inline]
    pub fn set_x(&mut self, x: f64) {
        self.coord.set_x(x);
    }

    /// Sets the Y coordinate.
    #[inline]
    pub fn set_y(&mut self, y: f64) {
        self.coord.set_y(y);
    }

    /// Sets both coordinates.
    #[inline]
    pub fn set_coord(&mut self, x: f64, y: f64) {
        self.coord.set_coord(x, y);
    }

    /// Sets the XY.
    #[inline]
    pub fn set_xy(&mut self, xy: XY) {
        self.coord = xy;
    }

    /// Returns the X coordinate.
    #[inline]
    pub const fn x(&self) -> f64 {
        self.coord.x()
    }

    /// Returns the Y coordinate.
    #[inline]
    pub const fn y(&self) -> f64 {
        self.coord.y()
    }

    /// Returns coordinates as tuple.
    #[inline]
    pub const fn coords(&self) -> (f64, f64) {
        self.coord.coords()
    }

    /// Returns the XY.
    #[inline]
    pub const fn xy(&self) -> XY {
        self.coord
    }

    /// Returns mutable reference to the underlying XY.
    #[inline]
    pub fn xy_mut(&mut self) -> &mut XY {
        &mut self.coord
    }

    /// Returns coordinate by index (1=X, 2=Y).
    #[inline]
    pub fn coord(&self, index: usize) -> f64 {
        self.coord.coord(index)
    }

    /// Sets coordinate by index (1=X, 2=Y).
    #[inline]
    pub fn set_coord_index(&mut self, index: usize, value: f64) {
        self.coord.set_coord_index(index, value);
    }

    /// Returns the distance to another point.
    #[inline]
    pub fn distance(&self, other: &Pnt2d) -> f64 {
        let dx = self.coord.x() - other.coord.x();
        let dy = self.coord.y() - other.coord.y();
        (dx * dx + dy * dy).sqrt()
    }

    /// Returns the squared distance to another point.
    #[inline]
    pub const fn square_distance(&self, other: &Pnt2d) -> f64 {
        let dx = self.coord.x() - other.coord.x();
        let dy = self.coord.y() - other.coord.y();
        dx * dx + dy * dy
    }

    /// Checks if this point is equal to another within tolerance.
    #[inline]
    pub fn is_equal(&self, other: &Pnt2d, tolerance: f64) -> bool {
        self.distance(other) <= tolerance
    }

    /// Translates by a vector.
    #[inline]
    pub fn translate(&mut self, vec: Vec2d) {
        self.coord = self.coord.added(&vec.xy());
    }

    /// Returns translated copy.
    #[inline]
    pub fn translated(&self, vec: Vec2d) -> Pnt2d {
        Pnt2d {
            coord: self.coord.added(&vec.xy()),
        }
    }

    /// Mirrors about a point.
    #[inline]
    pub fn mirror_point(&mut self, center: &Pnt2d) {
        let dx = center.coord.x() - self.coord.x();
        let dy = center.coord.y() - self.coord.y();
        self.coord.set_coord(center.coord.x() + dx, center.coord.y() + dy);
    }

    /// Returns mirrored copy about a point.
    #[inline]
    pub fn mirrored_point(&self, center: &Pnt2d) -> Pnt2d {
        let dx = center.coord.x() - self.coord.x();
        let dy = center.coord.y() - self.coord.y();
        Pnt2d::from_coords(center.coord.x() + dx, center.coord.y() + dy)
    }

    /// Mirrors about an axis.
    #[inline]
    pub fn mirror_axis(&mut self, axis: &Ax2d) {
        // Translate to axis origin, mirror, translate back
        let origin = axis.origin();
        let direction = axis.direction();
        
        // Vector from axis origin to point
        let v = Vec2d::from_coords(
            self.coord.x() - origin.x(),
            self.coord.y() - origin.y(),
        );
        
        // Project onto axis and compute mirror
        let d = direction.xy();
        let proj = v.xy().dot(&d) * 2.0;
        
        // Mirrored = origin + proj*direction - v
        let scaled_d = d.multiplied(proj);
        let mirrored_v = scaled_d.subtracted(&v.xy());
        
        self.coord.set_coord(
            origin.x() + mirrored_v.x(),
            origin.y() + mirrored_v.y(),
        );
    }

    /// Returns mirrored copy about an axis.
    #[inline]
    pub fn mirrored_axis(&self, axis: &Ax2d) -> Pnt2d {
        let mut result = *self;
        result.mirror_axis(axis);
        result
    }

    /// Rotates about a point by angle in radians.
    pub fn rotate(&mut self, center: &Pnt2d, angle: f64) {
        let cos = angle.cos();
        let sin = angle.sin();
        
        let dx = self.coord.x() - center.coord.x();
        let dy = self.coord.y() - center.coord.y();
        
        let new_x = dx * cos - dy * sin;
        let new_y = dx * sin + dy * cos;
        
        self.coord.set_coord(
            center.coord.x() + new_x,
            center.coord.y() + new_y,
        );
    }

    /// Returns rotated copy about a point.
    #[inline]
    pub fn rotated(&self, center: &Pnt2d, angle: f64) -> Pnt2d {
        let mut result = *self;
        result.rotate(center, angle);
        result
    }

    /// Scales from a center point.
    #[inline]
    pub fn scale(&mut self, center: &Pnt2d, factor: f64) {
        let dx = (self.coord.x() - center.coord.x()) * factor;
        let dy = (self.coord.y() - center.coord.y()) * factor;
        self.coord.set_coord(center.coord.x() + dx, center.coord.y() + dy);
    }

    /// Returns scaled copy.
    #[inline]
    pub fn scaled(&self, center: &Pnt2d, factor: f64) -> Pnt2d {
        let dx = (self.coord.x() - center.coord.x()) * factor;
        let dy = (self.coord.y() - center.coord.y()) * factor;
        Pnt2d::from_coords(center.coord.x() + dx, center.coord.y() + dy)
    }
}

impl Add<Vec2d> for Pnt2d {
    type Output = Pnt2d;
    #[inline]
    fn add(self, vec: Vec2d) -> Pnt2d {
        self.translated(vec)
    }
}

impl Sub for Pnt2d {
    type Output = Vec2d;
    #[inline]
    fn sub(self, other: Pnt2d) -> Vec2d {
        Vec2d::from_coords(
            self.x() - other.x(),
            self.y() - other.y(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pnt2d_new() {
        let p = Pnt2d::new();
        assert_eq!(p.x(), 0.0);
        assert_eq!(p.y(), 0.0);
    }

    #[test]
    fn test_pnt2d_from_coords() {
        let p = Pnt2d::from_coords(3.0, 4.0);
        assert_eq!(p.x(), 3.0);
        assert_eq!(p.y(), 4.0);
    }

    #[test]
    fn test_pnt2d_set() {
        let mut p = Pnt2d::new();
        p.set_x(5.0);
        p.set_y(6.0);
        assert_eq!(p.x(), 5.0);
        assert_eq!(p.y(), 6.0);
    }

    #[test]
    fn test_pnt2d_distance() {
        let p1 = Pnt2d::from_coords(0.0, 0.0);
        let p2 = Pnt2d::from_coords(3.0, 4.0);
        assert!((p1.distance(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pnt2d_square_distance() {
        let p1 = Pnt2d::from_coords(0.0, 0.0);
        let p2 = Pnt2d::from_coords(3.0, 4.0);
        assert_eq!(p1.square_distance(&p2), 25.0);
    }

    #[test]
    fn test_pnt2d_is_equal() {
        let p1 = Pnt2d::from_coords(1.0, 2.0);
        let p2 = Pnt2d::from_coords(1.0 + 1e-8, 2.0);
        assert!(p1.is_equal(&p2, 1e-7));
        assert!(!p1.is_equal(&p2, 1e-9));
    }

    #[test]
    fn test_pnt2d_rotate() {
        let center = Pnt2d::from_coords(0.0, 0.0);
        let mut p = Pnt2d::from_coords(1.0, 0.0);
        p.rotate(&center, std::f64::consts::PI / 2.0); // 90 degrees
        assert!(p.x().abs() < 1e-10);
        assert!((p.y() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pnt2d_scale() {
        let center = Pnt2d::from_coords(0.0, 0.0);
        let mut p = Pnt2d::from_coords(1.0, 1.0);
        p.scale(&center, 2.0);
        assert_eq!(p.x(), 2.0);
        assert_eq!(p.y(), 2.0);
    }

    #[test]
    fn test_pnt2d_mirror_point() {
        let center = Pnt2d::from_coords(1.0, 1.0);
        let mut p = Pnt2d::from_coords(0.0, 0.0);
        p.mirror_point(&center);
        assert_eq!(p.x(), 2.0);
        assert_eq!(p.y(), 2.0);
    }
}
