//! Circle in 3D space.
//!
//! Port of OCCT's gp_Circ class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Circ.hxx

use super::{Ax1, Ax2, Pnt, Vec3, Trsf};
use std::f64::consts::PI;

/// A circle in 3D space.
/// Defined by a coordinate system (Ax2) and a radius.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Circ {
    pos: Ax2,
    radius: f64,
}

impl Circ {
    /// Creates an indefinite circle (with RealLast radius).
    #[inline]
    pub const fn new() -> Self {
        Self {
            pos: Ax2::new(),
            radius: f64::MAX,
        }
    }

    /// Creates a circle from an axis system and radius.
    /// The axis system origin is the center of the circle.
    /// The main direction is the normal to the circle plane.
    #[inline]
    pub fn from_ax2_radius(ax2: Ax2, radius: f64) -> Self {
        if radius < 0.0 {
            panic!("Circ radius must be non-negative");
        }
        Self { pos: ax2, radius }
    }

    /// Returns the main axis (normal to the plane).
    #[inline]
    pub const fn axis(&self) -> Ax1 {
        self.pos.axis()
    }

    /// Returns the center of the circle.
    #[inline]
    pub const fn location(&self) -> Pnt {
        self.pos.location()
    }

    /// Returns the position (coordinate system).
    #[inline]
    pub const fn position(&self) -> Ax2 {
        self.pos
    }

    /// Returns the radius of the circle.
    #[inline]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// Returns the X axis of the circle.
    #[inline]
    pub fn x_axis(&self) -> Ax1 {
        Ax1::from_pnt_dir(self.pos.location(), self.pos.x_direction())
    }

    /// Returns the Y axis of the circle.
    #[inline]
    pub fn y_axis(&self) -> Ax1 {
        Ax1::from_pnt_dir(self.pos.location(), self.pos.y_direction())
    }

    /// Sets the main axis (normal direction).
    pub fn set_axis(&mut self, ax: &Ax1) -> Result<(), String> {
        self.pos.set_axis(ax)
    }

    /// Sets the center point.
    #[inline]
    pub fn set_location(&mut self, p: Pnt) {
        self.pos.set_location(p);
    }

    /// Sets the position (coordinate system).
    #[inline]
    pub fn set_position(&mut self, ax2: Ax2) {
        self.pos = ax2;
    }

    /// Sets the radius.
    #[inline]
    pub fn set_radius(&mut self, radius: f64) {
        if radius < 0.0 {
            panic!("Circ radius must be non-negative");
        }
        self.radius = radius;
    }

    /// Computes the area of the circle.
    #[inline]
    pub fn area(&self) -> f64 {
        PI * self.radius * self.radius
    }

    /// Computes the circumference (perimeter) of the circle.
    #[inline]
    pub fn length(&self) -> f64 {
        2.0 * PI * self.radius
    }

    /// Computes the minimum distance from the circle to a point.
    #[inline]
    pub fn distance(&self, p: &Pnt) -> f64 {
        self.square_distance(p).sqrt()
    }

    /// Computes the square distance from the circle to a point.
    #[inline]
    pub fn square_distance(&self, p: &Pnt) -> f64 {
        let v = Vec3::from_pnt_pnt(&self.location(), p);
        let x = v.dot(&self.pos.x_direction().to_vec());
        let y = v.dot(&self.pos.y_direction().to_vec());
        let z = v.dot(&self.pos.direction().to_vec());
        let t = (x * x + y * y).sqrt() - self.radius;
        t * t + z * z
    }

    /// Returns true if the point is on the circumference (within tolerance).
    #[inline]
    pub fn contains(&self, p: &Pnt, tol: f64) -> bool {
        self.distance(p) <= tol
    }

    /// Mirror (reflect) the circle through a point (in-place).
    pub fn mirror_pnt(&mut self, p: &Pnt) {
        self.pos.mirror_pnt(p);
    }

    /// Returns a mirrored circle through a point.
    pub fn mirrored_pnt(&self, p: &Pnt) -> Circ {
        let mut result = *self;
        result.mirror_pnt(p);
        result
    }

    /// Mirror (reflect) the circle through an axis (in-place).
    pub fn mirror_ax1(&mut self, ax: &Ax1) {
        self.pos.mirror_ax1(ax);
    }

    /// Returns a mirrored circle through an axis.
    pub fn mirrored_ax1(&self, ax: &Ax1) -> Circ {
        let mut result = *self;
        result.mirror_ax1(ax);
        result
    }

    /// Mirror (reflect) the circle through a plane (in-place).
    pub fn mirror_ax2(&mut self, ax: &Ax2) {
        self.pos.mirror_ax2(ax);
    }

    /// Returns a mirrored circle through a plane.
    pub fn mirrored_ax2(&self, ax: &Ax2) -> Circ {
        let mut result = *self;
        result.mirror_ax2(ax);
        result
    }

    /// Rotate the circle (in-place).
    pub fn rotate(&mut self, ax: &Ax1, ang: f64) {
        self.pos.rotate(ax, ang);
    }

    /// Returns a rotated circle.
    pub fn rotated(&self, ax: &Ax1, ang: f64) -> Circ {
        let mut result = *self;
        result.rotate(ax, ang);
        result
    }

    /// Scale the circle (in-place).
    pub fn scale(&mut self, p: &Pnt, s: f64) {
        let mut new_radius = self.radius * s;
        if new_radius < 0.0 {
            new_radius = -new_radius;
        }
        self.radius = new_radius;
        self.pos.scale(p, s);
    }

    /// Returns a scaled circle.
    pub fn scaled(&self, p: &Pnt, s: f64) -> Circ {
        let mut result = *self;
        result.scale(p, s);
        result
    }

    /// Transform the circle (in-place).
    pub fn transform(&mut self, t: &Trsf) {
        let mut new_radius = self.radius * t.scale_factor();
        if new_radius < 0.0 {
            new_radius = -new_radius;
        }
        self.radius = new_radius;
        self.pos.transform(t);
    }

    /// Returns a transformed circle.
    pub fn transformed(&self, t: &Trsf) -> Circ {
        let mut result = *self;
        result.transform(t);
        result
    }

    /// Translate the circle (in-place).
    pub fn translate(&mut self, v: &Vec3) {
        self.pos.translate(v);
    }

    /// Returns a translated circle.
    pub fn translated(&self, v: &Vec3) -> Circ {
        let mut result = *self;
        result.translate(v);
        result
    }

    /// Translate the circle between two points (in-place).
    pub fn translate_2pts(&mut self, p1: &Pnt, p2: &Pnt) {
        self.pos.translate_2pts(p1, p2);
    }

    /// Returns a circle translated between two points.
    pub fn translated_2pts(&self, p1: &Pnt, p2: &Pnt) -> Circ {
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
    fn test_circ_default() {
        let circ = Circ::new();
        assert_eq!(circ.location().x(), 0.0);
        assert_eq!(circ.radius(), f64::MAX);
    }

    #[test]
    fn test_circ_from_ax2_radius() {
        let ax2 = Ax2::new();
        let circ = Circ::from_ax2_radius(ax2, 5.0);

        assert!((circ.radius() - 5.0).abs() < 1e-10);
        assert_eq!(circ.location().x(), 0.0);
    }

    #[test]
    fn test_circ_set_radius() {
        let mut circ = Circ::from_ax2_radius(Ax2::new(), 3.0);
        circ.set_radius(7.0);

        assert!((circ.radius() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_circ_area() {
        let circ = Circ::from_ax2_radius(Ax2::new(), 1.0);
        let area = circ.area();

        assert!((area - PI).abs() < 1e-10);
    }

    #[test]
    fn test_circ_length() {
        let circ = Circ::from_ax2_radius(Ax2::new(), 1.0);
        let length = circ.length();

        assert!((length - 2.0 * PI).abs() < 1e-10);
    }

    #[test]
    fn test_circ_distance_to_point_on_circle() {
        let circ = Circ::from_ax2_radius(Ax2::new(), 5.0);
        let p = Pnt::from_coords(5.0, 0.0, 0.0);

        let dist = circ.distance(&p);
        assert!(dist < 1e-10);
    }

    #[test]
    fn test_circ_distance_to_point_inside() {
        let circ = Circ::from_ax2_radius(Ax2::new(), 5.0);
        let p = Pnt::from_coords(2.0, 0.0, 0.0);

        let dist = circ.distance(&p);
        assert!((dist - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_circ_distance_to_point_outside() {
        let circ = Circ::from_ax2_radius(Ax2::new(), 5.0);
        let p = Pnt::from_coords(8.0, 0.0, 0.0);

        let dist = circ.distance(&p);
        assert!((dist - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_circ_contains() {
        let circ = Circ::from_ax2_radius(Ax2::new(), 5.0);
        let p = Pnt::from_coords(5.0, 0.0, 0.0);

        assert!(circ.contains(&p, precision::CONFUSION));
    }

    #[test]
    fn test_circ_set_location() {
        let mut circ = Circ::from_ax2_radius(Ax2::new(), 5.0);
        let p = Pnt::from_coords(1.0, 2.0, 3.0);
        circ.set_location(p);

        assert!((circ.location().x() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_circ_translate() {
        let circ = Circ::from_ax2_radius(Ax2::new(), 5.0);
        let v = Vec3::from_coords(1.0, 2.0, 3.0);
        let circ_trans = circ.translated(&v);

        assert!((circ_trans.location().x() - 1.0).abs() < 1e-10);
        assert!((circ_trans.location().y() - 2.0).abs() < 1e-10);
        assert!((circ_trans.location().z() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_circ_scale() {
        let circ = Circ::from_ax2_radius(Ax2::new(), 5.0);
        let circ_scaled = circ.scaled(&Pnt::new(), 2.0);

        assert!((circ_scaled.radius() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_circ_rotated() {
        let ax2 = Ax2::new();
        let circ = Circ::from_ax2_radius(ax2, 5.0);
        let ax = Ax1::from_pnt_dir(Pnt::new(), Dir::from_xyz(0.0, 0.0, 1.0).unwrap());
        let circ_rot = circ.rotated(&ax, PI / 2.0);

        assert!((circ_rot.radius() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_circ_x_axis() {
        let circ = Circ::from_ax2_radius(Ax2::new(), 5.0);
        let x_axis = circ.x_axis();

        assert!((x_axis.direction().x() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_circ_y_axis() {
        let circ = Circ::from_ax2_radius(Ax2::new(), 5.0);
        let y_axis = circ.y_axis();

        assert!((y_axis.direction().y() - 1.0).abs() < 1e-10);
    }
}

use super::Dir;
