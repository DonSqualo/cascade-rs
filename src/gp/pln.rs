//! Plane in 3D space.
//!
//! Port of OCCT's gp_Pln class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Pln.hxx

use super::{Ax1, Ax2, Ax3, Dir, Pnt, Vec3, Lin, Trsf};
use std::f64::consts::PI;
use crate::precision;

/// A plane in 3D space.
/// Defined by an axis system (Ax3) with origin and normal direction.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pln {
    pos: Ax3,
}

impl Pln {
    /// Creates a plane coincident with the OXY plane of the reference coordinate system.
    #[inline]
    pub const fn new() -> Self {
        Self {
            pos: Ax3::standard(),
        }
    }

    /// Creates a plane from an axis system.
    /// The main direction of the axis system is the normal to the plane.
    #[inline]
    pub const fn from_ax3(ax3: Ax3) -> Self {
        Self { pos: ax3 }
    }

    /// Creates a plane from a point and a normal direction.
    #[inline]
    pub fn from_pnt_dir(p: Pnt, v: Dir) -> Self {
        let ax3 = Ax3::from_dir(v);
        let mut ax3 = ax3;
        ax3.set_location(p);
        Self { pos: ax3 }
    }

    /// Creates a plane from a cartesian equation: A*X + B*Y + C*Z + D = 0
    pub fn from_cartesian(a: f64, b: f64, c: f64, d: f64) -> Result<Self, String> {
        let norm = (a * a + b * b + c * c).sqrt();
        if norm <= precision::RESOLUTION {
            return Err("Plane normal is too small".to_string());
        }

        let nx = a / norm;
        let ny = b / norm;
        let nz = c / norm;

        // Find a point on the plane
        // Use the axis with largest absolute value for the normal
        let (x, y, z) = if nx.abs() >= ny.abs() && nx.abs() >= nz.abs() {
            ((-d) / a, 0.0, 0.0)
        } else if ny.abs() >= nz.abs() {
            (0.0, (-d) / b, 0.0)
        } else {
            (0.0, 0.0, (-d) / c)
        };

        let p = Pnt::from_coords(x, y, z);
        let dir = Dir::from_xyz(nx, ny, nz)?;
        let ax3 = Ax3::from_dir(dir);
        let mut ax3 = ax3;
        ax3.set_location(p);

        Ok(Self { pos: ax3 })
    }

    /// Returns the coefficients of the plane's cartesian equation: A*X + B*Y + C*Z + D = 0
    #[inline]
    pub fn coefficients(&self) -> (f64, f64, f64, f64) {
        let dir = self.pos.direction();
        let (a, b, c) = if self.pos.direct() {
            (dir.x(), dir.y(), dir.z())
        } else {
            (-dir.x(), -dir.y(), -dir.z())
        };

        let p = self.pos.location();
        let d = -(a * p.x() + b * p.y() + c * p.z());

        (a, b, c, d)
    }

    /// Returns the plane's normal axis (main direction with location).
    #[inline]
    pub const fn axis(&self) -> Ax1 {
        self.pos.axis()
    }

    /// Returns the plane's location (origin).
    #[inline]
    pub const fn location(&self) -> Pnt {
        self.pos.location()
    }

    /// Returns the local coordinate system of the plane.
    #[inline]
    pub const fn position(&self) -> Ax3 {
        self.pos
    }

    /// Sets the plane's location point.
    #[inline]
    pub fn set_location(&mut self, p: Pnt) {
        self.pos.set_location(p);
    }

    /// Sets the plane's position (coordinate system).
    #[inline]
    pub fn set_position(&mut self, ax3: Ax3) {
        self.pos = ax3;
    }

    /// Sets the plane's main axis (normal direction).
    pub fn set_axis(&mut self, ax: Ax1) -> Result<(), String> {
        self.pos.set_axis(ax)
    }

    /// Reverses the U parametrization (reverses the X axis).
    #[inline]
    pub fn u_reverse(&mut self) {
        self.posx_reverse();
    }

    /// Reverses the V parametrization (reverses the Y axis).
    #[inline]
    pub fn v_reverse(&mut self) {
        self.posy_reverse();
    }

    /// Returns true if the axis system is right-handed.
    #[inline]
    pub const fn direct(&self) -> bool {
        self.pos.direct()
    }

    /// Returns the X axis of the plane.
    #[inline]
    pub fn x_axis(&self) -> Ax1 {
        Ax1::from_pnt_dir(self.pos.location(), self.posxdirection())
    }

    /// Returns the Y axis of the plane.
    #[inline]
    pub fn y_axis(&self) -> Ax1 {
        Ax1::from_pnt_dir(self.pos.location(), self.posydirection())
    }

    /// Computes the distance from the plane to a point.
    #[inline]
    pub fn distance(&self, p: &Pnt) -> f64 {
        self.signed_distance(p).abs()
    }

    /// Computes the distance from the plane to a line.
    #[inline]
    pub fn distance_to_line(&self, lin: &Lin) -> f64 {
        self.signed_distance_to_line(lin).abs()
    }

    /// Computes the distance between two planes.
    #[inline]
    pub fn distance_to_plane(&self, other: &Pln) -> f64 {
        self.signed_distance_to_plane(other).abs()
    }

    /// Computes the signed distance from the plane to a point.
    /// Positive if point is in direction of normal, negative otherwise.
    #[inline]
    pub fn signed_distance(&self, p: &Pnt) -> f64 {
        let loc = self.pos.location();
        let dir = self.pos.direction();
        let v = Vec3::from_points(&loc, p);
        v.dot(&dir.to_vec())
    }

    /// Computes the signed distance from the plane to a line.
    /// Returns 0 if line intersects the plane.
    #[inline]
    pub fn signed_distance_to_line(&self, lin: &Lin) -> f64 {
        let dir = self.pos.direction();
        if !dir.is_normal(&lin.direction(), precision::RESOLUTION) {
            return 0.0;
        }
        self.signed_distance(&lin.location())
    }

    /// Computes the signed distance between two planes.
    /// Returns 0 if planes intersect.
    #[inline]
    pub fn signed_distance_to_plane(&self, other: &Pln) -> f64 {
        let dir = self.pos.direction();
        if !dir.is_parallel(&other.pos.direction(), precision::RESOLUTION) {
            return 0.0;
        }
        self.signed_distance(&other.pos.location())
    }

    /// Computes the square distance from the plane to a point.
    #[inline]
    pub fn square_distance(&self, p: &Pnt) -> f64 {
        let d = self.distance(p);
        d * d
    }

    /// Computes the square distance from the plane to a line.
    #[inline]
    pub fn square_distance_to_line(&self, lin: &Lin) -> f64 {
        let d = self.distance_to_line(lin);
        d * d
    }

    /// Computes the square distance between two planes.
    #[inline]
    pub fn square_distance_to_plane(&self, other: &Pln) -> f64 {
        let d = self.distance_to_plane(other);
        d * d
    }

    /// Returns true if the plane contains the point (within tolerance).
    #[inline]
    pub fn contains(&self, p: &Pnt, tol: f64) -> bool {
        self.distance(p) <= tol
    }

    /// Returns true if the plane contains the line (within tolerances).
    #[inline]
    pub fn contains_line(&self, lin: &Lin, lin_tol: f64, ang_tol: f64) -> bool {
        self.contains(&lin.location(), lin_tol)
            && self.pos.direction().is_normal(&lin.direction(), ang_tol)
    }

    /// Mirror (reflect) the plane through a point (in-place).
    pub fn mirror_pnt(&mut self, p: &Pnt) {
        self.pos.mirror_pnt(p);
    }

    /// Returns a mirrored plane through a point.
    pub fn mirrored_pnt(&self, p: &Pnt) -> Pln {
        let mut result = *self;
        result.mirror_pnt(p);
        result
    }

    /// Mirror (reflect) the plane through an axis (in-place).
    pub fn mirror_ax1(&mut self, ax: &Ax1) {
        self.pos.mirror_ax1(ax);
    }

    /// Returns a mirrored plane through an axis.
    pub fn mirrored_ax1(&self, ax: &Ax1) -> Pln {
        let mut result = *self;
        result.mirror_ax1(ax);
        result
    }

    /// Mirror (reflect) the plane through a plane (in-place).
    pub fn mirror_ax2(&mut self, ax: &Ax2) {
        self.pos.mirror_ax2(ax);
    }

    /// Returns a mirrored plane through a plane.
    pub fn mirrored_ax2(&self, ax: &Ax2) -> Pln {
        let mut result = *self;
        result.mirror_ax2(ax);
        result
    }

    /// Rotate the plane (in-place).
    pub fn rotate(&mut self, ax: &Ax1, ang: f64) {
        self.pos.rotate(ax, ang);
    }

    /// Returns a rotated plane.
    pub fn rotated(&self, ax: &Ax1, ang: f64) -> Pln {
        let mut result = *self;
        result.rotate(ax, ang);
        result
    }

    /// Scale the plane (in-place).
    pub fn scale(&mut self, p: &Pnt, s: f64) {
        self.pos.scale(p, s);
    }

    /// Returns a scaled plane.
    pub fn scaled(&self, p: &Pnt, s: f64) -> Pln {
        let mut result = *self;
        result.scale(p, s);
        result
    }

    /// Transform the plane (in-place).
    pub fn transform(&mut self, t: &Trsf) {
        self.pos.transform(t);
    }

    /// Returns a transformed plane.
    pub fn transformed(&self, t: &Trsf) -> Pln {
        let mut result = *self;
        result.transform(t);
        result
    }

    /// Translate the plane (in-place).
    pub fn translate(&mut self, v: &Vec3) {
        self.pos.translate(v);
    }

    /// Returns a translated plane.
    pub fn translated(&self, v: &Vec3) -> Pln {
        let mut result = *self;
        result.translate(v);
        result
    }

    /// Translate the plane between two points (in-place).
    pub fn translate_2pts(&mut self, p1: &Pnt, p2: &Pnt) {
        self.pos.translate_2pts(p1, p2);
    }

    /// Returns a plane translated between two points.
    pub fn translated_2pts(&self, p1: &Pnt, p2: &Pnt) -> Pln {
        let mut result = *self;
        result.translate_2pts(p1, p2);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pln_default() {
        let pln = Pln::new();
        assert_eq!(pln.location().x(), 0.0);
    }

    #[test]
    fn test_pln_from_pnt_dir() {
        let p = Pnt::from_coords(1.0, 2.0, 3.0);
        let d = Dir::from_xyz(0.0, 0.0, 1.0).unwrap();
        let pln = Pln::from_pnt_dir(p, d);

        assert!((pln.location().x() - 1.0).abs() < 1e-10);
        assert!((pln.location().y() - 2.0).abs() < 1e-10);
        assert!((pln.location().z() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pln_cartesian_equation() {
        // Z = 5: 0*X + 0*Y + 1*Z - 5 = 0
        let pln = Pln::from_cartesian(0.0, 0.0, 1.0, -5.0).unwrap();
        let p = Pnt::from_coords(1.0, 2.0, 5.0);

        assert!(pln.contains(&p, 1e-10));
    }

    #[test]
    fn test_pln_distance_to_point() {
        let pln = Pln::from_pnt_dir(
            Pnt::new(),
            Dir::from_xyz(0.0, 0.0, 1.0).unwrap(),
        );

        let p = Pnt::from_coords(0.0, 0.0, 5.0);
        assert!((pln.distance(&p) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pln_signed_distance() {
        let pln = Pln::from_pnt_dir(
            Pnt::new(),
            Dir::from_xyz(0.0, 0.0, 1.0).unwrap(),
        );

        let p_pos = Pnt::from_coords(0.0, 0.0, 5.0);
        let p_neg = Pnt::from_coords(0.0, 0.0, -5.0);

        assert!((pln.signed_distance(&p_pos) - 5.0).abs() < 1e-10);
        assert!((pln.signed_distance(&p_neg) + 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pln_coefficients() {
        let pln = Pln::from_pnt_dir(
            Pnt::new(),
            Dir::from_xyz(0.0, 0.0, 1.0).unwrap(),
        );

        let (a, b, c, d) = pln.coefficients();
        assert!(a.abs() < 1e-10);
        assert!(b.abs() < 1e-10);
        assert!((c - 1.0).abs() < 1e-10 || (c + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pln_set_location() {
        let mut pln = Pln::new();
        let p = Pnt::from_coords(5.0, 6.0, 7.0);
        pln.set_location(p);

        assert!((pln.location().x() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pln_contains_point() {
        let pln = Pln::from_pnt_dir(
            Pnt::new(),
            Dir::from_xyz(0.0, 0.0, 1.0).unwrap(),
        );

        let p = Pnt::from_coords(5.0, 6.0, 0.0);
        assert!(pln.contains(&p, precision::CONFUSION));
    }

    #[test]
    fn test_pln_contains_line() {
        let pln = Pln::from_pnt_dir(
            Pnt::new(),
            Dir::from_xyz(0.0, 0.0, 1.0).unwrap(),
        );

        let lin = Lin::from_pnt_dir(
            Pnt::from_coords(1.0, 2.0, 0.0),
            Dir::from_xyz(1.0, 0.0, 0.0).unwrap(),
        );

        assert!(pln.contains_line(&lin, precision::CONFUSION, precision::ANGULAR));
    }

    #[test]
    fn test_pln_translate() {
        let pln = Pln::from_pnt_dir(
            Pnt::from_coords(1.0, 2.0, 3.0),
            Dir::from_xyz(0.0, 0.0, 1.0).unwrap(),
        );

        let v = Vec3::from_coords(1.0, 1.0, 1.0);
        let pln_trans = pln.translated(&v);

        assert!((pln_trans.location().x() - 2.0).abs() < 1e-10);
        assert!((pln_trans.location().y() - 3.0).abs() < 1e-10);
        assert!((pln_trans.location().z() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_pln_parallel_planes_distance() {
        let pln1 = Pln::from_pnt_dir(
            Pnt::new(),
            Dir::from_xyz(0.0, 0.0, 1.0).unwrap(),
        );
        let pln2 = Pln::from_pnt_dir(
            Pnt::from_coords(0.0, 0.0, 5.0),
            Dir::from_xyz(0.0, 0.0, 1.0).unwrap(),
        );

        assert!((pln1.distance_to_plane(&pln2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pln_distance_to_intersecting_line() {
        let pln = Pln::from_pnt_dir(
            Pnt::new(),
            Dir::from_xyz(0.0, 0.0, 1.0).unwrap(),
        );

        let lin = Lin::from_pnt_dir(
            Pnt::from_coords(0.0, 0.0, 0.0),
            Dir::from_xyz(0.0, 0.0, 1.0).unwrap(),
        );

        assert!(pln.distance_to_line(&lin) < 1e-10);
    }

    #[test]
    fn test_pln_direct() {
        let pln = Pln::new();
        assert!(pln.direct());
    }

    #[test]
    fn test_pln_rotated() {
        let pln = Pln::from_pnt_dir(
            Pnt::new(),
            Dir::from_xyz(1.0, 0.0, 0.0).unwrap(),
        );

        let ax = Ax1::from_pnt_dir(Pnt::new(), Dir::from_xyz(0.0, 0.0, 1.0).unwrap());
        let pln_rot = pln.rotated(&ax, PI / 2.0);

        assert!((pln_rot.position().direction().y_val() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pln_scaled() {
        let pln = Pln::from_pnt_dir(
            Pnt::from_coords(1.0, 0.0, 0.0),
            Dir::from_xyz(0.0, 0.0, 1.0).unwrap(),
        );

        let pln_scaled = pln.scaled(&Pnt::new(), 2.0);

        assert!((pln_scaled.location().x() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_pln_u_reverse() {
        let mut pln = Pln::new();
        let old_xdir = pln.position()xdirection();
        pln.u_reverse();
        let new_xdir = pln.position()xdirection();

        assert!((old_xdir.x() + new_xdir.x()).abs() < 1e-10);
    }

    #[test]
    fn test_pln_v_reverse() {
        let mut pln = Pln::new();
        let old_ydir = pln.position()ydirection();
        pln.v_reverse();
        let new_ydir = pln.position()ydirection();

        assert!((old_ydir.y() + new_ydir.y()).abs() < 1e-10);
    }

    #[test]
    fn test_pln_x_axis() {
        let pln = Pln::new();
        let x_axis = pln.x_axis();
        assert!((x_axis.direction().x_val() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pln_y_axis() {
        let pln = Pln::new();
        let y_axis = pln.y_axis();
        assert!((y_axis.direction().y_val() - 1.0).abs() < 1e-10);
    }
}
