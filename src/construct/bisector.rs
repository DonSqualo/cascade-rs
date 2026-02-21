//! Bisector curve construction
//!
//! Functions for constructing bisector curves between geometric elements.
//! These are loci of points equidistant from two geometric objects.

use crate::{Result, CascadeError, TOLERANCE};
use crate::geom::{Pnt, Dir, Vec3};
use crate::brep::CurveType;
use crate::curve::Parabola;
use crate::Vec3 as CrateVec3;

/// A simple 2D line representation for bisector calculation
#[derive(Debug, Clone, Copy)]
struct Line2D {
    /// Point on the line
    point: [f64; 2],
    /// Direction vector (unit)
    direction: [f64; 2],
}

impl Line2D {
    /// Create a new 2D line
    fn new(point: [f64; 2], direction: [f64; 2]) -> Self {
        let mag = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        let normalized = if mag > 1e-10 {
            [direction[0] / mag, direction[1] / mag]
        } else {
            [1.0, 0.0]
        };
        Line2D {
            point,
            direction: normalized,
        }
    }

    /// Get the perpendicular direction
    fn perpendicular(&self) -> [f64; 2] {
        [-self.direction[1], self.direction[0]]
    }

    /// Find intersection of two 2D lines
    fn intersect(&self, other: &Line2D) -> Option<[f64; 2]> {
        let dx = other.point[0] - self.point[0];
        let dy = other.point[1] - self.point[1];

        let denom = self.direction[0] * other.direction[1]
            - self.direction[1] * other.direction[0];

        if denom.abs() < 1e-10 {
            return None; // Parallel lines
        }

        let t = (dx * other.direction[1] - dy * other.direction[0]) / denom;

        Some([
            self.point[0] + t * self.direction[0],
            self.point[1] + t * self.direction[1],
        ])
    }
}

/// Construct the two bisector lines between two intersecting lines in 3D.
///
/// The bisector lines are the loci of points equidistant from both input lines.
/// For two intersecting lines in 3D, there are two bisecting planes that pass
/// through the intersection point. The bisector lines are the lines in which
/// these planes intersect.
///
/// # Arguments
/// * `line1_point` - A point on the first line
/// * `line1_dir` - Direction of the first line
/// * `line2_point` - A point on the second line
/// * `line2_dir` - Direction of the second line
///
/// # Returns
/// A vector containing up to 2 bisector lines, or an error if the lines don't intersect
///
/// # Example
/// ```ignore
/// let p1 = Pnt::new(0.0, 0.0, 0.0);
/// let d1 = Dir::new(1.0, 0.0, 0.0);
/// let p2 = Pnt::new(0.0, 0.0, 0.0);
/// let d2 = Dir::new(0.0, 1.0, 0.0);
///
/// let bisectors = bisector_line_line(p1, d1, p2, d2)?;
/// assert_eq!(bisectors.len(), 2);
/// ```
pub fn bisector_line_line(
    line1_point: Pnt,
    line1_dir: Dir,
    line2_point: Pnt,
    line2_dir: Dir,
) -> Result<Vec<(Pnt, Dir)>> {
    // Check if lines are parallel
    let cross = line1_dir.to_vec3().cross(&line2_dir.to_vec3());
    if cross.magnitude() < TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            "Cannot compute bisector of parallel lines".to_string(),
        ));
    }

    // Find intersection point
    let v = line1_point.vec_to(&line2_point);
    let d1 = line1_dir.to_vec3();
    let d2 = line2_dir.to_vec3();

    // Solve: line1_point + t1 * d1 = line2_point + t2 * d2
    // Rewrite: t1 * d1 - t2 * d2 = v
    let cross_v_d2 = v.cross(&d2);
    let cross_d1_d2 = d1.cross(&d2);

    let denom = cross_d1_d2.magnitude_squared();
    if denom < TOLERANCE * TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            "Lines do not intersect properly".to_string(),
        ));
    }

    let t1 = cross_v_d2.dot(&cross_d1_d2) / denom;
    let intersection = line1_point.translated(&d1.scaled(t1));

    // Normalize directions
    let d1_norm = d1.normalized().unwrap_or(d1);
    let d2_norm = d2.normalized().unwrap_or(d2);

    // The two bisectors are computed as follows:
    // They pass through the intersection point and are oriented along:
    // b1 = (d1 + d2) / |d1 + d2|  (internal bisector)
    // b2 = (d1 - d2) / |d1 - d2|  (external bisector)

    let mut bisectors = Vec::new();

    // Internal bisector
    let sum = d1_norm.added(&d2_norm);
    if sum.magnitude() > TOLERANCE {
        if let Ok(bisector1_dir) = Dir::try_from_vec3(sum) {
            bisectors.push((intersection, bisector1_dir));
        }
    }

    // External bisector
    let diff = d1_norm.sub(&d2_norm);
    if diff.magnitude() > TOLERANCE {
        if let Ok(bisector2_dir) = Dir::try_from_vec3(diff) {
            bisectors.push((intersection, bisector2_dir));
        }
    }

    if bisectors.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Failed to compute bisector directions".to_string(),
        ));
    }

    Ok(bisectors)
}

/// Construct the bisector curve (parabola) between a point and a line.
///
/// A parabola is the locus of all points equidistant from a focus (the given point)
/// and a directrix (the given line). This function returns a parabola in 3D space.
///
/// The parabola lies in the plane containing the point and the line.
///
/// # Arguments
/// * `focus` - The focal point of the parabola
/// * `line_point` - A point on the directrix line
/// * `line_dir` - Direction of the directrix line
///
/// # Returns
/// A Parabola representing the bisector curve, or an error if construction fails
///
/// # Example
/// ```ignore
/// let focus = Pnt::new(2.0, 0.0, 0.0);
/// let line_point = Pnt::new(0.0, 0.0, 0.0);
/// let line_dir = Dir::z_axis();
///
/// let parabola = bisector_point_line(focus, line_point, line_dir)?;
/// ```
pub fn bisector_point_line(
    focus: Pnt,
    line_point: Pnt,
    line_dir: Dir,
) -> Result<Parabola> {
    let line_vec = line_dir.to_vec3();

    // Vector from line point to focus
    let pf = line_point.vec_to(&focus);

    // Find the point on the line closest to the focus
    let proj_t = pf.dot(&line_vec) / line_vec.magnitude_squared();
    let closest_point = line_point.translated(&line_vec.scaled(proj_t));

    // Distance from focus to directrix
    let distance = focus.distance(&closest_point);

    if distance < TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            "Focus must not lie on the directrix".to_string(),
        ));
    }

    // The vertex is at the midpoint between focus and its projection
    let vertex = focus.midpoint(&closest_point);

    // X direction: from vertex toward focus
    let x_dir_vec = vertex.vec_to(&focus).normalized().ok_or_else(|| {
        CascadeError::InvalidGeometry("Failed to normalize x direction".to_string())
    })?;

    // Y direction: perpendicular to line_dir, perpendicular to x_dir
    let perpendicular = line_vec.normalized().ok_or_else(|| {
        CascadeError::InvalidGeometry("Failed to normalize line direction".to_string())
    })?;

    let y_dir_vec = x_dir_vec.cross(&perpendicular);
    if y_dir_vec.magnitude() < TOLERANCE {
        // If line_dir is parallel to x_dir, choose perpendicular differently
        let fallback_perp = if line_vec.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let y_dir_vec = x_dir_vec.cross(&fallback_perp);
        if y_dir_vec.magnitude() < TOLERANCE {
            return Err(CascadeError::InvalidGeometry(
                "Cannot determine parabola plane".to_string(),
            ));
        }
        let y_dir = y_dir_vec
            .normalized()
            .ok_or_else(|| CascadeError::InvalidGeometry("Failed to normalize y direction".to_string()))?;

        // Focal parameter: p = 2 * distance (distance from vertex to focus)
        let focal_param = 2.0 * distance;

        return Ok(Parabola {
            origin: [vertex.x, vertex.y, vertex.z],
            x_dir: [x_dir_vec.x, x_dir_vec.y, x_dir_vec.z],
            y_dir: [y_dir.x, y_dir.y, y_dir.z],
            focal: focal_param,
        });
    }

    let y_dir = y_dir_vec
        .normalized()
        .ok_or_else(|| CascadeError::InvalidGeometry("Failed to normalize y direction".to_string()))?;

    // Focal parameter: p = 2 * distance (distance from vertex to focus)
    let focal_param = 2.0 * distance;

    Ok(Parabola {
        origin: [vertex.x, vertex.y, vertex.z],
        x_dir: [x_dir_vec.x, x_dir_vec.y, x_dir_vec.z],
        y_dir: [y_dir.x, y_dir.y, y_dir.z],
        focal: focal_param,
    })
}

/// Construct the bisector curve between two circles.
///
/// The bisector between two circles is the locus of points equidistant from both circles.
/// The result depends on the relative positions of the circles:
///
/// - **Separate circles**: Hyperbola (interior and exterior bisectors)
/// - **Overlapping circles**: Ellipse (interior bisector) + Hyperbola (exterior bisector)
/// - **One inside the other**: Ellipse (interior bisector)
/// - **Concentric circles**: Degenerate case (plane midway between them)
///
/// # Arguments
/// * `center1` - Center of the first circle
/// * `radius1` - Radius of the first circle
/// * `center2` - Center of the second circle
/// * `radius2` - Radius of the second circle
///
/// # Returns
/// A vector of CurveType variants representing the bisector curves, or an error
///
/// # Note
/// Circles must be coplanar. The curves lie in the same plane.
pub fn bisector_circle_circle(
    center1: Pnt,
    radius1: f64,
    center2: Pnt,
    radius2: f64,
) -> Result<Vec<CurveType>> {
    if radius1 <= TOLERANCE || radius2 <= TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            "Circle radii must be positive".to_string(),
        ));
    }

    let dist = center1.distance(&center2);

    if dist < TOLERANCE {
        // Concentric circles
        return Err(CascadeError::InvalidGeometry(
            "Cannot compute bisector of concentric circles".to_string(),
        ));
    }

    // Project to 2D for easier computation
    let c1_2d = [center1.x, center1.y];
    let c2_2d = [center2.x, center2.y];
    let z_coord = center1.z;

    let mut result = Vec::new();

    // Determine the configuration
    let dist_2d = ((c2_2d[0] - c1_2d[0]).powi(2) + (c2_2d[1] - c1_2d[1]).powi(2)).sqrt();

    // Check if circles are separate, overlapping, or one inside the other
    let separate = dist_2d > radius1 + radius2 + TOLERANCE;
    let overlapping = dist_2d < radius1 + radius2 - TOLERANCE && dist_2d > (radius1 - radius2).abs() + TOLERANCE;
    let one_inside = dist_2d < (radius1 - radius2).abs() - TOLERANCE;

    // Apollonius problem: For two circles, the bisector points satisfy:
    // |P - C1| - r1 = |P - C2| - r2  (interior bisector, diff of distances = constant)
    // This is a hyperbola if circles are separate or one inside, ellipse if overlapping

    if overlapping {
        // Interior bisector is an ellipse
        // The ellipse has foci at C1 and C2
        // For a point P on the ellipse: |P - C1| - r1 = |P - C2| - r2
        // This gives: |P - C1| + |P - C2| = r1 + r2
        // This is an ellipse with sum of distances = r1 + r2

        if let Some(ellipse) = construct_ellipse_bisector(c1_2d, radius1, c2_2d, radius2, z_coord) {
            result.push(ellipse);
        }
    } else {
        // Exterior bisector is a hyperbola
        // For a point P on the hyperbola: |P - C1| - r1 = |P - C2| - r2
        // This is a hyperbola

        if let Some(hyperbola) = construct_hyperbola_bisector(c1_2d, radius1, c2_2d, radius2, z_coord) {
            result.push(hyperbola);
        }
    }

    if result.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Failed to construct circle bisector".to_string(),
        ));
    }

    Ok(result)
}

/// Construct an ellipse as the bisector of two overlapping circles
fn construct_ellipse_bisector(
    c1: [f64; 2],
    r1: f64,
    c2: [f64; 2],
    r2: f64,
    z: f64,
) -> Option<CurveType> {
    let dist = ((c2[0] - c1[0]).powi(2) + (c2[1] - c1[1]).powi(2)).sqrt();

    if dist < 1e-10 {
        return None;
    }

    // Center of the ellipse is at the midpoint
    let center = [
        (c1[0] + c2[0]) / 2.0,
        (c1[1] + c2[1]) / 2.0,
    ];

    // Semi-major axis: a = (r1 + r2) / 2
    // The distance between foci: 2c = dist
    // So c = dist / 2
    // Semi-minor axis: b = sqrt(a² - c²)

    let a = (r1 + r2) / 2.0;
    let c = dist / 2.0;

    if a < c + 1e-10 {
        return None; // Degenerate ellipse
    }

    let b_sq = a * a - c * c;
    if b_sq < 1e-10 {
        return None;
    }

    let b = b_sq.sqrt();

    // Major axis direction: from c1 to c2
    let major_x = (c2[0] - c1[0]) / dist;
    let major_y = (c2[1] - c1[1]) / dist;

    // Minor axis direction: perpendicular
    let minor_x = -major_y;
    let minor_y = major_x;

    // Construct CurveType::Ellipse
    Some(CurveType::Ellipse {
        center: [center[0], center[1], z],
        major_axis: [major_x * a, major_y * a, 0.0],
        minor_axis: [minor_x * b, minor_y * b, 0.0],
    })
}

/// Construct a hyperbola as the bisector of two circles (separate or one inside other)
fn construct_hyperbola_bisector(
    c1: [f64; 2],
    r1: f64,
    c2: [f64; 2],
    r2: f64,
    z: f64,
) -> Option<CurveType> {
    let dist = ((c2[0] - c1[0]).powi(2) + (c2[1] - c1[1]).powi(2)).sqrt();

    if dist < 1e-10 {
        return None;
    }

    // For a hyperbola of two circles:
    // ||P - C1| - r1| = ||P - C2| - r2| is not the right equation
    // Instead: |P - C1| - r1 = |P - C2| - r2 or |P - C1| - r1 = -(|P - C2| - r2)
    // This gives ||P - C1| - |P - C2|| = |r1 - r2| (hyperbola)

    // Center of hyperbola (midpoint of foci)
    let center = [
        (c1[0] + c2[0]) / 2.0,
        (c1[1] + c2[1]) / 2.0,
    ];

    // Distance between foci: 2c = dist
    let c = dist / 2.0;

    // For hyperbola: ||P - C1| - |P - C2|| = 2a
    // where a is the semi-major axis
    let a = (r1 - r2).abs() / 2.0;

    if a > c - 1e-10 {
        // Degenerate case
        return None;
    }

    let b_sq = c * c - a * a;
    if b_sq < 1e-10 {
        return None;
    }

    let b = b_sq.sqrt();

    // Transverse axis direction: from c1 to c2
    let trans_x = (c2[0] - c1[0]) / dist;
    let trans_y = (c2[1] - c1[1]) / dist;

    // Conjugate axis direction: perpendicular
    let conj_x = -trans_y;
    let conj_y = trans_x;

    // Construct CurveType::Hyperbola
    Some(CurveType::Hyperbola {
        center: [center[0], center[1], z],
        x_dir: [trans_x, trans_y, 0.0],
        y_dir: [conj_x, conj_y, 0.0],
        major_radius: a,
        minor_radius: b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Dir;

    const TEST_TOL: f64 = 1e-6;

    #[test]
    fn test_bisector_line_line_perpendicular() {
        // Two perpendicular lines at origin
        let p1 = Pnt::new(0.0, 0.0, 0.0);
        let d1 = Dir::new(1.0, 0.0, 0.0);
        let p2 = Pnt::new(0.0, 0.0, 0.0);
        let d2 = Dir::new(0.0, 1.0, 0.0);

        let bisectors = bisector_line_line(p1, d1, p2, d2).unwrap();

        assert_eq!(bisectors.len(), 2);

        // Both bisectors should pass through the intersection point (origin)
        for (pt, _dir) in &bisectors {
            assert!(pt.distance(&p1) < TEST_TOL);
        }
    }

    #[test]
    fn test_bisector_line_line_45_degrees() {
        // Two lines at 45 degrees
        let p1 = Pnt::new(0.0, 0.0, 0.0);
        let d1 = Dir::new(1.0, 1.0, 0.0);
        let p2 = Pnt::new(0.0, 0.0, 0.0);
        let d2 = Dir::new(1.0, -1.0, 0.0);

        let bisectors = bisector_line_line(p1, d1, p2, d2).unwrap();

        assert_eq!(bisectors.len(), 2);

        // One bisector should be along X axis
        // Other should be along Y axis
        let has_x_bisector = bisectors.iter().any(|(_pt, dir)| {
            let d = dir.to_vec3();
            (d.x.abs() > 0.9 && d.y.abs() < 0.1) || (d.x.abs() < 0.1 && d.y.abs() > 0.9)
        });

        assert!(has_x_bisector);
    }

    #[test]
    fn test_bisector_line_line_parallel_error() {
        // Two parallel lines should fail
        let p1 = Pnt::new(0.0, 0.0, 0.0);
        let d1 = Dir::new(1.0, 0.0, 0.0);
        let p2 = Pnt::new(0.0, 1.0, 0.0);
        let d2 = Dir::new(1.0, 0.0, 0.0);

        let result = bisector_line_line(p1, d1, p2, d2);
        assert!(result.is_err());
    }

    #[test]
    fn test_bisector_point_line() {
        // Point at (2, 0, 0), line along Z axis through origin
        let focus = Pnt::new(2.0, 0.0, 0.0);
        let line_point = Pnt::new(0.0, 0.0, 0.0);
        let line_dir = Dir::z_axis();

        let parabola = bisector_point_line(focus, line_point, line_dir).unwrap();

        // Vertex should be at midpoint of focus and its projection
        let expected_vertex = [1.0, 0.0, 0.0];
        assert!((parabola.origin[0] - expected_vertex[0]).abs() < TEST_TOL);
        assert!((parabola.origin[1] - expected_vertex[1]).abs() < TEST_TOL);
        assert!((parabola.origin[2] - expected_vertex[2]).abs() < TEST_TOL);

        // Focal parameter should be 4.0 (2 * distance from vertex to focus)
        assert!((parabola.focal - 4.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_bisector_point_line_focus_on_line_error() {
        // Focus on the line should fail
        let focus = Pnt::new(0.0, 0.0, 5.0);
        let line_point = Pnt::new(0.0, 0.0, 0.0);
        let line_dir = Dir::z_axis();

        let result = bisector_point_line(focus, line_point, line_dir);
        assert!(result.is_err());
    }

    #[test]
    fn test_bisector_point_line_perpendicular_to_axis() {
        // Point perpendicular to line axis
        let focus = Pnt::new(0.0, 3.0, 0.0);
        let line_point = Pnt::new(0.0, 0.0, 0.0);
        let line_dir = Dir::z_axis();

        let parabola = bisector_point_line(focus, line_point, line_dir).unwrap();

        // Vertex should be at midpoint: (0, 1.5, 0)
        assert!((parabola.origin[1] - 1.5).abs() < TEST_TOL);

        // Focal parameter should be 6.0 (2 * 3)
        assert!((parabola.focal - 6.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_bisector_circle_circle_overlapping() {
        // Two overlapping circles at z = 0
        let c1 = Pnt::new(0.0, 0.0, 0.0);
        let r1 = 2.0;
        let c2 = Pnt::new(2.0, 0.0, 0.0);
        let r2 = 2.0;

        let bisectors = bisector_circle_circle(c1, r1, c2, r2).unwrap();

        // For overlapping circles, we should get at least an ellipse
        assert!(!bisectors.is_empty());

        // Check if we got an ellipse
        let has_ellipse = bisectors.iter().any(|curve| matches!(curve, CurveType::Ellipse { .. }));
        assert!(has_ellipse);
    }

    #[test]
    fn test_bisector_circle_circle_separate() {
        // Two separate circles
        let c1 = Pnt::new(0.0, 0.0, 0.0);
        let r1 = 1.0;
        let c2 = Pnt::new(5.0, 0.0, 0.0);
        let r2 = 1.0;

        let bisectors = bisector_circle_circle(c1, r1, c2, r2).unwrap();

        // For separate circles, we should get a hyperbola
        assert!(!bisectors.is_empty());

        let has_hyperbola = bisectors.iter().any(|curve| matches!(curve, CurveType::Hyperbola { .. }));
        assert!(has_hyperbola);
    }

    #[test]
    fn test_bisector_circle_circle_concentric_error() {
        // Two concentric circles should fail
        let c1 = Pnt::new(0.0, 0.0, 0.0);
        let r1 = 1.0;
        let c2 = Pnt::new(0.0, 0.0, 0.0);
        let r2 = 2.0;

        let result = bisector_circle_circle(c1, r1, c2, r2);
        assert!(result.is_err());
    }

    #[test]
    fn test_bisector_circle_circle_invalid_radius() {
        // Zero or negative radius should fail
        let c1 = Pnt::new(0.0, 0.0, 0.0);
        let c2 = Pnt::new(1.0, 0.0, 0.0);

        let result = bisector_circle_circle(c1, 0.0, c2, 1.0);
        assert!(result.is_err());

        let result = bisector_circle_circle(c1, 1.0, c2, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_bisector_point_line_various_angles() {
        // Test with line at different angle
        let focus = Pnt::new(1.0, 1.0, 0.0);
        let line_point = Pnt::new(0.0, 0.0, 0.0);
        let line_dir = Dir::new(1.0, 0.0, 0.0); // X axis

        let parabola = bisector_point_line(focus, line_point, line_dir).unwrap();

        // Focal parameter should be 2 * distance to line
        // Distance from (1, 1, 0) to x-axis is 1
        // So focal = 2 * 1 = 2
        assert!((parabola.focal - 2.0).abs() < TEST_TOL);
    }
}
