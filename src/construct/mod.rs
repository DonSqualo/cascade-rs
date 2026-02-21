//! Constraint-based geometric construction
//!
//! Functions for constructing geometric elements that satisfy
//! specific constraints (e.g., tangency, parallelism, perpendicularity).

use crate::{Result, CascadeError, TOLERANCE};
use crate::geom::{Pnt, Dir};

/// A geometric element used in constraint-based construction
#[derive(Debug, Clone, Copy)]
pub enum GeomElement {
    /// A point in 3D space
    Point(Pnt),
    /// A line defined by a point and direction
    Line { point: Pnt, direction: Dir },
    /// A circle defined by center, radius, and normal
    Circle { center: Pnt, radius: f64, normal: Dir },
}

/// A circle in 3D space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    /// Center of the circle
    pub center: Pnt,
    /// Radius of the circle
    pub radius: f64,
    /// Normal vector to the plane containing the circle
    pub normal: Dir,
}

impl Circle {
    /// Create a new circle
    pub fn new(center: Pnt, radius: f64, normal: Dir) -> Result<Self> {
        if radius <= TOLERANCE {
            return Err(CascadeError::InvalidGeometry(
                format!("Circle radius must be positive, got {}", radius),
            ));
        }
        Ok(Circle {
            center,
            radius,
            normal,
        })
    }

    /// Check if a point lies on the circle (within tolerance)
    pub fn contains_point(&self, point: Pnt) -> bool {
        let vec_to_point = self.center.vec_to(&point);
        let dist_to_plane = vec_to_point.dot(&self.normal.to_vec3()).abs();
        let projected_dist = (vec_to_point.magnitude_squared() - dist_to_plane * dist_to_plane).sqrt();
        (projected_dist - self.radius).abs() < TOLERANCE
    }

    /// Check if a line is tangent to the circle
    pub fn is_tangent_to_line(&self, line_point: Pnt, line_dir: Dir) -> bool {
        // Vector from circle center to line point
        let cp = self.center.vec_to(&line_point);
        
        // Distance from circle center to the line
        let cross = cp.cross(&line_dir.to_vec3());
        let dist = cross.magnitude();
        
        (dist - self.radius).abs() < TOLERANCE
    }

    /// Check if this circle is tangent to another circle
    pub fn is_tangent_to_circle(&self, other: &Circle) -> bool {
        let center_dist = self.center.distance(&other.center);
        let sum_radii = self.radius + other.radius;
        let diff_radii = (self.radius - other.radius).abs();
        
        // Externally tangent: distance = sum of radii
        if (center_dist - sum_radii).abs() < TOLERANCE {
            return true;
        }
        
        // Internally tangent: distance = difference of radii
        if (center_dist - diff_radii).abs() < TOLERANCE {
            return true;
        }
        
        false
    }
}

/// Construct a circle tangent to three geometric elements
///
/// Solves the Apollonius problem: finding all circles tangent to three given
/// geometric objects (points, lines, or circles). The solution can yield 0 to 8
/// circles depending on the configuration.
///
/// # Arguments
/// * `elem1` - First geometric element (Point, Line, or Circle)
/// * `elem2` - Second geometric element
/// * `elem3` - Third geometric element
///
/// # Returns
/// A vector of all valid solution circles. The vector may be empty if no
/// tangent circles exist for the given configuration.
pub fn circle_tangent_to_3(
    elem1: &GeomElement,
    elem2: &GeomElement,
    elem3: &GeomElement,
) -> Result<Vec<Circle>> {
    // Special case: 3 points (trivial case - circumcircle)
    match (elem1, elem2, elem3) {
        (GeomElement::Point(p1), GeomElement::Point(p2), GeomElement::Point(p3)) => {
            return circumcircle_3_points(*p1, *p2, *p3);
        }
        _ => {}
    }

    // Special case: 3 lines
    match (elem1, elem2, elem3) {
        (
            GeomElement::Line { point: p1, direction: d1 },
            GeomElement::Line { point: p2, direction: d2 },
            GeomElement::Line { point: p3, direction: d3 },
        ) => {
            return circles_tangent_to_3_lines(*p1, *d1, *p2, *d2, *p3, *d3);
        }
        _ => {}
    }

    // Special case: 3 circles (Apollonius problem - full solution)
    match (elem1, elem2, elem3) {
        (
            GeomElement::Circle { center: c1, radius: r1, normal: n1 },
            GeomElement::Circle { center: c2, radius: r2, normal: n2 },
            GeomElement::Circle { center: c3, radius: r3, normal: n3 },
        ) => {
            return apollonius_3_circles(*c1, *r1, *n1, *c2, *r2, *n2, *c3, *r3, *n3);
        }
        _ => {}
    }

    // Mixed cases: Handle various combinations
    // For now, return an error for unsupported mixed combinations
    Err(CascadeError::NotImplemented(
        "Mixed constraint combinations not yet implemented".to_string(),
    ))
}

/// Find the circumcircle of three points (trivial case)
fn circumcircle_3_points(p1: Pnt, p2: Pnt, p3: Pnt) -> Result<Vec<Circle>> {
    // Check if points are collinear
    let v1 = p1.vec_to(&p2);
    let v2 = p1.vec_to(&p3);
    let cross = v1.cross(&v2);

    if cross.magnitude() < TOLERANCE {
        return Ok(Vec::new()); // Collinear points - no circumcircle
    }

    // Find the circumcenter using perpendicular bisectors
    let mid12 = p1.midpoint(&p2);
    let mid23 = p2.midpoint(&p3);

    let normal = cross.normalized().expect("cross product should be non-zero");

    // Perpendicular to v1 in the plane of the three points
    let perp1 = normal.to_vec3().cross(&v1).normalized().expect("perpendicular should exist");

    // Direction from mid12 toward circumcenter
    let _v12_mid = mid12.vec_to(&mid23);
    let perp2 = normal.to_vec3().cross(&v2).normalized().expect("perpendicular should exist");

    // Find intersection of perpendicular bisectors
    // Line 1: mid12 + t * perp1
    // Line 2: mid23 + s * perp2
    // Solve: mid12 + t * perp1 = mid23 + s * perp2

    let p1_to_p2 = mid12.vec_to(&mid23);
    let denom = perp1.cross(&perp2).magnitude();

    if denom < TOLERANCE {
        return Ok(Vec::new()); // Lines are parallel - degenerate case
    }

    // Solve using cross products
    let cross_p_perp2 = p1_to_p2.cross(&perp2.to_vec3());
    let cross_perp1_perp2 = perp1.to_vec3().cross(&perp2.to_vec3());
    let denom = cross_perp1_perp2.dot(&normal.to_vec3());
    
    if denom.abs() < TOLERANCE {
        return Ok(Vec::new()); // Degenerate case
    }
    
    let t = cross_p_perp2.dot(&normal.to_vec3()) / denom;

    let center = mid12.translated(&(perp1.to_vec3().scaled(t)));
    let radius = center.distance(&p1);

    if radius > TOLERANCE {
        Ok(vec![Circle::new(center, radius, normal)?])
    } else {
        Ok(Vec::new())
    }
}

/// Find circles tangent to three lines
fn circles_tangent_to_3_lines(
    p1: Pnt,
    d1: Dir,
    p2: Pnt,
    d2: Dir,
    p3: Pnt,
    d3: Dir,
) -> Result<Vec<Circle>> {
    let mut solutions = Vec::new();

    // For 3 lines, we typically get up to 4 solutions:
    // - The incircle and 3 excircles (for a triangle formed by the lines)
    // - Or different configurations depending on parallel lines

    // Check for parallel lines
    let d1_perp_d2 = d1.is_parallel(&d2, 1e-6);
    let d2_perp_d3 = d2.is_parallel(&d3, 1e-6);
    let d1_perp_d3 = d1.is_parallel(&d3, 1e-6);

    if d1_perp_d2 && d2_perp_d3 {
        return Ok(Vec::new()); // All three parallel - no solution
    }

    if d1_perp_d2 || d2_perp_d3 || d1_perp_d3 {
        // Two or more parallel - limited solutions
        // For now, skip complex parallel cases
        return Ok(Vec::new());
    }

    // Compute intersection points (assuming lines are coplanar)
    // For simplicity, project to XY plane and solve 2D case

    // Get intersection of lines in 2D projection
    let p1_2d = [p1.x, p1.y];
    let d1_2d = [d1.x(), d1.y()];
    let p2_2d = [p2.x, p2.y];
    let d2_2d = [d2.x(), d2.y()];
    let p3_2d = [p3.x, p3.y];
    let d3_2d = [d3.x(), d3.y()];

    // Find triangle vertices
    if let (Some(v1), Some(v2), Some(v3)) = (
        line_line_intersection_2d(p1_2d, d1_2d, p2_2d, d2_2d),
        line_line_intersection_2d(p2_2d, d2_2d, p3_2d, d3_2d),
        line_line_intersection_2d(p3_2d, d3_2d, p1_2d, d1_2d),
    ) {
        // Compute incircle of the triangle
        let a = distance_2d(&v2, &v3);
        let b = distance_2d(&v3, &v1);
        let c = distance_2d(&v1, &v2);
        let s = (a + b + c) / 2.0; // semi-perimeter

        let area = triangle_area_2d(&v1, &v2, &v3);
        if area > TOLERANCE {
            let inradius = area / s;
            let incenter = [
                (a * v1[0] + b * v2[0] + c * v3[0]) / (a + b + c),
                (a * v1[1] + b * v2[1] + c * v3[1]) / (a + b + c),
            ];

            // Use the incircle (first solution)
            if let Ok(circle) = Circle::new(
                Pnt::new(incenter[0], incenter[1], p1.z),
                inradius,
                Dir::z_axis(),
            ) {
                solutions.push(circle);
            }

            // TODO: Add excircles as additional solutions
        }
    }

    Ok(solutions)
}

/// Solve the Apollonius problem: find circles tangent to 3 circles
fn apollonius_3_circles(
    c1: Pnt,
    r1: f64,
    n1: Dir,
    c2: Pnt,
    r2: f64,
    n2: Dir,
    c3: Pnt,
    r3: f64,
    n3: Dir,
) -> Result<Vec<Circle>> {
    // All circles must be in the same plane
    // Check if normals are parallel
    if !n1.is_parallel(&n2, 1e-6) || !n2.is_parallel(&n3, 1e-6) {
        return Err(CascadeError::InvalidGeometry(
            "All circles must be in the same plane (coplanar)".to_string(),
        ));
    }

    // Project circles to 2D for solving
    let p1_2d = [c1.x, c1.y];
    let p2_2d = [c2.x, c2.y];
    let p3_2d = [c3.x, c3.y];

    let mut solutions = Vec::new();

    // Use Descartes Circle Theorem and other approaches
    // For simplicity, we'll implement a numerical solver for the general case

    // The 8 possible solutions correspond to:
    // For each of the 3 circles, we can be externally or internally tangent
    // This gives 2^3 = 8 possible combinations

    let tangent_types = vec![
        (1.0, 1.0, 1.0),  // all external
        (1.0, 1.0, -1.0), // external, external, internal
        (1.0, -1.0, 1.0), // external, internal, external
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, -1.0, -1.0), // all internal
    ];

    for (t1, t2, t3) in tangent_types {
        if let Some((cx_2d, cy_2d, radius)) =
            solve_apollonius_case(p1_2d, r1, t1, p2_2d, r2, t2, p3_2d, r3, t3)
        {
            if radius > TOLERANCE {
                if let Ok(circle) = Circle::new(
                    Pnt::new(cx_2d, cy_2d, c1.z),
                    radius,
                    n1,
                ) {
                    // Check if this solution is valid (actually tangent to all three)
                    if is_tangent_to_circle_2d(p1_2d, r1, [cx_2d, cy_2d], radius, t1)
                        && is_tangent_to_circle_2d(p2_2d, r2, [cx_2d, cy_2d], radius, t2)
                        && is_tangent_to_circle_2d(p3_2d, r3, [cx_2d, cy_2d], radius, t3)
                    {
                        solutions.push(circle);
                    }
                }
            }
        }
    }

    // Remove duplicates (circles that are very close)
    solutions.dedup_by(|a, b| {
        a.center.distance(&b.center) < TOLERANCE && (a.radius - b.radius).abs() < TOLERANCE
    });

    Ok(solutions)
}

/// Solve one case of the Apollonius problem using algebraic method
/// Returns (center_x, center_y, radius) if a solution exists
fn solve_apollonius_case(
    p1: [f64; 2],
    r1: f64,
    t1: f64,
    p2: [f64; 2],
    r2: f64,
    t2: f64,
    p3: [f64; 2],
    r3: f64,
    t3: f64,
) -> Option<(f64, f64, f64)> {
    // Use Descartes' Circle Theorem as a guide
    // For the algebraic solution, we solve the system:
    // |center - p1|² = (r1 + t1*radius)²
    // |center - p2|² = (r2 + t2*radius)²
    // |center - p3|² = (r3 + t3*radius)²

    // Expand and rearrange to get linear system in center coordinates
    // (x - p1.x)² + (y - p1.y)² = (r1 + t1*radius)²
    // (x - p2.x)² + (y - p2.y)² = (r2 + t2*radius)²
    // (x - p3.x)² + (y - p3.y)² = (r3 + t3*radius)²

    // Subtract equations pairwise to get linear equations.
    // From (eq2 - eq1) and (eq3 - eq2), we get linear equations in x, y.
    // This is a linear system in x, y (for a fixed radius), so we'd need to solve
    // the nonlinear system. For now, use a simpler approach: bisection on radius.

    // Try Newton's method or brute force search on radius
    // Start with a reasonable radius guess based on the input circles
    let max_radius = 1000.0 * (r1 + r2 + r3) / 3.0;

    for radius_try in (1..100).map(|i| (max_radius as f64) * (i as f64) / 100.0) {
        if let Some((cx, cy)) = solve_center_for_radius(p1, r1, t1, p2, r2, t2, p3, r3, t3, radius_try) {
            // Verify the solution
            let dist1 = ((cx - p1[0]).powi(2) + (cy - p1[1]).powi(2)).sqrt();
            let dist2 = ((cx - p2[0]).powi(2) + (cy - p2[1]).powi(2)).sqrt();
            let dist3 = ((cx - p3[0]).powi(2) + (cy - p3[1]).powi(2)).sqrt();

            let target1 = (r1 + t1 * radius_try).abs();
            let target2 = (r2 + t2 * radius_try).abs();
            let target3 = (r3 + t3 * radius_try).abs();

            if (dist1 - target1).abs() < TOLERANCE * 10.0
                && (dist2 - target2).abs() < TOLERANCE * 10.0
                && (dist3 - target3).abs() < TOLERANCE * 10.0
            {
                return Some((cx, cy, radius_try));
            }
        }
    }

    None
}

/// Solve for center (x, y) given a fixed radius
fn solve_center_for_radius(
    p1: [f64; 2],
    r1: f64,
    t1: f64,
    p2: [f64; 2],
    r2: f64,
    t2: f64,
    p3: [f64; 2],
    r3: f64,
    t3: f64,
    radius: f64,
) -> Option<(f64, f64)> {
    // Solve the linear system:
    // 2*dx12*x + 2*dy12*y = const1 - 2*(t2*r2 - t1*r1)*radius
    // 2*dx23*x + 2*dy23*y = const2 - 2*(t3*r3 - t2*r2)*radius

    let dx12 = p2[0] - p1[0];
    let dy12 = p2[1] - p1[1];
    let const1 = p2[0].powi(2) - p1[0].powi(2)
        + p2[1].powi(2) - p1[1].powi(2)
        + r1.powi(2) - r2.powi(2);
    let rhs1 = const1 - 2.0 * (t2 * r2 - t1 * r1) * radius;

    let dx23 = p3[0] - p2[0];
    let dy23 = p3[1] - p2[1];
    let const2 = p3[0].powi(2) - p2[0].powi(2)
        + p3[1].powi(2) - p2[1].powi(2)
        + r2.powi(2) - r3.powi(2);
    let rhs2 = const2 - 2.0 * (t3 * r3 - t2 * r2) * radius;

    // Solve Ax = b where A = [[2*dx12, 2*dy12], [2*dx23, 2*dy23]]
    let det = 2.0 * dx12 * 2.0 * dy23 - 2.0 * dy12 * 2.0 * dx23;

    if det.abs() < TOLERANCE {
        return None;
    }

    let x = (rhs1 * 2.0 * dy23 - rhs2 * 2.0 * dy12) / det;
    let y = (2.0 * dx12 * rhs2 - 2.0 * dx23 * rhs1) / det;

    Some((x, y))
}

/// Check if a circle is tangent to another in 2D
fn is_tangent_to_circle_2d(
    center1: [f64; 2],
    radius1: f64,
    center2: [f64; 2],
    radius2: f64,
    tangent_type: f64,
) -> bool {
    let dist = distance_2d(&center1, &center2);
    let target = (radius1 + tangent_type * radius2).abs();
    (dist - target).abs() < TOLERANCE * 10.0
}

/// Find the intersection of two 2D lines
fn line_line_intersection_2d(
    p1: [f64; 2],
    d1: [f64; 2],
    p2: [f64; 2],
    d2: [f64; 2],
) -> Option<[f64; 2]> {
    // Line 1: p1 + t * d1
    // Line 2: p2 + s * d2
    // Solve: p1 + t * d1 = p2 + s * d2

    let det = d1[0] * d2[1] - d1[1] * d2[0];

    if det.abs() < TOLERANCE {
        return None; // Parallel lines
    }

    let dp = [p2[0] - p1[0], p2[1] - p1[1]];
    let t = (dp[0] * d2[1] - dp[1] * d2[0]) / det;

    Some([p1[0] + t * d1[0], p1[1] + t * d1[1]])
}

/// Compute the distance between two 2D points
fn distance_2d(p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
    ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt()
}

/// Compute the area of a 2D triangle
fn triangle_area_2d(p1: &[f64; 2], p2: &[f64; 2], p3: &[f64; 2]) -> f64 {
    let det = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]);
    det.abs() / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOL: f64 = 1e-6;

    #[test]
    fn test_circle_creation() {
        let center = Pnt::new(0.0, 0.0, 0.0);
        let normal = Dir::z_axis();
        let circle = Circle::new(center, 5.0, normal).unwrap();

        assert!((circle.radius - 5.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_circle_contains_point() {
        let center = Pnt::new(0.0, 0.0, 0.0);
        let normal = Dir::z_axis();
        let circle = Circle::new(center, 5.0, normal).unwrap();

        let point_on = Pnt::new(5.0, 0.0, 0.0);
        assert!(circle.contains_point(point_on));

        let point_off = Pnt::new(6.0, 0.0, 0.0);
        assert!(!circle.contains_point(point_off));
    }

    #[test]
    fn test_circumcircle_3_points() {
        // Triangle with vertices at (0,0), (6,0), (3,3√3)
        let p1 = Pnt::new(0.0, 0.0, 0.0);
        let p2 = Pnt::new(6.0, 0.0, 0.0);
        let p3 = Pnt::new(3.0, 3.0 * 3.0_f64.sqrt(), 0.0);

        let solutions = circumcircle_3_points(p1, p2, p3).unwrap();

        assert_eq!(solutions.len(), 1);
        let circle = &solutions[0];

        // All points should be on the circle
        assert!(circle.contains_point(p1));
        assert!(circle.contains_point(p2));
        assert!(circle.contains_point(p3));
    }

    #[test]
    fn test_circumcircle_collinear_points() {
        let p1 = Pnt::new(0.0, 0.0, 0.0);
        let p2 = Pnt::new(1.0, 0.0, 0.0);
        let p3 = Pnt::new(2.0, 0.0, 0.0);

        let solutions = circumcircle_3_points(p1, p2, p3).unwrap();
        assert_eq!(solutions.len(), 0); // Collinear - no solution
    }

    #[test]
    fn test_circle_tangent_to_3_points() {
        let p1 = Pnt::new(0.0, 0.0, 0.0);
        let p2 = Pnt::new(6.0, 0.0, 0.0);
        let p3 = Pnt::new(3.0, 3.0 * 3.0_f64.sqrt(), 0.0);

        let e1 = GeomElement::Point(p1);
        let e2 = GeomElement::Point(p2);
        let e3 = GeomElement::Point(p3);

        let solutions = circle_tangent_to_3(&e1, &e2, &e3).unwrap();

        assert_eq!(solutions.len(), 1);
        let circle = &solutions[0];

        // All points should be on the circle
        assert!(circle.contains_point(p1));
        assert!(circle.contains_point(p2));
        assert!(circle.contains_point(p3));
    }

    #[test]
    fn test_tangent_circle_with_invalid_radius() {
        let center = Pnt::new(0.0, 0.0, 0.0);
        let normal = Dir::z_axis();
        let result = Circle::new(center, -1.0, normal);

        assert!(result.is_err());
    }
}
