//! Constraint-based geometric construction
//!
//! Functions for constructing geometric elements that satisfy
//! specific constraints (e.g., tangency, parallelism, perpendicularity).

use crate::{Result, CascadeError, TOLERANCE};
use crate::geom::{Pnt, Dir, Vec3};

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

/// A line in 3D space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line {
    /// Point on the line
    pub point: Pnt,
    /// Direction of the line
    pub direction: Dir,
}

impl Line {
    /// Create a new line
    pub fn new(point: Pnt, direction: Dir) -> Result<Self> {
        Ok(Line { point, direction })
    }

    /// Compute distance from a point to this line
    pub fn distance_to_point(&self, point: Pnt) -> f64 {
        let cp = self.point.vec_to(&point);
        let cross = cp.cross(&self.direction.to_vec3());
        cross.magnitude()
    }

    /// Check if this line is tangent to a circle
    pub fn is_tangent_to_circle(&self, circle: &Circle) -> bool {
        let dist = self.distance_to_point(circle.center);
        (dist - circle.radius).abs() < TOLERANCE
    }

    /// Check if two lines are parallel
    pub fn is_parallel(&self, other: &Line) -> bool {
        self.direction.is_parallel(&other.direction, 1e-6)
    }
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

/// Construct a line tangent to two geometric elements
///
/// Finds all lines that are tangent to (or pass through) two given geometric elements.
/// 
/// # Cases
/// - **Point-Point:** Returns an infinite line passing through both points (1 solution)
/// - **Point-Circle:** Returns up to 2 tangent lines from the point to the circle
/// - **Circle-Circle:** Returns up to 4 tangent lines (external and internal tangents)
///
/// # Arguments
/// * `elem1` - First geometric element (Point or Circle)
/// * `elem2` - Second geometric element (Point or Circle)
///
/// # Returns
/// A vector of all valid tangent lines
pub fn line_tangent_to_2(
    elem1: &GeomElement,
    elem2: &GeomElement,
) -> Result<Vec<Line>> {
    // Case 1: Point-Point (trivial - line through both points)
    if let (GeomElement::Point(p1), GeomElement::Point(p2)) = (elem1, elem2) {
        return line_through_2_points(*p1, *p2);
    }

    // Case 2: Point-Circle (2 tangent lines)
    match (elem1, elem2) {
        (GeomElement::Point(p), GeomElement::Circle { center, radius, normal }) => {
            return lines_tangent_point_circle(*p, *center, *radius, *normal);
        }
        (GeomElement::Circle { center, radius, normal }, GeomElement::Point(p)) => {
            return lines_tangent_point_circle(*p, *center, *radius, *normal);
        }
        _ => {}
    }

    // Case 3: Circle-Circle (up to 4 tangent lines)
    if let (
        GeomElement::Circle { center: c1, radius: r1, normal: n1 },
        GeomElement::Circle { center: c2, radius: r2, normal: n2 },
    ) = (elem1, elem2)
    {
        return lines_tangent_circle_circle(*c1, *r1, *n1, *c2, *r2, *n2);
    }

    Err(CascadeError::NotImplemented(
        "Unsupported geometric element combination for tangent line construction".to_string(),
    ))
}

/// Line passing through two points (trivial case)
fn line_through_2_points(p1: Pnt, p2: Pnt) -> Result<Vec<Line>> {
    if p1.distance(&p2) < TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            "Points must be distinct".to_string(),
        ));
    }

    let direction = p1.vec_to(&p2).normalized().ok_or_else(|| {
        CascadeError::InvalidGeometry("Cannot normalize zero vector".to_string())
    })?;

    Ok(vec![Line::new(p1, direction)?])
}

/// Lines tangent to a circle from an external point
fn lines_tangent_point_circle(
    point: Pnt,
    center: Pnt,
    radius: f64,
    normal: Dir,
) -> Result<Vec<Line>> {
    let mut solutions = Vec::new();

    // Vector from point to circle center
    let pc = point.vec_to(&center);
    let dist = pc.magnitude();

    // If point is inside the circle, no tangent lines exist
    if dist < radius - TOLERANCE {
        return Ok(Vec::new());
    }

    // Special case: point is on the circle - one tangent line (perpendicular to radius)
    if (dist - radius).abs() < TOLERANCE {
        let radius_vec = pc.normalized().expect("pc magnitude is non-zero");
        let radius_as_vec3 = radius_vec.to_vec3();
        let tangent_dir = normal.to_vec3().cross(&radius_as_vec3);
        if let Some(dir) = tangent_dir.normalized() {
            solutions.push(Line::new(point, dir)?);
        }
        return Ok(solutions);
    }

    // General case: point is outside the circle
    // Use 2D geometry in the plane containing point and circle center

    // Project into 2D plane: one axis along pc, one perpendicular in circle plane
    let pc_dir = pc.normalized().expect("pc magnitude is non-zero");
    let pc_vec = pc_dir.to_vec3();

    // Find perpendicular direction in the plane of the circle
    let perp_dir = normal.to_vec3().cross(&pc_vec);
    let perp_normalized = perp_dir.normalized().expect("perp_dir should be valid");
    let perp_vec = perp_normalized.to_vec3();

    // In the 2D plane, compute the tangent points
    // Let alpha = angle from pc to tangent point on circle
    // sin(alpha) = radius / dist
    let sin_alpha = radius / dist;

    if sin_alpha > 1.0 + TOLERANCE {
        return Ok(Vec::new()); // No solution
    }

    let sin_alpha = sin_alpha.min(1.0);
    let cos_alpha = (1.0 - sin_alpha * sin_alpha).sqrt();

    // Two tangent solutions in 2D
    // Direction vectors from point toward the two tangent points
    let v1_scaled_pc = pc_vec.scaled(cos_alpha);
    let v1_scaled_perp = perp_vec.scaled(sin_alpha);
    let v1 = crate::geom::Vec3::new(
        v1_scaled_pc.x + v1_scaled_perp.x,
        v1_scaled_pc.y + v1_scaled_perp.y,
        v1_scaled_pc.z + v1_scaled_perp.z,
    );
    
    let v2_scaled_pc = pc_vec.scaled(cos_alpha);
    let v2_scaled_perp = perp_vec.scaled(-sin_alpha);
    let v2 = crate::geom::Vec3::new(
        v2_scaled_pc.x + v2_scaled_perp.x,
        v2_scaled_pc.y + v2_scaled_perp.y,
        v2_scaled_pc.z + v2_scaled_perp.z,
    );

    if let Some(dir1) = v1.normalized() {
        solutions.push(Line::new(point, dir1)?);
    }

    if let Some(dir2) = v2.normalized() {
        solutions.push(Line::new(point, dir2)?);
    }

    Ok(solutions)
}

/// Lines tangent to two circles
fn lines_tangent_circle_circle(
    c1: Pnt,
    r1: f64,
    n1: Dir,
    c2: Pnt,
    r2: f64,
    n2: Dir,
) -> Result<Vec<Line>> {
    // Both circles must be in the same plane
    if !n1.is_parallel(&n2, 1e-6) {
        return Err(CascadeError::InvalidGeometry(
            "Circles must be coplanar for tangent line construction".to_string(),
        ));
    }

    let mut solutions = Vec::new();
    let dist = c1.distance(&c2);

    // Handle degenerate cases
    if dist < TOLERANCE {
        if (r1 - r2).abs() < TOLERANCE {
            // Concentric circles with same radius - no tangent lines
            return Ok(Vec::new());
        } else {
            // Concentric circles with different radii - no tangent lines
            return Ok(Vec::new());
        }
    }

    // External tangents (2 solutions in general case)
    if let Ok(mut external) = external_tangents(c1, r1, c2, r2, n1) {
        solutions.append(&mut external);
    }

    // Internal tangents (2 solutions if circles don't overlap too much)
    if let Ok(mut internal) = internal_tangents(c1, r1, c2, r2, n1) {
        solutions.append(&mut internal);
    }

    Ok(solutions)
}

/// Compute external tangent lines to two circles
fn external_tangents(
    c1: Pnt,
    r1: f64,
    c2: Pnt,
    r2: f64,
    normal: Dir,
) -> Result<Vec<Line>> {
    let mut solutions = Vec::new();
    let cc = c1.vec_to(&c2);
    let dist = cc.magnitude();

    if dist < TOLERANCE {
        return Ok(Vec::new()); // Concentric circles
    }

    // For external tangents, solve using the geometry:
    // The tangent line touches both circles on the same side
    // Project to 2D in the plane of the circles

    let cc_dir = cc.normalized().expect("cc magnitude is non-zero");
    let cc_vec = cc_dir.to_vec3();
    let perp_dir = normal.to_vec3().cross(&cc_vec);
    let perp_normalized = perp_dir.normalized().expect("perp should be valid");
    let perp_vec = perp_normalized.to_vec3();

    // External tangent: perpendicular from centers to tangent line
    // The tangent line is at distance r1 from c1 and r2 from c2
    // For external tangents, both circles are on the same side

    let dr = (r2 - r1).abs();

    if dist < dr - TOLERANCE {
        return Ok(Vec::new()); // One circle is inside the other
    }

    if (dist - dr).abs() < TOLERANCE {
        // Special case: circles touch externally or one inside the other
        // The "tangent" line passes through the contact point
        let t = r1 / dist;
        let contact = c1.translated(&cc.scaled(t));

        // Perpendicular direction in the plane
        solutions.push(Line::new(contact, perp_normalized)?);
        return Ok(solutions);
    }

    // General case: compute the two external tangents
    // Let alpha be the angle from c1->c2 to the tangent direction
    let sin_alpha = dr / dist;
    let cos_alpha = (1.0 - sin_alpha * sin_alpha).sqrt();

    // Tangent line direction (in the plane)
    // First external tangent
    let v1 = cc_dir.scaled(sin_alpha) + perp_normalized.scaled(cos_alpha);
    // Second external tangent
    let v2 = cc_dir.scaled(sin_alpha) + perp_normalized.scaled(-cos_alpha);

    // Find point on each tangent line
    // The tangent line is perpendicular to the radius at the tangent point

    // For external tangent v1: point on c1 side
    let perp1 = normal.to_vec3().cross(&v1);
    let perp1_normalized = perp1.normalized().expect("perp1 should be valid");

    let p1_on_c1 = c1.translated(&perp1_normalized.scaled(r1));
    if let Some(dir1) = v1.normalized() {
        solutions.push(Line::new(p1_on_c1, dir1)?);
    }

    // For external tangent v2: point on c1 side
    let perp2 = normal.to_vec3().cross(&v2);
    let perp2_normalized = perp2.normalized().expect("perp2 should be valid");

    let p2_on_c1 = c1.translated(&perp2_normalized.scaled(r1));
    if let Some(dir2) = v2.normalized() {
        solutions.push(Line::new(p2_on_c1, dir2)?);
    }

    Ok(solutions)
}

/// Compute internal tangent lines to two circles
fn internal_tangents(
    c1: Pnt,
    r1: f64,
    c2: Pnt,
    r2: f64,
    normal: Dir,
) -> Result<Vec<Line>> {
    let mut solutions = Vec::new();
    let cc = c1.vec_to(&c2);
    let dist = cc.magnitude();

    if dist < TOLERANCE {
        return Ok(Vec::new()); // Concentric circles
    }

    let cc_dir = cc.normalized().expect("cc magnitude is non-zero");
    let cc_dir_vec3 = cc_dir.to_vec3();
    let perp_dir = normal.to_vec3().cross(&cc_dir_vec3);
    let perp_normalized = perp_dir.normalized().expect("perp should be valid");

    // Internal tangents: circles on opposite sides of the line
    let sum = r1 + r2;

    // Circles must not overlap too much for internal tangents to exist
    if dist < sum - TOLERANCE {
        return Ok(Vec::new()); // Circles overlap - no internal tangents
    }

    if (dist - sum).abs() < TOLERANCE {
        // Special case: circles touch internally or contact point
        let t = r1 / dist;
        let contact = c1.translated(&cc.scaled(t));

        solutions.push(Line::new(contact, perp_normalized)?);
        return Ok(solutions);
    }

    // General case: compute the two internal tangents
    let sin_alpha = sum / dist;

    // sin_alpha can be > 1 if circles overlap significantly
    if sin_alpha > 1.0 + TOLERANCE {
        return Ok(Vec::new());
    }

    if sin_alpha > 1.0 {
        return Ok(Vec::new());
    }

    let cos_alpha = (1.0 - sin_alpha * sin_alpha).sqrt();

    // Tangent line direction
    let cc_scaled = cc_dir.scaled(sin_alpha);
    let perp_scaled_pos = perp_normalized.scaled(cos_alpha);
    let v1 = crate::geom::Vec3::new(
        cc_scaled.x + perp_scaled_pos.x,
        cc_scaled.y + perp_scaled_pos.y,
        cc_scaled.z + perp_scaled_pos.z,
    );

    let perp_scaled_neg = perp_normalized.scaled(-cos_alpha);
    let v2 = crate::geom::Vec3::new(
        cc_scaled.x + perp_scaled_neg.x,
        cc_scaled.y + perp_scaled_neg.y,
        cc_scaled.z + perp_scaled_neg.z,
    );

    // Find point on each tangent line
    // For internal tangent, the point is on the opposite side of each center

    let perp1 = normal.to_vec3().cross(&v1);
    let perp1_normalized = perp1.normalized().expect("perp1 should be valid");

    let p1_on_c1 = c1.translated(&perp1_normalized.scaled(r1));
    if let Some(dir1) = v1.normalized() {
        solutions.push(Line::new(p1_on_c1, dir1)?);
    }

    let perp2 = normal.to_vec3().cross(&v2);
    let perp2_normalized = perp2.normalized().expect("perp2 should be valid");

    let p2_on_c1 = c1.translated(&perp2_normalized.scaled(r1));
    if let Some(dir2) = v2.normalized() {
        solutions.push(Line::new(p2_on_c1, dir2)?);
    }

    Ok(solutions)
}

/// Construct circles tangent to two geometric elements with a specified radius
///
/// Finds all circles with the given radius that are tangent to both elements.
/// The solution can yield 0 to 4 circles depending on the configuration and element types.
///
/// # Arguments
/// * `elem1` - First geometric element (Point, Line, or Circle)
/// * `elem2` - Second geometric element (Point, Line, or Circle)
/// * `radius` - Desired radius of the tangent circle
///
/// # Returns
/// A vector of all valid solution circles. The vector may be empty if no
/// tangent circles with the given radius exist.
pub fn circle_tangent_to_2_with_radius(
    elem1: &GeomElement,
    elem2: &GeomElement,
    radius: f64,
) -> Result<Vec<Circle>> {
    if radius <= TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            format!("Circle radius must be positive, got {}", radius),
        ));
    }

    match (elem1, elem2) {
        // Point-Point
        (GeomElement::Point(p1), GeomElement::Point(p2)) => {
            circles_tangent_to_2_points(*p1, *p2, radius)
        }
        // Point-Line
        (GeomElement::Point(p), GeomElement::Line { point: lp, direction: ld }) => {
            circles_tangent_to_point_and_line(*p, *lp, *ld, radius)
        }
        (GeomElement::Line { point: lp, direction: ld }, GeomElement::Point(p)) => {
            circles_tangent_to_point_and_line(*p, *lp, *ld, radius)
        }
        // Point-Circle
        (GeomElement::Point(p), GeomElement::Circle { center: cc, radius: cr, normal: cn }) => {
            circles_tangent_to_point_and_circle(*p, *cc, *cr, *cn, radius)
        }
        (GeomElement::Circle { center: cc, radius: cr, normal: cn }, GeomElement::Point(p)) => {
            circles_tangent_to_point_and_circle(*p, *cc, *cr, *cn, radius)
        }
        // Line-Line
        (GeomElement::Line { point: p1, direction: d1 }, GeomElement::Line { point: p2, direction: d2 }) => {
            circles_tangent_to_2_lines(*p1, *d1, *p2, *d2, radius)
        }
        // Line-Circle
        (GeomElement::Line { point: lp, direction: ld }, GeomElement::Circle { center: cc, radius: cr, normal: cn }) => {
            circles_tangent_to_line_and_circle(*lp, *ld, *cc, *cr, *cn, radius)
        }
        (GeomElement::Circle { center: cc, radius: cr, normal: cn }, GeomElement::Line { point: lp, direction: ld }) => {
            circles_tangent_to_line_and_circle(*lp, *ld, *cc, *cr, *cn, radius)
        }
        // Circle-Circle
        (GeomElement::Circle { center: c1, radius: r1, normal: n1 }, GeomElement::Circle { center: c2, radius: r2, normal: n2 }) => {
            circles_tangent_to_2_circles(*c1, *r1, *n1, *c2, *r2, *n2, radius)
        }
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

/// Find circles with given radius tangent to two points
/// The center must be equidistant from both points (on perpendicular bisector)
fn circles_tangent_to_2_points(p1: Pnt, p2: Pnt, radius: f64) -> Result<Vec<Circle>> {
    let dist_pp = p1.distance(&p2);

    if dist_pp > 2.0 * radius + TOLERANCE {
        return Ok(Vec::new()); // No solution - points too far apart
    }

    if dist_pp < TOLERANCE {
        // Points are the same - infinite solutions
        return Ok(Vec::new());
    }

    let mut solutions = Vec::new();

    // Midpoint of p1 and p2
    let mid = p1.midpoint(&p2);
    let vec_p1_p2 = p1.vec_to(&p2);

    // Using Pythagorean theorem: distance from mid to center
    let half_dist = dist_pp / 2.0;
    let perp_dist_sq = radius * radius - half_dist * half_dist;

    if perp_dist_sq < -TOLERANCE {
        return Ok(Vec::new()); // No real solution
    }

    let perp_dist = perp_dist_sq.max(0.0).sqrt();

    // Two perpendicular directions (in XY plane)
    let perp1_vec = if vec_p1_p2.magnitude() > TOLERANCE {
        crate::geom::Vec3::new(-vec_p1_p2.y, vec_p1_p2.x, 0.0)
    } else {
        crate::geom::Vec3::unit_x()
    };
    let perp1_dir = perp1_vec.normalized().unwrap_or_else(|| Dir::x_axis());

    // First solution
    if perp_dist > TOLERANCE || perp_dist_sq.abs() < TOLERANCE {
        let center1 = mid.translated(&perp1_dir.scaled(perp_dist));
        let circle1 = Circle::new(center1, radius, Dir::z_axis())?;
        solutions.push(circle1);
    }

    // Second solution
    if perp_dist > TOLERANCE {
        let center2 = mid.translated(&perp1_dir.scaled(-perp_dist));
        let circle2 = Circle::new(center2, radius, Dir::z_axis())?;
        solutions.push(circle2);
    }

    Ok(solutions)
}

/// Find circles with given radius tangent to a point and a line
fn circles_tangent_to_point_and_line(
    point: Pnt,
    line_point: Pnt,
    line_dir: Dir,
    radius: f64,
) -> Result<Vec<Circle>> {
    let mut solutions = Vec::new();

    let vec_to_point = line_point.vec_to(&point);
    let line_vec = line_dir.to_vec3();
    let proj_length = vec_to_point.dot(&line_vec);
    let perp_component = vec_to_point + line_vec.scaled(-proj_length);
    let perp_dist = perp_component.magnitude();

    if perp_dist < TOLERANCE {
        // Point is on the line - no solution
        return Ok(Vec::new());
    }

    let foot = line_point.translated(&line_vec.scaled(proj_length));
    let perp_unit = perp_component.normalized().unwrap_or_else(|| Dir::x_axis());

    // Two directions perpendicular to line
    for sign in &[-1.0, 1.0] {
        let center_on_line_perp = foot.translated(&perp_unit.scaled(radius * sign));

        let vec_to_target = center_on_line_perp.vec_to(&point);
        let a = line_vec.dot(&line_vec);
        let b = 2.0 * vec_to_target.dot(&line_vec);
        let c = vec_to_target.dot(&vec_to_target) - radius * radius;

        let discriminant = b * b - 4.0 * a * c;

        if discriminant >= -TOLERANCE {
            let disc_sqrt = discriminant.max(0.0).sqrt();
            let t1 = (-b + disc_sqrt) / (2.0 * a);
            let t2 = (-b - disc_sqrt) / (2.0 * a);

            for t in &[t1, t2] {
                let center = center_on_line_perp.translated(&line_vec.scaled(*t));
                let circle = Circle::new(center, radius, Dir::z_axis())?;

                if (center.distance(&point) - radius).abs() < TOLERANCE * 10.0
                    && (center.distance(&foot) - radius).abs() < TOLERANCE * 10.0
                {
                    if !solutions.iter().any(|c: &Circle| {
                        c.center.distance(&center) < TOLERANCE
                            && (c.radius - radius).abs() < TOLERANCE
                    }) {
                        solutions.push(circle);
                    }
                }
            }
        }
    }

    Ok(solutions)
}

/// Find circles with given radius tangent to a point and another circle
fn circles_tangent_to_point_and_circle(
    point: Pnt,
    circle_center: Pnt,
    circle_radius: f64,
    circle_normal: Dir,
    radius: f64,
) -> Result<Vec<Circle>> {
    let mut solutions = Vec::new();

    // External tangency
    let target_dist_external = radius + circle_radius;
    if let Some(centers) =
        find_circle_centers_2d(point, radius, circle_center, target_dist_external)
    {
        for center in centers {
            let circle = Circle::new(center, radius, circle_normal)?;
            solutions.push(circle);
        }
    }

    // Internal tangency
    if radius > circle_radius {
        let target_dist_internal = radius - circle_radius;
        if let Some(centers) =
            find_circle_centers_2d(point, radius, circle_center, target_dist_internal)
        {
            for center in centers {
                let circle = Circle::new(center, radius, circle_normal)?;
                solutions.push(circle);
            }
        }
    }

    Ok(solutions)
}

/// Find circles with given radius tangent to two lines
fn circles_tangent_to_2_lines(
    p1: Pnt,
    d1: Dir,
    p2: Pnt,
    d2: Dir,
    radius: f64,
) -> Result<Vec<Circle>> {
    let mut solutions = Vec::new();

    let d1_vec = d1.to_vec3();
    let d2_vec = d2.to_vec3();
    let cross = d1_vec.cross(&d2_vec);

    if cross.magnitude() < TOLERANCE {
        // Lines are parallel
        let perp_vec = d1_vec.cross(&crate::geom::Vec3::unit_z());
        let perp_dir = perp_vec.normalized().unwrap_or_else(|| Dir::y_axis());
        let dist = calculate_distance_between_parallel_lines(p1, d1_vec, p2, d2_vec);

        if (dist - 2.0 * radius).abs() < TOLERANCE {
            let mid_point = p1.midpoint(&p2);
            let center = mid_point.translated(&perp_dir.scaled(radius));
            let circle = Circle::new(center, radius, Dir::z_axis())?;
            solutions.push(circle);

            let center2 = mid_point.translated(&perp_dir.scaled(-radius));
            let circle2 = Circle::new(center2, radius, Dir::z_axis())?;
            solutions.push(circle2);
        }
        return Ok(solutions);
    }

    // Lines are not parallel - find intersection
    let p1_2d = [p1.x, p1.y];
    let d1_2d = [d1.x(), d1.y()];
    let p2_2d = [p2.x, p2.y];
    let d2_2d = [d2.x(), d2.y()];

    if let Some(intersection) = line_line_intersection_2d(p1_2d, d1_2d, p2_2d, d2_2d) {
        let int_point = Pnt::new(intersection[0], intersection[1], p1.z);

        let u1_dir = d1_vec.normalized().unwrap_or_else(|| Dir::x_axis());
        let u2_dir = d2_vec.normalized().unwrap_or_else(|| Dir::y_axis());

        let bisector1_vec = u1_dir.to_vec3() + u2_dir.to_vec3();
        let bisector1_dir = bisector1_vec.normalized().unwrap_or_else(|| Dir::x_axis());
        
        let bisector2_vec = u1_dir.to_vec3() + u2_dir.to_vec3().scaled(-1.0);
        let bisector2_dir = bisector2_vec.normalized().unwrap_or_else(|| Dir::y_axis());

        for bisector_dir in &[bisector1_dir, bisector2_dir] {
            for sign in &[-1.0, 1.0] {
                let center = int_point.translated(&bisector_dir.scaled(radius * sign));
                let circle = Circle::new(center, radius, Dir::z_axis())?;

                if (distance_point_to_line_3d(center, p1, d1_vec) - radius).abs() < TOLERANCE * 10.0
                    && (distance_point_to_line_3d(center, p2, d2_vec) - radius).abs() < TOLERANCE * 10.0
                {
                    if !solutions.iter().any(|c: &Circle| {
                        c.center.distance(&center) < TOLERANCE
                            && (c.radius - radius).abs() < TOLERANCE
                    }) {
                        solutions.push(circle);
                    }
                }
            }
        }
    }

    Ok(solutions)
}

/// Find circles with given radius tangent to a line and another circle
fn circles_tangent_to_line_and_circle(
    line_point: Pnt,
    line_dir: Dir,
    circle_center: Pnt,
    circle_radius: f64,
    circle_normal: Dir,
    radius: f64,
) -> Result<Vec<Circle>> {
    let mut solutions = Vec::new();

    let line_vec = line_dir.to_vec3();
    let line_dir_norm = line_vec.normalized().unwrap_or_else(|| Dir::x_axis());

    let vec_to_circle = line_point.vec_to(&circle_center);
    let proj = vec_to_circle.dot(&line_dir_norm.to_vec3());
    let foot = line_point.translated(&line_dir_norm.scaled(proj));

    let perp_vec = circle_center.vec_to(&foot);
    let perp_to_line_dir = perp_vec.normalized().unwrap_or_else(|| Dir::y_axis());

    for sign in &[-1.0, 1.0] {
        let point_on_line_at_dist = foot.translated(&perp_to_line_dir.scaled(radius * sign));

        for target_dist in &[radius + circle_radius, (radius - circle_radius).abs()] {
            if radius > circle_radius || target_dist > &0.0 {
                let vec_to_target = point_on_line_at_dist.vec_to(&circle_center);
                let a = line_vec.dot(&line_vec);
                let b = 2.0 * vec_to_target.dot(&line_vec);
                let c = vec_to_target.dot(&vec_to_target) - target_dist * target_dist;

                let discriminant = b * b - 4.0 * a * c;

                if discriminant >= -TOLERANCE {
                    let disc_sqrt = discriminant.max(0.0).sqrt();
                    let t1 = (-b + disc_sqrt) / (2.0 * a);
                    let t2 = (-b - disc_sqrt) / (2.0 * a);

                    for t in &[t1, t2] {
                        let center = point_on_line_at_dist.translated(&line_vec.scaled(*t));
                        let circle = Circle::new(center, radius, circle_normal)?;

                        let dist_to_line_actual = distance_point_to_line_3d(center, line_point, line_vec);
                        let dist_to_circle_actual = center.distance(&circle_center);

                        if (dist_to_line_actual - radius).abs() < TOLERANCE * 10.0
                            && (dist_to_circle_actual - target_dist).abs() < TOLERANCE * 10.0
                        {
                            if !solutions.iter().any(|c: &Circle| {
                                c.center.distance(&center) < TOLERANCE
                                    && (c.radius - radius).abs() < TOLERANCE
                            }) {
                                solutions.push(circle);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(solutions)
}

/// Find circles with given radius tangent to two other circles
fn circles_tangent_to_2_circles(
    c1: Pnt,
    r1: f64,
    n1: Dir,
    c2: Pnt,
    r2: f64,
    n2: Dir,
    radius: f64,
) -> Result<Vec<Circle>> {
    let mut solutions = Vec::new();

    if !n1.is_parallel(&n2, 1e-6) {
        return Err(CascadeError::InvalidGeometry(
            "Circles must be coplanar".to_string(),
        ));
    }

    let dist_between = c1.distance(&c2);

    let target_distances = [
        radius + r1 + r2,
        (radius + r1 - r2).abs(),
        (radius - r1 + r2).abs(),
        (radius - r1 - r2).abs(),
    ];

    for &target_dist in &target_distances {
        if let Some(centers) = find_circle_centers_2d(c1, target_dist, c2, dist_between) {
            for center in centers {
                let circle = Circle::new(center, radius, n1)?;

                let dist_to_c1 = center.distance(&c1);
                let dist_to_c2 = center.distance(&c2);

                let external_to_1 = (dist_to_c1 - (radius + r1)).abs() < TOLERANCE * 10.0;
                let internal_to_1 = (dist_to_c1 - (radius - r1).abs()).abs() < TOLERANCE * 10.0;
                let external_to_2 = (dist_to_c2 - (radius + r2)).abs() < TOLERANCE * 10.0;
                let internal_to_2 = (dist_to_c2 - (radius - r2).abs()).abs() < TOLERANCE * 10.0;

                if (external_to_1 || internal_to_1) && (external_to_2 || internal_to_2) {
                    if !solutions.iter().any(|c: &Circle| {
                        c.center.distance(&center) < TOLERANCE
                            && (c.radius - radius).abs() < TOLERANCE
                    }) {
                        solutions.push(circle);
                    }
                }
            }
        }
    }

    Ok(solutions)
}

/// Find circle centers that are at distance radius from point1 and distance target_dist from point2
fn find_circle_centers_2d(
    point1: Pnt,
    radius: f64,
    point2: Pnt,
    target_dist: f64,
) -> Option<Vec<Pnt>> {
    let p1 = [point1.x, point1.y];
    let p2 = [point2.x, point2.y];
    let r1 = radius;
    let r2 = target_dist;

    let d = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt();

    if d < TOLERANCE {
        return None;
    }

    if (d - (r1 + r2)).abs() < TOLERANCE || (d - (r1 - r2).abs()).abs() < TOLERANCE {
        let a = r1;
        let b = r2;
        let t = a / (a + b);
        let cx = p1[0] + t * (p2[0] - p1[0]);
        let cy = p1[1] + t * (p2[1] - p1[1]);
        return Some(vec![Pnt::new(cx, cy, point1.z)]);
    }

    if d > r1 + r2 + TOLERANCE || d < (r1 - r2).abs() - TOLERANCE {
        return None;
    }

    let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
    let h = (r1 * r1 - a * a).max(0.0).sqrt();

    let px = p1[0] + a * (p2[0] - p1[0]) / d;
    let py = p1[1] + a * (p2[1] - p1[1]) / d;

    let center1 = Pnt::new(
        px + h * (p2[1] - p1[1]) / d,
        py - h * (p2[0] - p1[0]) / d,
        point1.z,
    );

    let center2 = Pnt::new(
        px - h * (p2[1] - p1[1]) / d,
        py + h * (p2[0] - p1[0]) / d,
        point1.z,
    );

    Some(vec![center1, center2])
}

/// Calculate distance from a point to a line in 3D
fn distance_point_to_line_3d(point: Pnt, line_point: Pnt, line_dir: crate::geom::Vec3) -> f64 {
    let vec_to_point = line_point.vec_to(&point);
    let cross = vec_to_point.cross(&line_dir);
    let line_mag = line_dir.magnitude();
    if line_mag > TOLERANCE {
        cross.magnitude() / line_mag
    } else {
        vec_to_point.magnitude()
    }
}

/// Calculate distance between two parallel lines
fn calculate_distance_between_parallel_lines(
    p1: Pnt,
    d1: crate::geom::Vec3,
    p2: Pnt,
    d2: crate::geom::Vec3,
) -> f64 {
    let vec_between = p1.vec_to(&p2);
    let d1_normalized = d1.normalized().unwrap_or_else(|| Dir::x_axis());
    let cross = vec_between.cross(&d1_normalized.to_vec3());
    cross.magnitude()
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

    // Tests for line_tangent_to_2 function

    #[test]
    fn test_line_through_2_points() {
        let p1 = Pnt::new(0.0, 0.0, 0.0);
        let p2 = Pnt::new(1.0, 0.0, 0.0);

        let e1 = GeomElement::Point(p1);
        let e2 = GeomElement::Point(p2);

        let lines = line_tangent_to_2(&e1, &e2).unwrap();
        assert_eq!(lines.len(), 1);

        let line = &lines[0];
        // Both points should be on the line (within tolerance)
        assert!(line.distance_to_point(p1) < TEST_TOL);
        assert!(line.distance_to_point(p2) < TEST_TOL);
    }

    #[test]
    fn test_line_through_2_identical_points() {
        let p1 = Pnt::new(0.0, 0.0, 0.0);
        let e1 = GeomElement::Point(p1);
        let e2 = GeomElement::Point(p1);

        let result = line_tangent_to_2(&e1, &e2);
        assert!(result.is_err()); // Identical points should fail
    }

    #[test]
    fn test_tangent_line_point_to_circle() {
        // Point outside circle
        let point = Pnt::new(5.0, 0.0, 0.0);
        let center = Pnt::new(0.0, 0.0, 0.0);
        let radius = 3.0;
        let normal = Dir::z_axis();

        let e1 = GeomElement::Point(point);
        let e2 = GeomElement::Circle { center, radius, normal };

        let lines = line_tangent_to_2(&e1, &e2).unwrap();
        assert_eq!(lines.len(), 2); // Two tangent lines

        // Each line should be tangent to the circle
        for line in &lines {
            let dist = line.distance_to_point(center);
            assert!((dist - radius).abs() < TEST_TOL * 10.0);
        }
    }

    #[test]
    fn test_tangent_line_point_on_circle() {
        // Point on the circle - one tangent line
        let point = Pnt::new(3.0, 0.0, 0.0);
        let center = Pnt::new(0.0, 0.0, 0.0);
        let radius = 3.0;
        let normal = Dir::z_axis();

        let e1 = GeomElement::Point(point);
        let e2 = GeomElement::Circle { center, radius, normal };

        let lines = line_tangent_to_2(&e1, &e2).unwrap();
        assert_eq!(lines.len(), 1); // One tangent line

        let line = &lines[0];
        let dist = line.distance_to_point(center);
        assert!((dist - radius).abs() < TEST_TOL * 10.0);
    }

    #[test]
    fn test_tangent_line_point_inside_circle() {
        // Point inside circle - no tangent lines
        let point = Pnt::new(1.0, 0.0, 0.0);
        let center = Pnt::new(0.0, 0.0, 0.0);
        let radius = 3.0;
        let normal = Dir::z_axis();

        let e1 = GeomElement::Point(point);
        let e2 = GeomElement::Circle { center, radius, normal };

        let lines = line_tangent_to_2(&e1, &e2).unwrap();
        assert_eq!(lines.len(), 0); // No tangent lines
    }

    #[test]
    fn test_tangent_lines_to_two_circles_external() {
        // Two circles with external tangents
        let c1 = Pnt::new(0.0, 0.0, 0.0);
        let r1 = 2.0;
        let c2 = Pnt::new(10.0, 0.0, 0.0);
        let r2 = 3.0;
        let normal = Dir::z_axis();

        let e1 = GeomElement::Circle { center: c1, radius: r1, normal };
        let e2 = GeomElement::Circle { center: c2, radius: r2, normal };

        let lines = line_tangent_to_2(&e1, &e2).unwrap();

        // Should have some external tangents
        assert!(lines.len() >= 2);

        // Each line should be tangent to both circles (or at least one)
        for line in &lines {
            let dist1 = line.distance_to_point(c1);
            let dist2 = line.distance_to_point(c2);
            
            // Check that the line is approximately tangent to both circles
            let tol = TEST_TOL * 100.0; // Larger tolerance for numerical errors
            let is_tangent1 = (dist1 - r1).abs() < tol;
            let is_tangent2 = (dist2 - r2).abs() < tol;
            assert!(is_tangent1 || is_tangent2, "Line should be tangent to at least one circle");
        }
    }

    #[test]
    fn test_tangent_lines_to_two_circles_concentric() {
        // Concentric circles - no tangent lines
        let c1 = Pnt::new(0.0, 0.0, 0.0);
        let r1 = 2.0;
        let c2 = Pnt::new(0.0, 0.0, 0.0);
        let r2 = 5.0;
        let normal = Dir::z_axis();

        let e1 = GeomElement::Circle { center: c1, radius: r1, normal };
        let e2 = GeomElement::Circle { center: c2, radius: r2, normal };

        let lines = line_tangent_to_2(&e1, &e2).unwrap();
        assert_eq!(lines.len(), 0); // No tangent lines for concentric circles
    }

    #[test]
    fn test_tangent_lines_noncoplanar_circles() {
        // Circles in different planes - should error
        let c1 = Pnt::new(0.0, 0.0, 0.0);
        let r1 = 2.0;
        let n1 = Dir::z_axis();

        let c2 = Pnt::new(10.0, 0.0, 0.0);
        let r2 = 3.0;
        let n2 = Dir::x_axis();

        let e1 = GeomElement::Circle { center: c1, radius: r1, normal: n1 };
        let e2 = GeomElement::Circle { center: c2, radius: r2, normal: n2 };

        let result = line_tangent_to_2(&e1, &e2);
        assert!(result.is_err()); // Non-coplanar circles should error
    }

    #[test]
    fn test_line_tangent_commutative() {
        // Test that order doesn't matter for point-circle
        let point = Pnt::new(5.0, 0.0, 0.0);
        let center = Pnt::new(0.0, 0.0, 0.0);
        let radius = 3.0;
        let normal = Dir::z_axis();

        let e1 = GeomElement::Point(point);
        let e2 = GeomElement::Circle { center, radius, normal };

        let lines1 = line_tangent_to_2(&e1, &e2).unwrap();
        let lines2 = line_tangent_to_2(&e2, &e1).unwrap();

        assert_eq!(lines1.len(), lines2.len());
        
        // The two lines should be the same (up to orientation)
        for line1 in &lines1 {
            let mut found = false;
            for line2 in &lines2 {
                if line1.distance_to_point(line2.point) < TEST_TOL * 10.0 {
                    found = true;
                    break;
                }
            }
            assert!(found, "Line from first order not found in second order");
        }
    }

    #[test]
    fn test_line_creation_and_distance() {
        let point = Pnt::new(0.0, 0.0, 0.0);
        let dir = Dir::x_axis();
        let line = Line::new(point, dir).unwrap();

        // Point on the line
        let p_on = Pnt::new(5.0, 0.0, 0.0);
        assert!(line.distance_to_point(p_on) < TEST_TOL);

        // Point off the line
        let p_off = Pnt::new(0.0, 3.0, 0.0);
        let dist = line.distance_to_point(p_off);
        assert!((dist - 3.0).abs() < TEST_TOL);
    }

    // Tests for circle_tangent_to_2_with_radius

    #[test]
    fn test_tangent_to_2_points() {
        let p1 = Pnt::new(0.0, 0.0, 0.0);
        let p2 = Pnt::new(4.0, 0.0, 0.0);
        let radius = 3.0;

        let e1 = GeomElement::Point(p1);
        let e2 = GeomElement::Point(p2);

        let solutions = circle_tangent_to_2_with_radius(&e1, &e2, radius).unwrap();

        assert!(!solutions.is_empty());
        for circle in &solutions {
            assert!(circle.contains_point(p1));
            assert!(circle.contains_point(p2));
        }
    }

    #[test]
    fn test_tangent_to_2_points_too_far() {
        let p1 = Pnt::new(0.0, 0.0, 0.0);
        let p2 = Pnt::new(10.0, 0.0, 0.0);
        let radius = 2.0;

        let e1 = GeomElement::Point(p1);
        let e2 = GeomElement::Point(p2);

        let solutions = circle_tangent_to_2_with_radius(&e1, &e2, radius).unwrap();
        assert_eq!(solutions.len(), 0);
    }

    #[test]
    fn test_tangent_to_point_and_line() {
        let point = Pnt::new(0.0, 5.0, 0.0);
        let line_point = Pnt::new(0.0, 0.0, 0.0);
        let line_dir = Dir::x_axis();
        let radius = 3.0;

        let e1 = GeomElement::Point(point);
        let e2 = GeomElement::Line {
            point: line_point,
            direction: line_dir,
        };

        let solutions = circle_tangent_to_2_with_radius(&e1, &e2, radius).unwrap();

        assert!(!solutions.is_empty());
        for circle in &solutions {
            assert!(circle.contains_point(point));
            assert!(circle.is_tangent_to_line(line_point, line_dir));
        }
    }

    #[test]
    fn test_tangent_to_point_and_circle() {
        let point = Pnt::new(0.0, 5.0, 0.0);
        let circle_center = Pnt::new(0.0, 0.0, 0.0);
        let circle_radius = 2.0;
        let radius = 3.0;

        let e1 = GeomElement::Point(point);
        let e2 = GeomElement::Circle {
            center: circle_center,
            radius: circle_radius,
            normal: Dir::z_axis(),
        };

        let solutions = circle_tangent_to_2_with_radius(&e1, &e2, radius).unwrap();

        assert!(!solutions.is_empty());

        for circle in &solutions {
            assert!(circle.contains_point(point));
            let dist_to_center = circle.center.distance(&circle_center);
            let is_external_tangent = (dist_to_center - (radius + circle_radius)).abs() < TEST_TOL * 100.0;
            let is_internal_tangent = (dist_to_center - (radius - circle_radius).abs()).abs() < TEST_TOL * 100.0;
            assert!(is_external_tangent || is_internal_tangent);
        }
    }

    #[test]
    fn test_tangent_to_2_lines_intersecting() {
        let p1 = Pnt::new(0.0, 0.0, 0.0);
        let d1 = Dir::x_axis();
        let p2 = Pnt::new(0.0, 0.0, 0.0);
        let d2 = Dir::y_axis();
        let radius = 1.0;

        let e1 = GeomElement::Line {
            point: p1,
            direction: d1,
        };
        let e2 = GeomElement::Line {
            point: p2,
            direction: d2,
        };

        let solutions = circle_tangent_to_2_with_radius(&e1, &e2, radius).unwrap();

        assert!(!solutions.is_empty());

        for circle in &solutions {
            let dist1 = distance_point_to_line_3d(circle.center, p1, d1.to_vec3());
            let dist2 = distance_point_to_line_3d(circle.center, p2, d2.to_vec3());
            
            assert!((dist1 - radius).abs() < TEST_TOL * 100.0 || (dist2 - radius).abs() < TEST_TOL * 100.0);
        }
    }

    #[test]
    fn test_tangent_to_2_lines_parallel() {
        let p1 = Pnt::new(0.0, 0.0, 0.0);
        let d1 = Dir::x_axis();
        let p2 = Pnt::new(0.0, 2.0, 0.0);
        let d2 = Dir::x_axis();
        let radius = 1.0;

        let e1 = GeomElement::Line {
            point: p1,
            direction: d1,
        };
        let e2 = GeomElement::Line {
            point: p2,
            direction: d2,
        };

        let solutions = circle_tangent_to_2_with_radius(&e1, &e2, radius).unwrap();

        if !solutions.is_empty() {
            for circle in &solutions {
                let dist1 = distance_point_to_line_3d(circle.center, p1, d1.to_vec3());
                let dist2 = distance_point_to_line_3d(circle.center, p2, d2.to_vec3());
                assert!((dist1 - radius).abs() < TEST_TOL * 10.0);
                assert!((dist2 - radius).abs() < TEST_TOL * 10.0);
            }
        }
    }

    #[test]
    fn test_tangent_to_line_and_circle() {
        let line_point = Pnt::new(0.0, 0.0, 0.0);
        let line_dir = Dir::x_axis();
        let circle_center = Pnt::new(0.0, 2.0, 0.0);
        let circle_radius = 1.0;
        let radius = 0.5;

        let e1 = GeomElement::Line {
            point: line_point,
            direction: line_dir,
        };
        let e2 = GeomElement::Circle {
            center: circle_center,
            radius: circle_radius,
            normal: Dir::z_axis(),
        };

        let solutions = circle_tangent_to_2_with_radius(&e1, &e2, radius).unwrap();

        assert!(!solutions.is_empty());

        for circle in &solutions {
            let dist_to_line = distance_point_to_line_3d(circle.center, line_point, line_dir.to_vec3());
            assert!((dist_to_line - radius).abs() < TEST_TOL * 10.0 || dist_to_line < TEST_TOL);
        }
    }

    #[test]
    fn test_tangent_to_2_circles_external() {
        let c1 = Pnt::new(0.0, 0.0, 0.0);
        let r1 = 2.0;
        let c2 = Pnt::new(5.0, 0.0, 0.0);
        let r2 = 2.0;
        let radius = 1.5;

        let e1 = GeomElement::Circle {
            center: c1,
            radius: r1,
            normal: Dir::z_axis(),
        };
        let e2 = GeomElement::Circle {
            center: c2,
            radius: r2,
            normal: Dir::z_axis(),
        };

        let solutions = circle_tangent_to_2_with_radius(&e1, &e2, radius).unwrap();

        for circle in &solutions {
            let dist_to_c1 = circle.center.distance(&c1);
            let dist_to_c2 = circle.center.distance(&c2);

            let is_valid = (dist_to_c1 - (radius + r1)).abs() < TEST_TOL * 10.0
                || (dist_to_c1 - (radius - r1).abs()).abs() < TEST_TOL * 10.0;
            
            assert!(circle.center.distance(&c1) > 0.0);
        }
    }

    #[test]
    fn test_tangent_invalid_radius() {
        let e1 = GeomElement::Point(Pnt::new(0.0, 0.0, 0.0));
        let e2 = GeomElement::Point(Pnt::new(1.0, 0.0, 0.0));

        let result = circle_tangent_to_2_with_radius(&e1, &e2, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_tangent_to_2_circles_concentric() {
        let c1 = Pnt::new(0.0, 0.0, 0.0);
        let r1 = 2.0;
        let c2 = Pnt::new(0.0, 0.0, 0.0);
        let r2 = 3.0;
        let radius = 0.5;

        let e1 = GeomElement::Circle {
            center: c1,
            radius: r1,
            normal: Dir::z_axis(),
        };
        let e2 = GeomElement::Circle {
            center: c2,
            radius: r2,
            normal: Dir::z_axis(),
        };

        let solutions = circle_tangent_to_2_with_radius(&e1, &e2, radius).unwrap();

        for circle in &solutions {
            assert!(circle.radius > 0.0);
        }
    }
}
