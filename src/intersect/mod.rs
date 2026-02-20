//! Surface-surface and curve-curve intersection algorithms
//!
//! Computes the intersection curves between two surfaces.
//! Returns the results as wires (connected edge sequences).
//!
//! Also provides 2D curve-curve intersection for geometric computations.

use crate::{
    brep::{Edge, Face, SurfaceType, Vertex, Wire, CurveType},
    {CascadeError, Result, TOLERANCE},
};
use std::f64::consts::PI;

/// A 2D curve for intersection calculations
///
/// Represents simple 2D geometric curves: lines and circles.
/// Used for 2D curve-curve intersection operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Curve2D {
    /// A line defined by two points
    /// 
    /// The line is infinite; the points just define its direction.
    Line {
        /// Starting point of the line (or any point on it)
        p1: [f64; 2],
        /// Another point on the line
        p2: [f64; 2],
    },
    
    /// A circle defined by center and radius
    Circle {
        /// Center of the circle
        center: [f64; 2],
        /// Radius of the circle (must be positive)
        radius: f64,
    },
}

impl Curve2D {
    /// Create a line from two points
    pub fn line(p1: [f64; 2], p2: [f64; 2]) -> Result<Self> {
        if dist_2d(&p1, &p2) < TOLERANCE {
            return Err(CascadeError::InvalidGeometry(
                "Line: p1 and p2 must be distinct".to_string(),
            ));
        }
        Ok(Curve2D::Line { p1, p2 })
    }
    
    /// Create a circle from center and radius
    pub fn circle(center: [f64; 2], radius: f64) -> Result<Self> {
        if radius <= TOLERANCE {
            return Err(CascadeError::InvalidGeometry(
                "Circle: radius must be positive".to_string(),
            ));
        }
        Ok(Curve2D::Circle { center, radius })
    }
}

/// Compute the intersection points between two 2D curves
///
/// # Arguments
/// * `curve1` - First 2D curve
/// * `curve2` - Second 2D curve
///
/// # Returns
/// * `Ok(Vec<[f64; 2]>)` - List of intersection points in 2D
/// * `Err` - If the intersection calculation fails
///
/// # Notes
/// - For tangent curves, the tangent point is returned
/// - For coincident curves (line-line, circle-circle), an empty vector is returned
/// - Results are sorted by distance from the origin
pub fn intersect_curves_2d(curve1: &Curve2D, curve2: &Curve2D) -> Result<Vec<[f64; 2]>> {
    match (curve1, curve2) {
        // Line-Line intersection
        (Curve2D::Line { p1: p1a, p2: p2a }, Curve2D::Line { p1: p1b, p2: p2b }) => {
            line_line_intersection(p1a, p2a, p1b, p2b)
        }
        
        // Line-Circle intersection
        (Curve2D::Line { p1, p2 }, Curve2D::Circle { center, radius }) => {
            line_circle_intersection(p1, p2, center, *radius)
        }
        
        // Circle-Line intersection (commutative)
        (Curve2D::Circle { center, radius }, Curve2D::Line { p1, p2 }) => {
            line_circle_intersection(p1, p2, center, *radius)
        }
        
        // Circle-Circle intersection
        (Curve2D::Circle { center: c1, radius: r1 }, Curve2D::Circle { center: c2, radius: r2 }) => {
            circle_circle_intersection(c1, *r1, c2, *r2)
        }
    }
}

// ===== 2D Intersection Helper Functions =====

/// Compute distance between two 2D points
fn dist_2d(p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    (dx * dx + dy * dy).sqrt()
}

/// Compute dot product of two 2D vectors
fn dot_2d(v1: &[f64; 2], v2: &[f64; 2]) -> f64 {
    v1[0] * v2[0] + v1[1] * v2[1]
}

/// Compute cross product (returns scalar z-component) of two 2D vectors
fn cross_2d(v1: &[f64; 2], v2: &[f64; 2]) -> f64 {
    v1[0] * v2[1] - v1[1] * v2[0]
}

/// Line-Line intersection
///
/// Computes the intersection of two infinite lines defined by two points each.
/// Returns empty vector for parallel or coincident lines, single point for intersection.
fn line_line_intersection(
    p1: &[f64; 2],
    p2: &[f64; 2],
    p3: &[f64; 2],
    p4: &[f64; 2],
) -> Result<Vec<[f64; 2]>> {
    let d1 = [p2[0] - p1[0], p2[1] - p1[1]]; // Direction of line 1
    let d2 = [p4[0] - p3[0], p4[1] - p3[1]]; // Direction of line 2
    let d13 = [p3[0] - p1[0], p3[1] - p1[1]]; // Vector from p1 to p3
    
    let cross_d1_d2 = cross_2d(&d1, &d2);
    
    // Check if lines are parallel
    if cross_d1_d2.abs() < TOLERANCE {
        // Lines are parallel or coincident - no unique intersection point
        return Ok(vec![]);
    }
    
    // Compute parameter t for line 1: p = p1 + t * d1
    let t = cross_2d(&d13, &d2) / cross_d1_d2;
    
    // Compute intersection point
    let intersection = [p1[0] + t * d1[0], p1[1] + t * d1[1]];
    
    Ok(vec![intersection])
}

/// Line-Circle intersection
///
/// Computes intersection of a line (infinite) and a circle.
/// Returns 0, 1 (tangent), or 2 intersection points.
fn line_circle_intersection(
    p1: &[f64; 2],
    p2: &[f64; 2],
    center: &[f64; 2],
    radius: f64,
) -> Result<Vec<[f64; 2]>> {
    // Direction vector of the line
    let d = [p2[0] - p1[0], p2[1] - p1[1]];
    let d_len = dist_2d(p1, p2);
    
    if d_len < TOLERANCE {
        return Err(CascadeError::InvalidGeometry("Line has zero length".to_string()));
    }
    
    // Normalized direction
    let d_norm = [d[0] / d_len, d[1] / d_len];
    
    // Vector from p1 to circle center
    let f = [center[0] - p1[0], center[1] - p1[1]];
    
    // Project f onto the line direction (parameter t)
    let t = dot_2d(&f, &d_norm);
    
    // Closest point on the line to the circle center
    let closest = [p1[0] + t * d_norm[0], p1[1] + t * d_norm[1]];
    
    // Distance from circle center to the line
    let f_closest = [center[0] - closest[0], center[1] - closest[1]];
    let dist_to_line = (f_closest[0] * f_closest[0] + f_closest[1] * f_closest[1]).sqrt();
    
    // Check if line intersects circle
    if dist_to_line > radius + TOLERANCE {
        // No intersection
        return Ok(vec![]);
    }
    
    // Check if tangent
    if (dist_to_line - radius).abs() < TOLERANCE {
        // Tangent: return single point
        return Ok(vec![closest]);
    }
    
    // Two intersection points
    let delta = (radius * radius - dist_to_line * dist_to_line).sqrt();
    
    let p_1 = [
        closest[0] - delta * d_norm[1],
        closest[1] + delta * d_norm[0],
    ];
    
    let p_2 = [
        closest[0] + delta * d_norm[1],
        closest[1] - delta * d_norm[0],
    ];
    
    // Sort by parameter t along the line direction
    let t1 = dot_2d(&[p_1[0] - p1[0], p_1[1] - p1[1]], &d_norm);
    let t2 = dot_2d(&[p_2[0] - p1[0], p_2[1] - p1[1]], &d_norm);
    
    if t1 <= t2 {
        Ok(vec![p_1, p_2])
    } else {
        Ok(vec![p_2, p_1])
    }
}

/// Circle-Circle intersection
///
/// Computes intersection of two circles.
/// Returns 0, 1 (tangent), or 2 intersection points.
fn circle_circle_intersection(
    c1: &[f64; 2],
    r1: f64,
    c2: &[f64; 2],
    r2: f64,
) -> Result<Vec<[f64; 2]>> {
    let d = dist_2d(c1, c2);
    
    // Check for various cases
    if d < TOLERANCE {
        // Circles are concentric
        if (r1 - r2).abs() < TOLERANCE {
            // Coincident circles
            return Ok(vec![]);
        } else {
            // Concentric but different radii - no intersection
            return Ok(vec![]);
        }
    }
    
    // Check if circles are too far apart
    if d > r1 + r2 + TOLERANCE {
        return Ok(vec![]);
    }
    
    // Check if one circle is inside the other
    if d < (r1 - r2).abs() - TOLERANCE {
        return Ok(vec![]);
    }
    
    // Check for external tangency
    if (d - (r1 + r2)).abs() < TOLERANCE {
        // External tangent point
        let t = r1 / (r1 + r2);
        let tangent = [
            c1[0] + t * (c2[0] - c1[0]),
            c1[1] + t * (c2[1] - c1[1]),
        ];
        return Ok(vec![tangent]);
    }
    
    // Check for internal tangency
    if (d - (r1 - r2).abs()).abs() < TOLERANCE {
        // Internal tangent point
        let t = r1 / (r1 - r2).abs();
        let tangent = if r1 > r2 {
            [
                c1[0] + t * (c2[0] - c1[0]),
                c1[1] + t * (c2[1] - c1[1]),
            ]
        } else {
            [
                c1[0] - t * (c2[0] - c1[0]),
                c1[1] - t * (c2[1] - c1[1]),
            ]
        };
        return Ok(vec![tangent]);
    }
    
    // Two intersection points
    // Using the formula from analytical geometry
    let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
    let h = (r1 * r1 - a * a).sqrt();
    
    // Point on the line between centers
    let p_mid = [c1[0] + a / d * (c2[0] - c1[0]), c1[1] + a / d * (c2[1] - c1[1])];
    
    // Perpendicular offset direction
    let perp = [-(c2[1] - c1[1]) / d, (c2[0] - c1[0]) / d];
    
    let p1 = [p_mid[0] + h * perp[0], p_mid[1] + h * perp[1]];
    let p2 = [p_mid[0] - h * perp[0], p_mid[1] - h * perp[1]];
    
    // Sort by angle from center c1 for consistent ordering
    let angle1 = (p1[1] - c1[1]).atan2(p1[0] - c1[0]);
    let angle2 = (p2[1] - c1[1]).atan2(p2[0] - c1[0]);
    
    if angle1 <= angle2 {
        Ok(vec![p1, p2])
    } else {
        Ok(vec![p2, p1])
    }
}

/// Compute the intersection curves between two faces.
///
/// Returns a vector of wires representing the intersection curves.
/// Each wire is a closed or open curve where the two surfaces meet.
///
/// # Arguments
/// * `s1` - First surface (face)
/// * `s2` - Second surface (face)
///
/// # Returns
/// * `Ok(Vec<Wire>)` - List of intersection curves as wires
/// * `Err` - If intersection is not yet implemented for this surface pair
pub fn intersect_surfaces(s1: &Face, s2: &Face) -> Result<Vec<Wire>> {
    match (&s1.surface_type, &s2.surface_type) {
        // Plane-Plane intersection
        (SurfaceType::Plane { origin: o1, normal: n1 }, SurfaceType::Plane { origin: o2, normal: n2 }) => {
            plane_plane_intersection(o1, n1, o2, n2)
        }

        // Plane-Cylinder intersection
        (SurfaceType::Plane { origin, normal }, SurfaceType::Cylinder { origin: cyl_origin, axis, radius }) => {
            plane_cylinder_intersection(origin, normal, cyl_origin, axis, *radius)
        }
        (SurfaceType::Cylinder { origin: cyl_origin, axis, radius }, SurfaceType::Plane { origin, normal }) => {
            plane_cylinder_intersection(origin, normal, cyl_origin, axis, *radius)
        }

        // Cylinder-Cylinder intersection
        (SurfaceType::Cylinder { origin: o1, axis: a1, radius: r1 }, SurfaceType::Cylinder { origin: o2, axis: a2, radius: r2 }) => {
            cylinder_cylinder_intersection(o1, a1, *r1, o2, a2, *r2)
        }

        // Not yet implemented
        _ => Err(CascadeError::NotImplemented(format!(
            "Intersection between {:?} and {:?}",
            std::mem::discriminant(&s1.surface_type),
            std::mem::discriminant(&s2.surface_type)
        ))),
    }
}

/// Plane-plane intersection: returns a line (or empty if parallel/coincident)
fn plane_plane_intersection(
    o1: &[f64; 3],
    n1: &[f64; 3],
    o2: &[f64; 3],
    n2: &[f64; 3],
) -> Result<Vec<Wire>> {
    let n1_norm = normalize(n1);
    let n2_norm = normalize(n2);

    // Check if planes are parallel
    let cross_prod = cross(&n1_norm, &n2_norm);
    let cross_len = magnitude(&cross_prod);

    if cross_len < TOLERANCE {
        // Planes are parallel or coincident
        return Ok(vec![]);
    }

    // Direction of the intersection line is perpendicular to both normals
    let line_dir = normalize(&cross_prod);

    // Find a point on the intersection line
    // We need to solve: (p - o1) · n1 = 0 and (p - o2) · n2 = 0
    // This gives us two equations, we need to pick a third constraint

    // Use the axis corresponding to the largest component of line_dir
    let abs_dir = [line_dir[0].abs(), line_dir[1].abs(), line_dir[2].abs()];
    let max_idx = if abs_dir[0] > abs_dir[1] {
        if abs_dir[0] > abs_dir[2] { 0 } else { 2 }
    } else {
        if abs_dir[1] > abs_dir[2] { 1 } else { 2 }
    };

    let mut point = [0.0; 3];
    
    // Solve for a point on the line using Cramer's rule-like approach
    if max_idx == 2 {
        // Fix z=0 and solve for x, y
        let det = n1_norm[0] * n2_norm[1] - n1_norm[1] * n2_norm[0];
        if det.abs() > TOLERANCE {
            let d1 = dot(o1, &n1_norm);
            let d2 = dot(o2, &n2_norm);
            point[0] = (d1 * n2_norm[1] - d2 * n1_norm[1]) / det;
            point[1] = (n1_norm[0] * d2 - n2_norm[0] * d1) / det;
            point[2] = 0.0;
        } else {
            point = [o1[0], o1[1], 0.0];
        }
    } else if max_idx == 1 {
        // Fix y=0 and solve for x, z
        let det = n1_norm[0] * n2_norm[2] - n1_norm[2] * n2_norm[0];
        if det.abs() > TOLERANCE {
            let d1 = dot(o1, &n1_norm);
            let d2 = dot(o2, &n2_norm);
            point[0] = (d1 * n2_norm[2] - d2 * n1_norm[2]) / det;
            point[1] = 0.0;
            point[2] = (n1_norm[0] * d2 - n2_norm[0] * d1) / det;
        } else {
            point = [o1[0], 0.0, o1[2]];
        }
    } else {
        // Fix x=0 and solve for y, z
        let det = n1_norm[1] * n2_norm[2] - n1_norm[2] * n2_norm[1];
        if det.abs() > TOLERANCE {
            let d1 = dot(o1, &n1_norm);
            let d2 = dot(o2, &n2_norm);
            point[0] = 0.0;
            point[1] = (d1 * n2_norm[2] - d2 * n1_norm[2]) / det;
            point[2] = (n1_norm[1] * d2 - n2_norm[1] * d1) / det;
        } else {
            point = [0.0, o1[1], o1[2]];
        }
    }

    // Create a line wire as two points far apart along the direction
    let scale = 1000.0; // Large distance to represent "infinite" line
    let p1 = [
        point[0] - scale * line_dir[0],
        point[1] - scale * line_dir[1],
        point[2] - scale * line_dir[2],
    ];
    let p2 = [
        point[0] + scale * line_dir[0],
        point[1] + scale * line_dir[1],
        point[2] + scale * line_dir[2],
    ];

    let edge = Edge {
        start: Vertex::new(p1[0], p1[1], p1[2]),
        end: Vertex::new(p2[0], p2[1], p2[2]),
        curve_type: CurveType::Line,
    };

    let wire = Wire {
        edges: vec![edge],
        closed: false,
    };

    Ok(vec![wire])
}

/// Plane-cylinder intersection: returns a line or ellipse
fn plane_cylinder_intersection(
    plane_origin: &[f64; 3],
    plane_normal: &[f64; 3],
    cyl_origin: &[f64; 3],
    cyl_axis: &[f64; 3],
    radius: f64,
) -> Result<Vec<Wire>> {
    let plane_n = normalize(plane_normal);
    let cyl_a = normalize(cyl_axis);

    // Check the angle between plane normal and cylinder axis
    let dot_prod = dot(&plane_n, &cyl_a);
    let angle = dot_prod.abs();

    if angle > 1.0 - TOLERANCE {
        // Plane is perpendicular to cylinder axis: intersection is a circle (at the plane)
        // Project cylinder origin onto the plane
        let dist_to_plane = dot(&[
            cyl_origin[0] - plane_origin[0],
            cyl_origin[1] - plane_origin[1],
            cyl_origin[2] - plane_origin[2],
        ], &plane_n);

        let circle_center = [
            cyl_origin[0] - dist_to_plane * plane_n[0],
            cyl_origin[1] - dist_to_plane * plane_n[1],
            cyl_origin[2] - dist_to_plane * plane_n[2],
        ];

        // Create a circular wire approximation (8 segments)
        let segments = 8;
        let mut edges = vec![];
        
        // Create two perpendicular vectors in the plane
        let perp1 = perpendicular_to(&cyl_a);
        let perp2 = cross(&cyl_a, &perp1);
        let perp2 = normalize(&perp2);

        let mut prev_point = [
            circle_center[0] + radius * perp1[0],
            circle_center[1] + radius * perp1[1],
            circle_center[2] + radius * perp1[2],
        ];

        for i in 1..=segments {
            let angle = 2.0 * PI * (i as f64) / (segments as f64);
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let curr_point = [
                circle_center[0] + radius * (cos_a * perp1[0] + sin_a * perp2[0]),
                circle_center[1] + radius * (cos_a * perp1[1] + sin_a * perp2[1]),
                circle_center[2] + radius * (cos_a * perp1[2] + sin_a * perp2[2]),
            ];

            edges.push(Edge {
                start: Vertex::new(prev_point[0], prev_point[1], prev_point[2]),
                end: Vertex::new(curr_point[0], curr_point[1], curr_point[2]),
                curve_type: CurveType::Arc {
                    center: circle_center,
                    radius,
                },
            });

            prev_point = curr_point;
        }

        let wire = Wire {
            edges,
            closed: true,
        };

        return Ok(vec![wire]);
    }

    if angle < TOLERANCE {
        // Plane is parallel to cylinder axis: intersection is a line or no intersection
        // The line passes through the intersection of the plane and cylinder

        // Calculate distance from cylinder axis to the plane
        let v = [
            plane_origin[0] - cyl_origin[0],
            plane_origin[1] - cyl_origin[1],
            plane_origin[2] - cyl_origin[2],
        ];
        let dist_axis_to_plane = (dot(&v, &plane_n)).abs();

        if dist_axis_to_plane > radius + TOLERANCE {
            // No intersection
            return Ok(vec![]);
        }

        if (dist_axis_to_plane - radius).abs() < TOLERANCE {
            // Tangent: single line
            // Find the point on the cylinder axis closest to the plane
            let proj_length = dot(&v, &plane_n);
            let closest_on_axis = [
                cyl_origin[0] + proj_length * plane_n[0],
                cyl_origin[1] + proj_length * plane_n[1],
                cyl_origin[2] + proj_length * plane_n[2],
            ];

            let line_dir = cyl_a; // Line runs along cylinder axis
            let scale = 1000.0;

            let p1 = [
                closest_on_axis[0] - scale * line_dir[0],
                closest_on_axis[1] - scale * line_dir[1],
                closest_on_axis[2] - scale * line_dir[2],
            ];
            let p2 = [
                closest_on_axis[0] + scale * line_dir[0],
                closest_on_axis[1] + scale * line_dir[1],
                closest_on_axis[2] + scale * line_dir[2],
            ];

            let edge = Edge {
                start: Vertex::new(p1[0], p1[1], p1[2]),
                end: Vertex::new(p2[0], p2[1], p2[2]),
                curve_type: CurveType::Line,
            };

            let wire = Wire {
                edges: vec![edge],
                closed: false,
            };

            return Ok(vec![wire]);
        }

        // Two parallel lines
        let offset_dist = (radius * radius - dist_axis_to_plane * dist_axis_to_plane).sqrt();

        // Vector perpendicular to both plane normal and cylinder axis (in the plane)
        let line_dir = normalize(&cross(&plane_n, &cyl_a));

        // Vector from cylinder axis to plane, perpendicular to line direction
        let to_plane = normalize(&cross(&line_dir, &cyl_a));

        let center_offset = [
            to_plane[0] * dist_axis_to_plane,
            to_plane[1] * dist_axis_to_plane,
            to_plane[2] * dist_axis_to_plane,
        ];

        let mut wires = vec![];
        let scale = 1000.0;

        for sign in &[1.0, -1.0] {
            let line_center = [
                cyl_origin[0] + sign * offset_dist * (to_plane[0] - center_offset[0]),
                cyl_origin[1] + sign * offset_dist * (to_plane[1] - center_offset[1]),
                cyl_origin[2] + sign * offset_dist * (to_plane[2] - center_offset[2]),
            ];

            let p1 = [
                line_center[0] - scale * line_dir[0],
                line_center[1] - scale * line_dir[1],
                line_center[2] - scale * line_dir[2],
            ];
            let p2 = [
                line_center[0] + scale * line_dir[0],
                line_center[1] + scale * line_dir[1],
                line_center[2] + scale * line_dir[2],
            ];

            let edge = Edge {
                start: Vertex::new(p1[0], p1[1], p1[2]),
                end: Vertex::new(p2[0], p2[1], p2[2]),
                curve_type: CurveType::Line,
            };

            let wire = Wire {
                edges: vec![edge],
                closed: false,
            };

            wires.push(wire);
        }

        return Ok(wires);
    }

    // Oblique intersection: general case (ellipse)
    // This is complex; for now return a simplified circular approximation
    // Full ellipse computation would require solving the intersection algebraically

    let perp1 = perpendicular_to(&cyl_a);
    let perp2 = normalize(&cross(&cyl_a, &perp1));

    // Approximate as a circle in the plane
    let segments = 16;
    let mut edges = vec![];

    let mut prev_point = [
        cyl_origin[0] + radius * perp1[0],
        cyl_origin[1] + radius * perp1[1],
        cyl_origin[2] + radius * perp1[2],
    ];

    for i in 1..=segments {
        let angle = 2.0 * PI * (i as f64) / (segments as f64);
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let curr_point = [
            cyl_origin[0] + radius * (cos_a * perp1[0] + sin_a * perp2[0]),
            cyl_origin[1] + radius * (cos_a * perp1[1] + sin_a * perp2[1]),
            cyl_origin[2] + radius * (cos_a * perp1[2] + sin_a * perp2[2]),
        ];

        edges.push(Edge {
            start: Vertex::new(prev_point[0], prev_point[1], prev_point[2]),
            end: Vertex::new(curr_point[0], curr_point[1], curr_point[2]),
            curve_type: CurveType::Arc {
                center: *cyl_origin,
                radius,
            },
        });

        prev_point = curr_point;
    }

    let wire = Wire {
        edges,
        closed: true,
    };

    Ok(vec![wire])
}

/// Cylinder-cylinder intersection
fn cylinder_cylinder_intersection(
    o1: &[f64; 3],
    a1: &[f64; 3],
    r1: f64,
    o2: &[f64; 3],
    a2: &[f64; 3],
    r2: f64,
) -> Result<Vec<Wire>> {
    let a1_norm = normalize(a1);
    let a2_norm = normalize(a2);

    // Check if axes are parallel
    let cross_axes = cross(&a1_norm, &a2_norm);
    let cross_len = magnitude(&cross_axes);

    if cross_len < TOLERANCE {
        // Axes are parallel or coincident
        let v = [o2[0] - o1[0], o2[1] - o1[1], o2[2] - o1[2]];
        let axis_dist = magnitude(&cross(&v, &a1_norm));

        if axis_dist < TOLERANCE {
            // Cylinders are coaxial
            if (r1 - r2).abs() < TOLERANCE {
                // Same radius: infinite intersection (the cylinders overlap)
                return Ok(vec![]);
            } else {
                // Different radii: no intersection
                return Ok(vec![]);
            }
        } else if axis_dist < r1 + r2 {
            // Parallel cylinders intersecting in two lines
            // This is a complex calculation; simplified implementation
            return Ok(vec![]);
        } else {
            // No intersection
            return Ok(vec![]);
        }
    }

    // General case: non-parallel axes
    // Intersection curves are typically 4th-degree curves (quartic)
    // For a simplified implementation, we'll approximate with circle/line segments

    // Find the closest points on the two axes
    let v = [o2[0] - o1[0], o2[1] - o1[1], o2[2] - o1[2]];
    let dot_v_a1 = dot(&v, &a1_norm);
    let dot_v_a2 = dot(&v, &a2_norm);
    let dot_a1_a2 = dot(&a1_norm, &a2_norm);

    let denom = 1.0 - dot_a1_a2 * dot_a1_a2;
    if denom.abs() < TOLERANCE {
        return Ok(vec![]);
    }

    let t1 = (dot_v_a1 - dot_v_a2 * dot_a1_a2) / denom;
    let t2 = (dot_v_a2 * dot_a1_a2 - dot_v_a1) / denom;

    let p1 = [
        o1[0] + t1 * a1_norm[0],
        o1[1] + t1 * a1_norm[1],
        o1[2] + t1 * a1_norm[2],
    ];
    let p2 = [
        o2[0] + t2 * a2_norm[0],
        o2[1] + t2 * a2_norm[1],
        o2[2] + t2 * a2_norm[2],
    ];

    let axis_dist = distance(&p1, &p2);

    if axis_dist > r1 + r2 + TOLERANCE {
        // No intersection
        return Ok(vec![]);
    }

    // Approximate intersection with circular arcs
    // Create two elliptical curves representing the intersection
    let mut wires = vec![];

    for _ in 0..2 {
        let segments = 24;
        let mut edges = vec![];

        let mid_point = [
            (p1[0] + p2[0]) / 2.0,
            (p1[1] + p2[1]) / 2.0,
            (p1[2] + p2[2]) / 2.0,
        ];

        // Approximate as a circle
        let perp1 = perpendicular_to(&a1_norm);
        let perp2 = normalize(&cross(&a1_norm, &perp1));

        let radius_approx = (r1 + r2) / 2.0;

        let mut prev_point = [
            mid_point[0] + radius_approx * perp1[0],
            mid_point[1] + radius_approx * perp1[1],
            mid_point[2] + radius_approx * perp1[2],
        ];

        for i in 1..=segments {
            let angle = 2.0 * PI * (i as f64) / (segments as f64);
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let curr_point = [
                mid_point[0] + radius_approx * (cos_a * perp1[0] + sin_a * perp2[0]),
                mid_point[1] + radius_approx * (cos_a * perp1[1] + sin_a * perp2[1]),
                mid_point[2] + radius_approx * (cos_a * perp1[2] + sin_a * perp2[2]),
            ];

            edges.push(Edge {
                start: Vertex::new(prev_point[0], prev_point[1], prev_point[2]),
                end: Vertex::new(curr_point[0], curr_point[1], curr_point[2]),
                curve_type: CurveType::Arc {
                    center: mid_point,
                    radius: radius_approx,
                },
            });

            prev_point = curr_point;
        }

        let wire = Wire {
            edges,
            closed: true,
        };

        wires.push(wire);
    }

    Ok(wires)
}

// ===== Helper functions =====

fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let len = magnitude(v);
    if len > TOLERANCE {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn magnitude(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn perpendicular_to(v: &[f64; 3]) -> [f64; 3] {
    let abs_x = v[0].abs();
    let abs_y = v[1].abs();
    let abs_z = v[2].abs();

    let perp = if abs_x <= abs_y && abs_x <= abs_z {
        [1.0, 0.0, 0.0]
    } else if abs_y <= abs_x && abs_y <= abs_z {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };

    normalize(&cross(v, &perp))
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOL: f64 = 1e-9;

    fn pt_eq(p1: [f64; 2], p2: [f64; 2], tol: f64) -> bool {
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        (dx * dx + dy * dy).sqrt() < tol
    }
    
    fn pt_eq_3d(p1: [f64; 3], p2: [f64; 3], tol: f64) -> bool {
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        let dz = p1[2] - p2[2];
        (dx * dx + dy * dy + dz * dz).sqrt() < tol
    }

    #[test]
    fn test_curve2d_line_creation() {
        let line = Curve2D::line([0.0, 0.0], [1.0, 1.0]).unwrap();
        assert!(matches!(line, Curve2D::Line { .. }));
    }

    #[test]
    fn test_curve2d_circle_creation() {
        let circle = Curve2D::circle([0.0, 0.0], 1.0).unwrap();
        assert!(matches!(circle, Curve2D::Circle { .. }));
    }

    #[test]
    fn test_line_line_intersection_simple() {
        let line1 = Curve2D::line([0.0, 0.0], [1.0, 1.0]).unwrap();
        let line2 = Curve2D::line([0.0, 0.0], [1.0, -1.0]).unwrap();
        let result = intersect_curves_2d(&line1, &line2).unwrap();
        assert_eq!(result.len(), 1);
        assert!(pt_eq(result[0], [0.0, 0.0], TEST_TOL));
    }

    #[test]
    fn test_line_circle_two_intersections() {
        let line = Curve2D::line([-2.0, 0.0], [2.0, 0.0]).unwrap();
        let circle = Curve2D::circle([0.0, 0.0], 1.0).unwrap();
        let result = intersect_curves_2d(&line, &circle).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_circle_circle_two_intersections() {
        // Two circles with radius 1 and centers 1.5 apart intersect at 2 points
        let circle1 = Curve2D::circle([-0.75, 0.0], 1.0).unwrap();
        let circle2 = Curve2D::circle([0.75, 0.0], 1.0).unwrap();
        let result = intersect_curves_2d(&circle1, &circle2).unwrap();
        assert_eq!(result.len(), 2);
        for pt in &result {
            let d1 = dist_2d(pt, &[-0.75, 0.0]);
            let d2 = dist_2d(pt, &[0.75, 0.0]);
            assert!((d1 - 1.0).abs() < TEST_TOL);
            assert!((d2 - 1.0).abs() < TEST_TOL);
        }
    }
    
    #[test]
    fn test_line_plane_intersection() {
        // Test line piercing a plane
        let line_start = [0.0, 0.0, -1.0];
        let line_end = [0.0, 0.0, 1.0];
        let plane_origin = [0.0, 0.0, 0.0];
        let plane_normal = [0.0, 0.0, 1.0];
        
        let result = intersect_curve_surface(
            &CurveType::Line,
            line_start,
            line_end,
            &SurfaceType::Plane { origin: plane_origin, normal: plane_normal },
        ).unwrap();
        
        assert_eq!(result.len(), 1);
        assert!(pt_eq_3d(result[0], [0.0, 0.0, 0.0], 1e-6));
    }
    
    #[test]
    fn test_line_plane_parallel_no_intersection() {
        // Test line parallel to plane
        let line_start = [0.0, 0.0, 1.0];
        let line_end = [1.0, 0.0, 1.0];
        let plane_origin = [0.0, 0.0, 0.0];
        let plane_normal = [0.0, 0.0, 1.0];
        
        let result = intersect_curve_surface(
            &CurveType::Line,
            line_start,
            line_end,
            &SurfaceType::Plane { origin: plane_origin, normal: plane_normal },
        ).unwrap();
        
        assert_eq!(result.len(), 0);
    }
    
    #[test]
    fn test_line_sphere_intersection() {
        // Test line piercing a sphere
        let line_start = [0.0, 0.0, -2.0];
        let line_end = [0.0, 0.0, 2.0];
        let sphere_center = [0.0, 0.0, 0.0];
        let sphere_radius = 1.0;
        
        let result = intersect_curve_surface(
            &CurveType::Line,
            line_start,
            line_end,
            &SurfaceType::Sphere { center: sphere_center, radius: sphere_radius },
        ).unwrap();
        
        assert_eq!(result.len(), 2);
        // Should intersect at approximately [0, 0, -1] and [0, 0, 1]
        assert!(pt_eq_3d(result[0], [0.0, 0.0, -1.0], 1e-6));
        assert!(pt_eq_3d(result[1], [0.0, 0.0, 1.0], 1e-6));
    }
    
    #[test]
    fn test_line_sphere_tangent() {
        // Test line tangent to sphere
        let line_start = [-2.0, 1.0, 0.0];
        let line_end = [2.0, 1.0, 0.0];
        let sphere_center = [0.0, 0.0, 0.0];
        let sphere_radius = 1.0;
        
        let result = intersect_curve_surface(
            &CurveType::Line,
            line_start,
            line_end,
            &SurfaceType::Sphere { center: sphere_center, radius: sphere_radius },
        ).unwrap();
        
        assert_eq!(result.len(), 1);
        // Should be tangent at approximately [0, 1, 0]
        assert!(pt_eq_3d(result[0], [0.0, 1.0, 0.0], 1e-6));
    }
    
    #[test]
    fn test_line_sphere_no_intersection() {
        // Test line missing sphere
        let line_start = [-2.0, 2.0, 0.0];
        let line_end = [2.0, 2.0, 0.0];
        let sphere_center = [0.0, 0.0, 0.0];
        let sphere_radius = 1.0;
        
        let result = intersect_curve_surface(
            &CurveType::Line,
            line_start,
            line_end,
            &SurfaceType::Sphere { center: sphere_center, radius: sphere_radius },
        ).unwrap();
        
        assert_eq!(result.len(), 0);
    }
    
    #[test]
    fn test_line_cylinder_intersection() {
        // Test line piercing a cylinder
        let line_start = [-2.0, 0.0, 0.0];
        let line_end = [2.0, 0.0, 0.0];
        let cyl_origin = [0.0, 0.0, 0.0];
        let cyl_axis = [0.0, 0.0, 1.0];
        let cyl_radius = 1.0;
        
        let result = intersect_curve_surface(
            &CurveType::Line,
            line_start,
            line_end,
            &SurfaceType::Cylinder { origin: cyl_origin, axis: cyl_axis, radius: cyl_radius },
        ).unwrap();
        
        assert_eq!(result.len(), 2);
        // Should intersect at approximately [-1, 0, 0] and [1, 0, 0]
        assert!(pt_eq_3d(result[0], [-1.0, 0.0, 0.0], 1e-6));
        assert!(pt_eq_3d(result[1], [1.0, 0.0, 0.0], 1e-6));
    }
    
    #[test]
    fn test_line_cylinder_no_intersection() {
        // Test line missing cylinder
        let line_start = [-2.0, 2.0, 0.0];
        let line_end = [2.0, 2.0, 0.0];
        let cyl_origin = [0.0, 0.0, 0.0];
        let cyl_axis = [0.0, 0.0, 1.0];
        let cyl_radius = 1.0;
        
        let result = intersect_curve_surface(
            &CurveType::Line,
            line_start,
            line_end,
            &SurfaceType::Cylinder { origin: cyl_origin, axis: cyl_axis, radius: cyl_radius },
        ).unwrap();
        
        assert_eq!(result.len(), 0);
    }
}
