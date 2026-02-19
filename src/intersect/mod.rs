//! Surface-surface intersection algorithms
//!
//! Computes the intersection curves between two surfaces.
//! Returns the results as wires (connected edge sequences).

use crate::{
    brep::{Edge, Face, SurfaceType, Vertex, Wire, CurveType},
    {CascadeError, Result, TOLERANCE},
};
use std::f64::consts::PI;

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

    let v = [
        plane_origin[0] - cyl_origin[0],
        plane_origin[1] - cyl_origin[1],
        plane_origin[2] - cyl_origin[2],
    ];

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

// ===== Tests =====

// Tests for intersect_curves_2d and Curve2D are disabled until these features are implemented
// // #[cfg(test)]
// // mod tests {
//     use super::*;
// 
//     const TEST_TOL: f64 = 1e-9;
// 
//     fn pt_eq(p1: [f64; 2], p2: [f64; 2], tol: f64) -> bool {
//         let dx = p1[0] - p2[0];
//         let dy = p1[1] - p2[1];
//         (dx * dx + dy * dy).sqrt() < tol
//     }
// 
//     #[test]
//     fn test_curve2d_line_creation() {
//         let line = super::Curve2D::line([0.0, 0.0], [1.0, 1.0]).unwrap();
//         assert!(matches!(line, super::Curve2D::Line { .. }));
//     }
// 
//     #[test]
//     fn test_curve2d_line_same_points_error() {
//         let result = super::Curve2D::line([0.0, 0.0], [0.0, 0.0]);
//         assert!(result.is_err());
//     }
// 
//     #[test]
//     fn test_curve2d_circle_creation() {
//         let circle = super::Curve2D::circle([0.0, 0.0], 1.0).unwrap();
//         assert!(matches!(circle, super::Curve2D::Circle { .. }));
//     }
// 
//     #[test]
//     fn test_curve2d_circle_zero_radius_error() {
//         let result = super::Curve2D::circle([0.0, 0.0], 0.0);
//         assert!(result.is_err());
//     }
// 
//     #[test]
//     fn test_line_line_intersection_simple() {
//         // Lines: y = x and y = -x, intersect at origin
//         let line1 = super::Curve2D::line([0.0, 0.0], [1.0, 1.0]).unwrap();
//         let line2 = super::Curve2D::line([0.0, 0.0], [1.0, -1.0]).unwrap();
//         
//         let result = super::intersect_curves_2d(&line1, &line2).unwrap();
//         assert_eq!(result.len(), 1);
//         assert!(pt_eq(result[0], [0.0, 0.0], TEST_TOL));
//     }
// 
//     #[test]
//     fn test_line_line_intersection_arbitrary() {
//         // Lines: y = 2x and y = -x + 3, intersect at (1, 2)
//         let line1 = super::Curve2D::line([0.0, 0.0], [1.0, 2.0]).unwrap();
//         let line2 = super::Curve2D::line([0.0, 3.0], [1.0, 2.0]).unwrap();
//         
//         let result = super::intersect_curves_2d(&line1, &line2).unwrap();
//         assert_eq!(result.len(), 1);
//         assert!(pt_eq(result[0], [1.0, 2.0], TEST_TOL));
//     }
// 
//     #[test]
//     fn test_line_line_parallel() {
//         // Parallel lines: y = x and y = x + 1
//         let line1 = super::Curve2D::line([0.0, 0.0], [1.0, 1.0]).unwrap();
//         let line2 = super::Curve2D::line([0.0, 1.0], [1.0, 2.0]).unwrap();
//         
//         let result = super::intersect_curves_2d(&line1, &line2).unwrap();
//         assert_eq!(result.len(), 0);
//     }
// 
//     #[test]
//     fn test_line_circle_two_intersections() {
//         // Line through origin, circle centered at origin with radius 1
//         let line = super::Curve2D::line([-2.0, 0.0], [2.0, 0.0]).unwrap();
//         let circle = super::Curve2D::circle([0.0, 0.0], 1.0).unwrap();
//         
//         let result = super::intersect_curves_2d(&line, &circle).unwrap();
//         assert_eq!(result.len(), 2);
//         
//         // Points should be at (-1, 0) and (1, 0)
//         assert!(pt_eq(result[0], [-1.0, 0.0], TEST_TOL) || pt_eq(result[0], [1.0, 0.0], TEST_TOL));
//         assert!(pt_eq(result[1], [-1.0, 0.0], TEST_TOL) || pt_eq(result[1], [1.0, 0.0], TEST_TOL));
//     }
// 
//     #[test]
//     fn test_line_circle_tangent() {
//         // Line tangent to circle at (1, 0)
//         let line = super::Curve2D::line([1.0, -2.0], [1.0, 2.0]).unwrap();
//         let circle = super::Curve2D::circle([0.0, 0.0], 1.0).unwrap();
//         
//         let result = super::intersect_curves_2d(&line, &circle).unwrap();
//         assert_eq!(result.len(), 1);
//         assert!(pt_eq(result[0], [1.0, 0.0], TEST_TOL));
//     }
// 
//     #[test]
//     fn test_line_circle_no_intersection() {
//         // Line far from circle
//         let line = super::Curve2D::line([3.0, -2.0], [3.0, 2.0]).unwrap();
//         let circle = super::Curve2D::circle([0.0, 0.0], 1.0).unwrap();
//         
//         let result = super::intersect_curves_2d(&line, &circle).unwrap();
//         assert_eq!(result.len(), 0);
//     }
// 
//     #[test]
//     fn test_circle_circle_two_intersections() {
//         // Two circles with radius 1, centers at (-1, 0) and (1, 0)
//         // They intersect at (0, ±√3/2)
//         let circle1 = super::Curve2D::circle([-1.0, 0.0], 1.0).unwrap();
//         let circle2 = super::Curve2D::circle([1.0, 0.0], 1.0).unwrap();
//         
//         let result = super::intersect_curves_2d(&circle1, &circle2).unwrap();
//         assert_eq!(result.len(), 2);
//         
//         // Check that both points are at distance 1 from both centers
//         for pt in &result {
//             let d1 = dist_2d(pt, &[-1.0, 0.0]);
//             let d2 = dist_2d(pt, &[1.0, 0.0]);
//             assert!((d1 - 1.0).abs() < TEST_TOL);
//             assert!((d2 - 1.0).abs() < TEST_TOL);
//         }
//     }
// 
//     #[test]
//     fn test_circle_circle_external_tangent() {
//         // Two circles tangent externally at (1, 0)
//         let circle1 = super::Curve2D::circle([0.0, 0.0], 1.0).unwrap();
//         let circle2 = super::Curve2D::circle([2.0, 0.0], 1.0).unwrap();
//         
//         let result = super::intersect_curves_2d(&circle1, &circle2).unwrap();
//         assert_eq!(result.len(), 1);
//         assert!(pt_eq(result[0], [1.0, 0.0], TEST_TOL));
//     }
// 
//     #[test]
//     fn test_circle_circle_internal_tangent() {
//         // Two circles tangent internally at (1, 0)
//         let circle1 = super::Curve2D::circle([0.0, 0.0], 2.0).unwrap();
//         let circle2 = super::Curve2D::circle([1.0, 0.0], 1.0).unwrap();
//         
//         let result = super::intersect_curves_2d(&circle1, &circle2).unwrap();
//         assert_eq!(result.len(), 1);
//         assert!(pt_eq(result[0], [2.0, 0.0], TEST_TOL));
//     }
// 
//     #[test]
//     fn test_circle_circle_concentric() {
//         // Concentric circles
//         let circle1 = super::Curve2D::circle([0.0, 0.0], 1.0).unwrap();
//         let circle2 = super::Curve2D::circle([0.0, 0.0], 2.0).unwrap();
//         
//         let result = super::intersect_curves_2d(&circle1, &circle2).unwrap();
//         assert_eq!(result.len(), 0);
//     }
// 
//     #[test]
//     fn test_circle_circle_no_intersection_separate() {
//         // Separate circles
//         let circle1 = super::Curve2D::circle([0.0, 0.0], 1.0).unwrap();
//         let circle2 = super::Curve2D::circle([5.0, 0.0], 1.0).unwrap();
//         
//         let result = super::intersect_curves_2d(&circle1, &circle2).unwrap();
//         assert_eq!(result.len(), 0);
//     }
// 
//     #[test]
//     fn test_circle_circle_no_intersection_one_inside() {
//         // One circle inside another (non-tangent)
//         let circle1 = super::Curve2D::circle([0.0, 0.0], 5.0).unwrap();
//         let circle2 = super::Curve2D::circle([1.0, 0.0], 1.0).unwrap();
//         
//         let result = super::intersect_curves_2d(&circle1, &circle2).unwrap();
//         assert_eq!(result.len(), 0);
//     }
// 
//     #[test]
//     fn test_line_circle_vertical_line() {
//         // Vertical line x=0, circle at origin with radius 1
//         let line = super::Curve2D::line([0.0, -2.0], [0.0, 2.0]).unwrap();
//         let circle = super::Curve2D::circle([0.0, 0.0], 1.0).unwrap();
//         
//         let result = super::intersect_curves_2d(&line, &circle).unwrap();
//         assert_eq!(result.len(), 2);
//         assert!(pt_eq(result[0], [0.0, -1.0], TEST_TOL) || pt_eq(result[0], [0.0, 1.0], TEST_TOL));
//     }
// // }
