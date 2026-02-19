//! Local shape modification operations
//!
//! This module provides operations that modify individual faces and edges:
//! - **Split face**: Divides a face along a curve or edge

use crate::brep::{Face, Edge, Wire, Vertex, CurveType, SurfaceType};
use crate::{Result, CascadeError, TOLERANCE};

/// Split a face by a curve or edge.
///
/// Divides a face into multiple fragments where the splitting curve intersects it.
/// The splitting curve should either:
/// - Lie on the face surface, or
/// - Intersect the face's boundary edges
///
/// # Arguments
/// * `face` - The face to be split
/// * `splitting_curve` - An edge representing the splitting curve (must start and end on face boundary or lie on face)
///
/// # Returns
/// A vector of resulting face fragments. Returns an error if the splitting curve
/// doesn't properly intersect the face or if the operation is not supported for
/// the given surface type.
///
/// # Limitations
/// Currently supports splitting of planar faces. Support for other surface types
/// may be added in future versions.
///
/// # Example
/// ```ignore
/// let face = ... // Create or obtain a face
/// let splitting_edge = ... // Edge that lies on or intersects the face
/// let fragments = split_face(&face, &splitting_edge)?;
/// // fragments now contains the resulting face pieces
/// ```
pub fn split_face(face: &Face, splitting_curve: &Edge) -> Result<Vec<Face>> {
    // Match on the surface type to determine how to split
    match &face.surface_type {
        SurfaceType::Plane { origin, normal } => {
            split_planar_face(face, splitting_curve, origin, normal)
        }
        SurfaceType::Cylinder { .. } => {
            Err(CascadeError::NotImplemented(
                "Split face on cylindrical surfaces".to_string(),
            ))
        }
        SurfaceType::Sphere { .. } => {
            Err(CascadeError::NotImplemented(
                "Split face on spherical surfaces".to_string(),
            ))
        }
        SurfaceType::Cone { .. } => {
            Err(CascadeError::NotImplemented(
                "Split face on conical surfaces".to_string(),
            ))
        }
        SurfaceType::Torus { .. } => {
            Err(CascadeError::NotImplemented(
                "Split face on toroidal surfaces".to_string(),
            ))
        }
        SurfaceType::BezierSurface { .. } => {
            Err(CascadeError::NotImplemented(
                "Split face on Bezier surfaces".to_string(),
            ))
        }
        SurfaceType::BSpline { .. } => {
            Err(CascadeError::NotImplemented(
                "Split face on BSpline surfaces".to_string(),
            ))
        }
        SurfaceType::SurfaceOfRevolution { .. } => {
            Err(CascadeError::NotImplemented(
                "Split face on surfaces of revolution".to_string(),
            ))
        }
        SurfaceType::SurfaceOfLinearExtrusion { .. } => {
            Err(CascadeError::NotImplemented(
                "Split face on surfaces of linear extrusion".to_string(),
            ))
        }
        SurfaceType::RectangularTrimmedSurface { .. } => {
            Err(CascadeError::NotImplemented(
                "Split face on rectangular trimmed surfaces".to_string(),
            ))
        }
        SurfaceType::OffsetSurface { .. } => {
            Err(CascadeError::NotImplemented(
                "Split face on offset surfaces".to_string(),
            ))
        }
    }
}

/// Split a planar face by a splitting curve.
///
/// For a planar face, we split the face's boundary wires based on where
/// the splitting curve intersects them.
fn split_planar_face(
    face: &Face,
    splitting_curve: &Edge,
    plane_origin: &[f64; 3],
    plane_normal: &[f64; 3],
) -> Result<Vec<Face>> {
    // Validate that the splitting curve endpoints lie on or near the face
    let start_point = splitting_curve.start.point;
    let end_point = splitting_curve.end.point;

    // Check if endpoints are on the plane (within tolerance)
    if !is_point_on_plane(&start_point, plane_origin, plane_normal) {
        return Err(CascadeError::InvalidGeometry(
            "Splitting curve start point is not on the face plane".to_string(),
        ));
    }

    if !is_point_on_plane(&end_point, plane_origin, plane_normal) {
        return Err(CascadeError::InvalidGeometry(
            "Splitting curve end point is not on the face plane".to_string(),
        ));
    }

    // Check if endpoints are on the face boundary or edges
    let start_on_boundary = is_point_on_wire(&start_point, &face.outer_wire);
    let end_on_boundary = is_point_on_wire(&end_point, &face.outer_wire);

    if !start_on_boundary || !end_on_boundary {
        return Err(CascadeError::InvalidGeometry(
            "Splitting curve endpoints must be on face boundary edges".to_string(),
        ));
    }

    // Now split the outer wire using the splitting curve
    let (wire1, wire2) = split_wire_by_curve(&face.outer_wire, splitting_curve)?;

    // Create two new faces from the split wires
    let mut result_faces = Vec::new();

    // First face with split wire 1
    if !wire1.edges.is_empty() {
        result_faces.push(Face {
            outer_wire: wire1,
            inner_wires: face.inner_wires.clone(), // Keep inner wires (holes) for now
            surface_type: face.surface_type.clone(),
        });
    }

    // Second face with split wire 2
    if !wire2.edges.is_empty() {
        result_faces.push(Face {
            outer_wire: wire2,
            inner_wires: face.inner_wires.clone(), // Keep inner wires (holes) for now
            surface_type: face.surface_type.clone(),
        });
    }

    if result_faces.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Split operation resulted in no valid faces".to_string(),
        ));
    }

    Ok(result_faces)
}

/// Split a wire using a splitting curve.
///
/// Returns two new wires that together form a closed path including the splitting curve.
fn split_wire_by_curve(
    wire: &Wire,
    splitting_curve: &Edge,
) -> Result<(Wire, Wire)> {
    let split_start = splitting_curve.start.point;
    let split_end = splitting_curve.end.point;

    // Find the edges in the wire that contain the split start and end points
    let start_edge_idx = find_edge_containing_point(wire, &split_start)?;
    let end_edge_idx = find_edge_containing_point(wire, &split_end)?;

    if start_edge_idx == end_edge_idx {
        return Err(CascadeError::InvalidGeometry(
            "Split start and end points must be on different edges or different points on the wire"
                .to_string(),
        ));
    }

    // Split the wire at the two locations
    let mut edges1 = Vec::new();
    let mut edges2 = Vec::new();

    let num_edges = wire.edges.len();
    let mut current_idx = start_edge_idx;

    // Traverse from start_edge_idx to end_edge_idx
    loop {
        let edge = &wire.edges[current_idx];

        if current_idx == start_edge_idx {
            // Split the starting edge at split_start
            if !points_equal(&edge.start.point, &split_start) {
                let partial_edge = Edge {
                    start: edge.start.clone(),
                    end: Vertex::new(split_start[0], split_start[1], split_start[2]),
                    curve_type: edge.curve_type.clone(),
                };
                edges1.push(partial_edge);
            }
        } else if current_idx == end_edge_idx {
            // Split the ending edge at split_end
            let partial_edge = Edge {
                start: edge.start.clone(),
                end: Vertex::new(split_end[0], split_end[1], split_end[2]),
                curve_type: edge.curve_type.clone(),
            };
            edges1.push(partial_edge);
            break;
        } else {
            // Full edge, add to edges1
            edges1.push(edge.clone());
        }

        current_idx = (current_idx + 1) % num_edges;
    }

    // Add the splitting curve to close the first wire
    edges1.push(splitting_curve.clone());

    // Traverse the rest for the second wire
    current_idx = end_edge_idx;

    loop {
        let edge = &wire.edges[current_idx];

        if current_idx == end_edge_idx {
            // Split the ending edge at split_end (going the other direction)
            if !points_equal(&edge.end.point, &split_end) {
                let partial_edge = Edge {
                    start: Vertex::new(split_end[0], split_end[1], split_end[2]),
                    end: edge.end.clone(),
                    curve_type: edge.curve_type.clone(),
                };
                edges2.push(partial_edge);
            }
        } else if current_idx == start_edge_idx {
            // We've reached the start edge, add the reverse of the split curve
            let reverse_split = Edge {
                start: splitting_curve.end.clone(),
                end: splitting_curve.start.clone(),
                curve_type: splitting_curve.curve_type.clone(),
            };
            edges2.push(reverse_split);
            break;
        } else {
            // Full edge, add to edges2
            edges2.push(edge.clone());
        }

        current_idx = (current_idx + 1) % num_edges;
    }

    let wire1 = Wire {
        edges: edges1,
        closed: true,
    };

    let wire2 = Wire {
        edges: edges2,
        closed: true,
    };

    Ok((wire1, wire2))
}

/// Find the index of an edge in a wire that contains (or is very close to) a given point.
fn find_edge_containing_point(wire: &Wire, point: &[f64; 3]) -> Result<usize> {
    for (i, edge) in wire.edges.iter().enumerate() {
        // Check if point is close to edge start
        if distance(&edge.start.point, point) < TOLERANCE {
            return Ok(i);
        }
        // Check if point is close to edge end
        if distance(&edge.end.point, point) < TOLERANCE {
            return Ok(i);
        }
    }

    Err(CascadeError::InvalidGeometry(
        format!("Point {:?} not found on any edge of the wire", point),
    ))
}

/// Check if a point lies on the plane (within tolerance).
fn is_point_on_plane(point: &[f64; 3], plane_origin: &[f64; 3], plane_normal: &[f64; 3]) -> bool {
    let vec_to_point = [
        point[0] - plane_origin[0],
        point[1] - plane_origin[1],
        point[2] - plane_origin[2],
    ];

    let normal_norm = normalize(plane_normal);
    let distance_to_plane = (vec_to_point[0] * normal_norm[0]
        + vec_to_point[1] * normal_norm[1]
        + vec_to_point[2] * normal_norm[2])
        .abs();

    distance_to_plane < TOLERANCE
}

/// Check if a point lies on the boundary of a wire.
fn is_point_on_wire(point: &[f64; 3], wire: &Wire) -> bool {
    for edge in &wire.edges {
        if distance(&edge.start.point, point) < TOLERANCE
            || distance(&edge.end.point, point) < TOLERANCE
        {
            return true;
        }
    }
    false
}

/// Calculate the distance between two 3D points.
fn distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    let dz = p1[2] - p2[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Check if two points are equal (within tolerance).
fn points_equal(p1: &[f64; 3], p2: &[f64; 3]) -> bool {
    distance(p1, p2) < TOLERANCE
}

/// Normalize a vector to unit length.
fn normalize(vec: &[f64; 3]) -> [f64; 3] {
    let len = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();
    if len < TOLERANCE {
        [0.0, 0.0, 0.0]
    } else {
        [vec[0] / len, vec[1] / len, vec[2] / len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_planar_face_basic() {
        // Create a simple rectangular face
        let outer_wire = Wire {
            edges: vec![
                Edge {
                    start: Vertex::new(0.0, 0.0, 0.0),
                    end: Vertex::new(10.0, 0.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(10.0, 0.0, 0.0),
                    end: Vertex::new(10.0, 10.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(10.0, 10.0, 0.0),
                    end: Vertex::new(0.0, 10.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(0.0, 10.0, 0.0),
                    end: Vertex::new(0.0, 0.0, 0.0),
                    curve_type: CurveType::Line,
                },
            ],
            closed: true,
        };

        let face = Face {
            outer_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        };

        // Create a splitting curve from left edge to right edge
        let splitting_curve = Edge {
            start: Vertex::new(0.0, 5.0, 0.0),
            end: Vertex::new(10.0, 5.0, 0.0),
            curve_type: CurveType::Line,
        };

        let result = split_face(&face, &splitting_curve);
        assert!(result.is_ok(), "Split should succeed");

        let fragments = result.unwrap();
        assert_eq!(fragments.len(), 2, "Should produce exactly 2 fragments");

        // Both fragments should be valid faces
        for fragment in fragments {
            assert!(!fragment.outer_wire.edges.is_empty(), "Fragment should have edges");
            assert!(fragment.outer_wire.closed, "Fragment wire should be closed");
        }
    }

    #[test]
    fn test_split_face_endpoints_not_on_plane() {
        let outer_wire = Wire {
            edges: vec![
                Edge {
                    start: Vertex::new(0.0, 0.0, 0.0),
                    end: Vertex::new(10.0, 0.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(10.0, 0.0, 0.0),
                    end: Vertex::new(10.0, 10.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(10.0, 10.0, 0.0),
                    end: Vertex::new(0.0, 10.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(0.0, 10.0, 0.0),
                    end: Vertex::new(0.0, 0.0, 0.0),
                    curve_type: CurveType::Line,
                },
            ],
            closed: true,
        };

        let face = Face {
            outer_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        };

        // Create a splitting curve with endpoint NOT on the plane (z=5)
        let splitting_curve = Edge {
            start: Vertex::new(0.0, 5.0, 5.0), // Not on z=0 plane
            end: Vertex::new(10.0, 5.0, 0.0),
            curve_type: CurveType::Line,
        };

        let result = split_face(&face, &splitting_curve);
        assert!(
            result.is_err(),
            "Split should fail when curve endpoint is not on plane"
        );
    }

    #[test]
    fn test_split_face_endpoints_not_on_boundary() {
        let outer_wire = Wire {
            edges: vec![
                Edge {
                    start: Vertex::new(0.0, 0.0, 0.0),
                    end: Vertex::new(10.0, 0.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(10.0, 0.0, 0.0),
                    end: Vertex::new(10.0, 10.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(10.0, 10.0, 0.0),
                    end: Vertex::new(0.0, 10.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(0.0, 10.0, 0.0),
                    end: Vertex::new(0.0, 0.0, 0.0),
                    curve_type: CurveType::Line,
                },
            ],
            closed: true,
        };

        let face = Face {
            outer_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        };

        // Create a splitting curve with endpoints in the interior (not on boundary)
        let splitting_curve = Edge {
            start: Vertex::new(2.0, 5.0, 0.0), // Interior
            end: Vertex::new(8.0, 5.0, 0.0),   // Interior
            curve_type: CurveType::Line,
        };

        let result = split_face(&face, &splitting_curve);
        assert!(
            result.is_err(),
            "Split should fail when curve endpoints are not on boundary"
        );
    }

    #[test]
    fn test_split_face_same_edge() {
        let outer_wire = Wire {
            edges: vec![
                Edge {
                    start: Vertex::new(0.0, 0.0, 0.0),
                    end: Vertex::new(10.0, 0.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(10.0, 0.0, 0.0),
                    end: Vertex::new(10.0, 10.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(10.0, 10.0, 0.0),
                    end: Vertex::new(0.0, 10.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(0.0, 10.0, 0.0),
                    end: Vertex::new(0.0, 0.0, 0.0),
                    curve_type: CurveType::Line,
                },
            ],
            closed: true,
        };

        let face = Face {
            outer_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        };

        // Create a splitting curve with both endpoints on the same edge
        let splitting_curve = Edge {
            start: Vertex::new(0.0, 0.0, 0.0), // On bottom edge
            end: Vertex::new(5.0, 0.0, 0.0),   // Also on bottom edge
            curve_type: CurveType::Line,
        };

        let result = split_face(&face, &splitting_curve);
        assert!(
            result.is_err(),
            "Split should fail when both endpoints are on same edge"
        );
    }

    #[test]
    fn test_split_face_unsupported_surface() {
        let outer_wire = Wire {
            edges: vec![
                Edge {
                    start: Vertex::new(0.0, 0.0, 0.0),
                    end: Vertex::new(10.0, 0.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(10.0, 0.0, 0.0),
                    end: Vertex::new(10.0, 10.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(10.0, 10.0, 0.0),
                    end: Vertex::new(0.0, 10.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(0.0, 10.0, 0.0),
                    end: Vertex::new(0.0, 0.0, 0.0),
                    curve_type: CurveType::Line,
                },
            ],
            closed: true,
        };

        // Create a cylindrical face
        let face = Face {
            outer_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Cylinder {
                origin: [0.0, 0.0, 0.0],
                axis: [0.0, 0.0, 1.0],
                radius: 5.0,
            },
        };

        let splitting_curve = Edge {
            start: Vertex::new(0.0, 5.0, 0.0),
            end: Vertex::new(10.0, 5.0, 0.0),
            curve_type: CurveType::Line,
        };

        let result = split_face(&face, &splitting_curve);
        assert!(
            result.is_err(),
            "Split on cylindrical surfaces should not be implemented"
        );
    }
}
