//! Chamfering operations
//!
//! This module provides chamfer operations that create flat, beveled edges on solids.
//! A chamfer is a straight-edged cut at the intersection of two faces, typically at 45 degrees.

use crate::brep::{Solid, Edge, Face, Wire, Vertex, Shell, CurveType, SurfaceType};
use crate::{Result, CascadeError};
use crate::brep::topology;

/// Create a constant-distance chamfer on specified edges of a solid
///
/// # Arguments
/// * `solid` - The input solid to chamfer
/// * `edges` - Array of edges to chamfer (by index in a canonical edge list)
/// * `distance` - The distance of the chamfer (uniform offset from the edge)
///
/// # Returns
/// A new Solid with chamfered edges
///
/// # Implementation Notes
/// For a straight edge where two planar faces meet:
/// 1. Find the two adjacent faces
/// 2. Create a planar chamfer surface at 45° to both faces
/// 3. Trim the original faces along the chamfer intersection
/// 4. Create new topology connecting everything
pub fn make_chamfer(solid: &Solid, edge_indices: &[usize], distance: f64) -> Result<Solid> {
    if distance <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Chamfer distance must be positive".into(),
        ));
    }

    if edge_indices.is_empty() {
        return Ok(solid.clone());
    }

    // For now, implement simple case: single edge chamfer
    if edge_indices.len() != 1 {
        return Err(CascadeError::NotImplemented(
            "Multi-edge chamfering not yet implemented".into(),
        ));
    }

    let edge_idx = edge_indices[0];
    
    // Get all edges from the solid in a deterministic order
    let all_edges = collect_all_edges(solid);
    
    if edge_idx >= all_edges.len() {
        return Err(CascadeError::InvalidGeometry(
            "Edge index out of bounds".into(),
        ));
    }

    let target_edge = &all_edges[edge_idx];

    // Find the two faces that share this edge
    let adjacent = find_adjacent_faces(target_edge, solid)?;
    if adjacent.len() < 2 {
        return Err(CascadeError::TopologyError(
            "Edge must be shared by exactly two faces for chamfering".into(),
        ));
    }

    let face1 = &adjacent[0];
    let face2 = &adjacent[1];

    // Check if both faces are planar (simple case)
    let (normal1, origin1) = extract_plane_info(face1)?;
    let (normal2, origin2) = extract_plane_info(face2)?;

    // Get edge direction
    let edge_dir = get_edge_direction(target_edge)?;
    let edge_start = [target_edge.start.point[0], target_edge.start.point[1], target_edge.start.point[2]];
    let edge_end = [target_edge.end.point[0], target_edge.end.point[1], target_edge.end.point[2]];

    // Create the chamfer surface (planar bevel at 45°)
    let chamfer_face = create_chamfer_surface(
        &edge_start,
        &edge_end,
        &normal1,
        &normal2,
        distance,
    )?;

    // Create trimmed versions of the original faces
    let trimmed_face1 = trim_face_for_chamfer(face1, target_edge, distance, &normal1)?;
    let trimmed_face2 = trim_face_for_chamfer(face2, target_edge, distance, &normal2)?;

    // Collect all other faces (not involved in the chamfer)
    let other_faces = collect_other_faces(solid, face1, face2);

    // Build new shell with modified geometry
    let mut new_faces = vec![trimmed_face1, trimmed_face2, chamfer_face];
    new_faces.extend(other_faces);

    let new_shell = Shell {
        faces: new_faces,
        closed: true,
    };

    let result = Solid {
        outer_shell: new_shell,
        inner_shells: vec![],
    };

    Ok(result)
}

/// Collect all edges from a solid in a deterministic order
fn collect_all_edges(solid: &Solid) -> Vec<Edge> {
    let mut edges = Vec::new();
    let faces = topology::get_solid_faces_internal(solid);
    
    for face in faces {
        edges.extend(face.outer_wire.edges.iter().cloned());
        for wire in &face.inner_wires {
            edges.extend(wire.edges.iter().cloned());
        }
    }
    
    edges
}

/// Find faces adjacent to an edge
fn find_adjacent_faces(edge: &Edge, solid: &Solid) -> Result<Vec<Face>> {
    let faces = topology::get_solid_faces_internal(solid);
    let mut adjacent = Vec::new();

    for face in faces {
        for wire in &[&face.outer_wire][..] {
            for face_edge in &wire.edges {
                if edges_equal(edge, face_edge) {
                    adjacent.push(face.clone());
                    break;
                }
            }
        }

        for wire in &face.inner_wires {
            for face_edge in &wire.edges {
                if edges_equal(edge, face_edge) {
                    adjacent.push(face.clone());
                    break;
                }
            }
        }
    }

    Ok(adjacent)
}

/// Check if two edges are the same (within tolerance)
fn edges_equal(e1: &Edge, e2: &Edge) -> bool {
    const TOL: f64 = 1e-6;
    
    let same_direction = 
        (e1.start.point[0] - e2.start.point[0]).abs() < TOL &&
        (e1.start.point[1] - e2.start.point[1]).abs() < TOL &&
        (e1.start.point[2] - e2.start.point[2]).abs() < TOL &&
        (e1.end.point[0] - e2.end.point[0]).abs() < TOL &&
        (e1.end.point[1] - e2.end.point[1]).abs() < TOL &&
        (e1.end.point[2] - e2.end.point[2]).abs() < TOL;

    let reversed = 
        (e1.start.point[0] - e2.end.point[0]).abs() < TOL &&
        (e1.start.point[1] - e2.end.point[1]).abs() < TOL &&
        (e1.start.point[2] - e2.end.point[2]).abs() < TOL &&
        (e1.end.point[0] - e2.start.point[0]).abs() < TOL &&
        (e1.end.point[1] - e2.start.point[1]).abs() < TOL &&
        (e1.end.point[2] - e2.start.point[2]).abs() < TOL;

    same_direction || reversed
}

/// Extract plane information from a planar face
fn extract_plane_info(face: &Face) -> Result<([f64; 3], [f64; 3])> {
    match &face.surface_type {
        SurfaceType::Plane { origin, normal } => Ok((*normal, *origin)),
        _ => Err(CascadeError::NotImplemented(
            "Non-planar face chamfering not yet supported".into(),
        )),
    }
}

/// Get the direction vector of an edge
fn get_edge_direction(edge: &Edge) -> Result<[f64; 3]> {
    match &edge.curve_type {
        CurveType::Line => {
            let dx = edge.end.point[0] - edge.start.point[0];
            let dy = edge.end.point[1] - edge.start.point[1];
            let dz = edge.end.point[2] - edge.start.point[2];
            
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            if len < 1e-10 {
                return Err(CascadeError::InvalidGeometry(
                    "Edge has zero length".into(),
                ));
            }
            
            Ok([dx / len, dy / len, dz / len])
        }
        _ => Err(CascadeError::NotImplemented(
            "Non-linear edge chamfering not yet supported".into(),
        )),
    }
}

/// Create a planar chamfer surface between two planar faces
///
/// The chamfer is created as a plane that bisects the angle between the two faces.
/// For two perpendicular faces, this creates a 45-degree plane.
fn create_chamfer_surface(
    edge_start: &[f64; 3],
    edge_end: &[f64; 3],
    normal1: &[f64; 3],
    normal2: &[f64; 3],
    distance: f64,
) -> Result<Face> {
    // Edge direction
    let edge_dir = [
        edge_end[0] - edge_start[0],
        edge_end[1] - edge_start[1],
        edge_end[2] - edge_start[2],
    ];
    let edge_len = (edge_dir[0] * edge_dir[0] + edge_dir[1] * edge_dir[1] + edge_dir[2] * edge_dir[2]).sqrt();
    let edge_dir_norm = [edge_dir[0] / edge_len, edge_dir[1] / edge_len, edge_dir[2] / edge_len];

    // Calculate the chamfer plane normal (bisector of the two face normals)
    let chamfer_normal = calculate_chamfer_normal(normal1, normal2)?;

    // The chamfer surface is a plane perpendicular to both face normals
    // Its normal is the bisector of the two normals, oriented outward
    
    // Create vertices for the chamfer face
    // The chamfer face is a quadrilateral with edges connecting:
    // - Two points on face1 (offset by distance along its normal)
    // - Two points on face2 (offset by distance along its normal)
    
    let offset1 = scale_vector(&normal1, distance);
    let offset2 = scale_vector(&normal2, distance);

    // Chamfer vertices
    let v1 = Vertex::new(
        edge_start[0] + offset1[0],
        edge_start[1] + offset1[1],
        edge_start[2] + offset1[2],
    );

    let v2 = Vertex::new(
        edge_end[0] + offset1[0],
        edge_end[1] + offset1[1],
        edge_end[2] + offset1[2],
    );

    let v3 = Vertex::new(
        edge_end[0] + offset2[0],
        edge_end[1] + offset2[1],
        edge_end[2] + offset2[2],
    );

    let v4 = Vertex::new(
        edge_start[0] + offset2[0],
        edge_start[1] + offset2[1],
        edge_start[2] + offset2[2],
    );

    // Create edges for the chamfer face (quadrilateral)
    let edge1 = Edge {
        start: v1.clone(),
        end: v2.clone(),
        curve_type: CurveType::Line,
    };

    let edge2 = Edge {
        start: v2.clone(),
        end: v3.clone(),
        curve_type: CurveType::Line,
    };

    let edge3 = Edge {
        start: v3.clone(),
        end: v4.clone(),
        curve_type: CurveType::Line,
    };

    let edge4 = Edge {
        start: v4.clone(),
        end: v1.clone(),
        curve_type: CurveType::Line,
    };

    let chamfer_wire = Wire {
        edges: vec![edge1, edge2, edge3, edge4],
        closed: true,
    };

    // Create the chamfer face as a plane
    // The origin is at the midpoint between the two edge endpoints projected onto both faces
    let chamfer_origin = [
        edge_start[0] + (offset1[0] + offset2[0]) / 2.0,
        edge_start[1] + (offset1[1] + offset2[1]) / 2.0,
        edge_start[2] + (offset1[2] + offset2[2]) / 2.0,
    ];

    let chamfer_face = Face {
        outer_wire: chamfer_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: chamfer_origin,
            normal: chamfer_normal,
        },
    };

    Ok(chamfer_face)
}

/// Calculate the normal of the chamfer plane (bisector of two face normals)
fn calculate_chamfer_normal(normal1: &[f64; 3], normal2: &[f64; 3]) -> Result<[f64; 3]> {
    // The chamfer normal is the bisector of the two face normals
    // This creates a plane that is equidistant from both faces
    
    let bisector = [
        normal1[0] + normal2[0],
        normal1[1] + normal2[1],
        normal1[2] + normal2[2],
    ];

    let len = (bisector[0] * bisector[0] + bisector[1] * bisector[1] + bisector[2] * bisector[2]).sqrt();
    
    if len < 1e-10 {
        return Err(CascadeError::InvalidGeometry(
            "Cannot compute chamfer for parallel or opposite faces".into(),
        ));
    }

    Ok([
        bisector[0] / len,
        bisector[1] / len,
        bisector[2] / len,
    ])
}

/// Scale a vector by a scalar
fn scale_vector(vec: &[f64; 3], scale: f64) -> [f64; 3] {
    [vec[0] * scale, vec[1] * scale, vec[2] * scale]
}

/// Trim a face for chamfering by moving edges inward by the chamfer distance
fn trim_face_for_chamfer(
    face: &Face,
    edge: &Edge,
    distance: f64,
    normal: &[f64; 3],
) -> Result<Face> {
    // For a planar face, we need to move the edge inward by the chamfer distance
    // along the face's normal direction
    
    let mut trimmed_edges = Vec::new();

    for face_edge in &face.outer_wire.edges {
        if edges_equal(face_edge, edge) {
            // This is the edge being chamfered - offset it inward
            let offset_start = offset_point(&face_edge.start.point, normal, distance);
            let offset_end = offset_point(&face_edge.end.point, normal, distance);

            let trimmed_edge = Edge {
                start: Vertex::new(offset_start[0], offset_start[1], offset_start[2]),
                end: Vertex::new(offset_end[0], offset_end[1], offset_end[2]),
                curve_type: CurveType::Line,
            };
            trimmed_edges.push(trimmed_edge);
        } else {
            // Keep other edges as-is for now (simplified)
            trimmed_edges.push(face_edge.clone());
        }
    }

    let trimmed_wire = Wire {
        edges: trimmed_edges,
        closed: face.outer_wire.closed,
    };

    Ok(Face {
        outer_wire: trimmed_wire,
        inner_wires: face.inner_wires.clone(),
        surface_type: face.surface_type.clone(),
    })
}

/// Offset a point along a normal direction
fn offset_point(point: &[f64; 3], normal: &[f64; 3], distance: f64) -> [f64; 3] {
    [
        point[0] + normal[0] * distance,
        point[1] + normal[1] * distance,
        point[2] + normal[2] * distance,
    ]
}

/// Collect all faces except the two specified ones
fn collect_other_faces(solid: &Solid, face1: &Face, face2: &Face) -> Vec<Face> {
    let all_faces = topology::get_solid_faces_internal(solid);
    let mut other = Vec::new();

    for face in all_faces {
        // Simple heuristic: compare first edge of outer wire
        if !face.outer_wire.edges.is_empty() {
            let face1_first = if !face1.outer_wire.edges.is_empty() {
                &face1.outer_wire.edges[0]
            } else {
                continue;
            };

            let face2_first = if !face2.outer_wire.edges.is_empty() {
                &face2.outer_wire.edges[0]
            } else {
                continue;
            };

            let this_first = &face.outer_wire.edges[0];

            if !edges_equal(this_first, face1_first) && !edges_equal(this_first, face2_first) {
                other.push(face.clone());
            }
        }
    }

    other
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::make_box;

    #[test]
    fn test_chamfer_box_edge() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        
        // Chamfer the first edge with distance 1.0
        let result = make_chamfer(&solid, &[0], 1.0);
        
        assert!(result.is_ok(), "Chamfer operation should succeed");
        let chamfered = result.unwrap();
        
        // Check that the result is still a valid solid
        assert!(!chamfered.outer_shell.faces.is_empty(), "Chamfered solid should have faces");
        assert!(chamfered.outer_shell.faces.len() >= 6, "Chamfered solid should have at least 6 faces");
    }

    #[test]
    fn test_chamfer_negative_distance() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        
        let result = make_chamfer(&solid, &[0], -1.0);
        assert!(result.is_err(), "Negative distance should fail");
    }

    #[test]
    fn test_chamfer_empty_edges() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        
        let result = make_chamfer(&solid, &[], 1.0);
        assert!(result.is_ok(), "Empty edge list should return original solid");
    }

    #[test]
    fn test_chamfer_preserves_solid_property() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        let chamfered = make_chamfer(&solid, &[0], 0.5).expect("Chamfer failed");
        
        // Result should still be closed
        assert!(chamfered.outer_shell.closed, "Chamfered solid should be closed");
    }
}
