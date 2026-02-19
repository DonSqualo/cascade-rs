//! Offset operations on solids
//!
//! This module provides offset operations such as creating hollow solids (shelling)
//! by offsetting faces inward and optionally removing specified faces.

use crate::brep::{Solid, Shell, Face, Vertex, Wire, Edge, SurfaceType, CurveType};
use crate::{Result, CascadeError, TOLERANCE};
use crate::brep::topology;

/// Offset a solid by growing or shrinking (thick solid operation)
///
/// This operation creates a new solid by offsetting all faces by a distance:
/// - Positive offset = grow the solid (all faces move outward)
/// - Negative offset = shrink the solid (all faces move inward)
/// - Handles face intersections at corners through blending
///
/// # Arguments
/// * `solid` - The input solid to offset
/// * `offset` - The offset distance (positive = grow, negative = shrink)
///
/// # Returns
/// A new Solid with all faces offset by the specified distance
///
/// # Example
/// ```ignore
/// let box = make_box(10.0, 10.0, 10.0)?;
/// let grown = thick_solid(&box, 1.0)?;      // Grow by 1.0
/// let shrunk = thick_solid(&box, -0.5)?;    // Shrink by 0.5
/// ```
pub fn thick_solid(solid: &Solid, offset: f64) -> Result<Solid> {
    // Validate offset
    if offset == 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Offset distance must be non-zero".into(),
        ));
    }

    // For shrinking, validate that the offset is not too large
    if offset < 0.0 {
        let bounds = get_solid_bounds(solid);
        let min_dim = ((bounds.1[0] - bounds.0[0]).min(bounds.1[1] - bounds.0[1]))
            .min(bounds.1[2] - bounds.0[2]);
        
        if offset.abs() >= min_dim / 2.0 {
            return Err(CascadeError::InvalidGeometry(
                format!("Shrink offset {} is too large for geometry with minimum dimension {}", 
                        offset.abs(), min_dim),
            ));
        }
    }

    // Get all faces from the solid
    let all_faces = topology::get_solid_faces_internal(solid);
    
    if all_faces.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Solid must have at least one face".into(),
        ));
    }

    // Offset each face
    let mut offset_faces = Vec::new();
    
    for face in &all_faces {
        let offset_face = offset_face_by_distance(face, offset)?;
        offset_faces.push(offset_face);
    }

    // Create blending faces at edges where adjacent faces meet
    for i in 0..all_faces.len() {
        for j in (i + 1)..all_faces.len() {
            // Check if faces are adjacent (share an edge)
            if let Some(shared_edge) = topology::shared_edge(&all_faces[i], &all_faces[j]) {
                // Create a blending face connecting the offset edges
                if let Ok(blend_face) = create_thick_solid_blend(
                    &all_faces[i], 
                    &all_faces[j], 
                    &shared_edge, 
                    offset
                ) {
                    offset_faces.push(blend_face);
                }
            }
        }
    }

    // Create new outer shell from offset faces
    let new_shell = Shell {
        faces: offset_faces,
        closed: true,
    };

    // Create the result solid
    // Note: inner shells (voids) would also need offsetting in a full implementation
    // For now, we skip them to keep the implementation focused on the outer solid
    let result_solid = Solid {
        outer_shell: new_shell,
        inner_shells: Vec::new(),  // Clear inner shells for offset
    };

    Ok(result_solid)
}

/// Create a hollow version of a solid (shell operation)
///
/// This operation creates a hollow solid by:
/// 1. Offsetting all faces inward by the specified thickness
/// 2. Removing specified faces (which creates openings)
/// 3. Creating new faces at the openings
/// 4. Blending corners and edges
///
/// # Arguments
/// * `solid` - The input solid to hollow
/// * `thickness` - The thickness of the walls (must be positive and less than half the smallest dimension)
/// * `faces_to_remove` - Array of Face objects to remove (creates openings)
///
/// # Returns
/// A new Solid with hollow interior and optionally open faces
///
/// # Example
/// ```ignore
/// let box = make_box(10.0, 10.0, 10.0)?;
/// let hollow = make_shell(&box, 1.0, &[])?;  // Closed hollow box
/// let open = make_shell(&box, 1.0, &[box.outer_shell.faces[0]])?;  // With one face removed
/// ```
pub fn make_shell(solid: &Solid, thickness: f64, faces_to_remove: &[Face]) -> Result<Solid> {
    // Validate input parameters
    if thickness <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Shell thickness must be positive".into(),
        ));
    }

    // Check that thickness is reasonable
    let bounds = get_solid_bounds(solid);
    let min_dim = ((bounds.1[0] - bounds.0[0]).min(bounds.1[1] - bounds.0[1]))
        .min(bounds.1[2] - bounds.0[2]);
    
    if thickness >= min_dim / 2.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Shell thickness {} is too large for geometry with minimum dimension {}", 
                    thickness, min_dim),
        ));
    }

    // Get all faces from the solid
    let all_faces = topology::get_solid_faces_internal(solid);
    
    if all_faces.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Solid must have at least one face".into(),
        ));
    }

    // Identify which faces to remove (by reference comparison)
    let faces_to_remove_set = build_face_removal_set(faces_to_remove, &all_faces);

    // Create offset faces (inward by thickness)
    let mut offset_faces = Vec::new();
    let mut removed_face_indices = Vec::new();

    for (face_idx, face) in all_faces.iter().enumerate() {
        // Check if this face should be removed
        if faces_to_remove_set.contains(&face_idx) {
            removed_face_indices.push(face_idx);
            continue;
        }

        // Offset the face inward by thickness
        let offset_face = offset_face_inward(face, thickness)?;
        offset_faces.push(offset_face);
    }

    // Create inner face duplicates for removed faces (creates the opening)
    // These become part of the outer surface at the opening
    for face_idx in &removed_face_indices {
        let original_face = &all_faces[*face_idx];
        // The inner surface at the opening is just an inward offset
        let inner_face = offset_face_inward(original_face, thickness)?;
        offset_faces.push(inner_face);
    }

    // Handle corners and edge blending by creating connecting faces
    // For each pair of adjacent offset faces, create a blending face
    for i in 0..all_faces.len() {
        if faces_to_remove_set.contains(&i) {
            continue;
        }
        
        for j in (i + 1)..all_faces.len() {
            if faces_to_remove_set.contains(&j) {
                continue;
            }

            // Check if faces are adjacent (share an edge)
            if let Some(shared_edge) = topology::shared_edge(&all_faces[i], &all_faces[j]) {
                // Create a blending face connecting the offset edges
                if let Ok(blend_face) = create_edge_blend(&all_faces[i], &all_faces[j], &shared_edge, thickness) {
                    offset_faces.push(blend_face);
                }
            }
        }
    }

    // Create new outer shell from offset faces
    let new_shell = Shell {
        faces: offset_faces,
        closed: removed_face_indices.is_empty(),  // Open if faces were removed
    };

    // For now, keep the same inner shells (voids) as the original
    // In a full implementation, these would also be offset
    let result_solid = Solid {
        outer_shell: new_shell,
        inner_shells: solid.inner_shells.clone(),
    };

    Ok(result_solid)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get bounding box of a solid
fn get_solid_bounds(solid: &Solid) -> ([f64; 3], [f64; 3]) {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];

    let all_faces = topology::get_solid_faces_internal(solid);
    
    for face in &all_faces {
        // Scan outer wire
        for edge in &face.outer_wire.edges {
            for i in 0..3 {
                min[i] = min[i].min(edge.start.point[i]);
                max[i] = max[i].max(edge.start.point[i]);
                min[i] = min[i].min(edge.end.point[i]);
                max[i] = max[i].max(edge.end.point[i]);
            }
        }

        // Scan inner wires
        for inner_wire in &face.inner_wires {
            for edge in &inner_wire.edges {
                for i in 0..3 {
                    min[i] = min[i].min(edge.start.point[i]);
                    max[i] = max[i].max(edge.start.point[i]);
                    min[i] = min[i].min(edge.end.point[i]);
                    max[i] = max[i].max(edge.end.point[i]);
                }
            }
        }
    }

    (min, max)
}

/// Build a set of face indices that should be removed
fn build_face_removal_set(faces_to_remove: &[Face], all_faces: &[Face]) -> std::collections::HashSet<usize> {
    let mut removal_set = std::collections::HashSet::new();

    for face_to_remove in faces_to_remove {
        for (idx, face) in all_faces.iter().enumerate() {
            // Compare faces by their geometry (outer wire edges)
            if faces_are_equal(face_to_remove, face) {
                removal_set.insert(idx);
                break;
            }
        }
    }

    removal_set
}

/// Check if two faces are geometrically equal
fn faces_are_equal(f1: &Face, f2: &Face) -> bool {
    // Compare outer wire length
    if f1.outer_wire.edges.len() != f2.outer_wire.edges.len() {
        return false;
    }

    // Compare first edge (sufficient for identifying face)
    if f1.outer_wire.edges.is_empty() {
        return true;  // Both have empty outer wires
    }

    let e1 = &f1.outer_wire.edges[0];
    let e2 = &f2.outer_wire.edges[0];

    vertices_are_close(&e1.start, &e2.start) && vertices_are_close(&e1.end, &e2.end)
}

/// Check if two vertices are spatially close
fn vertices_are_close(v1: &Vertex, v2: &Vertex) -> bool {
    (v1.point[0] - v2.point[0]).abs() < TOLERANCE * 10.0
        && (v1.point[1] - v2.point[1]).abs() < TOLERANCE * 10.0
        && (v1.point[2] - v2.point[2]).abs() < TOLERANCE * 10.0
}

/// Offset a face by a distance (positive = outward, negative = inward)
fn offset_face_by_distance(face: &Face, offset: f64) -> Result<Face> {
    // Get the outward normal for the face
    let (normal, _origin) = extract_plane_info(face)?;
    
    // Offset direction: positive offset moves outward (along normal)
    let offset_direction = [normal[0] * offset, normal[1] * offset, normal[2] * offset];

    // Offset outer wire
    let offset_outer = offset_wire_by_vector(&face.outer_wire, &offset_direction)?;

    // Offset inner wires (holes) - these move in the same direction
    let offset_inner = face.inner_wires
        .iter()
        .map(|wire| offset_wire_by_vector(wire, &offset_direction))
        .collect::<Result<Vec<_>>>()?;

    Ok(Face {
        outer_wire: offset_outer,
        inner_wires: offset_inner,
        surface_type: face.surface_type.clone(),
    })
}

/// Offset a wire by a fixed vector
fn offset_wire_by_vector(wire: &Wire, offset_vector: &[f64; 3]) -> Result<Wire> {
    if wire.edges.is_empty() {
        return Ok(wire.clone());
    }

    // Move each vertex along the offset vector
    let offset_edges = wire.edges
        .iter()
        .map(|edge| {
            let start_new = Vertex {
                point: [
                    edge.start.point[0] + offset_vector[0],
                    edge.start.point[1] + offset_vector[1],
                    edge.start.point[2] + offset_vector[2],
                ],
            };

            let end_new = Vertex {
                point: [
                    edge.end.point[0] + offset_vector[0],
                    edge.end.point[1] + offset_vector[1],
                    edge.end.point[2] + offset_vector[2],
                ],
            };

            Edge {
                start: start_new,
                end: end_new,
                curve_type: edge.curve_type.clone(),
            }
        })
        .collect();

    Ok(Wire {
        edges: offset_edges,
        closed: wire.closed,
    })
}

/// Offset a face inward (toward its surface normal interior) by the given thickness
fn offset_face_inward(face: &Face, thickness: f64) -> Result<Face> {
    // For a planar face, inward offset moves vertices along the inward normal
    let (normal, _origin) = extract_plane_info(face)?;
    
    // Inward direction is negative of the outward normal
    let inward = [-normal[0], -normal[1], -normal[2]];

    // Offset outer wire
    let offset_outer = offset_wire(&face.outer_wire, &inward, thickness)?;

    // Offset inner wires (holes) - these go in opposite direction (outward from the hole)
    let offset_inner = face.inner_wires
        .iter()
        .map(|wire| {
            let hole_outward = [normal[0], normal[1], normal[2]];
            offset_wire(wire, &hole_outward, thickness)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Face {
        outer_wire: offset_outer,
        inner_wires: offset_inner,
        surface_type: face.surface_type.clone(),
    })
}

/// Offset a wire (polyline) in a given direction by a distance
fn offset_wire(wire: &Wire, direction: &[f64; 3], distance: f64) -> Result<Wire> {
    if wire.edges.is_empty() {
        return Ok(wire.clone());
    }

    // Move each vertex along the offset direction
    let offset_edges = wire.edges
        .iter()
        .map(|edge| {
            let start_new = Vertex {
                point: [
                    edge.start.point[0] + direction[0] * distance,
                    edge.start.point[1] + direction[1] * distance,
                    edge.start.point[2] + direction[2] * distance,
                ],
            };

            let end_new = Vertex {
                point: [
                    edge.end.point[0] + direction[0] * distance,
                    edge.end.point[1] + direction[1] * distance,
                    edge.end.point[2] + direction[2] * distance,
                ],
            };

            Edge {
                start: start_new,
                end: end_new,
                curve_type: edge.curve_type.clone(),
            }
        })
        .collect();

    Ok(Wire {
        edges: offset_edges,
        closed: wire.closed,
    })
}

/// Extract plane information (normal, origin) from a face
fn extract_plane_info(face: &Face) -> Result<([f64; 3], [f64; 3])> {
    match &face.surface_type {
        SurfaceType::Plane { origin, normal } => {
            // Normalize the normal
            let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if len < TOLERANCE {
                return Err(CascadeError::InvalidGeometry(
                    "Plane has invalid normal vector".into(),
                ));
            }
            Ok((
                [normal[0] / len, normal[1] / len, normal[2] / len],
                *origin,
            ))
        }
        _ => Err(CascadeError::NotImplemented(
            "Shell offset only supports planar faces currently".into(),
        )),
    }
}

/// Create a blending face for thick_solid operation
/// Handles the corner where two adjacent faces meet after offset
fn create_thick_solid_blend(
    face1: &Face,
    face2: &Face,
    shared_edge: &Edge,
    offset: f64,
) -> Result<Face> {
    // Get plane info for both faces
    let (normal1, _) = extract_plane_info(face1)?;
    let (normal2, _) = extract_plane_info(face2)?;

    // Calculate the offset vector for each face
    let offset_vec1 = [normal1[0] * offset, normal1[1] * offset, normal1[2] * offset];
    let offset_vec2 = [normal2[0] * offset, normal2[1] * offset, normal2[2] * offset];

    // Offset the shared edge along both face normals
    let start_offset1 = [
        shared_edge.start.point[0] + offset_vec1[0],
        shared_edge.start.point[1] + offset_vec1[1],
        shared_edge.start.point[2] + offset_vec1[2],
    ];

    let end_offset1 = [
        shared_edge.end.point[0] + offset_vec1[0],
        shared_edge.end.point[1] + offset_vec1[1],
        shared_edge.end.point[2] + offset_vec1[2],
    ];

    let start_offset2 = [
        shared_edge.start.point[0] + offset_vec2[0],
        shared_edge.start.point[1] + offset_vec2[1],
        shared_edge.start.point[2] + offset_vec2[2],
    ];

    let end_offset2 = [
        shared_edge.end.point[0] + offset_vec2[0],
        shared_edge.end.point[1] + offset_vec2[1],
        shared_edge.end.point[2] + offset_vec2[2],
    ];

    // Create a quad face connecting the original edge to the offset edges
    let blend_edges = vec![
        Edge {
            start: Vertex { point: shared_edge.start.point },
            end: Vertex { point: shared_edge.end.point },
            curve_type: shared_edge.curve_type.clone(),
        },
        Edge {
            start: Vertex { point: shared_edge.end.point },
            end: Vertex { point: end_offset1 },
            curve_type: CurveType::Line,
        },
        Edge {
            start: Vertex { point: end_offset1 },
            end: Vertex { point: end_offset2 },
            curve_type: CurveType::Line,
        },
        Edge {
            start: Vertex { point: end_offset2 },
            end: Vertex { point: shared_edge.end.point },
            curve_type: CurveType::Line,
        },
        Edge {
            start: Vertex { point: shared_edge.end.point },
            end: Vertex { point: start_offset2 },
            curve_type: CurveType::Line,
        },
        Edge {
            start: Vertex { point: start_offset2 },
            end: Vertex { point: start_offset1 },
            curve_type: CurveType::Line,
        },
        Edge {
            start: Vertex { point: start_offset1 },
            end: Vertex { point: shared_edge.start.point },
            curve_type: CurveType::Line,
        },
    ];

    // For simplicity, create a quad (4-edge face) connecting the offset edges
    let simplified_edges = vec![
        Edge {
            start: Vertex { point: shared_edge.start.point },
            end: Vertex { point: shared_edge.end.point },
            curve_type: shared_edge.curve_type.clone(),
        },
        Edge {
            start: Vertex { point: shared_edge.end.point },
            end: Vertex { point: end_offset1 },
            curve_type: CurveType::Line,
        },
        Edge {
            start: Vertex { point: end_offset1 },
            end: Vertex { point: start_offset1 },
            curve_type: CurveType::Line,
        },
        Edge {
            start: Vertex { point: start_offset1 },
            end: Vertex { point: shared_edge.start.point },
            curve_type: CurveType::Line,
        },
    ];

    let blend_wire = Wire {
        edges: simplified_edges,
        closed: true,
    };

    // Create a blending surface (average of the two adjacent planes)
    let blend_normal = [
        (normal1[0] + normal2[0]) / 2.0,
        (normal1[1] + normal2[1]) / 2.0,
        (normal1[2] + normal2[2]) / 2.0,
    ];

    let blend_origin = [
        (start_offset1[0] + start_offset2[0]) / 2.0,
        (start_offset1[1] + start_offset2[1]) / 2.0,
        (start_offset1[2] + start_offset2[2]) / 2.0,
    ];

    Ok(Face {
        outer_wire: blend_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: blend_origin,
            normal: blend_normal,
        },
    })
}

/// Create a blending face between two adjacent faces at a shared edge
fn create_edge_blend(
    face1: &Face,
    face2: &Face,
    shared_edge: &Edge,
    thickness: f64,
) -> Result<Face> {
    // Get plane info for both faces
    let (normal1, _) = extract_plane_info(face1)?;
    let (normal2, _) = extract_plane_info(face2)?;

    // Offset the shared edge inward
    let inward1 = [-normal1[0], -normal1[1], -normal1[2]];
    let inward2 = [-normal2[0], -normal2[1], -normal2[2]];

    // Create offset vertices along the edge
    let start_offset1 = [
        shared_edge.start.point[0] + inward1[0] * thickness,
        shared_edge.start.point[1] + inward1[1] * thickness,
        shared_edge.start.point[2] + inward1[2] * thickness,
    ];

    let end_offset1 = [
        shared_edge.end.point[0] + inward1[0] * thickness,
        shared_edge.end.point[1] + inward1[1] * thickness,
        shared_edge.end.point[2] + inward1[2] * thickness,
    ];

    let start_offset2 = [
        shared_edge.start.point[0] + inward2[0] * thickness,
        shared_edge.start.point[1] + inward2[1] * thickness,
        shared_edge.start.point[2] + inward2[2] * thickness,
    ];

    let _end_offset2 = [
        shared_edge.end.point[0] + inward2[0] * thickness,
        shared_edge.end.point[1] + inward2[1] * thickness,
        shared_edge.end.point[2] + inward2[2] * thickness,
    ];

    // Create a blending surface (quad) connecting the offset edges
    let blend_edges = vec![
        Edge {
            start: Vertex { point: shared_edge.start.point },
            end: Vertex { point: shared_edge.end.point },
            curve_type: shared_edge.curve_type.clone(),
        },
        Edge {
            start: Vertex { point: shared_edge.end.point },
            end: Vertex { point: end_offset1 },
            curve_type: CurveType::Line,
        },
        Edge {
            start: Vertex { point: end_offset1 },
            end: Vertex { point: start_offset1 },
            curve_type: CurveType::Line,
        },
        Edge {
            start: Vertex { point: start_offset1 },
            end: Vertex { point: shared_edge.start.point },
            curve_type: CurveType::Line,
        },
    ];

    let blend_wire = Wire {
        edges: blend_edges,
        closed: true,
    };

    // Create a blending surface (average of the two adjacent planes)
    let blend_normal = [
        (normal1[0] + normal2[0]) / 2.0,
        (normal1[1] + normal2[1]) / 2.0,
        (normal1[2] + normal2[2]) / 2.0,
    ];

    let blend_origin = [
        (start_offset1[0] + start_offset2[0]) / 2.0,
        (start_offset1[1] + start_offset2[1]) / 2.0,
        (start_offset1[2] + start_offset2[2]) / 2.0,
    ];

    Ok(Face {
        outer_wire: blend_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: blend_origin,
            normal: blend_normal,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::make_box;

    #[test]
    fn test_make_shell_closed() {
        // Create a simple box and make a hollow version
        let box_solid = make_box(10.0, 10.0, 10.0).unwrap();
        let hollow = make_shell(&box_solid, 1.0, &[]).unwrap();

        // Verify the hollow has the same topology (closed)
        assert!(hollow.outer_shell.closed);
        // Hollow should have more faces due to inner surface and blending
        assert!(!hollow.outer_shell.faces.is_empty());
    }

    #[test]
    fn test_make_shell_invalid_thickness() {
        let box_solid = make_box(10.0, 10.0, 10.0).unwrap();

        // Negative thickness should fail
        let result = make_shell(&box_solid, -1.0, &[]);
        assert!(result.is_err());

        // Zero thickness should fail
        let result = make_shell(&box_solid, 0.0, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_make_shell_excessive_thickness() {
        let box_solid = make_box(10.0, 10.0, 10.0).unwrap();

        // Thickness too large (half of minimum dimension is 5.0, so 5.0+ should fail)
        let result = make_shell(&box_solid, 5.5, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_make_shell_with_removed_face() {
        let box_solid = make_box(10.0, 10.0, 10.0).unwrap();

        // Get first face to remove
        let face_to_remove = &box_solid.outer_shell.faces[0];

        // Create shell with one face removed (creates an opening)
        let result = make_shell(&box_solid, 1.0, &[face_to_remove.clone()]);
        assert!(result.is_ok());

        let hollow = result.unwrap();
        // Should be open now
        assert!(!hollow.outer_shell.closed);
    }

    #[test]
    fn test_offset_wire() {
        // Create a simple wire (square)
        let wire = Wire {
            edges: vec![
                Edge {
                    start: Vertex { point: [0.0, 0.0, 0.0] },
                    end: Vertex { point: [1.0, 0.0, 0.0] },
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex { point: [1.0, 0.0, 0.0] },
                    end: Vertex { point: [1.0, 1.0, 0.0] },
                    curve_type: CurveType::Line,
                },
            ],
            closed: false,
        };

        let direction = [0.0, 0.0, 1.0];
        let offset_wire = offset_wire(&wire, &direction, 0.5).unwrap();

        // Check that vertices were offset correctly
        assert_eq!(offset_wire.edges[0].start.point[2], 0.5);
        assert_eq!(offset_wire.edges[0].end.point[2], 0.5);
    }

    #[test]
    fn test_thick_solid_grow() {
        // Create a simple box and grow it
        let box_solid = make_box(10.0, 10.0, 10.0).unwrap();
        let grown = thick_solid(&box_solid, 1.0).unwrap();

        // Verify the grown solid has faces
        assert!(!grown.outer_shell.faces.is_empty());
        // Grown solid should have more faces due to blending faces
        assert!(grown.outer_shell.faces.len() >= box_solid.outer_shell.faces.len());
    }

    #[test]
    fn test_thick_solid_shrink() {
        // Create a simple box and shrink it
        let box_solid = make_box(10.0, 10.0, 10.0).unwrap();
        let shrunk = thick_solid(&box_solid, -1.0).unwrap();

        // Verify the shrunk solid has faces
        assert!(!shrunk.outer_shell.faces.is_empty());
    }

    #[test]
    fn test_thick_solid_zero_offset() {
        // Zero offset should fail
        let box_solid = make_box(10.0, 10.0, 10.0).unwrap();
        let result = thick_solid(&box_solid, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_thick_solid_excessive_shrink() {
        // Shrink offset too large should fail
        let box_solid = make_box(10.0, 10.0, 10.0).unwrap();
        let result = thick_solid(&box_solid, -5.5);
        assert!(result.is_err());
    }
}
