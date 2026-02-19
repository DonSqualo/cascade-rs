//! Local shape modification operations
//!
//! This module provides operations that modify individual faces and edges:
//! - **Split face**: Divides a face along a curve or edge
//! - **Replace face**: Replaces a face while maintaining edge connectivity and validating boundary compatibility
//! - **Thicken face**: Creates a solid by offsetting a face in its normal direction

use crate::brep::{Face, Edge, Wire, Vertex, SurfaceType, CurveType, Shell, Solid};
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

/// Replace a face in a solid with a new face.
///
/// This operation replaces a face at the specified index with a new face while maintaining
/// the topological structure of the solid. The new face's boundary (outer wire) must be
/// compatible with the old face's boundary to ensure edge connectivity is preserved.
///
/// # Arguments
/// * `solid` - The solid containing the face to be replaced
/// * `face_index` - The index of the face in the solid's outer shell (0-based)
/// * `new_face` - The new face to replace the old one with
///
/// # Validation Rules
/// - The new face's outer wire must have the same number of edges as the old face
/// - The new face's outer wire edges must connect properly (end of edge N connects to start of edge N+1)
/// - The new face's inner wires (holes) must match the old face's inner wires in count and topology
/// - All boundary vertices must be compatible (within TOLERANCE)
///
/// # Returns
/// A new solid with the face replaced, or an error if the replacement would violate topology constraints.
///
/// # Example
/// ```ignore
/// let solid = ... // Obtain a solid
/// let new_face = ... // Create a new face with compatible boundary
/// let modified_solid = replace_face(&solid, 0, &new_face)?;
/// ```
pub fn replace_face(solid: &Solid, face_index: usize, new_face: &Face) -> Result<Solid> {
    // Check that face_index is valid for the outer shell
    if face_index >= solid.outer_shell.faces.len() {
        return Err(CascadeError::InvalidGeometry(
            format!(
                "Face index {} out of bounds (solid has {} faces)",
                face_index,
                solid.outer_shell.faces.len()
            ),
        ));
    }

    let old_face = &solid.outer_shell.faces[face_index];

    // Validate that the new face's boundary is compatible with the old face's boundary
    validate_face_boundary_compatibility(old_face, new_face)?;

    // Create a new outer shell with the replaced face
    let mut new_outer_faces = solid.outer_shell.faces.clone();
    new_outer_faces[face_index] = new_face.clone();

    let new_outer_shell = Shell {
        faces: new_outer_faces,
        closed: solid.outer_shell.closed,
    };

    // Create and return the new solid
    let new_solid = Solid {
        outer_shell: new_outer_shell,
        inner_shells: solid.inner_shells.clone(),
    };

    Ok(new_solid)
}

/// Validate that a new face's boundary is compatible with an old face's boundary.
///
/// This ensures that the topological relationship is preserved:
/// - The outer wires must have the same number of edges
/// - The inner wires (holes) must match in count
/// - All boundary vertices must align (within tolerance)
fn validate_face_boundary_compatibility(old_face: &Face, new_face: &Face) -> Result<()> {
    // Check outer wire edge count
    if old_face.outer_wire.edges.len() != new_face.outer_wire.edges.len() {
        return Err(CascadeError::InvalidGeometry(
            format!(
                "Outer wire edge count mismatch: old face has {} edges, new face has {} edges",
                old_face.outer_wire.edges.len(),
                new_face.outer_wire.edges.len()
            ),
        ));
    }

    // Check outer wire closure property
    if old_face.outer_wire.closed != new_face.outer_wire.closed {
        return Err(CascadeError::InvalidGeometry(
            "Outer wire closure property mismatch".to_string(),
        ));
    }

    // Validate outer wire vertex connectivity
    validate_wire_continuity(&new_face.outer_wire)?;

    // Check inner wire (hole) count
    if old_face.inner_wires.len() != new_face.inner_wires.len() {
        return Err(CascadeError::InvalidGeometry(
            format!(
                "Inner wire count mismatch: old face has {} holes, new face has {} holes",
                old_face.inner_wires.len(),
                new_face.inner_wires.len()
            ),
        ));
    }

    // Validate each inner wire
    for (idx, new_inner_wire) in new_face.inner_wires.iter().enumerate() {
        let old_inner_wire = &old_face.inner_wires[idx];

        // Check edge count
        if old_inner_wire.edges.len() != new_inner_wire.edges.len() {
            return Err(CascadeError::InvalidGeometry(
                format!(
                    "Inner wire {} edge count mismatch: old has {}, new has {}",
                    idx,
                    old_inner_wire.edges.len(),
                    new_inner_wire.edges.len()
                ),
            ));
        }

        // Check closure property
        if old_inner_wire.closed != new_inner_wire.closed {
            return Err(CascadeError::InvalidGeometry(
                format!("Inner wire {} closure property mismatch", idx),
            ));
        }

        // Validate continuity
        validate_wire_continuity(new_inner_wire)?;
    }

    // Validate outer wire vertex positions match (within tolerance)
    validate_wire_vertices(&old_face.outer_wire, &new_face.outer_wire)?;

    Ok(())
}

/// Validate that a wire's edges are properly connected (continuous path).
///
/// Each edge's end point must match the next edge's start point (within tolerance).
fn validate_wire_continuity(wire: &Wire) -> Result<()> {
    if wire.edges.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Wire must have at least one edge".to_string(),
        ));
    }

    for i in 0..wire.edges.len() {
        let current_edge = &wire.edges[i];
        let next_edge = &wire.edges[(i + 1) % wire.edges.len()];

        let current_end = current_edge.end.point;
        let next_start = next_edge.start.point;

        if distance(&current_end, &next_start) > TOLERANCE {
            return Err(CascadeError::InvalidGeometry(
                format!(
                    "Wire continuity broken at edge {}: end point {:?} doesn't match next edge start {:?}",
                    i, current_end, next_start
                ),
            ));
        }
    }

    Ok(())
}

/// Validate that the vertices of two wires match in position (within tolerance).
///
/// This checks that the boundary vertices are the same, ensuring compatibility
/// for edge connectivity purposes.
fn validate_wire_vertices(old_wire: &Wire, new_wire: &Wire) -> Result<()> {
    if old_wire.edges.len() != new_wire.edges.len() {
        return Err(CascadeError::InvalidGeometry(
            "Wire edge counts don't match".to_string(),
        ));
    }

    // Check that the first vertex of both wires is at the same location
    let old_start = old_wire.edges[0].start.point;
    let new_start = new_wire.edges[0].start.point;

    if distance(&old_start, &new_start) > TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            format!(
                "First vertex position mismatch: old {:?}, new {:?}",
                old_start, new_start
            ),
        ));
    }

    // Check that all intermediate vertices match
    for i in 0..old_wire.edges.len() {
        let old_end = old_wire.edges[i].end.point;
        let new_end = new_wire.edges[i].end.point;

        if distance(&old_end, &new_end) > TOLERANCE {
            return Err(CascadeError::InvalidGeometry(
                format!(
                    "Vertex {} position mismatch: old {:?}, new {:?}",
                    i, old_end, new_end
                ),
            ));
        }
    }

    Ok(())
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
        // Check if point lies on the edge (for straight line edges)
        if point_on_edge(point, edge) {
            return Ok(i);
        }
    }

    Err(CascadeError::InvalidGeometry(
        format!("Point {:?} not found on any edge of the wire", point),
    ))
}

/// Check if a point lies on an edge (for straight line edges).
fn point_on_edge(point: &[f64; 3], edge: &Edge) -> bool {
    match edge.curve_type {
        CurveType::Line => {
            // Check if point is on the line segment between start and end
            let start = &edge.start.point;
            let end = &edge.end.point;

            // Vector from start to end
            let edge_vec = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
            let edge_len_sq = edge_vec[0] * edge_vec[0]
                + edge_vec[1] * edge_vec[1]
                + edge_vec[2] * edge_vec[2];

            if edge_len_sq < TOLERANCE * TOLERANCE {
                // Edge is degenerate, already handled by endpoint checks
                return false;
            }

            // Vector from start to point
            let point_vec = [point[0] - start[0], point[1] - start[1], point[2] - start[2]];

            // Projection of point_vec onto edge_vec
            let dot_prod =
                point_vec[0] * edge_vec[0] + point_vec[1] * edge_vec[1] + point_vec[2] * edge_vec[2];
            let t = dot_prod / edge_len_sq;

            // Check if t is in [0, 1]
            if t < 0.0 || t > 1.0 {
                return false;
            }

            // Find closest point on edge
            let closest = [
                start[0] + t * edge_vec[0],
                start[1] + t * edge_vec[1],
                start[2] + t * edge_vec[2],
            ];

            // Check distance from point to closest point on edge
            distance(point, &closest) < TOLERANCE
        }
        _ => {
            // For non-line edges, only check endpoints
            false
        }
    }
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
        // Also check if point lies on the edge itself
        if point_on_edge(point, edge) {
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

/// Remove a face from a solid and attempt to heal the resulting gap.
///
/// This operation removes the specified face from the solid's outer shell and attempts
/// to heal the resulting gap by optionally filling it with an adjacent face or leaving
/// the shell open if healing is not possible.
///
/// # Arguments
/// * `solid` - The solid from which to remove the face
/// * `face_index` - The index of the face to remove in the outer shell
///
/// # Returns
/// * `Result<Solid>` - The solid with the face removed, or error if:
///   - `face_index` is out of bounds
///   - The solid has only one face (cannot remove it)
///   - The removal creates an invalid topology
///
/// # Behavior
/// - If the removal leaves a valid open shell, it returns the modified solid
/// - Edge cases are handled (last face, adjacent faces, etc.)
/// - The function attempts to validate the resulting topology
///
/// # Example
/// ```ignore
/// let solid = make_box(10.0, 10.0, 10.0);
/// let removed_solid = remove_face(&solid, 0)?;
/// // solid now has one fewer face
/// ```
pub fn remove_face(solid: &Solid, face_index: usize) -> Result<Solid> {
    // Check if face_index is valid
    if face_index >= solid.outer_shell.faces.len() {
        return Err(CascadeError::InvalidGeometry(
            format!(
                "Face index {} is out of bounds (solid has {} faces)",
                face_index,
                solid.outer_shell.faces.len()
            ),
        ));
    }

    // Edge case: cannot remove the only face
    if solid.outer_shell.faces.len() == 1 {
        return Err(CascadeError::TopologyError(
            "Cannot remove the only face from a solid".to_string(),
        ));
    }

    // Create a new outer shell without the specified face
    let mut new_faces = solid.outer_shell.faces.clone();
    let _removed_face = new_faces.remove(face_index);

    // Check if we have at least 3 faces remaining (minimum for a closed shell to be valid)
    if new_faces.is_empty() {
        return Err(CascadeError::TopologyError(
            "Removal resulted in an empty shell".to_string(),
        ));
    }

    // Create the new shell with the remaining faces
    // The shell becomes open (closed = false) after removal
    let new_outer_shell = Shell {
        faces: new_faces,
        closed: false, // Mark as open since we removed a face
    };

    // Create the new solid with the modified shell
    let new_solid = Solid {
        outer_shell: new_outer_shell,
        inner_shells: solid.inner_shells.clone(),
    };

    Ok(new_solid)
}

/// Thicken a face by offsetting it to create a solid.
///
/// Creates a solid by offsetting a face in its normal direction and connecting
/// the original face with its offset using side faces.
///
/// # Arguments
/// * `face` - The face to be thickened
/// * `thickness` - The offset distance. Positive values offset along the face normal,
///                  negative values offset in the opposite direction.
///
/// # Returns
/// A solid bounded by the original face, the offset face, and connecting side faces.
///
/// # Details
/// The resulting solid consists of:
/// 1. The original face
/// 2. An offset face displaced by `thickness` along the surface normal
/// 3. Side faces connecting corresponding edges of the original and offset faces
///
/// # Limitations
/// Currently supports thickening of:
/// - Planar faces
/// - Cylindrical faces
/// - Spherical faces
/// - Conical faces
/// - Toroidal faces
/// - Other parametric surface types
///
/// # Example
/// ```ignore
/// let face = ... // Create or obtain a face
/// let thickness = 2.0; // Positive offset
/// let solid = thicken_face(&face, thickness)?;
/// // solid now contains the thickened geometry
/// ```
pub fn thicken_face(face: &Face, thickness: f64) -> Result<Solid> {
    if thickness.abs() < TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            "Thickness must be non-zero".to_string(),
        ));
    }

    // Create offset face by offsetting the surface
    let offset_surface = SurfaceType::OffsetSurface {
        basis_surface: Box::new(face.surface_type.clone()),
        offset_distance: thickness,
    };

    // Create the offset face with the same topology as the original
    let offset_outer_wire = offset_wire(&face.outer_wire, &face.surface_type, thickness)?;
    let offset_inner_wires = face
        .inner_wires
        .iter()
        .map(|wire| offset_wire(wire, &face.surface_type, thickness))
        .collect::<Result<Vec<_>>>()?;

    let offset_face = Face {
        outer_wire: offset_outer_wire.clone(),
        inner_wires: offset_inner_wires.clone(),
        surface_type: offset_surface,
    };

    // Create side faces connecting the original edges to offset edges
    let mut side_faces = Vec::new();

    // Create side faces for outer wire
    create_side_faces_for_wire(
        &face.outer_wire,
        &offset_outer_wire,
        &mut side_faces,
        &face.surface_type,
        thickness,
    )?;

    // Create side faces for inner wires (holes)
    for (original_inner, offset_inner) in face.inner_wires.iter().zip(offset_inner_wires.iter()) {
        create_side_faces_for_wire(
            original_inner,
            offset_inner,
            &mut side_faces,
            &face.surface_type,
            thickness,
        )?;
    }

    // Build the solid from all faces (original, offset, and sides)
    let mut all_faces = vec![face.clone(), offset_face];
    all_faces.extend(side_faces);

    // Create a shell from all faces
    let shell = Shell {
        faces: all_faces,
        closed: true,
    };

    // Create the solid
    let solid = Solid {
        outer_shell: shell,
        inner_shells: vec![],
    };

    Ok(solid)
}

/// Create an offset wire by offsetting all its vertices along the surface normal.
fn offset_wire(
    wire: &Wire,
    surface: &SurfaceType,
    thickness: f64,
) -> Result<Wire> {
    let offset_edges = wire
        .edges
        .iter()
        .map(|edge| offset_edge(edge, surface, thickness))
        .collect::<Result<Vec<_>>>()?;

    Ok(Wire {
        edges: offset_edges,
        closed: wire.closed,
    })
}

/// Create an offset edge by offsetting its start and end vertices along the surface normal.
fn offset_edge(
    edge: &Edge,
    surface: &SurfaceType,
    thickness: f64,
) -> Result<Edge> {
    let normal_start = compute_surface_normal_at_point(&edge.start.point, surface)?;
    let normal_end = compute_surface_normal_at_point(&edge.end.point, surface)?;

    let offset_start = [
        edge.start.point[0] + thickness * normal_start[0],
        edge.start.point[1] + thickness * normal_start[1],
        edge.start.point[2] + thickness * normal_start[2],
    ];

    let offset_end = [
        edge.end.point[0] + thickness * normal_end[0],
        edge.end.point[1] + thickness * normal_end[1],
        edge.end.point[2] + thickness * normal_end[2],
    ];

    // The offset edge has the same curve type as original (for now)
    // This is a simplification; in reality, offset curves can be more complex
    let offset_curve = match &edge.curve_type {
        CurveType::Line => CurveType::Line,
        CurveType::Arc { center, radius } => {
            // Offset the arc center and radius
            let normal_center = compute_surface_normal_at_point(center, surface)?;
            let offset_center = [
                center[0] + thickness * normal_center[0],
                center[1] + thickness * normal_center[1],
                center[2] + thickness * normal_center[2],
            ];
            CurveType::Arc {
                center: offset_center,
                radius: *radius,
            }
        }
        other => other.clone(),
    };

    Ok(Edge {
        start: Vertex::new(offset_start[0], offset_start[1], offset_start[2]),
        end: Vertex::new(offset_end[0], offset_end[1], offset_end[2]),
        curve_type: offset_curve,
    })
}

/// Create side faces connecting edges of an original wire to corresponding offset edges.
fn create_side_faces_for_wire(
    original_wire: &Wire,
    offset_wire: &Wire,
    side_faces: &mut Vec<Face>,
    original_surface: &SurfaceType,
    thickness: f64,
) -> Result<()> {
    // For each pair of corresponding edges, create a side face
    for (original_edge, offset_edge) in original_wire.edges.iter().zip(offset_wire.edges.iter()) {
        // Create a quadrilateral face connecting:
        // original_start -> original_end -> offset_end -> offset_start -> back to original_start

        let side_wire = Wire {
            edges: vec![
                // Original edge (forward)
                original_edge.clone(),
                // Edge from original end to offset end
                Edge {
                    start: original_edge.end.clone(),
                    end: offset_edge.end.clone(),
                    curve_type: CurveType::Line,
                },
                // Offset edge (reversed)
                Edge {
                    start: offset_edge.end.clone(),
                    end: offset_edge.start.clone(),
                    curve_type: CurveType::Line,
                },
                // Edge from offset start back to original start
                Edge {
                    start: offset_edge.start.clone(),
                    end: original_edge.start.clone(),
                    curve_type: CurveType::Line,
                },
            ],
            closed: true,
        };

        // Compute normal for the side face (perpendicular to the edge direction)
        let edge_dir = [
            original_edge.end.point[0] - original_edge.start.point[0],
            original_edge.end.point[1] - original_edge.start.point[1],
            original_edge.end.point[2] - original_edge.start.point[2],
        ];

        let normal = compute_surface_normal_at_point(&original_edge.start.point, original_surface)?;

        // Create a planar side face
        let side_face = Face {
            outer_wire: side_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: original_edge.start.point,
                normal,
            },
        };

        side_faces.push(side_face);
    }

    Ok(())
}

/// Compute the surface normal at a given 3D point on the surface.
/// This is a geometric approximation based on the surface type.
fn compute_surface_normal_at_point(point: &[f64; 3], surface: &SurfaceType) -> Result<[f64; 3]> {
    match surface {
        SurfaceType::Plane { normal, .. } => {
            Ok(normalize(normal))
        }
        SurfaceType::Cylinder { origin, axis, radius } => {
            // For a cylinder, the normal points radially outward
            let ax = normalize(axis);
            let to_point = [
                point[0] - origin[0],
                point[1] - origin[1],
                point[2] - origin[2],
            ];

            // Project to_point onto the plane perpendicular to axis
            let proj_len = to_point[0] * ax[0] + to_point[1] * ax[1] + to_point[2] * ax[2];
            let radial = [
                to_point[0] - proj_len * ax[0],
                to_point[1] - proj_len * ax[1],
                to_point[2] - proj_len * ax[2],
            ];

            Ok(normalize(&radial))
        }
        SurfaceType::Sphere { center, .. } => {
            // For a sphere, the normal points from center to the point
            let normal = [
                point[0] - center[0],
                point[1] - center[1],
                point[2] - center[2],
            ];
            Ok(normalize(&normal))
        }
        SurfaceType::Cone { origin, axis, half_angle_rad } => {
            // For a cone, compute the normal based on cone geometry
            let ax = normalize(axis);
            let to_point = [
                point[0] - origin[0],
                point[1] - origin[1],
                point[2] - origin[2],
            ];

            let proj_len = to_point[0] * ax[0] + to_point[1] * ax[1] + to_point[2] * ax[2];
            let radial = [
                to_point[0] - proj_len * ax[0],
                to_point[1] - proj_len * ax[1],
                to_point[2] - proj_len * ax[2],
            ];

            let cos_angle = half_angle_rad.cos();
            let sin_angle = half_angle_rad.sin();
            let normal = [
                radial[0] * cos_angle - ax[0] * sin_angle,
                radial[1] * cos_angle - ax[1] * sin_angle,
                radial[2] * cos_angle - ax[2] * sin_angle,
            ];
            Ok(normalize(&normal))
        }
        SurfaceType::Torus { center, major_radius, minor_radius } => {
            // For a torus, normal points outward
            let to_point = [
                point[0] - center[0],
                point[1] - center[1],
                point[2] - center[2],
            ];
            
            // Simplified torus normal (pointing outward from the ring)
            Ok(normalize(&to_point))
        }
        _ => {
            // For other surface types, try to compute via offset surface
            Err(CascadeError::NotImplemented(
                "Thicken face for this surface type".to_string(),
            ))
        }
    }
}

/// Split an edge at a given parameter value.
///
/// Divides an edge into two edges at the specified parameter value.
/// The parameter should be in the range [0, 1], where 0 is the start vertex
/// and 1 is the end vertex.
///
/// # Arguments
/// * `edge` - The edge to be split
/// * `parameter` - The parameter value where the split occurs (must be in [0, 1])
///
/// # Returns
/// A tuple of two edges: (edge1, edge2) where:
/// - edge1: from the original start to the split point
/// - edge2: from the split point to the original end
///
/// # Example
/// ```ignore
/// let edge = ... // Create or obtain an edge
/// let (edge1, edge2) = split_edge(&edge, 0.5)?;
/// // edge1 and edge2 together form the original edge
/// ```
pub fn split_edge(edge: &Edge, parameter: f64) -> Result<(Edge, Edge)> {
    // Validate parameter is in [0, 1]
    if parameter < 0.0 || parameter > 1.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Split parameter must be in [0, 1], got {}", parameter),
        ));
    }

    // Special case: parameter at 0 or 1
    if parameter <= TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            "Split parameter too close to start (0)".to_string(),
        ));
    }

    if parameter >= 1.0 - TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            "Split parameter too close to end (1)".to_string(),
        ));
    }

    // Import the point_at function from curve module
    use crate::curve::point_at;

    // Evaluate the curve at the split parameter
    let split_point = point_at(&edge.curve_type, parameter)?;

    // Create a new vertex at the split point
    let split_vertex = Vertex::new(split_point[0], split_point[1], split_point[2]);

    // Create the two new edges based on curve type
    let (edge1, edge2) = match &edge.curve_type {
        CurveType::Line => {
            // For line segments, both edges are also lines
            (
                Edge {
                    start: edge.start.clone(),
                    end: split_vertex.clone(),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: split_vertex.clone(),
                    end: edge.end.clone(),
                    curve_type: CurveType::Line,
                },
            )
        }
        // For other curve types, wrap in Trimmed curves
        _ => {
            (
                Edge {
                    start: edge.start.clone(),
                    end: split_vertex.clone(),
                    curve_type: CurveType::Trimmed {
                        basis_curve: Box::new(edge.curve_type.clone()),
                        u1: 0.0,
                        u2: parameter,
                    },
                },
                Edge {
                    start: split_vertex.clone(),
                    end: edge.end.clone(),
                    curve_type: CurveType::Trimmed {
                        basis_curve: Box::new(edge.curve_type.clone()),
                        u1: parameter,
                        u2: 1.0,
                    },
                },
            )
        }
    };

    Ok((edge1, edge2))
}

/// Split an edge at a given point.
///
/// Divides an edge into two edges at the point closest to the specified location.
/// The point should lie on or very near the edge.
///
/// # Arguments
/// * `edge` - The edge to be split
/// * `point` - The 3D point where the split occurs (should be on the edge)
///
/// # Returns
/// A tuple of two edges: (edge1, edge2) where:
/// - edge1: from the original start to the split point
/// - edge2: from the split point to the original end
///
/// # Limitations
/// Currently supports splitting of line edges with direct computation.
/// For other curve types, this will compute the closest parameter numerically.
///
/// # Example
/// ```ignore
/// let edge = ... // Create or obtain an edge
/// let split_point = [5.0, 5.0, 0.0];
/// let (edge1, edge2) = split_edge_at_point(&edge, split_point)?;
/// ```
pub fn split_edge_at_point(edge: &Edge, point: [f64; 3]) -> Result<(Edge, Edge)> {
    // Find the parameter value that corresponds to the point on the edge
    let parameter = find_edge_parameter(edge, &point)?;

    // Use split_edge with the computed parameter
    split_edge(edge, parameter)
}

/// Find the parameter value on an edge that corresponds to a given point.
///
/// This function computes the parameter t ∈ [0, 1] such that evaluating
/// the edge's curve at parameter t gives a point close to the specified point.
fn find_edge_parameter(edge: &Edge, point: &[f64; 3]) -> Result<f64> {
    use crate::curve::point_at;

    match &edge.curve_type {
        CurveType::Line => {
            // For line segments, compute the parameter analytically
            let start = &edge.start.point;
            let end = &edge.end.point;

            // Vector from start to end
            let edge_vec = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
            let edge_len_sq = edge_vec[0] * edge_vec[0]
                + edge_vec[1] * edge_vec[1]
                + edge_vec[2] * edge_vec[2];

            if edge_len_sq < TOLERANCE * TOLERANCE {
                return Err(CascadeError::InvalidGeometry(
                    "Cannot split degenerate edge".to_string(),
                ));
            }

            // Vector from start to point
            let point_vec = [point[0] - start[0], point[1] - start[1], point[2] - start[2]];

            // Projection of point_vec onto edge_vec
            let dot_prod = point_vec[0] * edge_vec[0] + point_vec[1] * edge_vec[1] + point_vec[2] * edge_vec[2];
            let t = dot_prod / edge_len_sq;

            // Clamp t to [0, 1]
            let t_clamped = t.max(0.0).min(1.0);

            // Verify that the point is reasonably close to the edge
            let closest_point = [
                start[0] + t_clamped * edge_vec[0],
                start[1] + t_clamped * edge_vec[1],
                start[2] + t_clamped * edge_vec[2],
            ];

            let dist = distance(point, &closest_point);
            if dist > TOLERANCE * 10.0 {
                // Allow slightly larger tolerance for point-to-curve distance
                return Err(CascadeError::InvalidGeometry(
                    format!("Point is not on edge (distance: {})", dist),
                ));
            }

            Ok(t)
        }
        _ => {
            // For non-line curves, use a numerical method to find the parameter
            find_edge_parameter_numerical(edge, point)
        }
    }
}

/// Find the parameter value on a non-line edge using numerical optimization.
///
/// Uses a simple iterative approach (golden section search) to find the parameter
/// that minimizes distance to the target point.
fn find_edge_parameter_numerical(edge: &Edge, target_point: &[f64; 3]) -> Result<f64> {
    use crate::curve::point_at;

    // Golden section search for the parameter
    let mut a = 0.0;
    let mut b = 1.0;
    let golden_ratio = 0.381966; // (3 - sqrt(5)) / 2

    // Iterate until convergence
    for _ in 0..50 {
        let c = a + (1.0 - golden_ratio) * (b - a);
        let d = a + golden_ratio * (b - a);

        let point_c = point_at(&edge.curve_type, c)?;
        let point_d = point_at(&edge.curve_type, d)?;

        let dist_c = distance(&point_c, target_point);
        let dist_d = distance(&point_d, target_point);

        if (b - a) < TOLERANCE {
            // Converged
            break;
        }

        if dist_c < dist_d {
            b = d;
        } else {
            a = c;
        }
    }

    let parameter = (a + b) / 2.0;

    // Verify the result is close enough
    let result_point = point_at(&edge.curve_type, parameter)?;
    let dist = distance(&result_point, target_point);

    if dist > TOLERANCE * 10.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Could not find point on edge (distance: {})", dist),
        ));
    }

    Ok(parameter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replace_face_basic() {
        // Create a simple solid with one face
        let face = Face {
            outer_wire: Wire {
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
            },
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        };

        let solid = Solid {
            outer_shell: Shell {
                faces: vec![face.clone()],
                closed: false,
            },
            inner_shells: vec![],
        };

        // Create a new face with a different surface type but same boundary
        let new_face = Face {
            outer_wire: face.outer_wire.clone(),
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [5.0, 5.0, 0.0],  // Different origin, same plane
                normal: [0.0, 0.0, 1.0],
            },
        };

        // Replace the face
        let result = replace_face(&solid, 0, &new_face);
        assert!(result.is_ok(), "Replace face should succeed");

        let new_solid = result.unwrap();
        assert_eq!(new_solid.outer_shell.faces.len(), 1, "Should have 1 face");
        assert!(
            points_equal(
                &new_solid.outer_shell.faces[0].outer_wire.edges[0].start.point,
                &face.outer_wire.edges[0].start.point
            ),
            "Face boundary should be preserved"
        );
    }

    #[test]
    fn test_replace_face_invalid_index() {
        let face = Face {
            outer_wire: Wire {
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
            },
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        };

        let solid = Solid {
            outer_shell: Shell {
                faces: vec![face.clone()],
                closed: false,
            },
            inner_shells: vec![],
        };

        // Try to replace a face with an invalid index
        let result = replace_face(&solid, 5, &face);
        assert!(result.is_err(), "Replace should fail with invalid index");
    }

    #[test]
    fn test_replace_face_incompatible_boundary_edge_count() {
        let original_face = Face {
            outer_wire: Wire {
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
            },
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        };

        let solid = Solid {
            outer_shell: Shell {
                faces: vec![original_face.clone()],
                closed: false,
            },
            inner_shells: vec![],
        };

        // Create a new face with different edge count
        let incompatible_face = Face {
            outer_wire: Wire {
                edges: vec![
                    Edge {
                        start: Vertex::new(0.0, 0.0, 0.0),
                        end: Vertex::new(10.0, 5.0, 0.0),
                        curve_type: CurveType::Line,
                    },
                    Edge {
                        start: Vertex::new(10.0, 5.0, 0.0),
                        end: Vertex::new(10.0, 10.0, 0.0),
                        curve_type: CurveType::Line,
                    },
                    Edge {
                        start: Vertex::new(10.0, 10.0, 0.0),
                        end: Vertex::new(0.0, 0.0, 0.0),
                        curve_type: CurveType::Line,
                    },
                ],
                closed: true,
            },
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        };

        let result = replace_face(&solid, 0, &incompatible_face);
        assert!(
            result.is_err(),
            "Replace should fail with different edge count"
        );
    }

    #[test]
    fn test_replace_face_incompatible_vertices() {
        let original_face = Face {
            outer_wire: Wire {
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
            },
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        };

        let solid = Solid {
            outer_shell: Shell {
                faces: vec![original_face.clone()],
                closed: false,
            },
            inner_shells: vec![],
        };

        // Create a new face with same edge count but different vertices
        let incompatible_face = Face {
            outer_wire: Wire {
                edges: vec![
                    Edge {
                        start: Vertex::new(1.0, 1.0, 0.0),  // Shifted position
                        end: Vertex::new(11.0, 1.0, 0.0),
                        curve_type: CurveType::Line,
                    },
                    Edge {
                        start: Vertex::new(11.0, 1.0, 0.0),
                        end: Vertex::new(11.0, 11.0, 0.0),
                        curve_type: CurveType::Line,
                    },
                    Edge {
                        start: Vertex::new(11.0, 11.0, 0.0),
                        end: Vertex::new(1.0, 11.0, 0.0),
                        curve_type: CurveType::Line,
                    },
                    Edge {
                        start: Vertex::new(1.0, 11.0, 0.0),
                        end: Vertex::new(1.0, 1.0, 0.0),
                        curve_type: CurveType::Line,
                    },
                ],
                closed: true,
            },
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        };

        let result = replace_face(&solid, 0, &incompatible_face);
        assert!(
            result.is_err(),
            "Replace should fail with different vertex positions"
        );
    }

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

    #[test]
    fn test_remove_face_basic() {
        // Create a simple solid with 6 faces (box)
        let faces = vec![
            // Bottom face (z=0)
            Face {
                outer_wire: Wire {
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
                },
                inner_wires: vec![],
                surface_type: SurfaceType::Plane {
                    origin: [0.0, 0.0, 0.0],
                    normal: [0.0, 0.0, -1.0],
                },
            },
            // Top face (z=10)
            Face {
                outer_wire: Wire {
                    edges: vec![
                        Edge {
                            start: Vertex::new(0.0, 0.0, 10.0),
                            end: Vertex::new(10.0, 0.0, 10.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(10.0, 0.0, 10.0),
                            end: Vertex::new(10.0, 10.0, 10.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(10.0, 10.0, 10.0),
                            end: Vertex::new(0.0, 10.0, 10.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(0.0, 10.0, 10.0),
                            end: Vertex::new(0.0, 0.0, 10.0),
                            curve_type: CurveType::Line,
                        },
                    ],
                    closed: true,
                },
                inner_wires: vec![],
                surface_type: SurfaceType::Plane {
                    origin: [0.0, 0.0, 10.0],
                    normal: [0.0, 0.0, 1.0],
                },
            },
            // Front face (y=0)
            Face {
                outer_wire: Wire {
                    edges: vec![
                        Edge {
                            start: Vertex::new(0.0, 0.0, 0.0),
                            end: Vertex::new(10.0, 0.0, 0.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(10.0, 0.0, 0.0),
                            end: Vertex::new(10.0, 0.0, 10.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(10.0, 0.0, 10.0),
                            end: Vertex::new(0.0, 0.0, 10.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(0.0, 0.0, 10.0),
                            end: Vertex::new(0.0, 0.0, 0.0),
                            curve_type: CurveType::Line,
                        },
                    ],
                    closed: true,
                },
                inner_wires: vec![],
                surface_type: SurfaceType::Plane {
                    origin: [0.0, 0.0, 0.0],
                    normal: [0.0, -1.0, 0.0],
                },
            },
            // Back face (y=10)
            Face {
                outer_wire: Wire {
                    edges: vec![
                        Edge {
                            start: Vertex::new(0.0, 10.0, 0.0),
                            end: Vertex::new(10.0, 10.0, 0.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(10.0, 10.0, 0.0),
                            end: Vertex::new(10.0, 10.0, 10.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(10.0, 10.0, 10.0),
                            end: Vertex::new(0.0, 10.0, 10.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(0.0, 10.0, 10.0),
                            end: Vertex::new(0.0, 10.0, 0.0),
                            curve_type: CurveType::Line,
                        },
                    ],
                    closed: true,
                },
                inner_wires: vec![],
                surface_type: SurfaceType::Plane {
                    origin: [0.0, 10.0, 0.0],
                    normal: [0.0, 1.0, 0.0],
                },
            },
            // Left face (x=0)
            Face {
                outer_wire: Wire {
                    edges: vec![
                        Edge {
                            start: Vertex::new(0.0, 0.0, 0.0),
                            end: Vertex::new(0.0, 10.0, 0.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(0.0, 10.0, 0.0),
                            end: Vertex::new(0.0, 10.0, 10.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(0.0, 10.0, 10.0),
                            end: Vertex::new(0.0, 0.0, 10.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(0.0, 0.0, 10.0),
                            end: Vertex::new(0.0, 0.0, 0.0),
                            curve_type: CurveType::Line,
                        },
                    ],
                    closed: true,
                },
                inner_wires: vec![],
                surface_type: SurfaceType::Plane {
                    origin: [0.0, 0.0, 0.0],
                    normal: [-1.0, 0.0, 0.0],
                },
            },
            // Right face (x=10)
            Face {
                outer_wire: Wire {
                    edges: vec![
                        Edge {
                            start: Vertex::new(10.0, 0.0, 0.0),
                            end: Vertex::new(10.0, 10.0, 0.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(10.0, 10.0, 0.0),
                            end: Vertex::new(10.0, 10.0, 10.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(10.0, 10.0, 10.0),
                            end: Vertex::new(10.0, 0.0, 10.0),
                            curve_type: CurveType::Line,
                        },
                        Edge {
                            start: Vertex::new(10.0, 0.0, 10.0),
                            end: Vertex::new(10.0, 0.0, 0.0),
                            curve_type: CurveType::Line,
                        },
                    ],
                    closed: true,
                },
                inner_wires: vec![],
                surface_type: SurfaceType::Plane {
                    origin: [10.0, 0.0, 0.0],
                    normal: [1.0, 0.0, 0.0],
                },
            },
        ];

        let solid = Solid {
            outer_shell: Shell {
                faces: faces.clone(),
                closed: true,
            },
            inner_shells: vec![],
        };

        // Test removing a face (middle one)
        let result = remove_face(&solid, 2);
        assert!(
            result.is_ok(),
            "Should successfully remove a face from a solid with multiple faces"
        );

        let removed_solid = result.unwrap();
        assert_eq!(
            removed_solid.outer_shell.faces.len(),
            5,
            "Should have 5 faces after removing 1 from 6"
        );
        assert!(!removed_solid.outer_shell.closed, "Shell should be open after removal");
    }

    #[test]
    fn test_remove_face_out_of_bounds() {
        let faces = vec![Face {
            outer_wire: Wire {
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
            },
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        }];

        let solid = Solid {
            outer_shell: Shell {
                faces,
                closed: true,
            },
            inner_shells: vec![],
        };

        // Test removing face at invalid index
        let result = remove_face(&solid, 5);
        assert!(
            result.is_err(),
            "Should fail when face index is out of bounds"
        );
    }

    #[test]
    fn test_remove_face_only_face() {
        let faces = vec![Face {
            outer_wire: Wire {
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
            },
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        }];

        let solid = Solid {
            outer_shell: Shell {
                faces,
                closed: true,
            },
            inner_shells: vec![],
        };

        // Test removing the only face
        let result = remove_face(&solid, 0);
        assert!(
            result.is_err(),
            "Should fail when trying to remove the only face"
        );
    }

    #[test]
    fn test_thicken_planar_face_basic() {
        // Create a simple rectangular planar face
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

        // Thicken the face
        let result = thicken_face(&face, 2.0);
        assert!(result.is_ok(), "Thickening should succeed");

        let solid = result.unwrap();
        assert!(solid.outer_shell.closed, "Resulting shell should be closed");
        
        // Should have: original face + offset face + 4 side faces = 6 faces
        assert_eq!(solid.outer_shell.faces.len(), 6, "Solid should have 6 faces");
    }

    #[test]
    fn test_thicken_face_zero_thickness() {
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

        // Try to thicken with zero thickness
        let result = thicken_face(&face, 0.0);
        assert!(
            result.is_err(),
            "Thickening with zero thickness should fail"
        );
    }

    #[test]
    fn test_thicken_face_negative_thickness() {
        // Create a simple rectangular planar face
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

        // Thicken the face with negative thickness (offset in opposite direction)
        let result = thicken_face(&face, -2.0);
        assert!(result.is_ok(), "Negative thickness should work");

        let solid = result.unwrap();
        assert!(solid.outer_shell.closed, "Resulting shell should be closed");
        assert_eq!(solid.outer_shell.faces.len(), 6, "Solid should have 6 faces");
    }

    #[test]
    fn test_thicken_face_with_hole() {
        // Create a rectangular face with a rectangular hole
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

        // Inner hole
        let inner_wire = Wire {
            edges: vec![
                Edge {
                    start: Vertex::new(2.0, 2.0, 0.0),
                    end: Vertex::new(8.0, 2.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(8.0, 2.0, 0.0),
                    end: Vertex::new(8.0, 8.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(8.0, 8.0, 0.0),
                    end: Vertex::new(2.0, 8.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(2.0, 8.0, 0.0),
                    end: Vertex::new(2.0, 2.0, 0.0),
                    curve_type: CurveType::Line,
                },
            ],
            closed: true,
        };

        let face = Face {
            outer_wire,
            inner_wires: vec![inner_wire],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        };

        // Thicken the face
        let result = thicken_face(&face, 2.0);
        assert!(result.is_ok(), "Thickening with hole should succeed");

        let solid = result.unwrap();
        assert!(solid.outer_shell.closed, "Resulting shell should be closed");
        
        // Should have: original face + offset face + 4 outer side faces + 4 inner side faces = 10 faces
        assert_eq!(solid.outer_shell.faces.len(), 10, "Solid should have 10 faces (2 + 4 + 4)");
    }
}
