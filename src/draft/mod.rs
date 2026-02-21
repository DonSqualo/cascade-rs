//! Draft and taper operations
//!
//! This module provides draft operations that create tapered faces, commonly used for
//! mold release in manufacturing. A draft angle makes it easier to remove a part from a mold.
//!
//! The draft operation:
//! - Takes a face and tilts it by a specified angle relative to a neutral plane
//! - Points on the face above the neutral plane tilt in one direction
//! - Points below the neutral plane tilt in the opposite direction
//! - The neutral plane itself remains fixed (unchanged)

use crate::brep::topology;
use crate::brep::{Edge, Face, Shell, Solid, SurfaceType, Vertex, Wire};
use crate::{CascadeError, Result};

/// Add a draft (taper) angle to a face relative to a neutral plane
///
/// This operation tilts a specified face by a draft angle around the intersection line
/// with a neutral plane. This is commonly used in manufacturing for mold release.
///
/// # Arguments
/// * `solid` - The input solid to modify
/// * `face` - The face to draft (taper)
/// * `angle` - The draft angle in radians (typically small, like 1-5 degrees)
/// * `neutral_plane` - The plane around which the face rotates
///
/// # Returns
/// A new Solid with the drafted face
///
/// # Implementation Notes
/// The draft operation works as follows:
/// 1. Find the intersection line between the face and neutral plane
/// 2. For each point on the face:
///    - Project to the neutral plane
///    - Measure the perpendicular distance
///    - Rotate the point around the intersection line by the draft angle
/// 3. Create new geometry with the tilted face
///
/// # Example
/// ```ignore
/// let solid = make_box(10.0, 10.0, 10.0)?;
/// let box_faces = get_solid_faces(&solid);
/// let top_face = &box_faces[0];
/// let base_plane = Face { /* base plane at z=0 */ };
/// let angle = 5.0 * std::f64::consts::PI / 180.0; // 5 degrees
/// let drafted = add_draft(&solid, top_face, angle, &base_plane)?;
/// ```
pub fn add_draft(solid: &Solid, face: &Face, angle: f64, neutral_plane: &Face) -> Result<Solid> {
    if angle.abs() < 1e-10 {
        return Ok(solid.clone());
    }

    // Validate that both surfaces are planar
    let (face_normal, face_origin) = extract_plane_info(face)?;
    let (plane_normal, plane_origin) = extract_plane_info(neutral_plane)?;

    // Check that planes are not parallel (must intersect)
    let dot = face_normal[0] * plane_normal[0]
        + face_normal[1] * plane_normal[1]
        + face_normal[2] * plane_normal[2];

    if (dot.abs() - 1.0).abs() < 1e-6 {
        return Err(CascadeError::InvalidGeometry(
            "Draft face and neutral plane must not be parallel".into(),
        ));
    }

    // Find the intersection line between the face plane and neutral plane
    let intersection_line =
        compute_plane_intersection(&face_origin, &face_normal, &plane_origin, &plane_normal)?;

    // Apply draft to the face
    let drafted_face = apply_draft_to_face(
        face,
        angle,
        &intersection_line.0, // line point
        &intersection_line.1, // line direction
        &plane_origin,
        &plane_normal,
    )?;

    // Collect all faces from the solid
    let all_faces = topology::get_solid_faces_internal(solid);

    // Replace the target face with the drafted one
    let mut new_faces = Vec::new();
    let mut found_face = false;

    for existing_face in &all_faces {
        if faces_equal(existing_face, face) {
            new_faces.push(drafted_face.clone());
            found_face = true;
        } else {
            new_faces.push(existing_face.clone());
        }
    }

    if !found_face {
        return Err(CascadeError::InvalidGeometry(
            "Target face not found in solid".into(),
        ));
    }

    // Build the new solid
    let new_shell = Shell {
        faces: new_faces,
        closed: solid.outer_shell.closed,
    };

    let result_solid = Solid {
        outer_shell: new_shell,
        inner_shells: solid.inner_shells.clone(),
        attributes: Default::default(),
    };

    Ok(result_solid)
}

/// Apply a taper angle to multiple faces of a solid relative to a neutral plane
///
/// This operation tilts multiple specified faces by a taper angle around the intersection lines
/// with a neutral plane. This is similar to draft but applies to multiple faces of an existing solid.
///
/// # Arguments
/// * `solid` - The input solid to modify
/// * `face_indices` - Indices of the faces to taper
/// * `angle` - The taper angle in radians (typically small, like 1-5 degrees)
/// * `neutral_plane` - The plane around which the faces rotate
///
/// # Returns
/// A new Solid with the tapered faces
///
/// # Implementation Notes
/// The taper operation:
/// 1. Retrieves all faces from the solid
/// 2. For each specified face index:
///    - Finds the intersection line between the face and neutral plane
///    - Rotates all vertices of that face around the intersection line by the taper angle
///    - Preserves edge connectivity
/// 3. Creates new geometry with all tapered faces
///
/// # Example
/// ```ignore
/// let solid = make_box(10.0, 10.0, 10.0)?;
/// let all_faces = get_solid_faces(&solid);
/// let base_plane = &all_faces[4]; // Use bottom face as neutral plane
/// let angle = 5.0 * std::f64::consts::PI / 180.0; // 5 degrees
/// let face_indices = &[0, 1, 2, 3]; // Taper top and side faces
/// let tapered = taper(&solid, face_indices, angle, &base_plane)?;
/// ```
pub fn taper(
    solid: &Solid,
    face_indices: &[usize],
    angle: f64,
    neutral_plane: &Face,
) -> Result<Solid> {
    if angle.abs() < 1e-10 {
        return Ok(solid.clone());
    }

    if face_indices.is_empty() {
        return Ok(solid.clone());
    }

    // Validate that neutral plane is planar
    let (plane_normal, plane_origin) = extract_plane_info(neutral_plane)?;

    // Collect all faces from the solid
    let all_faces = topology::get_solid_faces_internal(solid);

    // Validate all indices are in range
    for &idx in face_indices {
        if idx >= all_faces.len() {
            return Err(CascadeError::InvalidGeometry(format!(
                "Face index {} out of range (solid has {} faces)",
                idx,
                all_faces.len()
            )));
        }
    }

    // Create a set of indices for quick lookup
    let indices_set: std::collections::HashSet<usize> = face_indices.iter().copied().collect();

    // Apply taper to selected faces
    let mut new_faces = Vec::new();

    for (idx, face) in all_faces.iter().enumerate() {
        if indices_set.contains(&idx) {
            // This face should be tapered
            let (face_normal, face_origin) = extract_plane_info(face)?;

            // Check that planes are not parallel (must intersect)
            let dot = face_normal[0] * plane_normal[0]
                + face_normal[1] * plane_normal[1]
                + face_normal[2] * plane_normal[2];

            if (dot.abs() - 1.0).abs() < 1e-6 {
                return Err(CascadeError::InvalidGeometry(format!(
                    "Taper face at index {} and neutral plane must not be parallel",
                    idx
                )));
            }

            // Find the intersection line between the face plane and neutral plane
            let intersection_line = compute_plane_intersection(
                &face_origin,
                &face_normal,
                &plane_origin,
                &plane_normal,
            )?;

            // Apply taper to the face
            let tapered_face = apply_draft_to_face(
                face,
                angle,
                &intersection_line.0, // line point
                &intersection_line.1, // line direction
                &plane_origin,
                &plane_normal,
            )?;

            new_faces.push(tapered_face);
        } else {
            // Keep face unchanged
            new_faces.push(face.clone());
        }
    }

    // Build the new solid
    let new_shell = Shell {
        faces: new_faces,
        closed: solid.outer_shell.closed,
    };

    let result_solid = Solid {
        outer_shell: new_shell,
        inner_shells: solid.inner_shells.clone(),
        attributes: Default::default(),
    };

    Ok(result_solid)
}

/// Extract plane information (normal and origin) from a planar face
fn extract_plane_info(face: &Face) -> Result<([f64; 3], [f64; 3])> {
    match &face.surface_type {
        SurfaceType::Plane { origin, normal } => Ok((*normal, *origin)),
        _ => Err(CascadeError::NotImplemented(
            "Draft operations on non-planar faces not yet supported".into(),
        )),
    }
}

/// Compute the intersection line of two planes
///
/// Returns (point_on_line, line_direction)
fn compute_plane_intersection(
    plane1_origin: &[f64; 3],
    plane1_normal: &[f64; 3],
    plane2_origin: &[f64; 3],
    plane2_normal: &[f64; 3],
) -> Result<([f64; 3], [f64; 3])> {
    // The intersection line has a direction that is perpendicular to both normals
    let line_direction = cross_product(plane1_normal, plane2_normal);
    let line_len_sq = line_direction[0] * line_direction[0]
        + line_direction[1] * line_direction[1]
        + line_direction[2] * line_direction[2];

    if line_len_sq < 1e-12 {
        return Err(CascadeError::InvalidGeometry(
            "Planes are parallel and do not intersect".into(),
        ));
    }

    let line_len = line_len_sq.sqrt();
    let line_dir_norm = [
        line_direction[0] / line_len,
        line_direction[1] / line_len,
        line_direction[2] / line_len,
    ];

    // Find a point on the intersection line using Cramer's rule
    // We need to solve: plane1_normal · (P - plane1_origin) = 0
    //                   plane2_normal · (P - plane2_origin) = 0
    // for the intersection

    let point_on_line = find_plane_intersection_point(
        plane1_origin,
        plane1_normal,
        plane2_origin,
        plane2_normal,
        &line_dir_norm,
    );

    Ok((point_on_line, line_dir_norm))
}

/// Find a point on the intersection line of two planes
fn find_plane_intersection_point(
    plane1_origin: &[f64; 3],
    plane1_normal: &[f64; 3],
    plane2_origin: &[f64; 3],
    plane2_normal: &[f64; 3],
    line_direction: &[f64; 3],
) -> [f64; 3] {
    // Use a simple approach: find the point closest to both planes along the intersection direction
    // For now, we use plane1_origin as the base and project it onto plane2

    // Vector from plane1 origin to plane2 origin
    let diff = [
        plane2_origin[0] - plane1_origin[0],
        plane2_origin[1] - plane1_origin[1],
        plane2_origin[2] - plane1_origin[2],
    ];

    // Distance from plane1_origin to plane2 along plane2's normal
    let dist_to_plane2 = dot_product(&diff, plane2_normal);

    // Point that lies on both planes (approximately)
    let point = [
        plane1_origin[0] + plane2_normal[0] * dist_to_plane2,
        plane1_origin[1] + plane2_normal[1] * dist_to_plane2,
        plane1_origin[2] + plane2_normal[2] * dist_to_plane2,
    ];

    point
}

/// Apply draft angle to all vertices of a face
fn apply_draft_to_face(
    face: &Face,
    angle: f64,
    line_point: &[f64; 3],
    line_direction: &[f64; 3],
    plane_origin: &[f64; 3],
    plane_normal: &[f64; 3],
) -> Result<Face> {
    // Draft the outer wire
    let drafted_outer_wire = draft_wire(
        &face.outer_wire,
        angle,
        line_point,
        line_direction,
        plane_origin,
        plane_normal,
    )?;

    // Draft any inner wires (holes)
    let mut drafted_inner_wires = Vec::new();
    for inner_wire in &face.inner_wires {
        let drafted_inner = draft_wire(
            inner_wire,
            angle,
            line_point,
            line_direction,
            plane_origin,
            plane_normal,
        )?;
        drafted_inner_wires.push(drafted_inner);
    }

    // The surface of the drafted face should be rotated around the intersection line
    // For simplicity, we assume the face remains planar and just update its normal
    // by rotating it around the intersection line
    let new_normal = rotate_vector_around_axis(&face.surface_type, angle, line_direction)?;

    let new_surface = match &face.surface_type {
        SurfaceType::Plane { .. } => SurfaceType::Plane {
            origin: *line_point,
            normal: new_normal,
        },
        _ => {
            return Err(CascadeError::NotImplemented(
                "Can only draft planar faces".into(),
            ))
        }
    };

    Ok(Face {
        outer_wire: drafted_outer_wire,
        inner_wires: drafted_inner_wires,
        surface_type: new_surface,
    })
}

/// Draft a wire by rotating its vertices around the intersection line
fn draft_wire(
    wire: &Wire,
    angle: f64,
    line_point: &[f64; 3],
    line_direction: &[f64; 3],
    plane_origin: &[f64; 3],
    plane_normal: &[f64; 3],
) -> Result<Wire> {
    let mut drafted_edges = Vec::new();

    for edge in &wire.edges {
        let drafted_edge = draft_edge(
            edge,
            angle,
            line_point,
            line_direction,
            plane_origin,
            plane_normal,
        )?;
        drafted_edges.push(drafted_edge);
    }

    Ok(Wire {
        edges: drafted_edges,
        closed: wire.closed,
    })
}

/// Draft an edge by rotating its endpoints around the intersection line
fn draft_edge(
    edge: &Edge,
    angle: f64,
    line_point: &[f64; 3],
    line_direction: &[f64; 3],
    plane_origin: &[f64; 3],
    plane_normal: &[f64; 3],
) -> Result<Edge> {
    let start_point = &edge.start.point;
    let end_point = &edge.end.point;

    // Project points onto the neutral plane to determine sign
    let start_distance = signed_distance_to_plane(start_point, plane_origin, plane_normal);
    let end_distance = signed_distance_to_plane(end_point, plane_origin, plane_normal);

    // Determine draft angle sign based on which side of the neutral plane
    let start_angle = if start_distance > 0.0 { angle } else { -angle };
    let end_angle = if end_distance > 0.0 { angle } else { -angle };

    // Rotate vertices around the intersection line
    let drafted_start =
        rotate_point_around_line(start_point, start_angle, line_point, line_direction)?;
    let drafted_end = rotate_point_around_line(end_point, end_angle, line_point, line_direction)?;

    Ok(Edge {
        start: Vertex::new(drafted_start[0], drafted_start[1], drafted_start[2]),
        end: Vertex::new(drafted_end[0], drafted_end[1], drafted_end[2]),
        curve_type: edge.curve_type.clone(),
    })
}

/// Calculate the signed distance from a point to a plane
fn signed_distance_to_plane(
    point: &[f64; 3],
    plane_origin: &[f64; 3],
    plane_normal: &[f64; 3],
) -> f64 {
    let vec = [
        point[0] - plane_origin[0],
        point[1] - plane_origin[1],
        point[2] - plane_origin[2],
    ];
    dot_product(&vec, plane_normal)
}

/// Rotate a point around a line by the given angle
fn rotate_point_around_line(
    point: &[f64; 3],
    angle: f64,
    line_point: &[f64; 3],
    line_direction: &[f64; 3],
) -> Result<[f64; 3]> {
    // Vector from line point to the point to rotate
    let vec = [
        point[0] - line_point[0],
        point[1] - line_point[1],
        point[2] - line_point[2],
    ];

    // Component of vec parallel to line direction
    let parallel_dot = dot_product(&vec, line_direction);
    let parallel = [
        line_direction[0] * parallel_dot,
        line_direction[1] * parallel_dot,
        line_direction[2] * parallel_dot,
    ];

    // Component perpendicular to line direction
    let perp = [
        vec[0] - parallel[0],
        vec[1] - parallel[1],
        vec[2] - parallel[2],
    ];

    let perp_len_sq = dot_product(&perp, &perp);

    if perp_len_sq < 1e-12 {
        // Point is on the line, no rotation needed
        return Ok(*point);
    }

    // Third axis perpendicular to both line_direction and perp
    let third_axis = cross_product(line_direction, &perp);

    // Rotate perp around line_direction by angle
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let rotated_perp = [
        perp[0] * cos_a + third_axis[0] * sin_a,
        perp[1] * cos_a + third_axis[1] * sin_a,
        perp[2] * cos_a + third_axis[2] * sin_a,
    ];

    // Recombine: rotated point = line_point + parallel + rotated_perp
    let result = [
        line_point[0] + parallel[0] + rotated_perp[0],
        line_point[1] + parallel[1] + rotated_perp[1],
        line_point[2] + parallel[2] + rotated_perp[2],
    ];

    Ok(result)
}

/// Rotate a normal vector around an axis by the given angle (for surface orientation)
fn rotate_vector_around_axis(
    surface: &SurfaceType,
    angle: f64,
    axis: &[f64; 3],
) -> Result<[f64; 3]> {
    let normal = match surface {
        SurfaceType::Plane { normal, .. } => *normal,
        _ => {
            return Err(CascadeError::NotImplemented(
                "Can only rotate normals of planar surfaces".into(),
            ))
        }
    };

    // Check if normal is parallel to axis (special case)
    let dot = dot_product(&normal, axis);
    if (dot.abs() - 1.0).abs() < 1e-6 {
        // Normal is parallel to axis, no rotation happens
        return Ok(normal);
    }

    // Component of normal parallel to axis
    let parallel_dot = dot_product(&normal, axis);
    let parallel = [
        axis[0] * parallel_dot,
        axis[1] * parallel_dot,
        axis[2] * parallel_dot,
    ];

    // Component perpendicular to axis
    let perp = [
        normal[0] - parallel[0],
        normal[1] - parallel[1],
        normal[2] - parallel[2],
    ];

    // Third axis
    let third_axis = cross_product(axis, &perp);

    // Rotate
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let rotated_perp = [
        perp[0] * cos_a + third_axis[0] * sin_a,
        perp[1] * cos_a + third_axis[1] * sin_a,
        perp[2] * cos_a + third_axis[2] * sin_a,
    ];

    let rotated = [
        parallel[0] + rotated_perp[0],
        parallel[1] + rotated_perp[1],
        parallel[2] + rotated_perp[2],
    ];

    // Normalize
    let len = (rotated[0] * rotated[0] + rotated[1] * rotated[1] + rotated[2] * rotated[2]).sqrt();
    if len < 1e-10 {
        return Err(CascadeError::InvalidGeometry(
            "Failed to rotate normal vector".into(),
        ));
    }

    Ok([rotated[0] / len, rotated[1] / len, rotated[2] / len])
}

/// Compute the cross product of two 3D vectors
fn cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Compute the dot product of two 3D vectors
fn dot_product(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Check if two faces are geometrically equal
fn faces_equal(f1: &Face, f2: &Face) -> bool {
    const TOL: f64 = 1e-6;

    // Compare surface origins based on surface type
    let origin_match = match (&f1.surface_type, &f2.surface_type) {
        (SurfaceType::Plane { origin: o1, .. }, SurfaceType::Plane { origin: o2, .. }) => {
            (o1[0] - o2[0]).abs() < TOL
                && (o1[1] - o2[1]).abs() < TOL
                && (o1[2] - o2[2]).abs() < TOL
        }
        (SurfaceType::Cylinder { origin: o1, .. }, SurfaceType::Cylinder { origin: o2, .. }) => {
            (o1[0] - o2[0]).abs() < TOL
                && (o1[1] - o2[1]).abs() < TOL
                && (o1[2] - o2[2]).abs() < TOL
        }
        (SurfaceType::Sphere { center: c1, .. }, SurfaceType::Sphere { center: c2, .. }) => {
            (c1[0] - c2[0]).abs() < TOL
                && (c1[1] - c2[1]).abs() < TOL
                && (c1[2] - c2[2]).abs() < TOL
        }
        _ => false,
    };

    // Compare outer wires (check first edge as quick proxy)
    let wire_match = if !f1.outer_wire.edges.is_empty() && !f2.outer_wire.edges.is_empty() {
        let e1 = &f1.outer_wire.edges[0];
        let e2 = &f2.outer_wire.edges[0];
        (e1.start.point[0] - e2.start.point[0]).abs() < TOL
            && (e1.start.point[1] - e2.start.point[1]).abs() < TOL
            && (e1.start.point[2] - e2.start.point[2]).abs() < TOL
    } else {
        false
    };

    origin_match && wire_match
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::make_box;
    use std::f64::consts::PI;

    #[test]
    fn test_draft_zero_angle() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        let faces = topology::get_solid_faces_internal(&solid);
        let neutral_plane = &faces[4]; // bottom face

        let result = add_draft(&solid, &faces[0], 0.0, neutral_plane);
        assert!(
            result.is_ok(),
            "Zero angle draft should return original solid"
        );
    }

    #[test]
    fn test_draft_small_angle() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        let faces = topology::get_solid_faces_internal(&solid);
        let neutral_plane = &faces[4]; // bottom face

        // Draft with 5 degrees
        let angle = 5.0 * PI / 180.0;
        let result = add_draft(&solid, &faces[0], angle, neutral_plane);

        assert!(result.is_ok(), "Small angle draft should succeed");
        let drafted = result.unwrap();
        assert!(
            !drafted.outer_shell.faces.is_empty(),
            "Drafted solid should have faces"
        );
    }

    #[test]
    fn test_draft_preserves_face_count() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        let faces = topology::get_solid_faces_internal(&solid);
        let neutral_plane = &faces[4];

        let angle = 3.0 * PI / 180.0;
        let result = add_draft(&solid, &faces[0], angle, neutral_plane);

        assert!(result.is_ok());
        let drafted = result.unwrap();

        let orig_count = solid.outer_shell.faces.len();
        let new_count = drafted.outer_shell.faces.len();
        assert_eq!(
            orig_count, new_count,
            "Draft should preserve number of faces"
        );
    }

    #[test]
    fn test_taper_zero_angle() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        let faces = topology::get_solid_faces_internal(&solid);
        let neutral_plane = &faces[4]; // bottom face

        let result = taper(&solid, &[0], 0.0, neutral_plane);
        assert!(
            result.is_ok(),
            "Zero angle taper should return original solid"
        );
    }

    #[test]
    fn test_taper_single_face() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        let faces = topology::get_solid_faces_internal(&solid);
        let neutral_plane = &faces[4]; // bottom face

        let angle = 5.0 * PI / 180.0;
        let result = taper(&solid, &[0], angle, neutral_plane);

        assert!(result.is_ok(), "Single face taper should succeed");
        let tapered = result.unwrap();
        assert!(
            !tapered.outer_shell.faces.is_empty(),
            "Tapered solid should have faces"
        );
    }

    #[test]
    fn test_taper_multiple_faces() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        let faces = topology::get_solid_faces_internal(&solid);
        let neutral_plane = &faces[4]; // bottom face

        let angle = 5.0 * PI / 180.0;
        let face_indices = [0, 1, 2, 3]; // Taper top and side faces
        let result = taper(&solid, &face_indices, angle, neutral_plane);

        assert!(result.is_ok(), "Multiple face taper should succeed");
        let tapered = result.unwrap();
        assert!(
            !tapered.outer_shell.faces.is_empty(),
            "Tapered solid should have faces"
        );
    }

    #[test]
    fn test_taper_preserves_face_count() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        let faces = topology::get_solid_faces_internal(&solid);
        let neutral_plane = &faces[4];

        let angle = 3.0 * PI / 180.0;
        let result = taper(&solid, &[0, 1], angle, neutral_plane);

        assert!(result.is_ok());
        let tapered = result.unwrap();

        let orig_count = solid.outer_shell.faces.len();
        let new_count = tapered.outer_shell.faces.len();
        assert_eq!(
            orig_count, new_count,
            "Taper should preserve number of faces"
        );
    }

    #[test]
    fn test_taper_empty_indices() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        let faces = topology::get_solid_faces_internal(&solid);
        let neutral_plane = &faces[4];

        let angle = 3.0 * PI / 180.0;
        let result = taper(&solid, &[], angle, neutral_plane);

        assert!(result.is_ok());
        let tapered = result.unwrap();
        // With empty indices, solid should be unchanged
        assert_eq!(
            solid.outer_shell.faces.len(),
            tapered.outer_shell.faces.len()
        );
    }

    #[test]
    fn test_taper_invalid_index() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        let faces = topology::get_solid_faces_internal(&solid);
        let neutral_plane = &faces[4];

        let angle = 3.0 * PI / 180.0;
        let result = taper(&solid, &[999], angle, neutral_plane);

        assert!(result.is_err(), "Taper with invalid face index should fail");
    }
}
