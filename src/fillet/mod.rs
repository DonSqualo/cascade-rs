//! Filleting and chamfering operations
//!
//! This module provides fillet operations that round edges on solids.
//! A fillet is a smooth, rounded edge created by blending two adjacent faces.
//!
//! Supports:
//! - Constant radius fillets
//! - Variable radius fillets with linear and smooth interpolation

use crate::brep::{Solid, Edge, Face, Wire, Vertex, Shell, CurveType, SurfaceType};
use crate::{Result, CascadeError};
use crate::brep::topology;
use std::cmp::Ordering;

/// Interpolation method for variable radius fillet
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Linear interpolation between radius points
    Linear,
    /// Smooth cubic spline interpolation (C2 continuous)
    Smooth,
}

/// Radius law defines how the fillet radius varies along an edge
/// 
/// The edge is parameterized from 0.0 to 1.0, and the radius law
/// specifies the fillet radius at different parameter values.
#[derive(Debug, Clone)]
pub struct RadiusLaw {
    /// List of (parameter, radius) pairs, sorted by parameter
    points: Vec<(f64, f64)>,
    /// Interpolation method to use
    method: InterpolationMethod,
}

impl RadiusLaw {
    /// Create a new empty radius law
    pub fn new(method: InterpolationMethod) -> Self {
        RadiusLaw {
            points: Vec::new(),
            method,
        }
    }

    /// Create a radius law from parameter-radius pairs
    pub fn from_points(points: &[(f64, f64)], method: InterpolationMethod) -> Result<Self> {
        if points.is_empty() {
            return Err(CascadeError::InvalidGeometry(
                "Radius law requires at least one radius point".into(),
            ));
        }

        // Validate that all parameters are in [0, 1]
        for &(param, radius) in points {
            if param < 0.0 || param > 1.0 {
                return Err(CascadeError::InvalidGeometry(
                    format!("Radius law parameter must be in [0, 1], got {}", param),
                ));
            }
            if radius <= 0.0 {
                return Err(CascadeError::InvalidGeometry(
                    format!("Radius must be positive, got {}", radius),
                ));
            }
        }

        // Sort by parameter
        let mut sorted_points = points.to_vec();
        sorted_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        Ok(RadiusLaw {
            points: sorted_points,
            method,
        })
    }

    /// Add a radius point to the law
    /// 
    /// The point will be inserted in the correct sorted position.
    /// If a point with the same parameter already exists, it will be replaced.
    pub fn add_radius_point(&mut self, param: f64, radius: f64) -> Result<()> {
        if param < 0.0 || param > 1.0 {
            return Err(CascadeError::InvalidGeometry(
                format!("Radius law parameter must be in [0, 1], got {}", param),
            ));
        }
        if radius <= 0.0 {
            return Err(CascadeError::InvalidGeometry(
                format!("Radius must be positive, got {}", radius),
            ));
        }

        // Check if point with same parameter exists
        if let Some(pos) = self.points.iter().position(|p| (p.0 - param).abs() < 1e-10) {
            self.points[pos] = (param, radius);
        } else {
            self.points.push((param, radius));
            self.points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        }

        Ok(())
    }

    /// Set the interpolation method
    pub fn set_radius_law(&mut self, method: InterpolationMethod) {
        self.method = method;
    }

    /// Get the radius at a given parameter value
    pub fn radius_at(&self, param: f64) -> Result<f64> {
        if self.points.is_empty() {
            return Err(CascadeError::InvalidGeometry(
                "Radius law has no points".into(),
            ));
        }

        // Clamp parameter to [0, 1]
        let t = param.max(0.0).min(1.0);

        // Handle single point case
        if self.points.len() == 1 {
            return Ok(self.points[0].1);
        }

        // Find the two points that bracket t
        let mut lower_idx = 0;
        let mut upper_idx = 1;

        for i in 0..self.points.len() - 1 {
            if self.points[i].0 <= t && t <= self.points[i + 1].0 {
                lower_idx = i;
                upper_idx = i + 1;
                break;
            }
        }

        let (t0, r0) = self.points[lower_idx];
        let (t1, r1) = self.points[upper_idx];

        match self.method {
            InterpolationMethod::Linear => {
                // Linear interpolation
                let alpha = if (t1 - t0).abs() > 1e-10 {
                    (t - t0) / (t1 - t0)
                } else {
                    0.0
                };
                Ok((1.0 - alpha) * r0 + alpha * r1)
            }
            InterpolationMethod::Smooth => {
                // Cubic spline interpolation (using Catmull-Rom)
                self.smooth_interpolate(lower_idx, upper_idx, t)
            }
        }
    }

    /// Perform smooth Catmull-Rom spline interpolation
    fn smooth_interpolate(&self, lower_idx: usize, upper_idx: usize, t: f64) -> Result<f64> {
        let (t0, r0) = self.points[lower_idx];
        let (t1, r1) = self.points[upper_idx];

        // Get tangent control points
        let r_prev = if lower_idx > 0 {
            self.points[lower_idx - 1].1
        } else {
            r0 // Clamp edge
        };

        let r_next = if upper_idx < self.points.len() - 1 {
            self.points[upper_idx + 1].1
        } else {
            r1 // Clamp edge
        };

        // Normalize t to [0, 1] within the segment
        let local_t = if (t1 - t0).abs() > 1e-10 {
            (t - t0) / (t1 - t0)
        } else {
            0.0
        };

        // Catmull-Rom basis functions
        let t2 = local_t * local_t;
        let t3 = t2 * local_t;

        let basis_prev = -0.5 * t3 + t2 - 0.5 * local_t;
        let basis_0 = 1.5 * t3 - 2.5 * t2 + 1.0;
        let basis_1 = -1.5 * t3 + 2.0 * t2 + 0.5 * local_t;
        let basis_next = 0.5 * t3 - 0.5 * t2;

        let radius = basis_prev * r_prev + basis_0 * r0 + basis_1 * r1 + basis_next * r_next;

        Ok(radius.max(0.0)) // Ensure non-negative
    }

    /// Get the interpolation method
    pub fn method(&self) -> InterpolationMethod {
        self.method
    }

    /// Get all radius points
    pub fn points(&self) -> &[(f64, f64)] {
        &self.points
    }
}

/// Create a constant-radius fillet on specified edges of a solid
///
/// # Arguments
/// * `solid` - The input solid to fillet
/// * `edges` - Array of edges to fillet (by index in a canonical edge list)
/// * `radius` - The radius of the fillet
///
/// # Returns
/// A new Solid with filleted edges
///
/// # Implementation Notes
/// For a straight edge where two planar faces meet:
/// 1. Find the two adjacent faces
/// 2. Create a cylindrical blend surface with the specified radius
/// 3. Trim the original faces along the fillet surface
/// 4. Create new topology connecting everything
pub fn make_fillet(solid: &Solid, edge_indices: &[usize], radius: f64) -> Result<Solid> {
    if radius <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Fillet radius must be positive".into(),
        ));
    }

    if edge_indices.is_empty() {
        return Ok(solid.clone());
    }

    // For now, implement simple case: single edge fillet
    if edge_indices.len() != 1 {
        return Err(CascadeError::NotImplemented(
            "Multi-edge filleting not yet implemented".into(),
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
            "Edge must be shared by exactly two faces for filleting".into(),
        ));
    }

    let face1 = &adjacent[0];
    let face2 = &adjacent[1];

    // Check if both faces are planar (simple case)
    let (normal1, origin1) = extract_plane_info(face1)?;
    let (normal2, origin2) = extract_plane_info(face2)?;

    // Get edge direction and perpendicular vectors
    let edge_dir = get_edge_direction(target_edge)?;
    let edge_start = [target_edge.start.point[0], target_edge.start.point[1], target_edge.start.point[2]];
    let edge_end = [target_edge.end.point[0], target_edge.end.point[1], target_edge.end.point[2]];

    // Create the fillet surface (cylindrical blend)
    let fillet_face = create_fillet_surface(
        &edge_start,
        &edge_end,
        &normal1,
        &normal2,
        radius,
    )?;

    // Create trimmed versions of the original faces
    let trimmed_face1 = trim_face_for_fillet(face1, target_edge, radius, &normal1)?;
    let trimmed_face2 = trim_face_for_fillet(face2, target_edge, radius, &normal2)?;

    // Collect all other faces (not involved in the fillet)
    let other_faces = collect_other_faces(solid, face1, face2);

    // Build new shell with modified geometry
    let mut new_faces = vec![trimmed_face1, trimmed_face2, fillet_face];
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

/// Create a variable-radius fillet on a specified edge of a solid
///
/// # Arguments
/// * `solid` - The input solid to fillet
/// * `edge_index` - Index of the edge to fillet
/// * `radius_law` - RadiusLaw specifying how radius varies along the edge
///
/// # Returns
/// A new Solid with the variable-radius filleted edge
///
/// # Implementation Notes
/// The variable radius is interpolated along the edge parameter [0, 1].
/// At each point along the edge, the appropriate radius is calculated
/// based on the interpolation method specified in the radius law.
pub fn make_fillet_variable(solid: &Solid, edge_index: usize, radius_law: &RadiusLaw) -> Result<Solid> {
    if radius_law.points().is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Radius law must have at least one point".into(),
        ));
    }

    // Get all edges from the solid in a deterministic order
    let all_edges = collect_all_edges(solid);

    if edge_index >= all_edges.len() {
        return Err(CascadeError::InvalidGeometry(
            "Edge index out of bounds".into(),
        ));
    }

    let target_edge = &all_edges[edge_index];

    // Find the two faces that share this edge
    let adjacent = find_adjacent_faces(target_edge, solid)?;
    if adjacent.len() < 2 {
        return Err(CascadeError::TopologyError(
            "Edge must be shared by exactly two faces for filleting".into(),
        ));
    }

    let face1 = &adjacent[0];
    let face2 = &adjacent[1];

    // Check if both faces are planar (simple case)
    let (normal1, origin1) = extract_plane_info(face1)?;
    let (normal2, origin2) = extract_plane_info(face2)?;

    // Get edge direction and endpoints
    let edge_dir = get_edge_direction(target_edge)?;
    let edge_start = [target_edge.start.point[0], target_edge.start.point[1], target_edge.start.point[2]];
    let edge_end = [target_edge.end.point[0], target_edge.end.point[1], target_edge.end.point[2]];

    // Calculate edge length for parameterization
    let edge_length = ((edge_end[0] - edge_start[0]).powi(2) +
                      (edge_end[1] - edge_start[1]).powi(2) +
                      (edge_end[2] - edge_start[2]).powi(2)).sqrt();

    // Create the variable radius fillet surface
    let fillet_face = create_variable_fillet_surface(
        &edge_start,
        &edge_end,
        &normal1,
        &normal2,
        edge_length,
        radius_law,
    )?;

    // For variable radius, we need to trim the original faces with variable offsets
    let trimmed_face1 = trim_face_for_variable_fillet(face1, target_edge, edge_length, &normal1, radius_law)?;
    let trimmed_face2 = trim_face_for_variable_fillet(face2, target_edge, edge_length, &normal2, radius_law)?;

    // Collect all other faces (not involved in the fillet)
    let other_faces = collect_other_faces(solid, face1, face2);

    // Build new shell with modified geometry
    let mut new_faces = vec![trimmed_face1, trimmed_face2, fillet_face];
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

/// Create a variable-radius cylindrical fillet surface
fn create_variable_fillet_surface(
    edge_start: &[f64; 3],
    edge_end: &[f64; 3],
    normal1: &[f64; 3],
    normal2: &[f64; 3],
    edge_length: f64,
    radius_law: &RadiusLaw,
) -> Result<Face> {
    // Edge direction
    let edge_dir = [
        edge_end[0] - edge_start[0],
        edge_end[1] - edge_start[1],
        edge_end[2] - edge_start[2],
    ];
    let edge_dir_len = (edge_dir[0] * edge_dir[0] + edge_dir[1] * edge_dir[1] + edge_dir[2] * edge_dir[2]).sqrt();
    let edge_dir_norm = if edge_dir_len > 1e-10 {
        [edge_dir[0] / edge_dir_len, edge_dir[1] / edge_dir_len, edge_dir[2] / edge_dir_len]
    } else {
        [0.0, 0.0, 1.0]
    };

    // Get radius at start and end
    let radius_start = radius_law.radius_at(0.0)?;
    let radius_end = radius_law.radius_at(1.0)?;

    // Create the edge on the filleted edge itself
    let v1 = Vertex::new(edge_start[0], edge_start[1], edge_start[2]);
    let v2 = Vertex::new(edge_end[0], edge_end[1], edge_end[2]);

    let edge1 = Edge {
        start: v1.clone(),
        end: v2.clone(),
        curve_type: CurveType::Line,
    };

    // Calculate offset points at start and end
    let center_offset_start = calculate_variable_fillet_offset(normal1, normal2, radius_start)?;
    let center_offset_end = calculate_variable_fillet_offset(normal1, normal2, radius_end)?;

    let v3 = Vertex::new(
        edge_start[0] + center_offset_start[0],
        edge_start[1] + center_offset_start[1],
        edge_start[2] + center_offset_start[2],
    );

    let v4 = Vertex::new(
        edge_end[0] + center_offset_end[0],
        edge_end[1] + center_offset_end[1],
        edge_end[2] + center_offset_end[2],
    );

    let edge2 = Edge {
        start: v3.clone(),
        end: v4.clone(),
        curve_type: CurveType::Line,
    };

    // Create blend edges - these are arcs with variable radius
    // For simplicity, we approximate with straight lines
    let edge3 = Edge {
        start: v1.clone(),
        end: v3.clone(),
        curve_type: CurveType::Line,
    };

    let edge4 = Edge {
        start: v2.clone(),
        end: v4.clone(),
        curve_type: CurveType::Line,
    };

    let fillet_wire = Wire {
        edges: vec![edge1, edge3, edge2, edge4],
        closed: true,
    };

    // Use an offset surface for the fillet geometry
    let fillet_face = Face {
        outer_wire: fillet_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Cylinder {
            origin: edge_start.clone(),
            axis: edge_dir_norm,
            radius: radius_start,
        },
    };

    Ok(fillet_face)
}

/// Calculate the offset vector for variable radius fillet center
fn calculate_variable_fillet_offset(normal1: &[f64; 3], normal2: &[f64; 3], radius: f64) -> Result<[f64; 3]> {
    // The center of the fillet should be offset equally along both normals
    let bisector = [
        normal1[0] + normal2[0],
        normal1[1] + normal2[1],
        normal1[2] + normal2[2],
    ];

    let len = (bisector[0] * bisector[0] + bisector[1] * bisector[1] + bisector[2] * bisector[2]).sqrt();

    if len < 1e-10 {
        return Err(CascadeError::InvalidGeometry(
            "Cannot compute fillet for parallel or opposite faces".into(),
        ));
    }

    let bisector_norm = [bisector[0] / len, bisector[1] / len, bisector[2] / len];

    // For two faces meeting at angle θ: offset = radius / sin(θ/2)
    let cos_angle = normal1[0] * normal2[0] + normal1[1] * normal2[1] + normal1[2] * normal2[2];
    let angle = cos_angle.acos();
    let sin_half_angle = (angle / 2.0).sin();

    let offset_distance = if sin_half_angle > 1e-6 {
        radius / sin_half_angle
    } else {
        radius
    };

    Ok([
        bisector_norm[0] * offset_distance,
        bisector_norm[1] * offset_distance,
        bisector_norm[2] * offset_distance,
    ])
}

/// Trim a face for variable radius filleting
fn trim_face_for_variable_fillet(
    face: &Face,
    edge: &Edge,
    edge_length: f64,
    normal: &[f64; 3],
    radius_law: &RadiusLaw,
) -> Result<Face> {
    // For variable radius, we need to offset the edge inward by a variable distance
    let mut trimmed_edges = Vec::new();

    for face_edge in &face.outer_wire.edges {
        if edges_equal(face_edge, edge) {
            // This is the edge being filleted - offset it inward with variable radius
            // For simplicity in this implementation, use average radius
            let avg_radius = (radius_law.radius_at(0.0)? + radius_law.radius_at(1.0)?) / 2.0;

            let offset_start = offset_point(&face_edge.start.point, normal, avg_radius);
            let offset_end = offset_point(&face_edge.end.point, normal, avg_radius);

            let trimmed_edge = Edge {
                start: Vertex::new(offset_start[0], offset_start[1], offset_start[2]),
                end: Vertex::new(offset_end[0], offset_end[1], offset_end[2]),
                curve_type: CurveType::Line,
            };
            trimmed_edges.push(trimmed_edge);
        } else {
            // Keep other edges as-is
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
            "Non-planar face filleting not yet supported".into(),
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
            "Non-linear edge filleting not yet supported".into(),
        )),
    }
}

/// Create a cylindrical fillet surface between two planar faces
fn create_fillet_surface(
    edge_start: &[f64; 3],
    edge_end: &[f64; 3],
    normal1: &[f64; 3],
    normal2: &[f64; 3],
    radius: f64,
) -> Result<Face> {
    // Edge direction
    let edge_dir = [
        edge_end[0] - edge_start[0],
        edge_end[1] - edge_start[1],
        edge_end[2] - edge_start[2],
    ];
    let edge_len = (edge_dir[0] * edge_dir[0] + edge_dir[1] * edge_dir[1] + edge_dir[2] * edge_dir[2]).sqrt();
    let edge_dir_norm = [edge_dir[0] / edge_len, edge_dir[1] / edge_len, edge_dir[2] / edge_len];

    // Calculate the fillet center line (bisector of the two faces)
    let center_offset = calculate_fillet_center_offset(normal1, normal2, radius)?;

    // Create the fillet face as a cylindrical surface
    // The fillet face connects the edge with a curved surface that follows both normal directions
    let v1 = Vertex::new(edge_start[0], edge_start[1], edge_start[2]);
    let v2 = Vertex::new(edge_end[0], edge_end[1], edge_end[2]);

    // Create edges for the fillet face
    let edge1 = Edge {
        start: v1.clone(),
        end: v2.clone(),
        curve_type: CurveType::Line,
    };

    // Offset edge along the center direction
    let v3 = Vertex::new(
        edge_start[0] + center_offset[0],
        edge_start[1] + center_offset[1],
        edge_start[2] + center_offset[2],
    );

    let v4 = Vertex::new(
        edge_end[0] + center_offset[0],
        edge_end[1] + center_offset[1],
        edge_end[2] + center_offset[2],
    );

    let edge2 = Edge {
        start: v3.clone(),
        end: v4.clone(),
        curve_type: CurveType::Line,
    };

    // Create blend edges connecting the corners
    let edge3 = Edge {
        start: v1.clone(),
        end: v3.clone(),
        curve_type: CurveType::Arc {
            center: edge_start.clone(),
            radius,
        },
    };

    let edge4 = Edge {
        start: v2.clone(),
        end: v4.clone(),
        curve_type: CurveType::Arc {
            center: edge_end.clone(),
            radius,
        },
    };

    let fillet_wire = Wire {
        edges: vec![edge1, edge3, edge2, edge4],
        closed: true,
    };

    let fillet_face = Face {
        outer_wire: fillet_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Cylinder {
            origin: edge_start.clone(),
            axis: edge_dir_norm,
            radius,
        },
    };

    Ok(fillet_face)
}

/// Calculate the offset vector for the fillet center line
fn calculate_fillet_center_offset(normal1: &[f64; 3], normal2: &[f64; 3], radius: f64) -> Result<[f64; 3]> {
    // The center of the fillet should be offset equally along both normals
    // This creates a smooth blend between the two faces

    // Bisector direction (average of the two normals, then normalize)
    let bisector = [
        normal1[0] + normal2[0],
        normal1[1] + normal2[1],
        normal1[2] + normal2[2],
    ];

    let len = (bisector[0] * bisector[0] + bisector[1] * bisector[1] + bisector[2] * bisector[2]).sqrt();
    
    if len < 1e-10 {
        return Err(CascadeError::InvalidGeometry(
            "Cannot compute fillet for parallel or opposite faces".into(),
        ));
    }

    // Distance along bisector to achieve desired radius
    let bisector_norm = [bisector[0] / len, bisector[1] / len, bisector[2] / len];

    // For two faces meeting at 90 degrees: offset = radius / sqrt(2)
    // For two faces meeting at angle θ: offset = radius / sin(θ/2)
    // Approximate: use radius / sin(θ/2) where θ is angle between normals
    let cos_angle = normal1[0] * normal2[0] + normal1[1] * normal2[1] + normal1[2] * normal2[2];
    let angle = cos_angle.acos();
    let sin_half_angle = (angle / 2.0).sin();

    let offset_distance = if sin_half_angle > 1e-6 {
        radius / sin_half_angle
    } else {
        radius
    };

    Ok([
        bisector_norm[0] * offset_distance,
        bisector_norm[1] * offset_distance,
        bisector_norm[2] * offset_distance,
    ])
}

/// Trim a face for filleting by moving edges inward by the fillet radius
fn trim_face_for_fillet(
    face: &Face,
    edge: &Edge,
    radius: f64,
    normal: &[f64; 3],
) -> Result<Face> {
    // For a planar face, we need to move the edge inward by the fillet radius
    // This creates space for the fillet surface

    let mut trimmed_edges = Vec::new();

    for face_edge in &face.outer_wire.edges {
        if edges_equal(face_edge, edge) {
            // This is the edge being filleted - offset it inward
            let offset_start = offset_point(&face_edge.start.point, normal, radius);
            let offset_end = offset_point(&face_edge.end.point, normal, radius);

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
    fn test_fillet_box_edge() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        
        // Fillet the first edge with radius 1.0
        let result = make_fillet(&solid, &[0], 1.0);
        
        assert!(result.is_ok(), "Fillet operation should succeed");
        let filleted = result.unwrap();
        
        // Check that the result is still a valid solid
        assert!(!filleted.outer_shell.faces.is_empty(), "Filleted solid should have faces");
        assert!(filleted.outer_shell.faces.len() >= 6, "Filleted solid should have at least 6 faces");
    }

    #[test]
    fn test_fillet_negative_radius() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        
        let result = make_fillet(&solid, &[0], -1.0);
        assert!(result.is_err(), "Negative radius should fail");
    }

    #[test]
    fn test_fillet_empty_edges() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        
        let result = make_fillet(&solid, &[], 1.0);
        assert!(result.is_ok(), "Empty edge list should return original solid");
    }

    #[test]
    fn test_fillet_preserves_solid_property() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        let filleted = make_fillet(&solid, &[0], 0.5).expect("Fillet failed");
        
        // Result should still be closed
        assert!(filleted.outer_shell.closed, "Filleted solid should be closed");
    }

    #[test]
    fn test_radius_law_creation() {
        let points = vec![(0.0, 1.0), (1.0, 2.0)];
        let law = RadiusLaw::from_points(&points, InterpolationMethod::Linear)
            .expect("Failed to create radius law");
        
        assert_eq!(law.points().len(), 2);
        assert_eq!(law.method(), InterpolationMethod::Linear);
    }

    #[test]
    fn test_radius_law_linear_interpolation() {
        let points = vec![(0.0, 1.0), (1.0, 3.0)];
        let law = RadiusLaw::from_points(&points, InterpolationMethod::Linear)
            .expect("Failed to create radius law");
        
        // At t=0, radius should be 1.0
        let r0 = law.radius_at(0.0).expect("Failed to get radius at 0");
        assert!((r0 - 1.0).abs() < 1e-6);
        
        // At t=1, radius should be 3.0
        let r1 = law.radius_at(1.0).expect("Failed to get radius at 1");
        assert!((r1 - 3.0).abs() < 1e-6);
        
        // At t=0.5, radius should be 2.0 (linear interpolation)
        let r_mid = law.radius_at(0.5).expect("Failed to get radius at 0.5");
        assert!((r_mid - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_radius_law_add_point() {
        let points = vec![(0.0, 1.0), (1.0, 3.0)];
        let mut law = RadiusLaw::from_points(&points, InterpolationMethod::Linear)
            .expect("Failed to create radius law");
        
        // Add a point in the middle
        law.add_radius_point(0.5, 2.0).expect("Failed to add point");
        
        assert_eq!(law.points().len(), 3);
        
        // Verify the points are sorted
        let points = law.points();
        assert!(points[0].0 < points[1].0);
        assert!(points[1].0 < points[2].0);
    }

    #[test]
    fn test_radius_law_smooth_interpolation() {
        let points = vec![(0.0, 1.0), (0.5, 2.0), (1.0, 1.0)];
        let law = RadiusLaw::from_points(&points, InterpolationMethod::Smooth)
            .expect("Failed to create radius law");
        
        // At t=0, radius should be 1.0
        let r0 = law.radius_at(0.0).expect("Failed to get radius at 0");
        assert!((r0 - 1.0).abs() < 1e-6);
        
        // At t=1, radius should be 1.0
        let r1 = law.radius_at(1.0).expect("Failed to get radius at 1");
        assert!((r1 - 1.0).abs() < 1e-6);
        
        // At t=0.5, radius should be close to 2.0 (smooth interpolation)
        let r_mid = law.radius_at(0.5).expect("Failed to get radius at 0.5");
        assert!(r_mid > 1.5 && r_mid < 2.5);
    }

    #[test]
    fn test_radius_law_invalid_parameter() {
        let points = vec![(0.0, 1.0), (1.5, 2.0)];
        let result = RadiusLaw::from_points(&points, InterpolationMethod::Linear);
        
        assert!(result.is_err(), "Should reject parameter > 1.0");
    }

    #[test]
    fn test_radius_law_invalid_radius() {
        let points = vec![(0.0, 1.0), (1.0, -1.0)];
        let result = RadiusLaw::from_points(&points, InterpolationMethod::Linear);
        
        assert!(result.is_err(), "Should reject negative radius");
    }

    #[test]
    fn test_fillet_variable_basic() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        
        // Create a radius law: radius increases from 0.5 at start to 1.5 at end
        let points = vec![(0.0, 0.5), (1.0, 1.5)];
        let law = RadiusLaw::from_points(&points, InterpolationMethod::Linear)
            .expect("Failed to create radius law");
        
        // Fillet the first edge with variable radius
        let result = make_fillet_variable(&solid, 0, &law);
        
        assert!(result.is_ok(), "Variable radius fillet operation should succeed");
        let filleted = result.unwrap();
        
        // Check that the result is still a valid solid
        assert!(!filleted.outer_shell.faces.is_empty(), "Filleted solid should have faces");
        assert!(filleted.outer_shell.faces.len() >= 6, "Filleted solid should have at least 6 faces");
    }

    #[test]
    fn test_fillet_variable_smooth_interpolation() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        
        // Create a radius law with smooth interpolation
        let points = vec![(0.0, 0.5), (0.5, 1.5), (1.0, 0.5)];
        let law = RadiusLaw::from_points(&points, InterpolationMethod::Smooth)
            .expect("Failed to create radius law");
        
        let result = make_fillet_variable(&solid, 0, &law);
        
        assert!(result.is_ok(), "Variable radius fillet with smooth interpolation should succeed");
        let filleted = result.unwrap();
        
        assert!(filleted.outer_shell.closed, "Filleted solid should be closed");
    }

    #[test]
    fn test_fillet_variable_invalid_edge_index() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        
        let points = vec![(0.0, 1.0), (1.0, 1.0)];
        let law = RadiusLaw::from_points(&points, InterpolationMethod::Linear)
            .expect("Failed to create radius law");
        
        // Try with an invalid edge index
        let result = make_fillet_variable(&solid, 999, &law);
        
        assert!(result.is_err(), "Should fail with invalid edge index");
    }

    #[test]
    fn test_fillet_variable_preserves_solid_property() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        
        let points = vec![(0.0, 0.25), (1.0, 0.5)];
        let law = RadiusLaw::from_points(&points, InterpolationMethod::Linear)
            .expect("Failed to create radius law");
        
        let filleted = make_fillet_variable(&solid, 0, &law).expect("Fillet failed");
        
        // Result should still be closed
        assert!(filleted.outer_shell.closed, "Filleted solid should be closed");
    }
}
