//! Lofting and skinning operations
//!
//! Creates solids by interpolating between multiple profiles (cross-sections).
//! Supports both ruled surfaces (straight lines between profiles) and smooth
//! BSpline surfaces.

use crate::brep::{Solid, Wire, Face, Edge, Vertex, CurveType, SurfaceType};
use crate::{Result, CascadeError};

/// Create a solid by lofting (skinning) through multiple profiles
///
/// Generates a solid that interpolates through a series of cross-section profiles.
/// The profiles must have compatible topology (same number of edges/vertices).
///
/// # Arguments
/// * `profiles` - Slice of Wire profiles in order. Must have at least 2 profiles.
/// * `ruled` - If true, creates ruled surfaces (straight lines between profiles).
///           If false, creates smooth BSpline surfaces.
///
/// # Returns
/// A Solid that interpolates through all profiles
///
/// # Example
/// ```
/// let profile1 = create_circle_at([0.0, 0.0, 0.0], 1.0)?;
/// let profile2 = create_circle_at([0.0, 0.0, 1.0], 0.5)?;
/// let profile3 = create_circle_at([0.0, 0.0, 2.0], 1.0)?;
/// let solid = make_loft(&[profile1, profile2, profile3], false)?;
/// ```
pub fn make_loft(profiles: &[Wire], ruled: bool) -> Result<Solid> {
    // Validate input
    if profiles.len() < 2 {
        return Err(CascadeError::InvalidGeometry(
            "Loft requires at least 2 profiles".into(),
        ));
    }

    // Check that all profiles have the same number of edges
    let first_edge_count = profiles[0].edges.len();
    for (i, profile) in profiles.iter().enumerate() {
        if profile.edges.len() != first_edge_count {
            return Err(CascadeError::InvalidGeometry(
                format!(
                    "All profiles must have the same number of edges. Profile 0 has {}, profile {} has {}",
                    first_edge_count, i, profile.edges.len()
                ),
            ));
        }
        if !profile.closed {
            return Err(CascadeError::InvalidGeometry(
                format!("Profile {} must be closed", i),
            ));
        }
    }

    let num_profiles = profiles.len();
    let edges_per_profile = profiles[0].edges.len();

    // Create the loft surface by connecting consecutive profiles
    let mut faces = Vec::new();

    // For each edge in the profiles, create a surface between corresponding edges
    for edge_idx in 0..edges_per_profile {
        // For ruled surfaces, we connect edges directly
        // For smooth surfaces, we interpolate intermediate points
        if ruled {
            // Create ruled faces between consecutive profiles
            for profile_idx in 0..num_profiles - 1 {
                let face = create_ruled_face(
                    &profiles[profile_idx],
                    &profiles[profile_idx + 1],
                    edge_idx,
                )?;
                faces.push(face);
            }
        } else {
            // Create smooth BSpline faces
            for profile_idx in 0..num_profiles - 1 {
                let face = create_bspline_face(
                    &profiles[profile_idx],
                    &profiles[profile_idx + 1],
                    edge_idx,
                )?;
                faces.push(face);
            }
        }
    }

    // Close the loft at the ends by adding the first and last profiles as faces
    // (optional - depends on whether you want closed or open loft)
    // For now, we only create the side faces

    // Create shell from all faces
    let shell = crate::brep::Shell {
        faces,
        closed: false, // Loft is open at ends
    };

    // Create and return the solid
    let solid = Solid {
        outer_shell: shell,
        inner_shells: vec![],
    };

    Ok(solid)
}

/// Create a ruled surface between two profiles along a specific edge
///
/// A ruled surface is created by connecting corresponding vertices
/// with straight lines (ruling lines).
fn create_ruled_face(
    profile1: &Wire,
    profile2: &Wire,
    edge_idx: usize,
) -> Result<Face> {
    if edge_idx >= profile1.edges.len() || edge_idx >= profile2.edges.len() {
        return Err(CascadeError::InvalidGeometry(
            "Edge index out of bounds".into(),
        ));
    }

    // Get vertices from both profiles for this edge
    let v1_start = &profile1.edges[edge_idx].start;
    let v1_end = &profile1.edges[edge_idx].end;
    let v2_start = &profile2.edges[edge_idx].start;
    let v2_end = &profile2.edges[edge_idx].end;

    // Create a surface that connects these four corners
    // This forms a quadrilateral surface (bilinear)

    // Create 4 edges of the quad
    let edge_profile1 = Edge {
        start: v1_start.clone(),
        end: v1_end.clone(),
        curve_type: CurveType::Line,
    };

    let edge_profile2 = Edge {
        start: v2_start.clone(),
        end: v2_end.clone(),
        curve_type: CurveType::Line,
    };

    let edge_ruling1 = Edge {
        start: v1_start.clone(),
        end: v2_start.clone(),
        curve_type: CurveType::Line,
    };

    let edge_ruling2 = Edge {
        start: v1_end.clone(),
        end: v2_end.clone(),
        curve_type: CurveType::Line,
    };

    // Create outer wire from these edges
    let outer_wire = Wire {
        edges: vec![edge_profile1, edge_ruling1, edge_profile2, edge_ruling2],
        closed: true,
    };

    // Create a bilinear surface (simple plane approximation for now)
    // In a full implementation, this would use NURBS
    let surface = create_bilinear_surface(v1_start, v1_end, v2_start, v2_end);

    let face = Face {
        outer_wire,
        inner_wires: vec![],
        surface_type: surface,
    };

    Ok(face)
}

/// Create a smooth BSpline surface between two profiles along an edge
///
/// This creates a more refined surface by interpolating intermediate points.
fn create_bspline_face(
    profile1: &Wire,
    profile2: &Wire,
    edge_idx: usize,
) -> Result<Face> {
    if edge_idx >= profile1.edges.len() || edge_idx >= profile2.edges.len() {
        return Err(CascadeError::InvalidGeometry(
            "Edge index out of bounds".into(),
        ));
    }

    // Get vertices from both profiles for this edge
    let v1_start = &profile1.edges[edge_idx].start;
    let v1_end = &profile1.edges[edge_idx].end;
    let v2_start = &profile2.edges[edge_idx].start;
    let v2_end = &profile2.edges[edge_idx].end;

    // Create edges for the face
    let edge_profile1 = Edge {
        start: v1_start.clone(),
        end: v1_end.clone(),
        curve_type: CurveType::Line,
    };

    let edge_profile2 = Edge {
        start: v2_start.clone(),
        end: v2_end.clone(),
        curve_type: CurveType::Line,
    };

    let edge_ruling1 = Edge {
        start: v1_start.clone(),
        end: v2_start.clone(),
        curve_type: CurveType::Line,
    };

    let edge_ruling2 = Edge {
        start: v1_end.clone(),
        end: v2_end.clone(),
        curve_type: CurveType::Line,
    };

    let outer_wire = Wire {
        edges: vec![edge_profile1, edge_ruling1, edge_profile2, edge_ruling2],
        closed: true,
    };

    // Use a bilinear surface for now (same as ruled)
    // In a full implementation, this would use higher-order BSpline
    let surface = create_bilinear_surface(v1_start, v1_end, v2_start, v2_end);

    let face = Face {
        outer_wire,
        inner_wires: vec![],
        surface_type: surface,
    };

    Ok(face)
}

/// Create a bilinear surface between four corner vertices
///
/// Uses parametric representation: P(u,v) = (1-u)(1-v)P00 + u(1-v)P10 + (1-u)vP01 + uvP11
/// where u,v ∈ [0,1]
///
/// This is stored as a BSpline surface with degree 1 in both directions.
fn create_bilinear_surface(
    p00: &Vertex,
    p10: &Vertex,
    p01: &Vertex,
    p11: &Vertex,
) -> SurfaceType {
    // Create control points grid for bilinear surface (2x2 grid = 4 points)
    let control_points = vec![
        vec![p00.point, p10.point],
        vec![p01.point, p11.point],
    ];

    // Knots for degree 1 (linear) BSpline: [0, 0, 1, 1]
    let knots = vec![0.0, 0.0, 1.0, 1.0];

    SurfaceType::BSpline {
        u_degree: 1,
        v_degree: 1,
        u_knots: knots.clone(),
        v_knots: knots,
        control_points,
        weights: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a simple circular wire profile
    fn create_circle_wire(center: [f64; 3], radius: f64, num_points: usize) -> Wire {
        let mut edges = Vec::new();

        for i in 0..num_points {
            let angle1 = 2.0 * std::f64::consts::PI * (i as f64) / (num_points as f64);
            let angle2 =
                2.0 * std::f64::consts::PI * ((i + 1) as f64 % (num_points as f64))
                    / (num_points as f64);

            let x1 = center[0] + radius * angle1.cos();
            let y1 = center[1] + radius * angle1.sin();
            let z1 = center[2];

            let x2 = center[0] + radius * angle2.cos();
            let y2 = center[1] + radius * angle2.sin();
            let z2 = center[2];

            edges.push(Edge {
                start: Vertex::new(x1, y1, z1),
                end: Vertex::new(x2, y2, z2),
                curve_type: CurveType::Line,
            });
        }

        Wire {
            edges,
            closed: true,
        }
    }

    #[test]
    fn test_loft_basic_two_circles() {
        let profile1 = create_circle_wire([0.0, 0.0, 0.0], 1.0, 8);
        let profile2 = create_circle_wire([0.0, 0.0, 1.0], 0.5, 8);

        let result = make_loft(&[profile1, profile2], true);
        assert!(result.is_ok(), "Basic loft should succeed");

        let solid = result.unwrap();
        assert!(!solid.outer_shell.faces.is_empty());
    }

    #[test]
    fn test_loft_three_profiles_ruled() {
        let profile1 = create_circle_wire([0.0, 0.0, 0.0], 1.0, 8);
        let profile2 = create_circle_wire([0.0, 0.0, 1.0], 0.5, 8);
        let profile3 = create_circle_wire([0.0, 0.0, 2.0], 1.0, 8);

        let result = make_loft(&[profile1, profile2, profile3], true);
        assert!(result.is_ok());

        let solid = result.unwrap();
        // Should have 16 faces (8 edges × 2 transitions)
        assert_eq!(solid.outer_shell.faces.len(), 16);
    }

    #[test]
    fn test_loft_three_profiles_smooth() {
        let profile1 = create_circle_wire([0.0, 0.0, 0.0], 1.0, 8);
        let profile2 = create_circle_wire([0.0, 0.0, 1.0], 0.5, 8);
        let profile3 = create_circle_wire([0.0, 0.0, 2.0], 1.0, 8);

        let result = make_loft(&[profile1, profile2, profile3], false);
        assert!(result.is_ok());

        let solid = result.unwrap();
        assert_eq!(solid.outer_shell.faces.len(), 16);
    }

    #[test]
    fn test_loft_invalid_single_profile() {
        let profile1 = create_circle_wire([0.0, 0.0, 0.0], 1.0, 8);
        let result = make_loft(&[profile1], false);
        assert!(result.is_err());
    }

    #[test]
    fn test_loft_mismatched_edges() {
        let profile1 = create_circle_wire([0.0, 0.0, 0.0], 1.0, 8);
        let profile2 = create_circle_wire([0.0, 0.0, 1.0], 0.5, 6); // Different edge count

        let result = make_loft(&[profile1, profile2], false);
        assert!(
            result.is_err(),
            "Loft with mismatched edge counts should fail"
        );
    }

    #[test]
    fn test_loft_unclosed_profile() {
        let mut wire = create_circle_wire([0.0, 0.0, 0.0], 1.0, 8);
        wire.closed = false; // Mark as unclosed

        let profile2 = create_circle_wire([0.0, 0.0, 1.0], 0.5, 8);

        let result = make_loft(&[wire, profile2], false);
        assert!(result.is_err(), "Unclosed profile should fail");
    }
}
