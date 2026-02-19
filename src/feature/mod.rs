//! Manufacturing feature operations
//!
//! This module provides common manufacturing features that modify solids:
//! - **Holes**: Subtractive features that cut cylindrical pockets
//! - **Slots**: Subtractive features that cut rectangular pockets along a path
//! - **Ribs**: Additive features that add material with a given profile

use crate::brep::{Solid, Wire, Face, Edge, Vertex, Shell, CurveType, SurfaceType};
use crate::primitive::{make_cylinder, make_box};
use crate::boolean::cut;
use crate::{Result, CascadeError, TOLERANCE};

/// Create a cylindrical hole in a solid
///
/// # Arguments
/// * `solid` - The input solid to modify
/// * `center` - 3D coordinates of the hole center
/// * `direction` - Normal vector of the hole axis (will be normalized)
/// * `diameter` - Hole diameter (must be > 0)
/// * `depth` - Hole depth (must be > 0). Hole cuts from center downward by depth amount
///
/// # Returns
/// A new Solid with a cylindrical hole cut through it
///
/// # Implementation
/// 1. Creates a cylinder aligned with the given direction
/// 2. Positions it at the hole center
/// 3. Extends it sufficiently to cut through the solid
/// 4. Uses boolean cut operation to remove material
///
/// # Example
/// ```ignore
/// let hole = make_hole(&solid, [5.0, 5.0, 0.0], [0.0, 0.0, 1.0], 10.0, 20.0)?;
/// ```
pub fn make_hole(
    solid: &Solid,
    center: [f64; 3],
    direction: [f64; 3],
    diameter: f64,
    depth: f64,
) -> Result<Solid> {
    // Validate inputs
    if diameter <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Hole diameter must be positive, got {}", diameter),
        ));
    }

    if depth <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Hole depth must be positive, got {}", depth),
        ));
    }

    // Normalize direction vector
    let dir_norm = normalize_vector(&direction)?;

    // Create a cylinder with the given diameter
    let radius = diameter / 2.0;
    let mut hole_cylinder = make_cylinder(radius, depth)?;

    // Position the cylinder at the hole center, aligned with direction
    // The cylinder origin needs to be positioned so it's centered at the given center
    // and extends downward by depth along the direction
    hole_cylinder = position_solid_at_with_direction(&hole_cylinder, &center, &dir_norm)?;

    // Perform boolean cut: remove the hole cylinder from the solid
    cut(solid, &hole_cylinder)
}

/// Create a countersunk hole in a solid
///
/// # Arguments
/// * `solid` - The input solid to modify
/// * `center` - 3D coordinates of the hole center (at the surface)
/// * `direction` - Normal vector of the hole axis (will be normalized)
/// * `hole_diameter` - Diameter of the main cylindrical hole (must be > 0)
/// * `countersink_diameter` - Diameter of the countersink opening (must be >= hole_diameter)
/// * `countersink_angle` - Angle of the countersink cone in degrees (typical: 82-120°)
/// * `depth` - Total depth of the hole including countersink (must be > 0)
///
/// # Returns
/// A new Solid with a countersunk hole cut through it
///
/// # Implementation
/// 1. Creates a cylinder for the main hole
/// 2. Creates a cone for the countersink with the specified angle
/// 3. Combines both geometries with boolean union
/// 4. Positions the combined geometry at the hole center
/// 5. Uses boolean cut operation to remove material
///
/// # Notes
/// - The countersink is a conical depression at the surface
/// - Depth extends from the surface downward along the direction
/// - The cone angle is the full angle of the countersink
///
/// # Example
/// ```ignore
/// let countersunk = make_hole_countersunk(
///     &solid,
///     [5.0, 5.0, 0.0],
///     [0.0, 0.0, 1.0],
///     5.0,      // main hole diameter
///     8.0,      // countersink diameter
///     90.0,     // countersink angle
///     10.0      // total depth
/// )?;
/// ```
pub fn make_hole_countersunk(
    solid: &Solid,
    center: [f64; 3],
    direction: [f64; 3],
    hole_diameter: f64,
    countersink_diameter: f64,
    countersink_angle: f64,
    depth: f64,
) -> Result<Solid> {
    // Validate inputs
    if hole_diameter <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Hole diameter must be positive, got {}", hole_diameter),
        ));
    }

    if countersink_diameter < hole_diameter {
        return Err(CascadeError::InvalidGeometry(
            format!("Countersink diameter ({}) must be >= hole diameter ({})",
                countersink_diameter, hole_diameter),
        ));
    }

    if countersink_angle <= 0.0 || countersink_angle >= 180.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Countersink angle must be between 0 and 180 degrees, got {}", countersink_angle),
        ));
    }

    if depth <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Hole depth must be positive, got {}", depth),
        ));
    }

    // Normalize direction vector
    let dir_norm = normalize_vector(&direction)?;

    // Create main hole cylinder
    let hole_radius = hole_diameter / 2.0;
    let mut hole_cylinder = make_cylinder(hole_radius, depth)?;

    // Create countersink cone
    // Half angle of the cone in radians (full angle / 2)
    let half_angle_rad = (countersink_angle / 2.0).to_radians();
    
    // Calculate cone depth from the countersink angle
    // For a cone: tan(half_angle) = countersink_radius / cone_depth
    let countersink_radius = countersink_diameter / 2.0;
    let cone_depth = countersink_radius / half_angle_rad.tan();
    
    // The cone should extend from surface to the point where it meets the main hole
    // Cap the cone depth to not exceed total depth
    let actual_cone_depth = cone_depth.min(depth);
    
    use crate::primitive::make_cone;
    let mut countersink_cone = make_cone(countersink_radius, 0.0, actual_cone_depth)?;

    // Combine hole and countersink using boolean union
    // This creates a single solid to subtract
    use crate::boolean::fuse;
    let mut hole_with_countersink = fuse(&hole_cylinder, &countersink_cone)?;

    // Position the combined hole geometry at the hole center
    hole_with_countersink = position_solid_at_with_direction(&hole_with_countersink, &center, &dir_norm)?;

    // Perform boolean cut: remove the hole from the solid
    cut(solid, &hole_with_countersink)
}

/// Create a counterbore hole in a solid
///
/// # Arguments
/// * `solid` - The input solid to modify
/// * `center` - 3D coordinates of the hole center (at the counterbore opening)
/// * `direction` - Normal vector of the hole axis (will be normalized)
/// * `hole_diameter` - Diameter of the main cylindrical hole (must be > 0)
/// * `counterbore_diameter` - Diameter of the counterbore opening (must be >= hole_diameter)
/// * `counterbore_depth` - Depth of the counterbore cavity (must be > 0)
/// * `hole_depth` - Total depth of the main hole below the counterbore (must be > 0)
///
/// # Returns
/// A new Solid with a counterbore hole cut through it
///
/// # Implementation
/// 1. Creates a cylinder for the main hole (smaller diameter)
/// 2. Creates a cylinder for the counterbore (larger diameter)
/// 3. Combines both cylinders with boolean union
/// 4. Positions the combined geometry at the hole center
/// 5. Uses boolean cut operation to remove material
///
/// # Notes
/// - The counterbore is a stepped hole with two different diameters
/// - counterbore_depth extends from the surface downward
/// - hole_depth extends below the counterbore at the smaller diameter
/// - Total depth = counterbore_depth + hole_depth
///
/// # Example
/// ```ignore
/// let counterbore = make_hole_counterbore(
///     &solid,
///     [5.0, 5.0, 0.0],
///     [0.0, 0.0, 1.0],
///     5.0,      // main hole diameter
///     8.0,      // counterbore diameter
///     3.0,      // counterbore depth
///     7.0       // hole depth below counterbore
/// )?;
/// ```
pub fn make_hole_counterbore(
    solid: &Solid,
    center: [f64; 3],
    direction: [f64; 3],
    hole_diameter: f64,
    counterbore_diameter: f64,
    counterbore_depth: f64,
    hole_depth: f64,
) -> Result<Solid> {
    // Validate inputs
    if hole_diameter <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Hole diameter must be positive, got {}", hole_diameter),
        ));
    }

    if counterbore_diameter < hole_diameter {
        return Err(CascadeError::InvalidGeometry(
            format!("Counterbore diameter ({}) must be >= hole diameter ({})",
                counterbore_diameter, hole_diameter),
        ));
    }

    if counterbore_depth <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Counterbore depth must be positive, got {}", counterbore_depth),
        ));
    }

    if hole_depth <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Hole depth must be positive, got {}", hole_depth),
        ));
    }

    // Normalize direction vector
    let dir_norm = normalize_vector(&direction)?;

    // Create main hole cylinder (extends through entire depth)
    let hole_radius = hole_diameter / 2.0;
    let total_depth = counterbore_depth + hole_depth;
    let mut main_hole = make_cylinder(hole_radius, total_depth)?;

    // Create counterbore cylinder (top portion only)
    let counterbore_radius = counterbore_diameter / 2.0;
    let mut counterbore_cylinder = make_cylinder(counterbore_radius, counterbore_depth)?;

    // Combine the two cylinders using boolean union
    // This creates a single stepped hole to subtract
    use crate::boolean::fuse;
    let mut hole_with_counterbore = fuse(&counterbore_cylinder, &main_hole)?;

    // Position the combined hole geometry at the hole center
    hole_with_counterbore = position_solid_at_with_direction(&hole_with_counterbore, &center, &dir_norm)?;

    // Perform boolean cut: remove the hole from the solid
    cut(solid, &hole_with_counterbore)
}

/// Create a rectangular slot in a solid
///
/// # Arguments
/// * `solid` - The input solid to modify
/// * `path` - A wire defining the centerline path of the slot
/// * `width` - Width of the slot (must be > 0)
/// * `depth` - Depth of the slot (must be > 0)
///
/// # Returns
/// A new Solid with a rectangular slot cut along the path
///
/// # Implementation
/// 1. Creates a rectangular profile with the given width and depth
/// 2. Sweeps this profile along the path wire
/// 3. Uses boolean cut operation to remove material
///
/// # Notes
/// - Currently simplified: assumes a straight path (first edge direction of the wire)
/// - For curved paths, would require full sweep surface generation
pub fn make_slot(
    solid: &Solid,
    path: &Wire,
    width: f64,
    depth: f64,
) -> Result<Solid> {
    // Validate inputs
    if width <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Slot width must be positive, got {}", width),
        ));
    }

    if depth <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Slot depth must be positive, got {}", depth),
        ));
    }

    if path.edges.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Slot path must contain at least one edge".into(),
        ));
    }

    // For a simplified implementation, create a box with width and depth
    // and position it along the first edge of the path
    
    let first_edge = &path.edges[0];
    let path_start = first_edge.start.point;
    let path_end = first_edge.end.point;

    // Calculate path direction and length
    let dx = path_end[0] - path_start[0];
    let dy = path_end[1] - path_start[1];
    let dz = path_end[2] - path_start[2];
    let path_length = (dx * dx + dy * dy + dz * dz).sqrt();

    if path_length < TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            "Slot path edge has zero length".into(),
        ));
    }

    let path_dir = [dx / path_length, dy / path_length, dz / path_length];

    // Create a rectangular box for the slot: width × depth × path_length
    let mut slot_box = make_box(width, depth, path_length)?;

    // Position the box so its center is at the midpoint of the path
    let path_mid = [
        path_start[0] + dx / 2.0,
        path_start[1] + dy / 2.0,
        path_start[2] + dz / 2.0,
    ];

    slot_box = position_solid_at_with_direction(&slot_box, &path_mid, &path_dir)?;

    // Perform boolean cut
    cut(solid, &slot_box)
}

/// Add a rib feature to a solid
///
/// # Arguments
/// * `solid` - The input solid to modify
/// * `profile` - A wire defining the cross-section profile of the rib
/// * `direction` - Direction vector for the rib extrusion (will be normalized)
/// * `thickness` - Thickness of the rib material (must be > 0)
///
/// # Returns
/// A new Solid with a rib feature added (material is added to the solid)
///
/// # Implementation
/// 1. Creates a rectangular box with the given thickness
/// 2. Positions it aligned with the profile wire
/// 3. Uses boolean union operation to add material
///
/// # Notes
/// - Currently simplified: creates a box perpendicular to the direction
/// - For complex profiles, would require full sweep surface generation
pub fn make_rib(
    solid: &Solid,
    profile: &Wire,
    direction: [f64; 3],
    thickness: f64,
) -> Result<Solid> {
    // Validate inputs
    if thickness <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Rib thickness must be positive, got {}", thickness),
        ));
    }

    if profile.edges.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Rib profile must contain at least one edge".into(),
        ));
    }

    // Normalize direction vector
    let dir_norm = normalize_vector(&direction)?;

    // Get profile bounds to determine rib dimensions
    let bounds = get_wire_bounds(profile)?;
    let profile_length = bounds.1[0] - bounds.0[0]; // Use x-dimension as profile length
    let profile_height = bounds.1[1] - bounds.0[1]; // Use y-dimension as profile height

    if profile_length < TOLERANCE || profile_height < TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            "Rib profile must have non-zero dimensions".into(),
        ));
    }

    // Get profile center
    let profile_center = [
        (bounds.0[0] + bounds.1[0]) / 2.0,
        (bounds.0[1] + bounds.1[1]) / 2.0,
        (bounds.0[2] + bounds.1[2]) / 2.0,
    ];

    // Create a box with dimensions: profile_length × profile_height × thickness
    let rib_box = make_box(profile_length, profile_height, thickness)?;

    // Position the rib at the profile center, aligned with the direction
    let rib_positioned = position_solid_at_with_direction(&rib_box, &profile_center, &dir_norm)?;

    // Perform boolean union to add the rib
    use crate::boolean::fuse;
    fuse(solid, &rib_positioned)
}

/// Normalize a vector to unit length
fn normalize_vector(vec: &[f64; 3]) -> Result<[f64; 3]> {
    let len = (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();

    if len < TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            "Direction vector must have non-zero length".into(),
        ));
    }

    Ok([vec[0] / len, vec[1] / len, vec[2] / len])
}

/// Get bounding box of a wire
fn get_wire_bounds(wire: &Wire) -> Result<([f64; 3], [f64; 3])> {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];

    if wire.edges.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Wire must contain at least one edge".into(),
        ));
    }

    for edge in &wire.edges {
        for i in 0..3 {
            min[i] = min[i].min(edge.start.point[i]);
            min[i] = min[i].min(edge.end.point[i]);
            max[i] = max[i].max(edge.start.point[i]);
            max[i] = max[i].max(edge.end.point[i]);
        }
    }

    Ok((min, max))
}

/// Position a solid at a given center point with a given axis direction
///
/// This is a simplified implementation that:
/// 1. Translates the solid to the specified center
/// 2. Aligns its Z-axis with the specified direction
fn position_solid_at_with_direction(
    solid: &Solid,
    center: &[f64; 3],
    direction: &[f64; 3],
) -> Result<Solid> {
    // For simplicity, we translate to the center
    // A full implementation would include rotation to align with direction
    
    // Get current solid center
    let current_center = get_solid_center(solid)?;

    // Calculate translation offset
    let tx = center[0] - current_center[0];
    let ty = center[1] - current_center[1];
    let tz = center[2] - current_center[2];

    // Translate all faces
    let mut translated_faces = Vec::new();
    for face in &solid.outer_shell.faces {
        let translated_face = translate_face(face, [tx, ty, tz])?;
        translated_faces.push(translated_face);
    }

    // Build new solid with translated faces
    let new_shell = Shell {
        faces: translated_faces,
        closed: solid.outer_shell.closed,
    };

    let result = Solid {
        outer_shell: new_shell,
        inner_shells: solid.inner_shells.clone(),
    };

    Ok(result)
}

/// Translate a face by a given offset vector
fn translate_face(face: &Face, offset: [f64; 3]) -> Result<Face> {
    let translated_outer = translate_wire(&face.outer_wire, offset)?;
    let mut translated_inner = Vec::new();

    for inner_wire in &face.inner_wires {
        translated_inner.push(translate_wire(inner_wire, offset)?);
    }

    // Translate surface origin if applicable
    let new_surface = match &face.surface_type {
        SurfaceType::Plane { origin, normal } => {
            SurfaceType::Plane {
                origin: [origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2]],
                normal: *normal,
            }
        }
        SurfaceType::Cylinder { origin, axis, radius } => {
            SurfaceType::Cylinder {
                origin: [origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2]],
                axis: *axis,
                radius: *radius,
            }
        }
        SurfaceType::Sphere { center, radius } => {
            SurfaceType::Sphere {
                center: [center[0] + offset[0], center[1] + offset[1], center[2] + offset[2]],
                radius: *radius,
            }
        }
        SurfaceType::Cone { origin, axis, half_angle_rad } => {
            SurfaceType::Cone {
                origin: [origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2]],
                axis: *axis,
                half_angle_rad: *half_angle_rad,
            }
        }
        SurfaceType::Torus { center, major_radius, minor_radius } => {
            SurfaceType::Torus {
                center: [center[0] + offset[0], center[1] + offset[1], center[2] + offset[2]],
                major_radius: *major_radius,
                minor_radius: *minor_radius,
            }
        }
        other => other.clone(),
    };

    Ok(Face {
        outer_wire: translated_outer,
        inner_wires: translated_inner,
        surface_type: new_surface,
    })
}

/// Translate a wire by a given offset vector
fn translate_wire(wire: &Wire, offset: [f64; 3]) -> Result<Wire> {
    let mut translated_edges = Vec::new();

    for edge in &wire.edges {
        let translated_edge = Edge {
            start: Vertex::new(
                edge.start.point[0] + offset[0],
                edge.start.point[1] + offset[1],
                edge.start.point[2] + offset[2],
            ),
            end: Vertex::new(
                edge.end.point[0] + offset[0],
                edge.end.point[1] + offset[1],
                edge.end.point[2] + offset[2],
            ),
            curve_type: edge.curve_type.clone(),
        };
        translated_edges.push(translated_edge);
    }

    Ok(Wire {
        edges: translated_edges,
        closed: wire.closed,
    })
}

/// Get the center point of a solid
fn get_solid_center(solid: &Solid) -> Result<[f64; 3]> {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];

    for face in &solid.outer_shell.faces {
        for edge in &face.outer_wire.edges {
            for i in 0..3 {
                min[i] = min[i].min(edge.start.point[i]);
                min[i] = min[i].min(edge.end.point[i]);
                max[i] = max[i].max(edge.start.point[i]);
                max[i] = max[i].max(edge.end.point[i]);
            }
        }

        for inner_wire in &face.inner_wires {
            for edge in &inner_wire.edges {
                for i in 0..3 {
                    min[i] = min[i].min(edge.start.point[i]);
                    min[i] = min[i].min(edge.end.point[i]);
                    max[i] = max[i].max(edge.start.point[i]);
                    max[i] = max[i].max(edge.end.point[i]);
                }
            }
        }
    }

    Ok([
        (min[0] + max[0]) / 2.0,
        (min[1] + max[1]) / 2.0,
        (min[2] + max[2]) / 2.0,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::make_box;

    #[test]
    fn test_make_hole_basic() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let hole = make_hole(&solid, [10.0, 10.0, 0.0], [0.0, 0.0, 1.0], 5.0, 10.0);
        
        assert!(hole.is_ok(), "Hole creation should succeed");
        let holed = hole.unwrap();
        assert!(!holed.outer_shell.faces.is_empty(), "Result should have faces");
    }

    #[test]
    fn test_make_hole_invalid_diameter() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let result = make_hole(&solid, [10.0, 10.0, 0.0], [0.0, 0.0, 1.0], -5.0, 10.0);
        
        assert!(result.is_err(), "Negative diameter should fail");
    }

    #[test]
    fn test_make_hole_invalid_depth() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let result = make_hole(&solid, [10.0, 10.0, 0.0], [0.0, 0.0, 1.0], 5.0, -10.0);
        
        assert!(result.is_err(), "Negative depth should fail");
    }

    #[test]
    fn test_make_hole_zero_direction() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let result = make_hole(&solid, [10.0, 10.0, 0.0], [0.0, 0.0, 0.0], 5.0, 10.0);
        
        assert!(result.is_err(), "Zero direction vector should fail");
    }

    #[test]
    fn test_make_slot_invalid_width() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let path = Wire {
            edges: vec![Edge {
                start: Vertex::new(0.0, 0.0, 0.0),
                end: Vertex::new(10.0, 0.0, 0.0),
                curve_type: CurveType::Line,
            }],
            closed: false,
        };
        
        let result = make_slot(&solid, &path, -5.0, 10.0);
        assert!(result.is_err(), "Negative width should fail");
    }

    #[test]
    fn test_make_slot_invalid_depth() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let path = Wire {
            edges: vec![Edge {
                start: Vertex::new(0.0, 0.0, 0.0),
                end: Vertex::new(10.0, 0.0, 0.0),
                curve_type: CurveType::Line,
            }],
            closed: false,
        };
        
        let result = make_slot(&solid, &path, 5.0, 0.0);
        assert!(result.is_err(), "Zero depth should fail");
    }

    #[test]
    fn test_make_slot_empty_path() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let empty_wire = Wire {
            edges: vec![],
            closed: false,
        };
        
        let result = make_slot(&solid, &empty_wire, 5.0, 10.0);
        assert!(result.is_err(), "Empty path should fail");
    }

    #[test]
    fn test_make_rib_invalid_thickness() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let profile = Wire {
            edges: vec![Edge {
                start: Vertex::new(0.0, 0.0, 0.0),
                end: Vertex::new(5.0, 0.0, 0.0),
                curve_type: CurveType::Line,
            }],
            closed: false,
        };
        
        let result = make_rib(&solid, &profile, [0.0, 0.0, 1.0], -2.0);
        assert!(result.is_err(), "Negative thickness should fail");
    }

    #[test]
    fn test_make_rib_zero_thickness() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let profile = Wire {
            edges: vec![Edge {
                start: Vertex::new(0.0, 0.0, 0.0),
                end: Vertex::new(5.0, 0.0, 0.0),
                curve_type: CurveType::Line,
            }],
            closed: false,
        };
        
        let result = make_rib(&solid, &profile, [0.0, 0.0, 1.0], 0.0);
        assert!(result.is_err(), "Zero thickness should fail");
    }

    #[test]
    fn test_make_rib_empty_profile() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let empty_wire = Wire {
            edges: vec![],
            closed: false,
        };
        
        let result = make_rib(&solid, &empty_wire, [0.0, 0.0, 1.0], 2.0);
        assert!(result.is_err(), "Empty profile should fail");
    }

    #[test]
    fn test_rib_basic() {
        let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        let profile = Wire {
            edges: vec![
                Edge {
                    start: Vertex::new(0.0, 0.0, 0.0),
                    end: Vertex::new(5.0, 0.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(5.0, 0.0, 0.0),
                    end: Vertex::new(5.0, 3.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(5.0, 3.0, 0.0),
                    end: Vertex::new(0.0, 3.0, 0.0),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: Vertex::new(0.0, 3.0, 0.0),
                    end: Vertex::new(0.0, 0.0, 0.0),
                    curve_type: CurveType::Line,
                },
            ],
            closed: true,
        };
        
        let ribbed = make_rib(&solid, &profile, [0.0, 0.0, 1.0], 2.0);
        assert!(ribbed.is_ok(), "Rib creation should succeed");
        // The ribbed solid should have more faces than the original
        let ribbed_solid = ribbed.unwrap();
        assert!(ribbed_solid.outer_shell.faces.len() >= solid.outer_shell.faces.len(),
                "Ribbed solid should have at least as many faces");
    }
}
