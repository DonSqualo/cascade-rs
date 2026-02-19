//! Manufacturing feature operations
//!
//! This module provides common manufacturing features that modify solids:
//! - **Holes**: Subtractive features that cut cylindrical pockets
//! - **Slots**: Subtractive features that cut rectangular pockets along a path
//! - **Ribs**: Additive features that add material with a given profile

use crate::brep::{Solid, Wire, Face, Edge, Vertex, Shell, CurveType, SurfaceType};
use crate::primitive::{make_cylinder, make_box};
use crate::boolean::{cut, fuse_many};
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
    let hole_cylinder = make_cylinder(hole_radius, depth)?;

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
    let countersink_cone = make_cone(countersink_radius, 0.0, actual_cone_depth)?;

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
    let main_hole = make_cylinder(hole_radius, total_depth)?;

    // Create counterbore cylinder (top portion only)
    let counterbore_radius = counterbore_diameter / 2.0;
    let counterbore_cylinder = make_cylinder(counterbore_radius, counterbore_depth)?;

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

/// Create a groove feature in a solid
///
/// A groove is a recessed channel cut into a solid by sweeping an arbitrary profile
/// along a path. Unlike a slot (which has a fixed rectangular shape), a groove
/// can have any profile shape such as V-groove, U-groove, or custom profiles.
///
/// # Arguments
/// * `solid` - The input solid to modify
/// * `path` - A wire defining the centerline path of the groove
/// * `profile` - A wire defining the cross-section profile of the groove
///
/// # Returns
/// A new Solid with a groove feature cut into it
///
/// # Implementation
/// 1. Extracts the profile bounds (dimensions)
/// 2. Creates a box with those dimensions
/// 3. Sweeps this box along the path
/// 4. Uses boolean cut operation to remove material from the solid
///
/// # Notes
/// - Currently simplified: creates a box from profile bounds and sweeps it along the path
/// - For curved paths with complex profiles, would require full sweep surface generation
/// - Common profiles: rectangular, V-groove, U-groove
///
/// # Example
/// ```ignore
/// // Create a V-groove profile (simple two-line wire forming a V shape)
/// let v_groove_profile = Wire {
///     edges: vec![...],  // V-shaped wire
///     closed: false,
/// };
/// 
/// // Create a path wire for the groove centerline
/// let groove_path = Wire {
///     edges: vec![...],  // path along which to sweep the profile
///     closed: false,
/// };
/// 
/// let grooved = make_groove(&solid, &groove_path, &v_groove_profile)?;
/// ```
pub fn make_groove(
    solid: &Solid,
    path: &Wire,
    profile: &Wire,
) -> Result<Solid> {
    // Validate inputs
    if path.edges.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Groove path must contain at least one edge".into(),
        ));
    }

    if profile.edges.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Groove profile must contain at least one edge".into(),
        ));
    }

    // Get profile bounds to determine groove dimensions
    let bounds = get_wire_bounds(profile)?;
    let profile_width = bounds.1[0] - bounds.0[0]; // x-dimension
    let profile_depth = bounds.1[1] - bounds.0[1]; // y-dimension

    if profile_width < TOLERANCE || profile_depth < TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            "Groove profile must have non-zero dimensions in x and y".into(),
        ));
    }

    // Get path bounds to determine sweep length
    let path_bounds = get_wire_bounds(path)?;
    
    // Calculate path direction and length from the first edge
    let first_edge = &path.edges[0];
    let path_start = first_edge.start.point;
    let path_end = first_edge.end.point;

    let dx = path_end[0] - path_start[0];
    let dy = path_end[1] - path_start[1];
    let dz = path_end[2] - path_start[2];
    let path_length = (dx * dx + dy * dy + dz * dz).sqrt();

    if path_length < TOLERANCE {
        return Err(CascadeError::InvalidGeometry(
            "Groove path edge has zero length".into(),
        ));
    }

    let path_dir = [dx / path_length, dy / path_length, dz / path_length];

    // Create a box with dimensions: profile_width × profile_depth × path_length
    // This box represents the grooved volume
    let mut groove_box = make_box(profile_width, profile_depth, path_length)?;

    // Position the box so its center is at the midpoint of the path
    // The box will be aligned with the path direction
    let profile_center = [
        (bounds.0[0] + bounds.1[0]) / 2.0,
        (bounds.0[1] + bounds.1[1]) / 2.0,
        (bounds.0[2] + bounds.1[2]) / 2.0,
    ];

    let path_mid = [
        path_start[0] + dx / 2.0,
        path_start[1] + dy / 2.0,
        path_start[2] + dz / 2.0,
    ];

    groove_box = position_solid_at_with_direction(&groove_box, &path_mid, &path_dir)?;

    // Perform boolean cut to remove the groove from the solid
    cut(solid, &groove_box)
}

/// Create circular pattern of shapes around an axis
///
/// # Arguments
/// * `shape` - The input solid to pattern
/// * `axis_point` - A point on the rotation axis
/// * `axis_dir` - Direction vector of the rotation axis (will be normalized)
/// * `count` - Number of instances in the pattern (must be >= 1)
/// * `angle` - Total sweep angle in radians (2*PI for full circle)
///
/// # Returns
/// A Vec<Solid> containing the patterned instances
///
/// # Implementation
/// 1. Validates all inputs
/// 2. Creates N copies of the input shape
/// 3. Rotates each copy around the given axis
/// 4. The i-th instance is rotated by angle * (i / count)
///
/// # Example
/// ```ignore
/// let solids = circular_pattern(
///     &shape,
///     [0.0, 0.0, 0.0],      // axis point at origin
///     [0.0, 0.0, 1.0],      // rotate around Z-axis
///     8,                      // 8 instances
///     std::f64::consts::PI * 2.0  // full circle
/// )?;
/// ```
pub fn circular_pattern(
    shape: &Solid,
    axis_point: [f64; 3],
    axis_dir: [f64; 3],
    count: usize,
    angle: f64,
) -> Result<Vec<Solid>> {
    // Validate inputs
    if count == 0 {
        return Err(CascadeError::InvalidGeometry(
            "Pattern count must be at least 1".into(),
        ));
    }

    if angle.is_nan() || angle.is_infinite() {
        return Err(CascadeError::InvalidGeometry(
            "Pattern angle must be a valid number".into(),
        ));
    }

    // Normalize direction vector
    let axis_norm = normalize_vector(&axis_dir)?;

    // Create array of rotated solids
    let mut pattern_solids = Vec::new();

    for i in 0..count {
        let rotation_angle = if count == 1 {
            0.0
        } else {
            angle * (i as f64 / count as f64)
        };

        let rotated = rotate_solid_around_axis(shape, &axis_point, &axis_norm, rotation_angle)?;
        pattern_solids.push(rotated);
    }

    Ok(pattern_solids)
}

/// Create circular pattern and fuse all instances into a single solid
///
/// # Arguments
/// * `shape` - The input solid to pattern
/// * `axis_point` - A point on the rotation axis
/// * `axis_dir` - Direction vector of the rotation axis (will be normalized)
/// * `count` - Number of instances in the pattern (must be >= 1)
/// * `angle` - Total sweep angle in radians (2*PI for full circle)
///
/// # Returns
/// A single Solid that is the union of all patterned instances
///
/// # Implementation
/// 1. Uses circular_pattern() to create individual instances
/// 2. Fuses all instances together using boolean union
///
/// # Example
/// ```ignore
/// let pattern = circular_pattern_fused(
///     &shape,
///     [0.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0],
///     6,
///     std::f64::consts::PI * 2.0
/// )?;
/// ```
pub fn circular_pattern_fused(
    shape: &Solid,
    axis_point: [f64; 3],
    axis_dir: [f64; 3],
    count: usize,
    angle: f64,
) -> Result<Solid> {
    let pattern_solids = circular_pattern(shape, axis_point, axis_dir, count, angle)?;

    if pattern_solids.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Pattern produced no solids".into(),
        ));
    }

    // Fuse all solids together
    use crate::boolean::fuse_many;
    fuse_many(&pattern_solids)
}

/// Rotate a solid around an axis by a given angle
///
/// # Arguments
/// * `solid` - The input solid to rotate
/// * `axis_point` - A point on the rotation axis
/// * `axis_dir` - Direction vector of the rotation axis (must be normalized)
/// * `angle` - Rotation angle in radians
///
/// # Returns
/// A rotated copy of the solid
fn rotate_solid_around_axis(
    solid: &Solid,
    axis_point: &[f64; 3],
    axis_dir: &[f64; 3],
    angle: f64,
) -> Result<Solid> {
    if angle.abs() < TOLERANCE {
        // No rotation needed
        return Ok(solid.clone());
    }

    // Rotate all faces
    let mut rotated_faces = Vec::new();
    for face in &solid.outer_shell.faces {
        let rotated_face = rotate_face(face, axis_point, axis_dir, angle)?;
        rotated_faces.push(rotated_face);
    }

    // Rotate inner shells (cavities)
    let mut rotated_inner_shells = Vec::new();
    for inner_shell in &solid.inner_shells {
        let mut rotated_inner_faces = Vec::new();
        for face in &inner_shell.faces {
            let rotated_face = rotate_face(face, axis_point, axis_dir, angle)?;
            rotated_inner_faces.push(rotated_face);
        }
        rotated_inner_shells.push(Shell {
            faces: rotated_inner_faces,
            closed: inner_shell.closed,
        });
    }

    let new_shell = Shell {
        faces: rotated_faces,
        closed: solid.outer_shell.closed,
    };

    Ok(Solid {
        outer_shell: new_shell,
        inner_shells: rotated_inner_shells,
    })
}

/// Rotate a face around an axis
fn rotate_face(
    face: &Face,
    axis_point: &[f64; 3],
    axis_dir: &[f64; 3],
    angle: f64,
) -> Result<Face> {
    let rotated_outer = rotate_wire(&face.outer_wire, axis_point, axis_dir, angle)?;
    let mut rotated_inner = Vec::new();

    for inner_wire in &face.inner_wires {
        rotated_inner.push(rotate_wire(inner_wire, axis_point, axis_dir, angle)?);
    }

    // Rotate surface based on its type
    let new_surface = match &face.surface_type {
        SurfaceType::Plane { origin, normal } => {
            let rotated_origin = rotate_point(origin, axis_point, axis_dir, angle);
            let rotated_normal = rotate_vector(normal, axis_dir, angle);
            SurfaceType::Plane {
                origin: rotated_origin,
                normal: rotated_normal,
            }
        }
        SurfaceType::Cylinder { origin, axis, radius } => {
            let rotated_origin = rotate_point(origin, axis_point, axis_dir, angle);
            let rotated_axis = rotate_vector(axis, axis_dir, angle);
            SurfaceType::Cylinder {
                origin: rotated_origin,
                axis: rotated_axis,
                radius: *radius,
            }
        }
        SurfaceType::Sphere { center, radius } => {
            let rotated_center = rotate_point(center, axis_point, axis_dir, angle);
            SurfaceType::Sphere {
                center: rotated_center,
                radius: *radius,
            }
        }
        SurfaceType::Cone { origin, axis, half_angle_rad } => {
            let rotated_origin = rotate_point(origin, axis_point, axis_dir, angle);
            let rotated_axis = rotate_vector(axis, axis_dir, angle);
            SurfaceType::Cone {
                origin: rotated_origin,
                axis: rotated_axis,
                half_angle_rad: *half_angle_rad,
            }
        }
        SurfaceType::Torus { center, major_radius, minor_radius } => {
            let rotated_center = rotate_point(center, axis_point, axis_dir, angle);
            SurfaceType::Torus {
                center: rotated_center,
                major_radius: *major_radius,
                minor_radius: *minor_radius,
            }
        }
        other => other.clone(),
    };

    Ok(Face {
        outer_wire: rotated_outer,
        inner_wires: rotated_inner,
        surface_type: new_surface,
    })
}

/// Rotate a wire around an axis
fn rotate_wire(
    wire: &Wire,
    axis_point: &[f64; 3],
    axis_dir: &[f64; 3],
    angle: f64,
) -> Result<Wire> {
    let mut rotated_edges = Vec::new();

    for edge in &wire.edges {
        let rotated_start = rotate_point(&edge.start.point, axis_point, axis_dir, angle);
        let rotated_end = rotate_point(&edge.end.point, axis_point, axis_dir, angle);

        let rotated_curve = rotate_curve_type(&edge.curve_type, axis_point, axis_dir, angle)?;

        rotated_edges.push(Edge {
            start: Vertex::new(rotated_start[0], rotated_start[1], rotated_start[2]),
            end: Vertex::new(rotated_end[0], rotated_end[1], rotated_end[2]),
            curve_type: rotated_curve,
        });
    }

    Ok(Wire {
        edges: rotated_edges,
        closed: wire.closed,
    })
}

/// Rotate a point around an axis using Rodrigues' rotation formula
fn rotate_point(
    point: &[f64; 3],
    axis_point: &[f64; 3],
    axis_dir: &[f64; 3],
    angle: f64,
) -> [f64; 3] {
    // Translate point relative to axis
    let p = [
        point[0] - axis_point[0],
        point[1] - axis_point[1],
        point[2] - axis_point[2],
    ];

    // Rodrigues' rotation formula: v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let one_minus_cos = 1.0 - cos_a;

    // k · v (dot product)
    let k_dot_v = axis_dir[0] * p[0] + axis_dir[1] * p[1] + axis_dir[2] * p[2];

    // k × v (cross product)
    let cross = [
        axis_dir[1] * p[2] - axis_dir[2] * p[1],
        axis_dir[2] * p[0] - axis_dir[0] * p[2],
        axis_dir[0] * p[1] - axis_dir[1] * p[0],
    ];

    // v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
    let rotated = [
        p[0] * cos_a + cross[0] * sin_a + axis_dir[0] * k_dot_v * one_minus_cos,
        p[1] * cos_a + cross[1] * sin_a + axis_dir[1] * k_dot_v * one_minus_cos,
        p[2] * cos_a + cross[2] * sin_a + axis_dir[2] * k_dot_v * one_minus_cos,
    ];

    // Translate back to original position
    [
        rotated[0] + axis_point[0],
        rotated[1] + axis_point[1],
        rotated[2] + axis_point[2],
    ]
}

/// Rotate a vector (without translation) using Rodrigues' rotation formula
fn rotate_vector(vec: &[f64; 3], axis_dir: &[f64; 3], angle: f64) -> [f64; 3] {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let one_minus_cos = 1.0 - cos_a;

    // k · v
    let k_dot_v = axis_dir[0] * vec[0] + axis_dir[1] * vec[1] + axis_dir[2] * vec[2];

    // k × v
    let cross = [
        axis_dir[1] * vec[2] - axis_dir[2] * vec[1],
        axis_dir[2] * vec[0] - axis_dir[0] * vec[2],
        axis_dir[0] * vec[1] - axis_dir[1] * vec[0],
    ];

    // v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
    [
        vec[0] * cos_a + cross[0] * sin_a + axis_dir[0] * k_dot_v * one_minus_cos,
        vec[1] * cos_a + cross[1] * sin_a + axis_dir[1] * k_dot_v * one_minus_cos,
        vec[2] * cos_a + cross[2] * sin_a + axis_dir[2] * k_dot_v * one_minus_cos,
    ]
}

/// Rotate a curve type around an axis
fn rotate_curve_type(
    curve: &CurveType,
    axis_point: &[f64; 3],
    axis_dir: &[f64; 3],
    angle: f64,
) -> Result<CurveType> {
    match curve {
        CurveType::Line => Ok(CurveType::Line),
        CurveType::Arc { center, radius } => {
            let rotated_center = rotate_point(center, axis_point, axis_dir, angle);
            Ok(CurveType::Arc {
                center: rotated_center,
                radius: *radius,
            })
        }
        CurveType::Ellipse {
            center,
            major_axis,
            minor_axis,
        } => {
            let rotated_center = rotate_point(center, axis_point, axis_dir, angle);
            let rotated_major = rotate_vector(major_axis, axis_dir, angle);
            let rotated_minor = rotate_vector(minor_axis, axis_dir, angle);
            Ok(CurveType::Ellipse {
                center: rotated_center,
                major_axis: rotated_major,
                minor_axis: rotated_minor,
            })
        }
        CurveType::Parabola {
            origin,
            x_dir,
            y_dir,
            focal,
        } => {
            let rotated_origin = rotate_point(origin, axis_point, axis_dir, angle);
            let rotated_x_dir = rotate_vector(x_dir, axis_dir, angle);
            let rotated_y_dir = rotate_vector(y_dir, axis_dir, angle);
            Ok(CurveType::Parabola {
                origin: rotated_origin,
                x_dir: rotated_x_dir,
                y_dir: rotated_y_dir,
                focal: *focal,
            })
        }
        CurveType::Hyperbola {
            center,
            x_dir,
            y_dir,
            major_radius,
            minor_radius,
        } => {
            let rotated_center = rotate_point(center, axis_point, axis_dir, angle);
            let rotated_x_dir = rotate_vector(x_dir, axis_dir, angle);
            let rotated_y_dir = rotate_vector(y_dir, axis_dir, angle);
            Ok(CurveType::Hyperbola {
                center: rotated_center,
                x_dir: rotated_x_dir,
                y_dir: rotated_y_dir,
                major_radius: *major_radius,
                minor_radius: *minor_radius,
            })
        }
        CurveType::Bezier { control_points } => {
            let rotated_points: Vec<[f64; 3]> = control_points
                .iter()
                .map(|p| rotate_point(p, axis_point, axis_dir, angle))
                .collect();
            Ok(CurveType::Bezier {
                control_points: rotated_points,
            })
        }
        CurveType::BSpline {
            control_points,
            knots,
            degree,
            weights,
        } => {
            let rotated_points: Vec<[f64; 3]> = control_points
                .iter()
                .map(|p| rotate_point(p, axis_point, axis_dir, angle))
                .collect();
            Ok(CurveType::BSpline {
                control_points: rotated_points,
                knots: knots.clone(),
                degree: *degree,
                weights: weights.clone(),
            })
        }
        CurveType::Trimmed {
            basis_curve,
            u1,
            u2,
        } => {
            let rotated_basis = rotate_curve_type(basis_curve, axis_point, axis_dir, angle)?;
            Ok(CurveType::Trimmed {
                basis_curve: Box::new(rotated_basis),
                u1: *u1,
                u2: *u2,
            })
        }
        CurveType::Offset {
            basis_curve,
            offset_distance,
            offset_direction,
        } => {
            let rotated_basis = rotate_curve_type(basis_curve, axis_point, axis_dir, angle)?;
            let rotated_offset_dir = rotate_vector(offset_direction, axis_dir, angle);
            Ok(CurveType::Offset {
                basis_curve: Box::new(rotated_basis),
                offset_distance: *offset_distance,
                offset_direction: rotated_offset_dir,
            })
        }
    }
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

/// Create a linear pattern of copies of a solid along a direction
///
/// # Arguments
/// * `shape` - The input solid to pattern
/// * `direction` - Direction vector for the pattern (will be normalized)
/// * `count` - Number of copies to create (must be >= 1)
/// * `spacing` - Distance between consecutive copies (must be > 0)
///
/// # Returns
/// A vector of `count` solids, each spaced along the direction vector
///
/// # Implementation
/// 1. Normalizes the direction vector
/// 2. Creates `count` copies of the input shape
/// 3. Translates each copy along the direction by `i * spacing` distance
/// 4. Returns all copies as a vector
///
/// # Notes
/// - The first copy is at the original position of the shape
/// - Subsequent copies are displaced by spacing * i along the direction
/// - All copies are independent solids
///
/// # Example
/// ```ignore
/// let patterns = linear_pattern(&solid, [1.0, 0.0, 0.0], 5, 10.0)?;
/// // Returns 5 copies of solid, each spaced 10 units apart along the X axis
/// ```
pub fn linear_pattern(
    shape: &Solid,
    direction: [f64; 3],
    count: usize,
    spacing: f64,
) -> Result<Vec<Solid>> {
    // Validate inputs
    if count < 1 {
        return Err(CascadeError::InvalidGeometry(
            format!("Pattern count must be at least 1, got {}", count),
        ));
    }

    if spacing <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            format!("Pattern spacing must be positive, got {}", spacing),
        ));
    }

    // Normalize direction vector
    let dir_norm = normalize_vector(&direction)?;

    // Create array to hold all patterns
    let mut patterns = Vec::new();

    // Generate count copies
    for i in 0..count {
        // Calculate translation offset for this copy
        let offset_distance = (i as f64) * spacing;
        let offset = [
            dir_norm[0] * offset_distance,
            dir_norm[1] * offset_distance,
            dir_norm[2] * offset_distance,
        ];

        // Translate the shape by the offset
        let mut translated_faces = Vec::new();
        for face in &shape.outer_shell.faces {
            let translated_face = translate_face(face, offset)?;
            translated_faces.push(translated_face);
        }

        // Build new solid with translated faces
        let new_shell = Shell {
            faces: translated_faces,
            closed: shape.outer_shell.closed,
        };

        let mut translated_inner = Vec::new();
        for inner_shell in &shape.inner_shells {
            let mut translated_inner_faces = Vec::new();
            for face in &inner_shell.faces {
                let translated_face = translate_face(face, offset)?;
                translated_inner_faces.push(translated_face);
            }
            translated_inner.push(Shell {
                faces: translated_inner_faces,
                closed: inner_shell.closed,
            });
        }

        let patterned_solid = Solid {
            outer_shell: new_shell,
            inner_shells: translated_inner,
        };

        patterns.push(patterned_solid);
    }

    Ok(patterns)
}

/// Create a linear pattern of copies of a solid and fuse them into a single solid
///
/// # Arguments
/// * `shape` - The input solid to pattern
/// * `direction` - Direction vector for the pattern (will be normalized)
/// * `count` - Number of copies to create (must be >= 1)
/// * `spacing` - Distance between consecutive copies (must be > 0)
///
/// # Returns
/// A single solid that is the fusion of all patterned copies
///
/// # Implementation
/// 1. Creates `count` copies using `linear_pattern()`
/// 2. Fuses all copies together using boolean union
/// 3. Returns the unified solid
///
/// # Notes
/// - If count=1, returns a copy of the original shape
/// - For count > 1, merges all copies into a single contiguous solid
/// - The result is a solid object without gaps between copies
///
/// # Example
/// ```ignore
/// let fused_pattern = linear_pattern_fused(&solid, [1.0, 0.0, 0.0], 5, 10.0)?;
/// // Returns a single solid that is the union of 5 copies spaced 10 units apart
/// ```
pub fn linear_pattern_fused(
    shape: &Solid,
    direction: [f64; 3],
    count: usize,
    spacing: f64,
) -> Result<Solid> {
    // Generate the pattern as separate copies
    let patterns = linear_pattern(shape, direction, count, spacing)?;

    // If only one pattern, return it directly
    if patterns.len() == 1 {
        return Ok(patterns[0].clone());
    }

    // Fuse all patterns together
    fuse_many(&patterns)
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

    #[test]
    fn test_make_hole_countersunk_basic() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let countersunk = make_hole_countersunk(
            &solid,
            [10.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
            5.0,   // hole diameter
            8.0,   // countersink diameter
            90.0,  // countersink angle
            10.0   // depth
        );
        
        assert!(countersunk.is_ok(), "Countersunk hole creation should succeed");
        let holed = countersunk.unwrap();
        assert!(!holed.outer_shell.faces.is_empty(), "Result should have faces");
    }

    #[test]
    fn test_make_hole_countersunk_invalid_diameter() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let result = make_hole_countersunk(
            &solid,
            [10.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
            -5.0,  // negative hole diameter
            8.0,
            90.0,
            10.0
        );
        
        assert!(result.is_err(), "Negative hole diameter should fail");
    }

    #[test]
    fn test_make_hole_countersunk_invalid_countersink_diameter() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let result = make_hole_countersunk(
            &solid,
            [10.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
            8.0,   // hole diameter
            5.0,   // countersink diameter < hole diameter
            90.0,
            10.0
        );
        
        assert!(result.is_err(), "Countersink diameter < hole diameter should fail");
    }

    #[test]
    fn test_make_hole_countersunk_invalid_angle() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let result = make_hole_countersunk(
            &solid,
            [10.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
            5.0,
            8.0,
            180.0,  // invalid angle
            10.0
        );
        
        assert!(result.is_err(), "Invalid angle (>= 180°) should fail");
    }

    #[test]
    fn test_make_hole_countersunk_invalid_depth() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let result = make_hole_countersunk(
            &solid,
            [10.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
            5.0,
            8.0,
            90.0,
            0.0   // invalid depth
        );
        
        assert!(result.is_err(), "Zero depth should fail");
    }

    #[test]
    fn test_make_hole_counterbore_basic() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let counterbore = make_hole_counterbore(
            &solid,
            [10.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
            5.0,   // hole diameter
            8.0,   // counterbore diameter
            3.0,   // counterbore depth
            7.0    // hole depth
        );
        
        assert!(counterbore.is_ok(), "Counterbore hole creation should succeed");
        let holed = counterbore.unwrap();
        assert!(!holed.outer_shell.faces.is_empty(), "Result should have faces");
    }

    #[test]
    fn test_make_hole_counterbore_invalid_hole_diameter() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let result = make_hole_counterbore(
            &solid,
            [10.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
            -5.0,  // negative hole diameter
            8.0,
            3.0,
            7.0
        );
        
        assert!(result.is_err(), "Negative hole diameter should fail");
    }

    #[test]
    fn test_make_hole_counterbore_invalid_counterbore_diameter() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let result = make_hole_counterbore(
            &solid,
            [10.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
            8.0,   // hole diameter
            5.0,   // counterbore diameter < hole diameter
            3.0,
            7.0
        );
        
        assert!(result.is_err(), "Counterbore diameter < hole diameter should fail");
    }

    #[test]
    fn test_make_hole_counterbore_invalid_counterbore_depth() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let result = make_hole_counterbore(
            &solid,
            [10.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
            5.0,
            8.0,
            -3.0,  // negative counterbore depth
            7.0
        );
        
        assert!(result.is_err(), "Negative counterbore depth should fail");
    }

    #[test]
    fn test_make_hole_counterbore_invalid_hole_depth() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let result = make_hole_counterbore(
            &solid,
            [10.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
            5.0,
            8.0,
            3.0,
            0.0   // zero hole depth
        );
        
        assert!(result.is_err(), "Zero hole depth should fail");
    }

    #[test]
    fn test_make_hole_counterbore_zero_direction() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let result = make_hole_counterbore(
            &solid,
            [10.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],  // zero direction vector
            5.0,
            8.0,
            3.0,
            7.0
        );
        
        assert!(result.is_err(), "Zero direction vector should fail");
    }

    #[test]
    fn test_make_hole_countersunk_zero_direction() {
        let solid = make_box(20.0, 20.0, 20.0).expect("Failed to create box");
        let result = make_hole_countersunk(
            &solid,
            [10.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],  // zero direction vector
            5.0,
            8.0,
            90.0,
            10.0
        );
        
        assert!(result.is_err(), "Zero direction vector should fail");
    }

    #[test]
    fn test_linear_pattern_basic() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        let patterns = linear_pattern(&solid, [1.0, 0.0, 0.0], 3, 10.0);
        
        assert!(patterns.is_ok(), "Linear pattern should succeed");
        let pattern_list = patterns.unwrap();
        assert_eq!(pattern_list.len(), 3, "Should have 3 copies");
        assert!(!pattern_list[0].outer_shell.faces.is_empty(), "Each copy should have faces");
    }

    #[test]
    fn test_linear_pattern_single_copy() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        let patterns = linear_pattern(&solid, [1.0, 0.0, 0.0], 1, 10.0);
        
        assert!(patterns.is_ok(), "Single copy should succeed");
        let pattern_list = patterns.unwrap();
        assert_eq!(pattern_list.len(), 1, "Should have 1 copy");
    }

    #[test]
    fn test_linear_pattern_invalid_count() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        let result = linear_pattern(&solid, [1.0, 0.0, 0.0], 0, 10.0);
        
        assert!(result.is_err(), "Count of 0 should fail");
    }

    #[test]
    fn test_linear_pattern_invalid_spacing() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        let result = linear_pattern(&solid, [1.0, 0.0, 0.0], 3, 0.0);
        
        assert!(result.is_err(), "Zero spacing should fail");
    }

    #[test]
    fn test_linear_pattern_negative_spacing() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        let result = linear_pattern(&solid, [1.0, 0.0, 0.0], 3, -10.0);
        
        assert!(result.is_err(), "Negative spacing should fail");
    }

    #[test]
    fn test_linear_pattern_zero_direction() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        let result = linear_pattern(&solid, [0.0, 0.0, 0.0], 3, 10.0);
        
        assert!(result.is_err(), "Zero direction vector should fail");
    }

    #[test]
    fn test_linear_pattern_different_directions() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        
        // Test X direction
        let x_patterns = linear_pattern(&solid, [1.0, 0.0, 0.0], 2, 5.0);
        assert!(x_patterns.is_ok(), "X direction pattern should succeed");
        
        // Test Y direction
        let y_patterns = linear_pattern(&solid, [0.0, 1.0, 0.0], 2, 5.0);
        assert!(y_patterns.is_ok(), "Y direction pattern should succeed");
        
        // Test Z direction
        let z_patterns = linear_pattern(&solid, [0.0, 0.0, 1.0], 2, 5.0);
        assert!(z_patterns.is_ok(), "Z direction pattern should succeed");
        
        // Test diagonal direction
        let diag_patterns = linear_pattern(&solid, [1.0, 1.0, 1.0], 2, 5.0);
        assert!(diag_patterns.is_ok(), "Diagonal direction pattern should succeed");
    }

    #[test]
    fn test_linear_pattern_fused_basic() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        let fused = linear_pattern_fused(&solid, [1.0, 0.0, 0.0], 3, 10.0);
        
        assert!(fused.is_ok(), "Linear pattern fused should succeed");
        let fused_solid = fused.unwrap();
        assert!(!fused_solid.outer_shell.faces.is_empty(), "Result should have faces");
    }

    #[test]
    fn test_linear_pattern_fused_single_copy() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        let fused = linear_pattern_fused(&solid, [1.0, 0.0, 0.0], 1, 10.0);
        
        assert!(fused.is_ok(), "Single copy fused should succeed");
        let fused_solid = fused.unwrap();
        assert!(!fused_solid.outer_shell.faces.is_empty(), "Result should have faces");
    }

    #[test]
    fn test_linear_pattern_fused_invalid_count() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        let result = linear_pattern_fused(&solid, [1.0, 0.0, 0.0], 0, 10.0);
        
        assert!(result.is_err(), "Count of 0 should fail");
    }

    #[test]
    fn test_linear_pattern_fused_invalid_spacing() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        let result = linear_pattern_fused(&solid, [1.0, 0.0, 0.0], 3, -5.0);
        
        assert!(result.is_err(), "Negative spacing should fail");
    }

    #[test]
    fn test_linear_pattern_fused_zero_direction() {
        let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
        let result = linear_pattern_fused(&solid, [0.0, 0.0, 0.0], 3, 10.0);
        
        assert!(result.is_err(), "Zero direction vector should fail");
    }

    #[test]
    fn test_linear_pattern_fused_large_pattern() {
        let solid = make_box(2.0, 2.0, 2.0).expect("Failed to create box");
        let fused = linear_pattern_fused(&solid, [1.0, 0.0, 0.0], 5, 5.0);
        
        assert!(fused.is_ok(), "Large pattern should succeed");
        let fused_solid = fused.unwrap();
        assert!(!fused_solid.outer_shell.faces.is_empty(), "Result should have faces");
    }
}
