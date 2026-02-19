//! Sweep and extrusion operations
//!
//! These functions create complex shapes by sweeping 2D profiles along curves.

use crate::brep::{Vertex, Edge, Wire, Face, Solid, Shell, CurveType, SurfaceType};
use crate::curve;
use crate::{Result, CascadeError};

/// Linear extrusion - sweeps a 2D profile along a straight line
///
/// # Arguments
/// * `profile` - A 2D face to extrude
/// * `direction` - Direction vector for extrusion
/// * `distance` - Distance to extrude
///
/// # Returns
/// A Solid representing the extruded shape
pub fn make_prism(profile: &Face, direction: &[f64; 3], distance: f64) -> Result<Solid> {
    if distance <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Extrusion distance must be positive".into()
        ));
    }

    let dir_norm = normalize(direction);
    
    // Create offset profile at the end of extrusion
    let offset = [
        dir_norm[0] * distance,
        dir_norm[1] * distance,
        dir_norm[2] * distance,
    ];
    
    let base_face = profile.clone();
    let top_face = translate_face(&profile, &offset)?;
    
    // Create side faces connecting the two profiles
    let mut side_faces = vec![];
    
    // Connect outer wire edges
    for i in 0..profile.outer_wire.edges.len() {
        let edge = &profile.outer_wire.edges[i];
        let next_idx = (i + 1) % profile.outer_wire.edges.len();
        let _next_edge = &profile.outer_wire.edges[next_idx];
        
        // Create 4 vertices for a side face
        let v0 = edge.start.clone();
        let v1 = edge.end.clone();
        let v2 = Vertex::new(
            edge.end.point[0] + offset[0],
            edge.end.point[1] + offset[1],
            edge.end.point[2] + offset[2],
        );
        let v3 = Vertex::new(
            edge.start.point[0] + offset[0],
            edge.start.point[1] + offset[1],
            edge.start.point[2] + offset[2],
        );
        
        // Create the side face wire (quad face as two triangles worth of edges)
        let side_wire = Wire {
            edges: vec![
                Edge {
                    start: v0.clone(),
                    end: v1.clone(),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: v1.clone(),
                    end: v2.clone(),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: v2.clone(),
                    end: v3.clone(),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: v3.clone(),
                    end: v0.clone(),
                    curve_type: CurveType::Line,
                },
            ],
            closed: true,
        };
        
        // Approximate as a plane for now (simplified)
        let side_face = Face {
            outer_wire: side_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: v0.point,
                normal: compute_quad_normal(&v0.point, &v1.point, &v2.point, &v3.point),
            },
        };
        
        side_faces.push(side_face);
    }
    
    // TODO: Handle inner wires (holes)
    
    let mut faces = vec![base_face, top_face];
    faces.extend(side_faces);
    
    let shell = Shell { faces, closed: true };
    Ok(Solid {
        outer_shell: shell,
        inner_shells: vec![],
    })
}

/// Path sweep - sweeps a 2D profile along a 3D curve
///
/// This is a key CAD operation for creating tubes, rails, and other swept shapes.
/// Uses the Frenet frame (moving frame) to orient the profile along the path.
///
/// # Arguments
/// * `profile` - A 2D face to sweep (should be perpendicular to initial path direction)
/// * `path` - A 3D wire representing the path curve
///
/// # Returns
/// A Solid representing the swept shape
///
/// # Algorithm
/// 1. Sample the path at regular intervals
/// 2. At each point, compute the Frenet frame (T, N, B)
/// 3. Transform and position the profile at each frame
/// 4. Create side faces connecting consecutive profiles
/// 5. Add end caps (start and end profiles)
pub fn make_pipe(profile: &Face, path: &Wire) -> Result<Solid> {
    if path.edges.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Path wire must have at least one edge".into()
        ));
    }
    
    if !path.closed {
        // Open paths are OK for pipes
    }
    
    // Number of samples along the path
    let num_samples = 20;
    
    // Collect points and frames along the path
    let mut path_points = vec![];
    let mut frames = vec![];
    
    // For simplicity, we'll work with the first edge of the path
    // A full implementation would concatenate all edges
    let edge = &path.edges[0];
    
    // Sample the edge
    for i in 0..=num_samples {
        let t = i as f64 / num_samples as f64;
        
        // Evaluate point on path
        let point = evaluate_edge_at(&edge, t)?;
        path_points.push(point);
        
        // Compute tangent at this point
        let tangent = compute_edge_tangent(&edge, t)?;
        
        // Compute Frenet frame at this point
        let frame = if i == 0 {
            // First frame: use initial normal
            compute_frenet_frame(&tangent, None)
        } else {
            // Subsequent frames: parallel transport from previous frame
            compute_frenet_frame(&tangent, Some(&frames[i - 1]))
        };
        
        frames.push(frame);
    }
    
    // Create cross-sectional profiles at each path point
    let mut profile_sections = vec![];
    
    for i in 0..path_points.len() {
        let point = path_points[i];
        let frame = &frames[i];
        
        // Transform profile to this location
        let section = transform_profile_to_frame(&profile, &point, frame)?;
        profile_sections.push(section);
    }
    
    // Create side faces connecting consecutive sections
    let mut side_faces = vec![];
    
    for i in 0..profile_sections.len() - 1 {
        let current_section = &profile_sections[i];
        let next_section = &profile_sections[i + 1];
        
        // Create faces by connecting edges of consecutive profiles
        let section_faces = create_sweep_faces(current_section, next_section)?;
        side_faces.extend(section_faces);
    }
    
    // Add end caps (start and end profiles)
    let mut all_faces = vec![];
    
    // Start cap (flip normal to point outward)
    let mut start_cap = profile_sections[0].clone();
    start_cap = flip_face(&start_cap);
    all_faces.push(start_cap);
    
    // Side faces
    all_faces.extend(side_faces);
    
    // End cap
    all_faces.push(profile_sections[profile_sections.len() - 1].clone());
    
    let shell = Shell { faces: all_faces, closed: true };
    Ok(Solid {
        outer_shell: shell,
        inner_shells: vec![],
    })
}

/// Helper: Evaluate a point on an edge at parameter t ∈ [0, 1]
fn evaluate_edge_at(edge: &Edge, t: f64) -> Result<[f64; 3]> {
    match &edge.curve_type {
        CurveType::Line => {
            // Linear interpolation
            let p = [
                edge.start.point[0] * (1.0 - t) + edge.end.point[0] * t,
                edge.start.point[1] * (1.0 - t) + edge.end.point[1] * t,
                edge.start.point[2] * (1.0 - t) + edge.end.point[2] * t,
            ];
            Ok(p)
        }
        _ => {
            // For other curve types, use the curve module
            curve::point_at(&edge.curve_type, t)
        }
    }
}

/// Helper: Compute tangent vector to an edge at parameter t
fn compute_edge_tangent(edge: &Edge, t: f64) -> Result<[f64; 3]> {
    match &edge.curve_type {
        CurveType::Line => {
            let tangent = [
                edge.end.point[0] - edge.start.point[0],
                edge.end.point[1] - edge.start.point[1],
                edge.end.point[2] - edge.start.point[2],
            ];
            Ok(normalize(&tangent))
        }
        _ => {
            // For other curve types, use finite differences or the curve module
            let delta = 0.001;
            let t_minus = (t - delta).max(0.0);
            let t_plus = (t + delta).min(1.0);
            
            let p_minus = evaluate_edge_at(edge, t_minus)?;
            let p_plus = evaluate_edge_at(edge, t_plus)?;
            
            let tangent = [
                p_plus[0] - p_minus[0],
                p_plus[1] - p_minus[1],
                p_plus[2] - p_minus[2],
            ];
            Ok(normalize(&tangent))
        }
    }
}

/// Frenet frame: (T, N, B) - tangent, normal, binormal
#[derive(Debug, Clone)]
struct FrenetFrame {
    tangent: [f64; 3],      // T
    normal: [f64; 3],       // N (principal normal)
    binormal: [f64; 3],     // B (binormal)
}

/// Compute Frenet frame given a tangent vector
/// If a previous frame is provided, use parallel transport for continuity
fn compute_frenet_frame(tangent: &[f64; 3], prev_frame: Option<&FrenetFrame>) -> FrenetFrame {
    let t = normalize(tangent);
    
    let (n, b) = if let Some(prev) = prev_frame {
        // Parallel transport: keep the previous normal/binormal but re-orthogonalize
        let mut n = prev.normal.clone();
        let mut b = prev.binormal.clone();
        
        // Remove component along new tangent
        let tn_dot = dot(&t, &n);
        n = [
            n[0] - tn_dot * t[0],
            n[1] - tn_dot * t[1],
            n[2] - tn_dot * t[2],
        ];
        n = normalize(&n);
        
        // Recompute binormal
        b = cross(&t, &n);
        b = normalize(&b);
        
        (n, b)
    } else {
        // Initial frame: find an arbitrary normal perpendicular to tangent
        let n = if (t[0].abs() < 0.9) {
            let v = [1.0, 0.0, 0.0];
            let n = cross(&t, &v);
            normalize(&n)
        } else {
            let v = [0.0, 1.0, 0.0];
            let n = cross(&t, &v);
            normalize(&n)
        };
        
        let b = cross(&t, &n);
        (n, normalize(&b))
    };
    
    FrenetFrame {
        tangent: t,
        normal: n,
        binormal: b,
    }
}

/// Transform a 2D profile to a 3D location with a given orientation frame
fn transform_profile_to_frame(
    profile: &Face,
    location: &[f64; 3],
    frame: &FrenetFrame,
) -> Result<Face> {
    // The profile is assumed to be in the N-B plane, with its center at origin
    // We transform it to the actual 3D location using the frame
    
    let mut transformed_edges = vec![];
    
    for edge in &profile.outer_wire.edges {
        let start = transform_point_to_frame(&edge.start.point, location, frame);
        let end = transform_point_to_frame(&edge.end.point, location, frame);
        
        transformed_edges.push(Edge {
            start: Vertex::new(start[0], start[1], start[2]),
            end: Vertex::new(end[0], end[1], end[2]),
            curve_type: edge.curve_type.clone(),
        });
    }
    
    let transformed_wire = Wire {
        edges: transformed_edges,
        closed: profile.outer_wire.closed,
    };
    
    Ok(Face {
        outer_wire: transformed_wire,
        inner_wires: vec![], // TODO: handle holes
        surface_type: SurfaceType::BSpline {
            u_degree: 1,
            v_degree: 1,
            u_knots: vec![0.0, 1.0],
            v_knots: vec![0.0, 1.0],
            control_points: vec![vec![[0.0; 3]; 2]; 2],
            weights: None,
        },
    })
}

/// Transform a 2D point (in N-B plane) to 3D using the frame
fn transform_point_to_frame(
    point_2d: &[f64; 3],  // Assumes point_2d[2] is unused (2D profile)
    location: &[f64; 3],
    frame: &FrenetFrame,
) -> [f64; 3] {
    let x = point_2d[0] * frame.normal[0] + point_2d[1] * frame.binormal[0] + location[0];
    let y = point_2d[0] * frame.normal[1] + point_2d[1] * frame.binormal[1] + location[1];
    let z = point_2d[0] * frame.normal[2] + point_2d[1] * frame.binormal[2] + location[2];
    [x, y, z]
}

/// Create faces connecting two profile sections
fn create_sweep_faces(section1: &Face, section2: &Face) -> Result<Vec<Face>> {
    let mut faces = vec![];
    
    let edges1 = &section1.outer_wire.edges;
    let edges2 = &section2.outer_wire.edges;
    
    if edges1.len() != edges2.len() {
        return Err(CascadeError::InvalidGeometry(
            "Profile sections must have same number of edges".into()
        ));
    }
    
    for i in 0..edges1.len() {
        let e1 = &edges1[i];
        let e2 = &edges2[i];
        
        // Create a quad face (4 vertices)
        let v0 = e1.start.clone();
        let v1 = e1.end.clone();
        let v2 = e2.end.clone();
        let v3 = e2.start.clone();
        
        let face_wire = Wire {
            edges: vec![
                Edge {
                    start: v0.clone(),
                    end: v1.clone(),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: v1.clone(),
                    end: v2.clone(),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: v2.clone(),
                    end: v3.clone(),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: v3.clone(),
                    end: v0.clone(),
                    curve_type: CurveType::Line,
                },
            ],
            closed: true,
        };
        
        let normal = compute_quad_normal(&v0.point, &v1.point, &v2.point, &v3.point);
        
        let face = Face {
            outer_wire: face_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: v0.point,
                normal,
            },
        };
        
        faces.push(face);
    }
    
    Ok(faces)
}

/// Flip a face (reverse its normal direction)
fn flip_face(face: &Face) -> Face {
    let mut flipped_edges = face.outer_wire.edges.clone();
    flipped_edges.reverse();
    
    // Reverse each edge direction
    for edge in &mut flipped_edges {
        let tmp = edge.start.clone();
        edge.start = edge.end.clone();
        edge.end = tmp;
    }
    
    let flipped_normal = [
        -match &face.surface_type {
            SurfaceType::Plane { normal, .. } => normal[0],
            _ => 0.0,
        },
        -match &face.surface_type {
            SurfaceType::Plane { normal, .. } => normal[1],
            _ => 0.0,
        },
        -match &face.surface_type {
            SurfaceType::Plane { normal, .. } => normal[2],
            _ => 0.0,
        },
    ];
    
    Face {
        outer_wire: Wire {
            edges: flipped_edges,
            closed: face.outer_wire.closed,
        },
        inner_wires: face.inner_wires.clone(),
        surface_type: match &face.surface_type {
            SurfaceType::Plane { origin, .. } => SurfaceType::Plane {
                origin: *origin,
                normal: flipped_normal,
            },
            _ => face.surface_type.clone(),
        },
    }
}

/// Translate a face by a vector
fn translate_face(face: &Face, offset: &[f64; 3]) -> Result<Face> {
    let mut translated_edges = vec![];
    
    for edge in &face.outer_wire.edges {
        translated_edges.push(Edge {
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
        });
    }
    
    let translated_wire = Wire {
        edges: translated_edges,
        closed: face.outer_wire.closed,
    };
    
    let origin = match &face.surface_type {
        SurfaceType::Plane { origin, normal: _ } => [
            origin[0] + offset[0],
            origin[1] + offset[1],
            origin[2] + offset[2],
        ],
        _ => [0.0, 0.0, 0.0],
    };
    
    Ok(Face {
        outer_wire: translated_wire,
        inner_wires: face.inner_wires.clone(),
        surface_type: match &face.surface_type {
            SurfaceType::Plane { normal, .. } => SurfaceType::Plane {
                origin,
                normal: *normal,
            },
            _ => face.surface_type.clone(),
        },
    })
}

// ============================================================================
// Vector math helpers
// ============================================================================

fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        [0.0, 0.0, 1.0] // Fallback
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
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

fn compute_quad_normal(p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3], p3: &[f64; 3]) -> [f64; 3] {
    let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v2 = [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]];
    let normal = cross(&v1, &v2);
    normalize(&normal)
}

/// Draft prism - linear extrusion with taper angle (tapered extrusion)
///
/// Extrudes a 2D profile profile upward by a specified height while applying a draft angle.
/// A draft angle creates a taper: faces are tilted from vertical.
///
/// # Arguments
/// * `profile` - A 2D profile wire (should be a closed planar wire)
/// * `direction` - Direction vector for extrusion (defines "up")
/// * `height` - Extrusion height (must be positive)
/// * `draft_angle` - Taper angle in radians
///   - Positive angle: outward taper (profile grows larger)
///   - Negative angle: inward taper (profile shrinks)
///
/// # Returns
/// A Solid representing the tapered extrusion
///
/// # Algorithm
/// 1. Create a face from the input wire (base profile)
/// 2. Calculate the scale factor at the top based on draft_angle and height
/// 3. Create a top profile scaled/offset based on the taper
/// 4. Create tapered side faces connecting base and top
/// 5. Assemble into a solid
///
/// # Example
/// ```ignore
/// // Create a square profile
/// let profile = create_square_wire(1.0)?;
/// // Extrude upward 10 units with 5-degree outward taper
/// let draft_angle = 5.0 * std::f64::consts::PI / 180.0;
/// let solid = make_draft_prism(&profile, &[0.0, 0.0, 1.0], 10.0, draft_angle)?;
/// ```
pub fn make_draft_prism(
    profile: &Wire,
    direction: &[f64; 3],
    height: f64,
    draft_angle: f64,
) -> crate::Result<Solid> {
    if height <= 0.0 {
        return Err(crate::CascadeError::InvalidGeometry(
            "Extrusion height must be positive".into()
        ));
    }

    if profile.edges.is_empty() {
        return Err(crate::CascadeError::InvalidGeometry(
            "Profile wire must have at least one edge".into()
        ));
    }

    let dir_norm = normalize(direction);
    
    // Calculate extrusion offset
    let offset = [
        dir_norm[0] * height,
        dir_norm[1] * height,
        dir_norm[2] * height,
    ];
    
    // Find the center of the profile to use as reference for scaling
    let center = calculate_wire_center(profile);
    
    // Calculate scale factor based on draft angle
    // For a draft angle θ and height h:
    // The horizontal offset from vertical is: h * tan(θ)
    // This changes the scale of the profile
    let horizontal_offset = height * draft_angle.tan();
    
    // The scale factor at the top is determined by:
    // new_size = original_size + 2 * horizontal_offset
    // This is because both sides expand/contract
    let scale_factor = 1.0 + (2.0 * horizontal_offset) / calculate_wire_characteristic_size(profile);
    
    // Create base face from profile
    let base_face = Face {
        outer_wire: profile.clone(),
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: center,
            normal: dir_norm,
        },
    };
    
    // Create top profile: translate + scale
    let top_wire = scale_and_translate_wire(profile, &center, scale_factor, &offset)?;
    
    let top_face = Face {
        outer_wire: top_wire.clone(),
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [center[0] + offset[0], center[1] + offset[1], center[2] + offset[2]],
            normal: dir_norm,
        },
    };
    
    // Create tapered side faces
    let side_faces = create_tapered_faces(&profile, &top_wire, &center, &offset, draft_angle)?;
    
    // Assemble the solid
    let mut faces = vec![base_face];
    faces.extend(side_faces);
    faces.push(top_face);
    
    let shell = Shell { faces, closed: true };
    Ok(Solid {
        outer_shell: shell,
        inner_shells: vec![],
    })
}

/// Helper: Calculate the geometric center of a wire
fn calculate_wire_center(wire: &Wire) -> [f64; 3] {
    if wire.edges.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;
    let mut count = 0;

    for edge in &wire.edges {
        sum_x += edge.start.point[0];
        sum_y += edge.start.point[1];
        sum_z += edge.start.point[2];
        count += 1;
    }

    [
        sum_x / count as f64,
        sum_y / count as f64,
        sum_z / count as f64,
    ]
}

/// Helper: Calculate characteristic size (radius) of a wire
fn calculate_wire_characteristic_size(wire: &Wire) -> f64 {
    let center = calculate_wire_center(wire);
    
    let mut max_dist = 0.0;
    for edge in &wire.edges {
        let dx = edge.start.point[0] - center[0];
        let dy = edge.start.point[1] - center[1];
        let dz = edge.start.point[2] - center[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        max_dist = max_dist.max(dist);
    }

    max_dist.max(1.0) // Avoid division by zero
}

/// Helper: Scale and translate a wire for the top profile
fn scale_and_translate_wire(
    wire: &Wire,
    center: &[f64; 3],
    scale_factor: f64,
    offset: &[f64; 3],
) -> crate::Result<Wire> {
    let mut scaled_edges = vec![];

    for edge in &wire.edges {
        let start_scaled = scale_point_around_center(&edge.start.point, center, scale_factor);
        let end_scaled = scale_point_around_center(&edge.end.point, center, scale_factor);

        let start_translated = [
            start_scaled[0] + offset[0],
            start_scaled[1] + offset[1],
            start_scaled[2] + offset[2],
        ];

        let end_translated = [
            end_scaled[0] + offset[0],
            end_scaled[1] + offset[1],
            end_scaled[2] + offset[2],
        ];

        scaled_edges.push(Edge {
            start: Vertex::new(start_translated[0], start_translated[1], start_translated[2]),
            end: Vertex::new(end_translated[0], end_translated[1], end_translated[2]),
            curve_type: edge.curve_type.clone(),
        });
    }

    Ok(Wire {
        edges: scaled_edges,
        closed: wire.closed,
    })
}

/// Helper: Scale a point around a center
fn scale_point_around_center(point: &[f64; 3], center: &[f64; 3], scale: f64) -> [f64; 3] {
    [
        center[0] + (point[0] - center[0]) * scale,
        center[1] + (point[1] - center[1]) * scale,
        center[2] + (point[2] - center[2]) * scale,
    ]
}

/// Helper: Create tapered side faces connecting base and top profiles
fn create_tapered_faces(
    base_wire: &Wire,
    top_wire: &Wire,
    center: &[f64; 3],
    offset: &[f64; 3],
    _draft_angle: f64,
) -> crate::Result<Vec<Face>> {
    let mut faces = vec![];

    let base_edges = &base_wire.edges;
    let top_edges = &top_wire.edges;

    if base_edges.len() != top_edges.len() {
        return Err(crate::CascadeError::InvalidGeometry(
            "Base and top profiles must have the same number of edges".into()
        ));
    }

    for i in 0..base_edges.len() {
        let base_edge = &base_edges[i];
        let top_edge = &top_edges[i];

        // Create a tapered side face as a quad
        let v0 = base_edge.start.clone();
        let v1 = base_edge.end.clone();
        let v2 = top_edge.end.clone();
        let v3 = top_edge.start.clone();

        let face_wire = Wire {
            edges: vec![
                Edge {
                    start: v0.clone(),
                    end: v1.clone(),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: v1.clone(),
                    end: v2.clone(),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: v2.clone(),
                    end: v3.clone(),
                    curve_type: CurveType::Line,
                },
                Edge {
                    start: v3.clone(),
                    end: v0.clone(),
                    curve_type: CurveType::Line,
                },
            ],
            closed: true,
        };

        // Compute normal for the tapered face
        let normal = compute_quad_normal(&v0.point, &v1.point, &v2.point, &v3.point);

        let face = Face {
            outer_wire: face_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: v0.point,
                normal,
            },
        };

        faces.push(face);
    }

    Ok(faces)
}

/// Revol sweep - rotates a 2D profile around an axis
/// 
/// # Arguments
/// * `profile` - A 2D face to rotate
/// * `axis_origin` - A point on the rotation axis
/// * `axis_direction` - The direction of the rotation axis
/// * `angle` - Rotation angle in radians
///
/// # Returns
/// A Solid representing the revolved shape
pub fn make_revol(
    profile: &Face,
    axis_origin: &[f64; 3],
    axis_direction: &[f64; 3],
    angle: f64,
) -> Result<Solid> {
    if angle <= 0.0 || angle > 2.0 * std::f64::consts::PI {
        return Err(CascadeError::InvalidGeometry(
            "Rotation angle must be between 0 and 2π".into()
        ));
    }
    
    let axis_dir = normalize(axis_direction);
    let num_samples = 20;
    
    let mut profile_sections = vec![];
    
    // Sample the rotation
    for i in 0..=num_samples {
        let current_angle = (i as f64 / num_samples as f64) * angle;
        
        // Rotate profile around the axis
        let rotated = rotate_face_around_axis(profile, axis_origin, &axis_dir, current_angle)?;
        profile_sections.push(rotated);
    }
    
    // Create side faces connecting consecutive sections
    let mut side_faces = vec![];
    
    for i in 0..profile_sections.len() - 1 {
        let current_section = &profile_sections[i];
        let next_section = &profile_sections[i + 1];
        
        let section_faces = create_sweep_faces(current_section, next_section)?;
        side_faces.extend(section_faces);
    }
    
    let mut all_faces = vec![];
    
    // Start cap
    let mut start_cap = profile_sections[0].clone();
    start_cap = flip_face(&start_cap);
    all_faces.push(start_cap);
    
    // Side faces
    all_faces.extend(side_faces);
    
    // End cap
    all_faces.push(profile_sections[profile_sections.len() - 1].clone());
    
    let shell = Shell { faces: all_faces, closed: true };
    Ok(Solid {
        outer_shell: shell,
        inner_shells: vec![],
    })
}

/// Rotate a face around an axis
fn rotate_face_around_axis(
    face: &Face,
    axis_origin: &[f64; 3],
    axis_dir: &[f64; 3],
    angle: f64,
) -> Result<Face> {
    let mut rotated_edges = vec![];
    
    for edge in &face.outer_wire.edges {
        let start = rotate_point_around_axis(&edge.start.point, axis_origin, axis_dir, angle);
        let end = rotate_point_around_axis(&edge.end.point, axis_origin, axis_dir, angle);
        
        rotated_edges.push(Edge {
            start: Vertex::new(start[0], start[1], start[2]),
            end: Vertex::new(end[0], end[1], end[2]),
            curve_type: edge.curve_type.clone(),
        });
    }
    
    let rotated_wire = Wire {
        edges: rotated_edges,
        closed: face.outer_wire.closed,
    };
    
    Ok(Face {
        outer_wire: rotated_wire,
        inner_wires: vec![],
        surface_type: face.surface_type.clone(),
    })
}

/// Rotate a point around an axis using Rodrigues' rotation formula
fn rotate_point_around_axis(
    point: &[f64; 3],
    axis_origin: &[f64; 3],
    axis_dir: &[f64; 3],
    angle: f64,
) -> [f64; 3] {
    let k = normalize(axis_dir);
    
    // Vector from axis origin to point
    let v = [
        point[0] - axis_origin[0],
        point[1] - axis_origin[1],
        point[2] - axis_origin[2],
    ];
    
    // Rodrigues' formula:
    // v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let k_dot_v = dot(&k, &v);
    let k_cross_v = cross(&k, &v);
    
    let term1 = [v[0] * cos_a, v[1] * cos_a, v[2] * cos_a];
    let term2 = [k_cross_v[0] * sin_a, k_cross_v[1] * sin_a, k_cross_v[2] * sin_a];
    let term3 = [
        k[0] * k_dot_v * (1.0 - cos_a),
        k[1] * k_dot_v * (1.0 - cos_a),
        k[2] * k_dot_v * (1.0 - cos_a),
    ];
    
    [
        axis_origin[0] + term1[0] + term2[0] + term3[0],
        axis_origin[1] + term1[1] + term2[1] + term3[1],
        axis_origin[2] + term1[2] + term2[2] + term3[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_revol_rectangle_cylinder() {
        // Create a simple rectangle profile in XZ plane
        let v1 = Vertex::new(1.0, 0.0, 0.0);
        let v2 = Vertex::new(1.0, 0.0, 3.0);
        let v3 = Vertex::new(2.0, 0.0, 3.0);
        let v4 = Vertex::new(2.0, 0.0, 0.0);

        let edges = vec![
            Edge {
                start: v1.clone(),
                end: v2.clone(),
                curve_type: CurveType::Line,
            },
            Edge {
                start: v2.clone(),
                end: v3.clone(),
                curve_type: CurveType::Line,
            },
            Edge {
                start: v3.clone(),
                end: v4.clone(),
                curve_type: CurveType::Line,
            },
            Edge {
                start: v4.clone(),
                end: v1.clone(),
                curve_type: CurveType::Line,
            },
        ];

        let wire = Wire {
            edges,
            closed: true,
        };

        let face = Face {
            outer_wire: wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
            },
        };

        // Revolve 360 degrees around Z axis to create cylinder-like solid
        let result = make_revol(&face, &[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0], 2.0 * std::f64::consts::PI);

        assert!(result.is_ok(), "Failed to create 360° revolution solid");
        let solid = result.unwrap();
        assert!(!solid.outer_shell.faces.is_empty(), "Solid should have faces");
    }

    #[test]
    fn test_revol_90_degrees() {
        // Create a rectangle profile for partial revolution
        let v1 = Vertex::new(1.0, 0.0, 0.0);
        let v2 = Vertex::new(1.0, 0.0, 1.0);
        let v3 = Vertex::new(2.0, 0.0, 1.0);
        let v4 = Vertex::new(2.0, 0.0, 0.0);

        let edges = vec![
            Edge {
                start: v1.clone(),
                end: v2.clone(),
                curve_type: CurveType::Line,
            },
            Edge {
                start: v2.clone(),
                end: v3.clone(),
                curve_type: CurveType::Line,
            },
            Edge {
                start: v3.clone(),
                end: v4.clone(),
                curve_type: CurveType::Line,
            },
            Edge {
                start: v4.clone(),
                end: v1.clone(),
                curve_type: CurveType::Line,
            },
        ];

        let wire = Wire {
            edges,
            closed: true,
        };

        let face = Face {
            outer_wire: wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
            },
        };

        // Revolve 90 degrees
        let result = make_revol(&face, &[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0], std::f64::consts::PI / 2.0);
        
        assert!(result.is_ok(), "Failed to create 90° revolution solid");
        let solid = result.unwrap();
        assert!(!solid.outer_shell.faces.is_empty(), "Revolved solid should have faces");
    }
}
