//! Primitive shape creation
//!
//! These functions create basic solid shapes that can be used
//! as building blocks for more complex geometry.

use crate::brep::{Solid, Vertex, Edge, Wire, Face, Shell, CurveType, SurfaceType};
use crate::{Result, CascadeError};

/// Create a box (cuboid) solid
///
/// # Arguments
/// * `dx` - Size in X direction
/// * `dy` - Size in Y direction  
/// * `dz` - Size in Z direction
///
/// # Returns
/// A Solid representing a box centered at origin
pub fn make_box(dx: f64, dy: f64, dz: f64) -> Result<Solid> {
    if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Box dimensions must be positive".into()
        ));
    }
    
    // Create 8 vertices at the corners of the box
    // Bottom face (z=0): v0-v1-v2-v3
    // Top face (z=dz): v4-v5-v6-v7
    let v0 = Vertex::new(0.0, 0.0, 0.0);
    let v1 = Vertex::new(dx, 0.0, 0.0);
    let v2 = Vertex::new(dx, dy, 0.0);
    let v3 = Vertex::new(0.0, dy, 0.0);
    let v4 = Vertex::new(0.0, 0.0, dz);
    let v5 = Vertex::new(dx, 0.0, dz);
    let v6 = Vertex::new(dx, dy, dz);
    let v7 = Vertex::new(0.0, dy, dz);
    
    // Helper to create an edge
    let edge = |start: &Vertex, end: &Vertex| Edge {
        start: start.clone(),
        end: end.clone(),
        curve_type: CurveType::Line,
    };
    
    // Create faces with properly connected edges forming closed loops
    // Each wire's edges form a connected chain: e0.end = e1.start, etc.
    
    // Bottom face (z=0) - vertices v0, v1, v2, v3 - normal pointing down (-Z)
    let bottom_wire = Wire {
        edges: vec![edge(&v0, &v1), edge(&v1, &v2), edge(&v2, &v3), edge(&v3, &v0)],
        closed: true,
    };
    let bottom_face = Face {
        outer_wire: bottom_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, -1.0],
        },
    };
    
    // Top face (z=dz) - vertices v4, v5, v6, v7 - normal pointing up (+Z)
    let top_wire = Wire {
        edges: vec![edge(&v4, &v5), edge(&v5, &v6), edge(&v6, &v7), edge(&v7, &v4)],
        closed: true,
    };
    let top_face = Face {
        outer_wire: top_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, 0.0, dz],
            normal: [0.0, 0.0, 1.0],
        },
    };
    
    // Front face (y=0) - vertices v0, v1, v5, v4 - normal pointing front (-Y)
    let front_wire = Wire {
        edges: vec![edge(&v0, &v1), edge(&v1, &v5), edge(&v5, &v4), edge(&v4, &v0)],
        closed: true,
    };
    let front_face = Face {
        outer_wire: front_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, 0.0, 0.0],
            normal: [0.0, -1.0, 0.0],
        },
    };
    
    // Back face (y=dy) - vertices v3, v2, v6, v7 - normal pointing back (+Y)
    let back_wire = Wire {
        edges: vec![edge(&v3, &v2), edge(&v2, &v6), edge(&v6, &v7), edge(&v7, &v3)],
        closed: true,
    };
    let back_face = Face {
        outer_wire: back_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, dy, 0.0],
            normal: [0.0, 1.0, 0.0],
        },
    };
    
    // Left face (x=0) - vertices v0, v4, v7, v3 - normal pointing left (-X)
    let left_wire = Wire {
        edges: vec![edge(&v0, &v4), edge(&v4, &v7), edge(&v7, &v3), edge(&v3, &v0)],
        closed: true,
    };
    let left_face = Face {
        outer_wire: left_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, 0.0, 0.0],
            normal: [-1.0, 0.0, 0.0],
        },
    };
    
    // Right face (x=dx) - vertices v1, v2, v6, v5 - normal pointing right (+X)
    let right_wire = Wire {
        edges: vec![edge(&v1, &v2), edge(&v2, &v6), edge(&v6, &v5), edge(&v5, &v1)],
        closed: true,
    };
    let right_face = Face {
        outer_wire: right_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [dx, 0.0, 0.0],
            normal: [1.0, 0.0, 0.0],
        },
    };
    
    // Create shell with all 6 faces
    let shell = Shell {
        faces: vec![bottom_face, top_face, front_face, back_face, left_face, right_face],
        closed: true,
    };
    
    // Create solid
    let solid = Solid {
        outer_shell: shell,
        inner_shells: vec![],
    };
    
    Ok(solid)
}

/// Create a sphere solid
///
/// # Arguments
/// * `radius` - Sphere radius
///
/// # Returns
/// A Solid representing a sphere centered at origin
pub fn make_sphere(radius: f64) -> Result<Solid> {
    if radius <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Sphere radius must be positive".into()
        ));
    }
    
    // Create a sphere BREP topology
    // North pole vertex
    let v_north = Vertex::new(0.0, 0.0, radius);
    // South pole vertex
    let v_south = Vertex::new(0.0, 0.0, -radius);
    
    // Create edges forming meridians that connect the poles
    // We'll create 4 meridian edges positioned at 90-degree intervals around the sphere
    // This forms a closed loop of edges that bounds the spherical surface
    
    // Meridian 1: from north to south via positive X side
    let edge1 = Edge {
        start: v_north.clone(),
        end: v_south.clone(),
        curve_type: CurveType::Arc {
            center: [0.0, 0.0, 0.0],
            radius,
        },
    };
    
    // Meridian 2: from south back to north via positive Y side
    let edge2 = Edge {
        start: v_south.clone(),
        end: v_north.clone(),
        curve_type: CurveType::Arc {
            center: [0.0, 0.0, 0.0],
            radius,
        },
    };
    
    // Meridian 3: from north to south via negative X side
    let edge3 = Edge {
        start: v_north.clone(),
        end: v_south.clone(),
        curve_type: CurveType::Arc {
            center: [0.0, 0.0, 0.0],
            radius,
        },
    };
    
    // Meridian 4: from south back to north via negative Y side
    let edge4 = Edge {
        start: v_south.clone(),
        end: v_north.clone(),
        curve_type: CurveType::Arc {
            center: [0.0, 0.0, 0.0],
            radius,
        },
    };
    
    // Create the outer wire from these edges
    let outer_wire = Wire {
        edges: vec![edge1, edge2, edge3, edge4],
        closed: true,
    };
    
    // Create the spherical face
    let sphere_face = Face {
        outer_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Sphere {
            center: [0.0, 0.0, 0.0],
            radius,
        },
    };
    
    // Create shell containing the spherical face
    let shell = Shell {
        faces: vec![sphere_face],
        closed: true,
    };
    
    // Create and return the solid
    let solid = Solid {
        outer_shell: shell,
        inner_shells: vec![],
    };
    
    Ok(solid)
}

/// Create a cylinder solid
///
/// # Arguments
/// * `radius` - Cylinder radius
/// * `height` - Cylinder height
///
/// # Returns
/// A Solid representing a cylinder along Z axis, base at origin
pub fn make_cylinder(radius: f64, height: f64) -> Result<Solid> {
    if radius <= 0.0 || height <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Cylinder dimensions must be positive".into()
        ));
    }
    
    // Approximate circular edges with 8 arc segments
    let n_segs = 8;
    let angle_step = 2.0 * std::f64::consts::PI / (n_segs as f64);
    
    // Create vertices for bottom circle (z=0)
    let mut bottom_vertices = Vec::new();
    for i in 0..n_segs {
        let angle = angle_step * (i as f64);
        let x = radius * angle.cos();
        let y = radius * angle.sin();
        bottom_vertices.push(Vertex::new(x, y, 0.0));
    }
    
    // Create vertices for top circle (z=height)
    let mut top_vertices = Vec::new();
    for i in 0..n_segs {
        let angle = angle_step * (i as f64);
        let x = radius * angle.cos();
        let y = radius * angle.sin();
        top_vertices.push(Vertex::new(x, y, height));
    }
    
    // Create arc edges for bottom circle
    let mut bottom_edges = Vec::new();
    for i in 0..n_segs {
        let start = bottom_vertices[i].clone();
        let end = bottom_vertices[(i + 1) % n_segs].clone();
        let edge = Edge {
            start,
            end,
            curve_type: CurveType::Arc {
                center: [0.0, 0.0, 0.0],
                radius,
            },
        };
        bottom_edges.push(edge);
    }
    
    // Create arc edges for top circle
    let mut top_edges = Vec::new();
    for i in 0..n_segs {
        let start = top_vertices[i].clone();
        let end = top_vertices[(i + 1) % n_segs].clone();
        let edge = Edge {
            start,
            end,
            curve_type: CurveType::Arc {
                center: [0.0, 0.0, height],
                radius,
            },
        };
        top_edges.push(edge);
    }
    
    // Create wires for the circular boundaries
    let bottom_wire = Wire {
        edges: bottom_edges.clone(),
        closed: true,
    };
    
    let top_wire = Wire {
        edges: top_edges,
        closed: true,
    };
    
    // Create bottom planar face (z=0, normal pointing down)
    let bottom_face = Face {
        outer_wire: bottom_wire.clone(),
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, -1.0],
        },
    };
    
    // Create top planar face (z=height, normal pointing up)
    let top_face = Face {
        outer_wire: top_wire.clone(),
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, 0.0, height],
            normal: [0.0, 0.0, 1.0],
        },
    };
    
    // Create cylindrical side face
    // The outer wire is the bottom circle, inner wire is the top circle
    // This represents a cylindrical surface bounded by those two circles
    let side_face = Face {
        outer_wire: bottom_wire,
        inner_wires: vec![top_wire],
        surface_type: SurfaceType::Cylinder {
            origin: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            radius,
        },
    };
    
    // Create closed shell with 3 faces (bottom cap + top cap + cylindrical side)
    let shell = Shell {
        faces: vec![bottom_face, top_face, side_face],
        closed: true,
    };
    
    // Create solid
    let solid = Solid {
        outer_shell: shell,
        inner_shells: vec![],
    };
    
    Ok(solid)
}

/// Create a cone solid
///
/// # Arguments
/// * `radius1` - Base radius
/// * `radius2` - Top radius (0 for pointed cone)
/// * `height` - Cone height
///
/// # Returns
/// A Solid representing a cone along Z axis
pub fn make_cone(radius1: f64, radius2: f64, height: f64) -> Result<Solid> {
    if radius1 < 0.0 || radius2 < 0.0 || height <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Cone dimensions must be non-negative (height positive)".into()
        ));
    }
    if radius1 == 0.0 && radius2 == 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "At least one cone radius must be positive".into()
        ));
    }
    
    // Approximate circular edges with 8 arc segments
    let n_segs = 8;
    let angle_step = 2.0 * std::f64::consts::PI / (n_segs as f64);
    
    let mut faces = Vec::new();
    
    // Create vertices for bottom circle (z=0)
    let mut bottom_vertices = Vec::new();
    for i in 0..n_segs {
        let angle = angle_step * (i as f64);
        let x = radius1 * angle.cos();
        let y = radius1 * angle.sin();
        bottom_vertices.push(Vertex::new(x, y, 0.0));
    }
    
    // Create vertices for top circle (z=height)
    let mut top_vertices = Vec::new();
    for i in 0..n_segs {
        let angle = angle_step * (i as f64);
        let x = radius2 * angle.cos();
        let y = radius2 * angle.sin();
        top_vertices.push(Vertex::new(x, y, height));
    }
    
    // Create arc edges for bottom circle
    let mut bottom_edges = Vec::new();
    for i in 0..n_segs {
        let start = bottom_vertices[i].clone();
        let end = bottom_vertices[(i + 1) % n_segs].clone();
        let edge = Edge {
            start,
            end,
            curve_type: CurveType::Arc {
                center: [0.0, 0.0, 0.0],
                radius: radius1,
            },
        };
        bottom_edges.push(edge);
    }
    
    // Create arc edges for top circle (only if radius2 > 0)
    let mut top_edges = Vec::new();
    if radius2 > 0.0 {
        for i in 0..n_segs {
            let start = top_vertices[i].clone();
            let end = top_vertices[(i + 1) % n_segs].clone();
            let edge = Edge {
                start,
                end,
                curve_type: CurveType::Arc {
                    center: [0.0, 0.0, height],
                    radius: radius2,
                },
            };
            top_edges.push(edge);
        }
    }
    
    // Create wires for the circular boundaries
    let bottom_wire = Wire {
        edges: bottom_edges.clone(),
        closed: true,
    };
    
    // Create bottom planar face (z=0, normal pointing down) - only if radius1 > 0
    if radius1 > 0.0 {
        let bottom_face = Face {
            outer_wire: bottom_wire.clone(),
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, -1.0],
            },
        };
        faces.push(bottom_face);
    }
    
    // Create top planar face (z=height, normal pointing up) - only if radius2 > 0
    if radius2 > 0.0 {
        let top_wire = Wire {
            edges: top_edges.clone(),
            closed: true,
        };
        let top_face = Face {
            outer_wire: top_wire.clone(),
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: [0.0, 0.0, height],
                normal: [0.0, 0.0, 1.0],
            },
        };
        faces.push(top_face);
    }
    
    // Create conical side face
    // The outer wire is the bottom circle, inner wire is the top circle
    // For a pointed cone (radius2 == 0), inner_wires is empty
    let inner_wires = if radius2 > 0.0 && !top_edges.is_empty() {
        vec![Wire {
            edges: top_edges,
            closed: true,
        }]
    } else {
        vec![]
    };
    
    let side_face = Face {
        outer_wire: bottom_wire,
        inner_wires,
        surface_type: SurfaceType::Cone {
            origin: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            half_angle_rad: if height > 0.0 && radius1 > 0.0 {
                (radius1 / height).atan()
            } else {
                0.0
            },
        },
    };
    faces.push(side_face);
    
    // Create closed shell
    let shell = Shell {
        faces,
        closed: true,
    };
    
    // Create solid
    let solid = Solid {
        outer_shell: shell,
        inner_shells: vec![],
    };
    
    Ok(solid)
}

/// Create a torus solid
///
/// # Arguments
/// * `major_radius` - Distance from center to tube center
/// * `minor_radius` - Tube radius
///
/// # Returns
/// A Solid representing a torus centered at origin, in XY plane
pub fn make_torus(major_radius: f64, minor_radius: f64) -> Result<Solid> {
    if major_radius <= 0.0 || minor_radius <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Torus radii must be positive".into()
        ));
    }
    if minor_radius >= major_radius {
        return Err(CascadeError::InvalidGeometry(
            "Minor radius must be less than major radius".into()
        ));
    }
    
    // Create a torus BREP topology with a single toroidal surface
    // The torus is bounded by 4 edges that form a closed curve around the major circle
    // at the outer equator of the torus (distance major_radius + minor_radius from center)
    
    // Create 4 points around the outer equator of the torus
    // These are positioned at 90-degree intervals
    let v0 = Vertex::new(major_radius + minor_radius, 0.0, 0.0);
    let v1 = Vertex::new(0.0, major_radius + minor_radius, 0.0);
    let v2 = Vertex::new(-(major_radius + minor_radius), 0.0, 0.0);
    let v3 = Vertex::new(0.0, -(major_radius + minor_radius), 0.0);
    
    // Create 4 arc edges connecting these points around the major circle
    // Each edge is an arc on a circle of radius (major_radius + minor_radius) in the XY plane
    let edge0 = Edge {
        start: v0.clone(),
        end: v1.clone(),
        curve_type: CurveType::Arc {
            center: [0.0, 0.0, 0.0],
            radius: major_radius + minor_radius,
        },
    };
    
    let edge1 = Edge {
        start: v1.clone(),
        end: v2.clone(),
        curve_type: CurveType::Arc {
            center: [0.0, 0.0, 0.0],
            radius: major_radius + minor_radius,
        },
    };
    
    let edge2 = Edge {
        start: v2.clone(),
        end: v3.clone(),
        curve_type: CurveType::Arc {
            center: [0.0, 0.0, 0.0],
            radius: major_radius + minor_radius,
        },
    };
    
    let edge3 = Edge {
        start: v3.clone(),
        end: v0.clone(),
        curve_type: CurveType::Arc {
            center: [0.0, 0.0, 0.0],
            radius: major_radius + minor_radius,
        },
    };
    
    // Create the outer wire from these edges
    let outer_wire = Wire {
        edges: vec![edge0, edge1, edge2, edge3],
        closed: true,
    };
    
    // Create the toroidal face
    let torus_face = Face {
        outer_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Torus {
            center: [0.0, 0.0, 0.0],
            major_radius,
            minor_radius,
        },
    };
    
    // Create shell containing the toroidal face
    let shell = Shell {
        faces: vec![torus_face],
        closed: true,
    };
    
    // Create and return the solid
    let solid = Solid {
        outer_shell: shell,
        inner_shells: vec![],
    };
    
    Ok(solid)
}

/// Create a wedge (prism) solid
///
/// A wedge is a box-like shape with the top edge offset in the X direction.
/// The top face is smaller than the bottom, creating a sloped side face.
///
/// # Arguments
/// * `dx` - Size in X direction at the base
/// * `dy` - Size in Y direction
/// * `dz` - Size in Z direction
/// * `ltx` - Length of the top edge in X direction (0 = pyramid, dx = box)
///
/// # Returns
/// A Solid representing a wedge
pub fn make_wedge(dx: f64, dy: f64, dz: f64, ltx: f64) -> Result<Solid> {
    if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Wedge dimensions must be positive".into()
        ));
    }
    
    if ltx < 0.0 || ltx > dx {
        return Err(CascadeError::InvalidGeometry(
            "ltx must be between 0 and dx".into()
        ));
    }
    
    // Create 8 vertices
    // Bottom face (z=0) - full size rectangle
    let v0 = Vertex::new(0.0, 0.0, 0.0);
    let v1 = Vertex::new(dx, 0.0, 0.0);
    let v2 = Vertex::new(dx, dy, 0.0);
    let v3 = Vertex::new(0.0, dy, 0.0);
    
    // Top face (z=dz) - offset in X direction by (dx-ltx)/2
    let offset_x = (dx - ltx) / 2.0;
    let v4 = Vertex::new(offset_x, 0.0, dz);
    let v5 = Vertex::new(offset_x + ltx, 0.0, dz);
    let v6 = Vertex::new(offset_x + ltx, dy, dz);
    let v7 = Vertex::new(offset_x, dy, dz);
    
    // Create 12 edges
    // Bottom face edges (z=0)
    let e0 = Edge { start: v0.clone(), end: v1.clone(), curve_type: CurveType::Line };
    let e1 = Edge { start: v1.clone(), end: v2.clone(), curve_type: CurveType::Line };
    let e2 = Edge { start: v2.clone(), end: v3.clone(), curve_type: CurveType::Line };
    let e3 = Edge { start: v3.clone(), end: v0.clone(), curve_type: CurveType::Line };
    
    // Top face edges (z=dz)
    let e4 = Edge { start: v4.clone(), end: v5.clone(), curve_type: CurveType::Line };
    let e5 = Edge { start: v5.clone(), end: v6.clone(), curve_type: CurveType::Line };
    let e6 = Edge { start: v6.clone(), end: v7.clone(), curve_type: CurveType::Line };
    let e7 = Edge { start: v7.clone(), end: v4.clone(), curve_type: CurveType::Line };
    
    // Vertical edges connecting bottom to top
    let e8 = Edge { start: v0.clone(), end: v4.clone(), curve_type: CurveType::Line };
    let e9 = Edge { start: v1.clone(), end: v5.clone(), curve_type: CurveType::Line };
    let e10 = Edge { start: v2.clone(), end: v6.clone(), curve_type: CurveType::Line };
    let e11 = Edge { start: v3.clone(), end: v7.clone(), curve_type: CurveType::Line };
    
    // Create 6 faces
    
    // Bottom face (z=0) - normal pointing down
    let bottom_wire = Wire {
        edges: vec![e0.clone(), e1.clone(), e2.clone(), e3.clone()],
        closed: true,
    };
    let bottom_face = Face {
        outer_wire: bottom_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, -1.0],
        },
    };
    
    // Top face (z=dz) - normal pointing up
    let top_wire = Wire {
        edges: vec![e4.clone(), e5.clone(), e6.clone(), e7.clone()],
        closed: true,
    };
    let top_face = Face {
        outer_wire: top_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [offset_x, 0.0, dz],
            normal: [0.0, 0.0, 1.0],
        },
    };
    
    // Front face (y=0) - normal pointing back (negative Y)
    let front_wire = Wire {
        edges: vec![e0.clone(), e9.clone(), e4.clone(), e8.clone()],
        closed: true,
    };
    let front_face = Face {
        outer_wire: front_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, 0.0, 0.0],
            normal: [0.0, -1.0, 0.0],
        },
    };
    
    // Back face (y=dy) - normal pointing forward (positive Y)
    let back_wire = Wire {
        edges: vec![e3.clone(), e7.clone(), e6.clone(), e2.clone()],
        closed: true,
    };
    let back_face = Face {
        outer_wire: back_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, dy, 0.0],
            normal: [0.0, 1.0, 0.0],
        },
    };
    
    // Left face (x=offset_x) - normal pointing left (negative X)
    let left_wire = Wire {
        edges: vec![e8.clone(), e7.clone(), e3.clone(), e11.clone()],
        closed: true,
    };
    let left_face = Face {
        outer_wire: left_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [offset_x, 0.0, 0.0],
            normal: [-1.0, 0.0, 0.0],
        },
    };
    
    // Right face (sloped) - connecting v1, v2, v6, v5
    // This is a planar quadrilateral face with normal computed from the plane
    let right_wire = Wire {
        edges: vec![e1.clone(), e10.clone(), e5.clone(), e9.clone()],
        closed: true,
    };
    let right_face = Face {
        outer_wire: right_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [dx, 0.0, 0.0],
            normal: [1.0, 0.0, 0.0],
        },
    };
    
    // Create shell with all faces
    let shell = Shell {
        faces: vec![bottom_face, top_face, front_face, back_face, left_face, right_face],
        closed: true,
    };
    
    // Create solid
    let solid = Solid {
        outer_shell: shell,
        inner_shells: vec![],
    };
    
    Ok(solid)
}

/// Represents a half-space: an infinite solid bounded by a plane
/// 
/// A half-space is defined by a plane (origin + normal).
/// The half-space extends infinitely in the direction of the normal.
/// It can be used in boolean operations to cut solids by a plane.
#[derive(Debug, Clone)]
pub struct HalfSpace {
    /// Origin point of the plane
    pub origin: [f64; 3],
    /// Normal vector of the plane (should be normalized)
    /// Points toward the "inside" of the half-space
    pub normal: [f64; 3],
}

impl HalfSpace {
    /// Create a new half-space from a plane (origin + normal)
    /// 
    /// # Arguments
    /// * `origin` - A point on the plane
    /// * `normal` - Normal vector (will be normalized)
    /// 
    /// # Returns
    /// A new HalfSpace
    pub fn new(origin: [f64; 3], normal: [f64; 3]) -> Result<Self> {
        // Normalize the normal vector
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        
        if len < 1e-10 {
            return Err(CascadeError::InvalidGeometry(
                "Half-space normal vector must be non-zero".into()
            ));
        }
        
        let normalized_normal = [
            normal[0] / len,
            normal[1] / len,
            normal[2] / len,
        ];
        
        Ok(HalfSpace {
            origin,
            normal: normalized_normal,
        })
    }
    
    /// Get the plane defining this half-space
    /// Returns (origin, normal) where normal is the inward-pointing normal
    pub fn plane(&self) -> ([f64; 3], [f64; 3]) {
        (self.origin, self.normal)
    }
    
    /// Test if a point is contained in the half-space
    /// 
    /// A point is inside if: dot(point - origin, normal) >= 0
    pub fn contains_point(&self, pt: [f64; 3]) -> bool {
        let to_pt = [
            pt[0] - self.origin[0],
            pt[1] - self.origin[1],
            pt[2] - self.origin[2],
        ];
        
        let dot_product = to_pt[0] * self.normal[0] + to_pt[1] * self.normal[1] + to_pt[2] * self.normal[2];
        
        dot_product >= -1e-10  // Use small tolerance for floating point
    }
    
    /// Convert a half-space to a solid by creating a large box aligned with the plane
    /// This creates a finite approximation of the infinite half-space for boolean operations
    /// 
    /// # Arguments
    /// * `bbox_size` - Half-size of the bounding box to create
    /// 
    /// # Returns
    /// A Solid representing a bounded half-space
    pub fn to_solid(&self, bbox_size: f64) -> Result<Solid> {
        if bbox_size <= 0.0 {
            return Err(CascadeError::InvalidGeometry(
                "Bounding box size must be positive".into()
            ));
        }
        
        // Create an orthonormal basis with normal as Z axis
        let nz = self.normal;
        
        // Find perpendicular vectors
        let nx = if nz[0].abs() < 0.9 {
            let v = [0.0, -nz[2], nz[1]];
            let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
            [v[0]/len, v[1]/len, v[2]/len]
        } else {
            let v = [-nz[1], nz[0], 0.0];
            let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
            [v[0]/len, v[1]/len, v[2]/len]
        };
        
        let ny = [
            nz[1]*nx[2] - nz[2]*nx[1],
            nz[2]*nx[0] - nz[0]*nx[2],
            nz[0]*nx[1] - nz[1]*nx[0],
        ];
        
        // Create 8 vertices of the bounding box
        // The box extends bbox_size in -z direction (behind the plane)
        // and bbox_size in +z direction (into the half-space)
        
        let vertices: Vec<[f64; 3]> = vec![
            // Back face (z = -bbox_size)
            [
                self.origin[0] - bbox_size*nx[0] - bbox_size*ny[0] - bbox_size*nz[0],
                self.origin[1] - bbox_size*nx[1] - bbox_size*ny[1] - bbox_size*nz[1],
                self.origin[2] - bbox_size*nx[2] - bbox_size*ny[2] - bbox_size*nz[2],
            ],
            [
                self.origin[0] + bbox_size*nx[0] - bbox_size*ny[0] - bbox_size*nz[0],
                self.origin[1] + bbox_size*nx[1] - bbox_size*ny[1] - bbox_size*nz[1],
                self.origin[2] + bbox_size*nx[2] - bbox_size*ny[2] - bbox_size*nz[2],
            ],
            [
                self.origin[0] + bbox_size*nx[0] + bbox_size*ny[0] - bbox_size*nz[0],
                self.origin[1] + bbox_size*nx[1] + bbox_size*ny[1] - bbox_size*nz[1],
                self.origin[2] + bbox_size*nx[2] + bbox_size*ny[2] - bbox_size*nz[2],
            ],
            [
                self.origin[0] - bbox_size*nx[0] + bbox_size*ny[0] - bbox_size*nz[0],
                self.origin[1] - bbox_size*nx[1] + bbox_size*ny[1] - bbox_size*nz[1],
                self.origin[2] - bbox_size*nx[2] + bbox_size*ny[2] - bbox_size*nz[2],
            ],
            // Front face (z = +bbox_size)
            [
                self.origin[0] - bbox_size*nx[0] - bbox_size*ny[0] + bbox_size*nz[0],
                self.origin[1] - bbox_size*nx[1] - bbox_size*ny[1] + bbox_size*nz[1],
                self.origin[2] - bbox_size*nx[2] - bbox_size*ny[2] + bbox_size*nz[2],
            ],
            [
                self.origin[0] + bbox_size*nx[0] - bbox_size*ny[0] + bbox_size*nz[0],
                self.origin[1] + bbox_size*nx[1] - bbox_size*ny[1] + bbox_size*nz[1],
                self.origin[2] + bbox_size*nx[2] - bbox_size*ny[2] + bbox_size*nz[2],
            ],
            [
                self.origin[0] + bbox_size*nx[0] + bbox_size*ny[0] + bbox_size*nz[0],
                self.origin[1] + bbox_size*nx[1] + bbox_size*ny[1] + bbox_size*nz[1],
                self.origin[2] + bbox_size*nx[2] + bbox_size*ny[2] + bbox_size*nz[2],
            ],
            [
                self.origin[0] - bbox_size*nx[0] + bbox_size*ny[0] + bbox_size*nz[0],
                self.origin[1] - bbox_size*nx[1] + bbox_size*ny[1] + bbox_size*nz[1],
                self.origin[2] - bbox_size*nx[2] + bbox_size*ny[2] + bbox_size*nz[2],
            ],
        ];
        
        let v: Vec<Vertex> = vertices.iter().map(|p| Vertex::new(p[0], p[1], p[2])).collect();
        
        // Create faces of the bounding box (oriented so normals point outward)
        let edge = |start: &Vertex, end: &Vertex| Edge {
            start: start.clone(),
            end: end.clone(),
            curve_type: CurveType::Line,
        };
        
        // Back face (z = -bbox_size), normal pointing outward (-Z direction in plane basis)
        let back_wire = Wire {
            edges: vec![edge(&v[0], &v[1]), edge(&v[1], &v[2]), edge(&v[2], &v[3]), edge(&v[3], &v[0])],
            closed: true,
        };
        let back_face = Face {
            outer_wire: back_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: self.origin,
                normal: [-nz[0], -nz[1], -nz[2]],
            },
        };
        
        // Front face (z = +bbox_size), normal pointing outward (+Z direction)
        let front_wire = Wire {
            edges: vec![edge(&v[4], &v[7]), edge(&v[7], &v[6]), edge(&v[6], &v[5]), edge(&v[5], &v[4])],
            closed: true,
        };
        let front_face = Face {
            outer_wire: front_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: self.origin,
                normal: nz,
            },
        };
        
        // Side faces (perpendicular to the plane)
        // Bottom (Y = -bbox_size)
        let bottom_wire = Wire {
            edges: vec![edge(&v[0], &v[4]), edge(&v[4], &v[5]), edge(&v[5], &v[1]), edge(&v[1], &v[0])],
            closed: true,
        };
        let bottom_face = Face {
            outer_wire: bottom_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: self.origin,
                normal: [-ny[0], -ny[1], -ny[2]],
            },
        };
        
        // Top (Y = +bbox_size)
        let top_wire = Wire {
            edges: vec![edge(&v[3], &v[2]), edge(&v[2], &v[6]), edge(&v[6], &v[7]), edge(&v[7], &v[3])],
            closed: true,
        };
        let top_face = Face {
            outer_wire: top_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: self.origin,
                normal: ny,
            },
        };
        
        // Left (X = -bbox_size)
        let left_wire = Wire {
            edges: vec![edge(&v[0], &v[3]), edge(&v[3], &v[7]), edge(&v[7], &v[4]), edge(&v[4], &v[0])],
            closed: true,
        };
        let left_face = Face {
            outer_wire: left_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: self.origin,
                normal: [-nx[0], -nx[1], -nx[2]],
            },
        };
        
        // Right (X = +bbox_size)
        let right_wire = Wire {
            edges: vec![edge(&v[1], &v[5]), edge(&v[5], &v[6]), edge(&v[6], &v[2]), edge(&v[2], &v[1])],
            closed: true,
        };
        let right_face = Face {
            outer_wire: right_wire,
            inner_wires: vec![],
            surface_type: SurfaceType::Plane {
                origin: self.origin,
                normal: nx,
            },
        };
        
        let shell = Shell {
            faces: vec![back_face, front_face, bottom_face, top_face, left_face, right_face],
            closed: true,
        };
        
        let solid = Solid {
            outer_shell: shell,
            inner_shells: vec![],
        };
        
        Ok(solid)
    }
}

/// Create a half-space solid
/// 
/// A half-space is an infinite solid bounded by a plane.
/// It can be used to cut other solids via boolean operations.
/// 
/// # Arguments
/// * `origin` - A point on the plane
/// * `normal` - Normal vector pointing toward the inside of the half-space
/// * `bbox_size` - Size of the bounding box used to approximate the infinite half-space
/// 
/// # Returns
/// A Solid representing the half-space (as a large box)
pub fn make_half_space(origin: [f64; 3], normal: [f64; 3], bbox_size: f64) -> Result<Solid> {
    let half_space = HalfSpace::new(origin, normal)?;
    half_space.to_solid(bbox_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_box_invalid_dimensions() {
        assert!(make_box(-1.0, 1.0, 1.0).is_err());
        assert!(make_box(0.0, 1.0, 1.0).is_err());
    }
    
    #[test]
    fn test_sphere_invalid_radius() {
        assert!(make_sphere(-1.0).is_err());
        assert!(make_sphere(0.0).is_err());
    }
    
    #[test]
    fn test_half_space_creation() {
        // Create a half-space with normal pointing up
        let hs = HalfSpace::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(hs.is_ok());
        let hs = hs.unwrap();
        assert_eq!(hs.origin, [0.0, 0.0, 0.0]);
        assert_eq!(hs.normal, [0.0, 0.0, 1.0]);
    }
    
    #[test]
    fn test_half_space_invalid_normal() {
        // Create a half-space with zero normal (should fail)
        let hs = HalfSpace::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(hs.is_err());
    }
    
    #[test]
    fn test_half_space_contains_point() {
        let hs = HalfSpace::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        
        // Points above the plane should be inside
        assert!(hs.contains_point([0.0, 0.0, 1.0]));
        assert!(hs.contains_point([1.0, 1.0, 1.0]));
        
        // Points below should be outside
        assert!(!hs.contains_point([0.0, 0.0, -1.0]));
        assert!(!hs.contains_point([1.0, 1.0, -1.0]));
        
        // Point on the plane should be inside (boundary)
        assert!(hs.contains_point([0.0, 0.0, 0.0]));
    }
    
    #[test]
    fn test_half_space_normal_normalization() {
        // Create with non-unit normal
        let hs = HalfSpace::new([0.0, 0.0, 0.0], [0.0, 0.0, 5.0]).unwrap();
        
        // Normal should be normalized
        let len = hs.normal[0]*hs.normal[0] + hs.normal[1]*hs.normal[1] + hs.normal[2]*hs.normal[2];
        assert!((len - 1.0).abs() < 1e-6);
        assert!((hs.normal[2] - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_half_space_to_solid() {
        let hs = HalfSpace::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        let solid = hs.to_solid(10.0);
        assert!(solid.is_ok());
        let solid = solid.unwrap();
        
        // Check that it's a closed solid
        assert!(solid.outer_shell.closed);
        assert_eq!(solid.outer_shell.faces.len(), 6); // Box has 6 faces
    }
    
    #[test]
    fn test_make_half_space() {
        let solid = make_half_space([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 5.0);
        assert!(solid.is_ok());
    }
    
    #[test]
    fn test_make_half_space_invalid_bbox() {
        let solid = make_half_space([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0);
        assert!(solid.is_err());
    }
}
