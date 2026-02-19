use cascade::brep::{SurfaceType, Vertex, Edge, Wire, Face, Shell, Solid};

#[test]
fn test_bspline_surface_creation() {
    // Create a simple quadratic BSpline surface
    let u_degree = 2;
    let v_degree = 2;
    
    // Knots for a 3x3 grid of control points
    let u_knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let v_knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    
    // 3x3 grid of control points forming a simple patch
    let control_points = vec![
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        vec![[0.0, 1.0, 0.5], [1.0, 1.0, 1.0], [2.0, 1.0, 0.5]],
        vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
    ];
    
    let surface = SurfaceType::BSpline {
        u_degree,
        v_degree,
        u_knots,
        v_knots,
        control_points,
        weights: None,
    };
    
    // Test point evaluation at corners
    let pt_00 = surface.point_at(0.0, 0.0);
    assert!(pt_00[0] >= 0.0 && pt_00[0] <= 2.0);
    assert!(pt_00[1] >= 0.0 && pt_00[1] <= 2.0);
    
    let pt_11 = surface.point_at(1.0, 1.0);
    assert!(pt_11[0] >= 0.0 && pt_11[0] <= 2.0);
    assert!(pt_11[1] >= 0.0 && pt_11[1] <= 2.0);
    
    // Test at mid-point
    let pt_mid = surface.point_at(0.5, 0.5);
    assert!(pt_mid[0] >= 0.0 && pt_mid[0] <= 2.0);
    assert!(pt_mid[1] >= 0.0 && pt_mid[1] <= 2.0);
}

#[test]
fn test_bspline_surface_normal() {
    let u_degree = 2;
    let v_degree = 2;
    
    let u_knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let v_knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    
    // Flat surface (should have normals pointing in Z direction)
    let control_points = vec![
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        vec![[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
    ];
    
    let surface = SurfaceType::BSpline {
        u_degree,
        v_degree,
        u_knots,
        v_knots,
        control_points,
        weights: None,
    };
    
    // Test normal at center
    let normal = surface.normal_at(0.5, 0.5);
    
    // Normal should be unit vector (length close to 1)
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    assert!((length - 1.0).abs() < 0.01, "Normal should be unit vector, got length {}", length);
}

#[test]
fn test_bspline_surface_in_face() {
    // Create a BSpline surface as part of a Face
    let u_degree = 2;
    let v_degree = 2;
    
    let u_knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let v_knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    
    let control_points = vec![
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        vec![[0.0, 1.0, 0.5], [1.0, 1.0, 1.0], [2.0, 1.0, 0.5]],
        vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
    ];
    
    let surface_type = SurfaceType::BSpline {
        u_degree,
        v_degree,
        u_knots,
        v_knots,
        control_points,
        weights: None,
    };
    
    // Create a simple rectangular wire (boundary)
    let outer_wire = Wire {
        edges: vec![
            Edge {
                start: Vertex::new(0.0, 0.0, 0.0),
                end: Vertex::new(2.0, 0.0, 0.0),
                curve_type: cascade::brep::CurveType::Line,
            },
            Edge {
                start: Vertex::new(2.0, 0.0, 0.0),
                end: Vertex::new(2.0, 2.0, 0.0),
                curve_type: cascade::brep::CurveType::Line,
            },
            Edge {
                start: Vertex::new(2.0, 2.0, 0.0),
                end: Vertex::new(0.0, 2.0, 0.0),
                curve_type: cascade::brep::CurveType::Line,
            },
            Edge {
                start: Vertex::new(0.0, 2.0, 0.0),
                end: Vertex::new(0.0, 0.0, 0.0),
                curve_type: cascade::brep::CurveType::Line,
            },
        ],
        closed: true,
    };
    
    let face = Face {
        outer_wire,
        inner_wires: vec![],
        surface_type,
    };
    
    // Verify face was created successfully
    assert_eq!(face.outer_wire.edges.len(), 4);
}
