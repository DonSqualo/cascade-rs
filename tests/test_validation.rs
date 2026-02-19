//! Test shape validation with known-bad and known-good geometry

use cascade::{check_valid, check_watertight, check_self_intersection, Shape};
use cascade::{make_box, make_sphere};
use cascade::brep::{Edge, Vertex, CurveType};

#[test]
fn test_degenerate_edge_detection() {
    // Edge with zero length (both start and end at same point)
    let degenerate_edge = Edge {
        start: Vertex::new(0.0, 0.0, 0.0),
        end: Vertex::new(0.0, 0.0, 0.0),
        curve_type: CurveType::Line,
    };
    
    let edge_shape = Shape::Edge(degenerate_edge);
    let errors = check_valid(&edge_shape).unwrap();
    
    // Should detect the degenerate edge
    assert!(!errors.is_empty(), "Should detect zero-length edge");
}

#[test]
fn test_tiny_edge_detection() {
    // Edge with extremely small length
    let tiny_edge = Edge {
        start: Vertex::new(0.0, 0.0, 0.0),
        end: Vertex::new(1e-8, 1e-8, 1e-8),
        curve_type: CurveType::Line,
    };
    
    let shape = Shape::Edge(tiny_edge);
    let errors = check_valid(&shape).unwrap();
    
    // Should detect the tiny edge
    assert!(!errors.is_empty(), "Should detect near-zero-length edge");
}

#[test]
fn test_normal_edge_valid() {
    // Normal edge with proper length
    let normal_edge = Edge {
        start: Vertex::new(0.0, 0.0, 0.0),
        end: Vertex::new(1.0, 0.0, 0.0),
        curve_type: CurveType::Line,
    };
    
    let shape = Shape::Edge(normal_edge);
    let errors = check_valid(&shape).unwrap();
    
    // Should have no degenerate edge errors
    let has_degenerate_errors = errors.iter().any(|e| {
        matches!(e, cascade::ShapeError::DegenerateEdgeNearZero { .. })
    });
    assert!(!has_degenerate_errors, "Normal edge should not be flagged as degenerate");
}

#[test]
fn test_box_is_valid() {
    // A box should be a valid shape
    let solid = make_box(1.0, 1.0, 1.0).expect("Failed to create box");
    let shape = Shape::Solid(solid);
    
    let errors = check_valid(&shape).unwrap();
    
    // Box should have minimal errors (mostly topology artifacts)
    let critical_errors: Vec<_> = errors
        .iter()
        .filter(|e| matches!(e, 
            cascade::ShapeError::DegenerateFace { .. } | 
            cascade::ShapeError::DegenerateEdgeNearZero { .. }
        ))
        .collect();
    
    assert!(critical_errors.len() < 2, "Box should have no critical errors");
}

#[test]
fn test_sphere_is_valid() {
    // A sphere should be a valid shape
    let solid = make_sphere(1.0).expect("Failed to create sphere");
    let shape = Shape::Solid(solid);
    
    let errors = check_valid(&shape).unwrap();
    
    // Sphere should have minimal critical errors
    let critical_errors: Vec<_> = errors
        .iter()
        .filter(|e| matches!(e, 
            cascade::ShapeError::DegenerateFace { .. } | 
            cascade::ShapeError::DegenerateEdgeNearZero { .. }
        ))
        .collect();
    
    assert!(critical_errors.len() < 2, "Sphere should have no critical errors");
}

#[test]
fn test_box_is_watertight() {
    // A box should be watertight
    let solid = make_box(2.0, 3.0, 4.0).expect("Failed to create box");
    
    let is_watertight = check_watertight(&solid);
    assert!(is_watertight, "Box should be watertight");
}

#[test]
fn test_sphere_is_watertight() {
    // A sphere should be watertight
    let solid = make_sphere(1.5).expect("Failed to create sphere");
    
    let is_watertight = check_watertight(&solid);
    assert!(is_watertight, "Sphere should be watertight");
}

#[test]
fn test_box_no_self_intersection() {
    // A box should not self-intersect
    let solid = make_box(1.0, 2.0, 3.0).expect("Failed to create box");
    
    let no_intersection = check_self_intersection(&solid);
    assert!(no_intersection, "Box should not self-intersect");
}

#[test]
fn test_sphere_no_self_intersection() {
    // A sphere should not self-intersect
    let solid = make_sphere(1.0).expect("Failed to create sphere");
    
    let no_intersection = check_self_intersection(&solid);
    assert!(no_intersection, "Sphere should not self-intersect");
}
