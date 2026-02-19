//! Test validation with known-bad geometry

use cascade::{check_valid, check_watertight, check_self_intersection};
use cascade::{Shape, Solid, Shell, Face, Wire, Edge, Vertex};
use cascade::brep::CurveType;

fn main() {
    println!("Testing shape validation with known-bad geometry...\n");
    
    // Test 1: Degenerate edge (zero length)
    println!("Test 1: Degenerate edge");
    let degenerate_edge = Edge {
        start: Vertex::new(0.0, 0.0, 0.0),
        end: Vertex::new(0.0, 0.0, 0.0),  // Same point!
        curve_type: CurveType::Line,
    };
    
    let edge_shape = Shape::Edge(degenerate_edge.clone());
    match check_valid(&edge_shape) {
        Ok(errors) => {
            if errors.is_empty() {
                println!("  ✗ Expected errors but got none");
            } else {
                println!("  ✓ Found degenerate edge: {:?}", errors[0]);
            }
        }
        Err(e) => println!("  ✗ Error: {}", e),
    }
    
    // Test 2: Very small edge
    println!("\nTest 2: Very small edge");
    let tiny_edge = Edge {
        start: Vertex::new(0.0, 0.0, 0.0),
        end: Vertex::new(1e-8, 1e-8, 1e-8),
        curve_type: CurveType::Line,
    };
    
    let tiny_shape = Shape::Edge(tiny_edge);
    match check_valid(&tiny_shape) {
        Ok(errors) => {
            if errors.is_empty() {
                println!("  ✗ Expected errors but got none");
            } else {
                println!("  ✓ Found tiny edge: {:?}", errors[0]);
            }
        }
        Err(e) => println!("  ✗ Error: {}", e),
    }
    
    // Test 3: Normal edge (should pass)
    println!("\nTest 3: Normal edge (should be valid)");
    let normal_edge = Edge {
        start: Vertex::new(0.0, 0.0, 0.0),
        end: Vertex::new(1.0, 0.0, 0.0),
        curve_type: CurveType::Line,
    };
    
    let normal_shape = Shape::Edge(normal_edge);
    match check_valid(&normal_shape) {
        Ok(errors) => {
            if errors.is_empty() {
                println!("  ✓ Edge is valid");
            } else {
                println!("  ✗ Unexpected errors: {:?}", errors);
            }
        }
        Err(e) => println!("  ✗ Error: {}", e),
    }
    
    // Test 4: Test watertight check with a box (should be watertight)
    println!("\nTest 4: Watertight check with box");
    match cascade::make_box(1.0, 1.0, 1.0) {
        Ok(solid) => {
            let watertight = check_watertight(&solid);
            if watertight {
                println!("  ✓ Box is watertight");
            } else {
                println!("  ✗ Box should be watertight");
            }
        }
        Err(e) => println!("  ✗ Error creating box: {}", e),
    }
    
    // Test 5: Self-intersection check
    println!("\nTest 5: Self-intersection check with box");
    match cascade::make_box(1.0, 1.0, 1.0) {
        Ok(solid) => {
            let no_self_intersection = check_self_intersection(&solid);
            if no_self_intersection {
                println!("  ✓ Box has no self-intersections");
            } else {
                println!("  ✗ Box should have no self-intersections");
            }
        }
        Err(e) => println!("  ✗ Error creating box: {}", e),
    }
    
    println!("\nAll tests completed!");
}
