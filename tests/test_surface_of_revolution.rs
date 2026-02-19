//! Tests for SurfaceOfRevolution

use cascade::brep::{SurfaceType, CurveType};
use std::f64::consts::PI;

#[test]
fn test_surface_of_revolution_line_creates_cone() {
    // Revolve a line segment around Y-axis to create a cone-like surface
    // Line from (1, 0, 0) to (2, 1, 0)
    let surface = SurfaceType::SurfaceOfRevolution {
        basis_curve: CurveType::Line,
        curve_start: [1.0, 0.0, 0.0],
        curve_end: [2.0, 1.0, 0.0],
        axis_location: [0.0, 0.0, 0.0],
        axis_direction: [0.0, 1.0, 0.0],
    };

    // At u=0, v=0, should be at curve start (1, 0, 0)
    let p00 = surface.point_at(0.0, 0.0);
    assert!((p00[0] - 1.0).abs() < 1e-10);
    assert!(p00[1].abs() < 1e-10);
    assert!(p00[2].abs() < 1e-10);

    // At u=1, v=0, should be at curve end (2, 1, 0)
    let p10 = surface.point_at(1.0, 0.0);
    assert!((p10[0] - 2.0).abs() < 1e-10);
    assert!((p10[1] - 1.0).abs() < 1e-10);
    assert!(p10[2].abs() < 1e-10);

    // At u=0, v=π/2, should be rotated 90° around Y-axis
    // Point (1, 0, 0) rotated 90° around Y-axis = (0, 0, -1)
    // This is because positive rotation follows right-hand rule
    let p_rot = surface.point_at(0.0, PI / 2.0);
    assert!(p_rot[0].abs() < 1e-10, "x should be ~0, got {}", p_rot[0]);
    assert!(p_rot[1].abs() < 1e-10, "y should be ~0, got {}", p_rot[1]);
    assert!((p_rot[2].abs() - 1.0).abs() < 1e-10, "z should be ~±1, got {}", p_rot[2]);
}

#[test]
fn test_surface_of_revolution_vertical_line_cylinder() {
    // Revolve a vertical line around Y-axis to create a cylinder
    // Line from (2, 0, 0) to (2, 3, 0)
    let surface = SurfaceType::SurfaceOfRevolution {
        basis_curve: CurveType::Line,
        curve_start: [2.0, 0.0, 0.0],
        curve_end: [2.0, 3.0, 0.0],
        axis_location: [0.0, 0.0, 0.0],
        axis_direction: [0.0, 1.0, 0.0],
    };

    // At any point, distance from Y-axis should be 2
    for v in [0.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0] {
        let p = surface.point_at(0.5, v);
        let radius = (p[0] * p[0] + p[2] * p[2]).sqrt();
        assert!((radius - 2.0).abs() < 1e-10, "Expected radius 2, got {}", radius);
    }
}

#[test]
fn test_surface_of_revolution_circle_creates_torus() {
    // Revolve a circle around Z-axis to create a torus
    // Circle centered at (2, 0, 0) with radius 0.5 in the XY plane
    let surface = SurfaceType::SurfaceOfRevolution {
        basis_curve: CurveType::Arc { center: [2.0, 0.0, 0.0], radius: 0.5 },
        curve_start: [2.5, 0.0, 0.0],  // Not used for Arc
        curve_end: [2.5, 0.0, 0.0],
        axis_location: [0.0, 0.0, 0.0],
        axis_direction: [0.0, 0.0, 1.0],
    };

    // At u=0 (angle=0 on circle), point should be at (2.5, 0, 0) in XY plane
    let p0 = surface.point_at(0.0, 0.0);
    assert!((p0[0] - 2.5).abs() < 1e-10, "Expected x=2.5, got x={}", p0[0]);
    
    // After rotating by π around Z-axis, x should become -2.5
    let p_rot = surface.point_at(0.0, PI);
    assert!((p_rot[0] + 2.5).abs() < 1e-10, "Expected x=-2.5, got x={}", p_rot[0]);
}

#[test]
fn test_surface_of_revolution_normal() {
    // Revolve a vertical line around Y-axis (creates cylinder)
    let surface = SurfaceType::SurfaceOfRevolution {
        basis_curve: CurveType::Line,
        curve_start: [1.0, 0.0, 0.0],
        curve_end: [1.0, 1.0, 0.0],
        axis_location: [0.0, 0.0, 0.0],
        axis_direction: [0.0, 1.0, 0.0],
    };

    // For a cylinder, the normal should point radially outward
    let normal = surface.normal_at(0.5, 0.0);
    // Normalize and check it has significant radial component
    let radial_component = (normal[0] * normal[0] + normal[2] * normal[2]).sqrt();
    assert!(radial_component > 0.5, "Normal should have significant radial component");
}

#[test]
fn test_surface_of_revolution_constructor() {
    let surface = SurfaceType::revolution(
        CurveType::Line,
        [1.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );

    match surface {
        SurfaceType::SurfaceOfRevolution { axis_direction, .. } => {
            assert_eq!(axis_direction, [0.0, 1.0, 0.0]);
        }
        _ => panic!("Expected SurfaceOfRevolution"),
    }
}

#[test]
fn test_surface_of_revolution_full_rotation() {
    // Verify that rotating by 2π returns to the same point
    let surface = SurfaceType::SurfaceOfRevolution {
        basis_curve: CurveType::Line,
        curve_start: [1.0, 0.0, 0.0],
        curve_end: [2.0, 1.0, 0.0],
        axis_location: [0.0, 0.0, 0.0],
        axis_direction: [0.0, 1.0, 0.0],
    };

    let p0 = surface.point_at(0.5, 0.0);
    let p_full = surface.point_at(0.5, 2.0 * PI);
    
    assert!((p0[0] - p_full[0]).abs() < 1e-10);
    assert!((p0[1] - p_full[1]).abs() < 1e-10);
    assert!((p0[2] - p_full[2]).abs() < 1e-10);
}
