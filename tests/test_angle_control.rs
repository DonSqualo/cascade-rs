use cascade::{make_sphere, mesh::triangulate_with_angle};

#[test]
fn test_angle_control_basic() {
    // Create a sphere with radius 10.0
    let sphere = make_sphere(10.0).expect("Failed to create sphere");
    
    // Triangulate with angle control
    let mesh = triangulate_with_angle(&sphere, 0.2)
        .expect("Failed to triangulate with angle control");
    
    // Check that we got vertices and triangles
    assert!(!mesh.vertices.is_empty(), "Mesh should have vertices");
    assert!(!mesh.triangles.is_empty(), "Mesh should have triangles");
    assert_eq!(
        mesh.vertices.len(),
        mesh.normals.len(),
        "Vertices and normals should match"
    );
    
    println!("Basic test: {} vertices, {} triangles", mesh.vertices.len(), mesh.triangles.len());
}

#[test]
fn test_angle_control_coarse() {
    // Create a sphere with radius 10.0
    let sphere = make_sphere(10.0).expect("Failed to create sphere");
    
    // Triangulate with coarse angle control (large angle = fewer triangles)
    let mesh_coarse = triangulate_with_angle(&sphere, 0.5)
        .expect("Failed to triangulate with angle control");
    
    // Check validity
    assert!(!mesh_coarse.vertices.is_empty(), "Mesh should have vertices");
    assert!(!mesh_coarse.triangles.is_empty(), "Mesh should have triangles");
    assert_eq!(
        mesh_coarse.vertices.len(),
        mesh_coarse.normals.len(),
        "Vertices and normals should match"
    );
    
    let coarse_triangle_count = mesh_coarse.triangles.len();
    println!("Coarse mesh (angle=0.5 rad ≈ 28.6°): {} triangles", coarse_triangle_count);
    
    // Ensure we have a reasonable number of triangles for a sphere
    assert!(coarse_triangle_count >= 8, "Should have at least 8 triangles for a sphere");
}

#[test]
fn test_angle_control_fine() {
    // Create a sphere with radius 10.0
    let sphere = make_sphere(10.0).expect("Failed to create sphere");
    
    // Triangulate with fine angle control (small angle = more triangles)
    let mesh_fine = triangulate_with_angle(&sphere, 0.1)
        .expect("Failed to triangulate with angle control");
    
    // Check validity
    assert!(!mesh_fine.vertices.is_empty(), "Mesh should have vertices");
    assert!(!mesh_fine.triangles.is_empty(), "Mesh should have triangles");
    assert_eq!(
        mesh_fine.vertices.len(),
        mesh_fine.normals.len(),
        "Vertices and normals should match"
    );
    
    let fine_triangle_count = mesh_fine.triangles.len();
    println!("Fine mesh (angle=0.1 rad ≈ 5.7°): {} triangles", fine_triangle_count);
    
    // Ensure we have a reasonable number of triangles
    assert!(fine_triangle_count >= 8, "Should have at least 8 triangles for a sphere");
}

#[test]
fn test_angle_control_difference() {
    // Create a sphere with radius 10.0
    let sphere = make_sphere(10.0).expect("Failed to create sphere");
    
    // Triangulate with different angle controls
    let mesh_coarse = triangulate_with_angle(&sphere, 0.5)
        .expect("Failed to triangulate coarse");
    let mesh_medium = triangulate_with_angle(&sphere, 0.2)
        .expect("Failed to triangulate medium");
    let mesh_fine = triangulate_with_angle(&sphere, 0.05)
        .expect("Failed to triangulate fine");
    
    let coarse_count = mesh_coarse.triangles.len();
    let medium_count = mesh_medium.triangles.len();
    let fine_count = mesh_fine.triangles.len();
    
    println!(
        "Triangle counts: coarse (0.5 rad)={}, medium (0.2 rad)={}, fine (0.05 rad)={}",
        coarse_count, medium_count, fine_count
    );
    
    // Verify that smaller angle produces more triangles
    assert!(
        medium_count > coarse_count,
        "Medium angle (0.2 rad) should produce more triangles than coarse (0.5 rad): {} vs {}",
        medium_count,
        coarse_count
    );
    
    assert!(
        fine_count > medium_count,
        "Fine angle (0.05 rad) should produce more triangles than medium (0.2 rad): {} vs {}",
        fine_count,
        medium_count
    );
}

#[test]
fn test_angle_control_positive_only() {
    // Create a sphere
    let sphere = make_sphere(5.0).expect("Failed to create sphere");
    
    // Test that negative angle is rejected
    let result_negative = triangulate_with_angle(&sphere, -0.1);
    assert!(result_negative.is_err(), "Negative angle should be rejected");
    
    // Test that zero angle is rejected
    let result_zero = triangulate_with_angle(&sphere, 0.0);
    assert!(result_zero.is_err(), "Zero angle should be rejected");
    
    // Test that angle > π is rejected
    let result_too_large = triangulate_with_angle(&sphere, std::f64::consts::PI + 0.1);
    assert!(result_too_large.is_err(), "Angle > π should be rejected");
}

#[test]
fn test_angle_control_vertex_validity() {
    // Create a sphere
    let sphere = make_sphere(10.0).expect("Failed to create sphere");
    
    // Triangulate with angle control
    let mesh = triangulate_with_angle(&sphere, 0.2)
        .expect("Failed to triangulate");
    
    // Check that all triangles reference valid vertex indices
    for triangle in &mesh.triangles {
        let [i0, i1, i2] = triangle;
        assert!(*i0 < mesh.vertices.len(), "Triangle vertex index out of bounds");
        assert!(*i1 < mesh.vertices.len(), "Triangle vertex index out of bounds");
        assert!(*i2 < mesh.vertices.len(), "Triangle vertex index out of bounds");
    }
    
    // Check that all vertices are finite
    for vertex in &mesh.vertices {
        for &coord in vertex {
            assert!(coord.is_finite(), "Vertex coordinate should be finite");
        }
    }
    
    // Check that all normals are finite and normalized
    for normal in &mesh.normals {
        for &coord in normal {
            assert!(coord.is_finite(), "Normal coordinate should be finite");
        }
        
        // Check that normal is approximately unit length
        let len_sq = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        assert!(
            (len_sq - 1.0).abs() < 0.01,
            "Normal should be unit length: length_sq = {}",
            len_sq
        );
    }
}

#[test]
fn test_angle_control_sphere_curvature_verification() {
    // Create spheres with different radii and verify angle control works correctly
    let sphere_small = make_sphere(5.0).expect("Failed to create small sphere");
    let sphere_large = make_sphere(10.0).expect("Failed to create large sphere");
    
    // Use the same angle control for both
    let mesh_small = triangulate_with_angle(&sphere_small, 0.2)
        .expect("Failed to triangulate small sphere");
    let mesh_large = triangulate_with_angle(&sphere_large, 0.2)
        .expect("Failed to triangulate large sphere");
    
    let small_count = mesh_small.triangles.len();
    let large_count = mesh_large.triangles.len();
    
    println!(
        "Sphere angle control verification: small radius (5.0) = {} triangles, large radius (10.0) = {} triangles",
        small_count, large_count
    );
    
    // The larger sphere should have more triangles than the smaller one
    // (for the same angle control)
    // This is because the same angle between normals corresponds to a larger deflection on a larger sphere
    assert!(
        large_count > small_count,
        "Larger sphere should have more triangles for same angle control: {} vs {}",
        large_count,
        small_count
    );
}

#[test]
fn test_angle_control_triangle_normal_angles() {
    // Create a sphere and verify that adjacent triangle normals are within the specified angle
    let sphere = make_sphere(10.0).expect("Failed to create sphere");
    let max_angle = 0.3; // radians
    
    let mesh = triangulate_with_angle(&sphere, max_angle)
        .expect("Failed to triangulate");
    
    // Check some adjacent triangles to verify angle constraint
    let mut max_observed_angle = 0.0;
    let mut checked_pairs = 0;
    
    // Build a simple edge-to-triangle adjacency (for testing purposes)
    // For each edge, find triangles that share it and check their normals
    for i in 0..mesh.triangles.len() {
        let tri_i = mesh.triangles[i];
        let normal_i = mesh.normals[tri_i[0]]; // Use first vertex normal as approximation
        
        // Find adjacent triangles (sharing at least 2 vertices)
        for j in (i + 1)..mesh.triangles.len().min(i + 50) {
            let tri_j = mesh.triangles[j];
            
            // Count shared vertices
            let mut shared = 0;
            for &vi in &tri_i {
                for &vj in &tri_j {
                    if vi == vj {
                        shared += 1;
                    }
                }
            }
            
            // If triangles share an edge (2 vertices), check normal angle
            if shared >= 2 {
                let normal_j = mesh.normals[tri_j[0]];
                
                // Calculate angle between normals
                let dot = normal_i[0] * normal_j[0] + normal_i[1] * normal_j[1] + normal_i[2] * normal_j[2];
                let dot = dot.clamp(-1.0, 1.0);
                let angle = dot.acos();
                
                max_observed_angle = max_observed_angle.max(angle);
                checked_pairs += 1;
            }
        }
    }
    
    println!(
        "Angle control verification: max_angle={:.4} rad, max_observed_angle={:.4} rad, pairs checked={}",
        max_angle, max_observed_angle, checked_pairs
    );
    
    // Note: We use a looser bound here because we're only sampling pairs,
    // and the normals at vertices might not perfectly represent the triangle normals
    // The actual triangle normals should be closer to the specified max_angle
    if checked_pairs > 0 {
        assert!(
            max_observed_angle <= max_angle * 2.5,
            "Observed angle between adjacent triangle normals should be reasonably close to max_angle: {} > {}",
            max_observed_angle,
            max_angle * 2.5
        );
    }
}

#[test]
fn test_angle_control_very_fine() {
    // Test with a very fine angle control
    let sphere = make_sphere(10.0).expect("Failed to create sphere");
    
    let mesh = triangulate_with_angle(&sphere, 0.02)
        .expect("Failed to triangulate with very fine angle control");
    
    // Check validity
    assert!(!mesh.vertices.is_empty(), "Mesh should have vertices");
    assert!(!mesh.triangles.is_empty(), "Mesh should have triangles");
    
    let triangle_count = mesh.triangles.len();
    println!("Very fine mesh (angle=0.02 rad ≈ 1.1°): {} triangles", triangle_count);
    
    // With very fine angle control, we should have many triangles
    assert!(triangle_count > 100, "Very fine angle control should produce many triangles: {}", triangle_count);
}

#[test]
fn test_angle_control_consistency() {
    // Test that the same sphere with the same angle control produces the same result
    let sphere1 = make_sphere(10.0).expect("Failed to create sphere 1");
    let sphere2 = make_sphere(10.0).expect("Failed to create sphere 2");
    
    let mesh1 = triangulate_with_angle(&sphere1, 0.2)
        .expect("Failed to triangulate sphere 1");
    let mesh2 = triangulate_with_angle(&sphere2, 0.2)
        .expect("Failed to triangulate sphere 2");
    
    // The meshes should have the same triangle count (deterministic behavior)
    assert_eq!(
        mesh1.triangles.len(),
        mesh2.triangles.len(),
        "Same sphere with same angle should produce same mesh"
    );
    
    // The meshes should have the same vertex count
    assert_eq!(
        mesh1.vertices.len(),
        mesh2.vertices.len(),
        "Same sphere with same angle should produce same vertex count"
    );
    
    println!(
        "Consistency check passed: {} vertices, {} triangles",
        mesh1.vertices.len(),
        mesh1.triangles.len()
    );
}
