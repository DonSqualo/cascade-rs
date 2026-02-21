use cascade::{make_sphere, mesh::triangulate_with_deflection};

#[test]
fn test_sphere_deflection_coarse() {
    // Create a sphere with radius 10.0
    let sphere = make_sphere(10.0).expect("Failed to create sphere");
    
    // Triangulate with coarse deflection
    let mesh_coarse = triangulate_with_deflection(&sphere, 1.0)
        .expect("Failed to triangulate with deflection");
    
    // Check that we got vertices and triangles
    assert!(!mesh_coarse.vertices.is_empty(), "Mesh should have vertices");
    assert!(!mesh_coarse.triangles.is_empty(), "Mesh should have triangles");
    assert_eq!(
        mesh_coarse.vertices.len(),
        mesh_coarse.normals.len(),
        "Vertices and normals should match"
    );
    
    // Store the count for later comparison
    let coarse_triangle_count = mesh_coarse.triangles.len();
    println!("Coarse mesh (deflection=1.0): {} triangles", coarse_triangle_count);
    
    // Ensure we have a reasonable number of triangles for a sphere
    assert!(coarse_triangle_count >= 8, "Should have at least 8 triangles for a sphere");
}

#[test]
fn test_sphere_deflection_fine() {
    // Create a sphere with radius 10.0
    let sphere = make_sphere(10.0).expect("Failed to create sphere");
    
    // Triangulate with fine deflection
    let mesh_fine = triangulate_with_deflection(&sphere, 0.1)
        .expect("Failed to triangulate with deflection");
    
    // Check that we got vertices and triangles
    assert!(!mesh_fine.vertices.is_empty(), "Mesh should have vertices");
    assert!(!mesh_fine.triangles.is_empty(), "Mesh should have triangles");
    assert_eq!(
        mesh_fine.vertices.len(),
        mesh_fine.normals.len(),
        "Vertices and normals should match"
    );
    
    let fine_triangle_count = mesh_fine.triangles.len();
    println!("Fine mesh (deflection=0.1): {} triangles", fine_triangle_count);
    
    // Ensure we have a reasonable number of triangles for a sphere
    assert!(fine_triangle_count >= 8, "Should have at least 8 triangles for a sphere");
}

#[test]
fn test_deflection_difference() {
    // Create a sphere with radius 10.0
    let sphere = make_sphere(10.0).expect("Failed to create sphere");
    
    // Triangulate with different deflections
    let mesh_coarse = triangulate_with_deflection(&sphere, 1.0)
        .expect("Failed to triangulate coarse");
    let mesh_medium = triangulate_with_deflection(&sphere, 0.5)
        .expect("Failed to triangulate medium");
    let mesh_fine = triangulate_with_deflection(&sphere, 0.1)
        .expect("Failed to triangulate fine");
    
    let coarse_count = mesh_coarse.triangles.len();
    let medium_count = mesh_medium.triangles.len();
    let fine_count = mesh_fine.triangles.len();
    
    println!(
        "Triangle counts: coarse={}, medium={}, fine={}",
        coarse_count, medium_count, fine_count
    );
    
    // Verify that smaller deflection produces more triangles
    assert!(
        medium_count > coarse_count,
        "Medium deflection (0.5) should produce more triangles than coarse (1.0): {} vs {}",
        medium_count,
        coarse_count
    );
    
    assert!(
        fine_count > medium_count,
        "Fine deflection (0.1) should produce more triangles than medium (0.5): {} vs {}",
        fine_count,
        medium_count
    );
}

#[test]
fn test_deflection_positive_only() {
    // Create a sphere
    let sphere = make_sphere(5.0).expect("Failed to create sphere");
    
    // Test that negative deflection is rejected
    let result_negative = triangulate_with_deflection(&sphere, -1.0);
    assert!(result_negative.is_err(), "Negative deflection should be rejected");
    
    // Test that zero deflection is rejected
    let result_zero = triangulate_with_deflection(&sphere, 0.0);
    assert!(result_zero.is_err(), "Zero deflection should be rejected");
}

#[test]
fn test_deflection_vertex_validity() {
    // Create a sphere
    let sphere = make_sphere(10.0).expect("Failed to create sphere");
    
    // Triangulate with deflection
    let mesh = triangulate_with_deflection(&sphere, 0.5)
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
fn test_deflection_scales_properly() {
    // Test that doubling the sphere radius doesn't drastically change triangle count
    // with the same deflection (should scale appropriately)
    
    let sphere_small = make_sphere(5.0).expect("Failed to create small sphere");
    let sphere_large = make_sphere(10.0).expect("Failed to create large sphere");
    
    // Use the same deflection for both
    let mesh_small = triangulate_with_deflection(&sphere_small, 0.5)
        .expect("Failed to triangulate small sphere");
    let mesh_large = triangulate_with_deflection(&sphere_large, 0.5)
        .expect("Failed to triangulate large sphere");
    
    let small_count = mesh_small.triangles.len();
    let large_count = mesh_large.triangles.len();
    
    println!(
        "Scaled sphere: small radius (5.0) = {} triangles, large radius (10.0) = {} triangles",
        small_count, large_count
    );
    
    // The larger sphere should have more triangles than the smaller one
    // (for the same deflection tolerance)
    assert!(
        large_count > small_count,
        "Larger sphere should have more triangles for same deflection: {} vs {}",
        large_count,
        small_count
    );
}
