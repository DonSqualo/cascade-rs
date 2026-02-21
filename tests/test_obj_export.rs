use cascade::{make_box, make_sphere, make_cylinder, mesh::triangulate, io::write_obj};
use std::path::Path;

#[test]
fn test_obj_export_box() {
    // Create a simple box
    let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
    
    // Triangulate with tolerance
    let mesh = triangulate(&solid, 1.0).expect("Failed to triangulate");
    
    // Export to OBJ
    let output_path = "/tmp/test_obj_box.obj";
    write_obj(&mesh, output_path).expect("Failed to export OBJ");
    
    // Verify file exists
    assert!(Path::new(output_path).exists(), "OBJ file should be created");
    
    // Check file has proper content
    let content = std::fs::read_to_string(output_path).expect("Failed to read OBJ");
    
    // Should have header comment
    assert!(content.contains("# Wavefront OBJ"), "OBJ should have header comment");
    
    // Should have vertices
    assert!(content.contains("v "), "OBJ should contain vertices");
    
    // Should have normals
    assert!(content.contains("vn "), "OBJ should contain normals");
    
    // Should have faces (1-indexed)
    assert!(content.contains("f "), "OBJ should contain faces");
}

#[test]
fn test_obj_export_sphere() {
    let solid = make_sphere(5.0).expect("Failed to create sphere");
    let mesh = triangulate(&solid, 0.5).expect("Failed to triangulate");
    
    let output_path = "/tmp/test_obj_sphere.obj";
    write_obj(&mesh, output_path).expect("Failed to export OBJ");
    
    let content = std::fs::read_to_string(output_path).expect("Failed to read OBJ");
    
    // Sphere should have many triangles
    let triangle_count: usize = content.lines().filter(|l| l.starts_with("f ")).count();
    assert!(triangle_count > 10, "Sphere should have multiple triangles, got {}", triangle_count);
}

#[test]
fn test_obj_export_cylinder() {
    let solid = make_cylinder(3.0, 10.0).expect("Failed to create cylinder");
    let mesh = triangulate(&solid, 0.5).expect("Failed to triangulate");
    
    let output_path = "/tmp/test_obj_cylinder.obj";
    write_obj(&mesh, output_path).expect("Failed to export OBJ");
    
    assert!(Path::new(output_path).exists(), "OBJ file should be created");
}

#[test]
fn test_obj_face_indices_are_one_indexed() {
    // Create minimal geometry
    let solid = make_box(1.0, 1.0, 1.0).expect("Failed to create box");
    let mesh = triangulate(&solid, 0.5).expect("Failed to triangulate");
    
    let output_path = "/tmp/test_obj_indices.obj";
    write_obj(&mesh, output_path).expect("Failed to export OBJ");
    
    let content = std::fs::read_to_string(output_path).expect("Failed to read OBJ");
    
    // Find all face lines and check indices
    for line in content.lines() {
        if line.starts_with("f ") {
            // Parse face indices - they should all be >= 1 (1-indexed)
            let parts: Vec<&str> = line.split_whitespace().collect();
            for part in &parts[1..] {
                // Handle both "f 1 2 3" and "f 1//1 2//2 3//3" formats
                let idx_str = part.split("//").next().unwrap();
                let idx: usize = idx_str.parse().expect("Face index should be a number");
                assert!(idx >= 1, "OBJ face indices must be 1-indexed, found {}", idx);
            }
        }
    }
}
