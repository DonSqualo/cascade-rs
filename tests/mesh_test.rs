use cascade::{make_box, mesh::triangulate, mesh::export_stl};
use std::path::Path;

#[test]
fn test_triangulate_box() {
    // Create a simple box
    let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
    
    // Triangulate with tolerance
    let mesh = triangulate(&solid, 1.0).expect("Failed to triangulate");
    
    // Check that we got vertices and triangles
    assert!(!mesh.vertices.is_empty(), "Mesh should have vertices");
    assert!(!mesh.triangles.is_empty(), "Mesh should have triangles");
    assert_eq!(mesh.vertices.len(), mesh.normals.len(), "Vertices and normals should match");
}

#[test]
fn test_export_stl_ascii() {
    // Create a simple box
    let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
    
    // Triangulate
    let mesh = triangulate(&solid, 0.5).expect("Failed to triangulate");
    
    // Export to STL
    let output_path = "/tmp/test_mesh.stl";
    export_stl(&mesh, output_path).expect("Failed to export STL");
    
    // Verify file exists
    assert!(Path::new(output_path).exists(), "STL file should be created");
    
    // Check file has content
    let content = std::fs::read_to_string(output_path).expect("Failed to read STL");
    assert!(content.starts_with("solid mesh"), "STL should start with 'solid mesh'");
    assert!(content.contains("facet normal"), "STL should contain facet data");
    assert!(content.ends_with("endsolid mesh\n"), "STL should end with 'endsolid mesh'");
}
