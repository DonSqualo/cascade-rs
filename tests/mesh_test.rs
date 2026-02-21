use cascade::{make_box, mesh::triangulate, mesh::export_stl, mesh::write_stl_binary};
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

#[test]
fn test_stl_binary_export() {
    // Create a simple box
    let solid = make_box(5.0, 5.0, 5.0).expect("Failed to create box");
    
    // Triangulate
    let mesh = triangulate(&solid, 0.5).expect("Failed to triangulate");
    let triangle_count = mesh.triangles.len();
    
    // Export to binary STL
    let output_path = "/tmp/test_mesh_binary.stl";
    write_stl_binary(&mesh, output_path).expect("Failed to export binary STL");
    
    // Verify file exists
    assert!(Path::new(output_path).exists(), "Binary STL file should be created");
    
    // Read and verify binary format
    let content = std::fs::read(output_path).expect("Failed to read binary STL");
    
    // Check header (80 bytes)
    assert!(content.len() >= 84, "Binary STL should have at least 84 bytes (header + count)");
    assert!(content[..28].starts_with(b"cascade-rs binary STL"), "Header should contain signature");
    
    // Check triangle count (bytes 80-83, little-endian u32)
    let count_bytes: [u8; 4] = content[80..84].try_into().unwrap();
    let file_triangle_count = u32::from_le_bytes(count_bytes) as usize;
    assert_eq!(file_triangle_count, triangle_count, "Triangle count should match");
    
    // Verify total file size: 80 (header) + 4 (count) + 50 * triangles
    let expected_size = 80 + 4 + 50 * triangle_count;
    assert_eq!(content.len(), expected_size, "Binary STL file size should be correct");
}

#[test]
fn test_stl_binary_triangle_data() {
    // Create a minimal mesh manually for precise verification
    let mesh = cascade::mesh::TriangleMesh {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        normals: vec![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        triangles: vec![[0, 1, 2]],
    };
    
    let output_path = "/tmp/test_single_triangle.stl";
    write_stl_binary(&mesh, output_path).expect("Failed to export binary STL");
    
    let content = std::fs::read(output_path).expect("Failed to read binary STL");
    
    // Expected size: 80 + 4 + 50 = 134 bytes
    assert_eq!(content.len(), 134, "Single triangle STL should be 134 bytes");
    
    // Check triangle count
    let count_bytes: [u8; 4] = content[80..84].try_into().unwrap();
    assert_eq!(u32::from_le_bytes(count_bytes), 1, "Should have 1 triangle");
    
    // Check normal (first 12 bytes of triangle data, starting at byte 84)
    // Normal should be [0, 0, 1] (calculated from vertices)
    let nx = f32::from_le_bytes(content[84..88].try_into().unwrap());
    let ny = f32::from_le_bytes(content[88..92].try_into().unwrap());
    let nz = f32::from_le_bytes(content[92..96].try_into().unwrap());
    assert!((nx - 0.0).abs() < 0.001, "Normal X should be 0");
    assert!((ny - 0.0).abs() < 0.001, "Normal Y should be 0");
    assert!((nz - 1.0).abs() < 0.001, "Normal Z should be 1");
    
    // Check first vertex (0, 0, 0)
    let v1x = f32::from_le_bytes(content[96..100].try_into().unwrap());
    let v1y = f32::from_le_bytes(content[100..104].try_into().unwrap());
    let v1z = f32::from_le_bytes(content[104..108].try_into().unwrap());
    assert!((v1x - 0.0).abs() < 0.001, "V1.X should be 0");
    assert!((v1y - 0.0).abs() < 0.001, "V1.Y should be 0");
    assert!((v1z - 0.0).abs() < 0.001, "V1.Z should be 0");
    
    // Check second vertex (1, 0, 0)
    let v2x = f32::from_le_bytes(content[108..112].try_into().unwrap());
    let v2y = f32::from_le_bytes(content[112..116].try_into().unwrap());
    let v2z = f32::from_le_bytes(content[116..120].try_into().unwrap());
    assert!((v2x - 1.0).abs() < 0.001, "V2.X should be 1");
    assert!((v2y - 0.0).abs() < 0.001, "V2.Y should be 0");
    assert!((v2z - 0.0).abs() < 0.001, "V2.Z should be 0");
    
    // Check third vertex (0, 1, 0)
    let v3x = f32::from_le_bytes(content[120..124].try_into().unwrap());
    let v3y = f32::from_le_bytes(content[124..128].try_into().unwrap());
    let v3z = f32::from_le_bytes(content[128..132].try_into().unwrap());
    assert!((v3x - 0.0).abs() < 0.001, "V3.X should be 0");
    assert!((v3y - 1.0).abs() < 0.001, "V3.Y should be 1");
    assert!((v3z - 0.0).abs() < 0.001, "V3.Z should be 0");
    
    // Check attribute byte count (should be 0)
    let attr = u16::from_le_bytes(content[132..134].try_into().unwrap());
    assert_eq!(attr, 0, "Attribute byte count should be 0");
}
