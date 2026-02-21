//! Integration test for PLY export functionality

use cascade::make_box;
use cascade::mesh::triangulate;
use cascade::io::write_ply;
use std::fs;

#[test]
fn test_ply_export_box() {
    // Create a box and triangulate it
    let solid = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
    let mesh = triangulate(&solid, 1.0).expect("Failed to triangulate box");
    
    let output_path = "/tmp/test_box.ply";
    write_ply(&mesh, output_path).expect("Failed to write PLY");
    
    // Verify file exists
    assert!(fs::metadata(output_path).is_ok(), "PLY file was not created");
    
    // Read and verify content
    let content = fs::read_to_string(output_path).expect("Failed to read PLY file");
    
    // Check header
    assert!(content.starts_with("ply\n"), "PLY file should start with 'ply'");
    assert!(content.contains("format ascii 1.0"), "Should be ASCII format");
    assert!(content.contains("element vertex"), "Should have vertex element");
    assert!(content.contains("element face"), "Should have face element");
    assert!(content.contains("end_header"), "Should have end_header");
    
    // Verify we have vertices and faces
    assert!(mesh.vertices.len() > 0, "Mesh should have vertices");
    assert!(mesh.triangles.len() > 0, "Mesh should have triangles");
}

#[test]
fn test_ply_file_format_validity() {
    // Create a simple mesh
    let mesh = cascade::mesh::TriangleMesh {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        normals: vec![],
        triangles: vec![
            [0, 1, 2],
            [0, 2, 3],
        ],
    };
    
    let output_path = "/tmp/test_format.ply";
    write_ply(&mesh, output_path).expect("Failed to write PLY");
    
    let content = fs::read_to_string(output_path).expect("Failed to read PLY file");
    let lines: Vec<&str> = content.lines().collect();
    
    // Parse and validate header
    let mut header_end = 0;
    for (i, line) in lines.iter().enumerate() {
        if *line == "end_header" {
            header_end = i;
            break;
        }
    }
    
    assert!(header_end > 0, "Should have end_header");
    
    // Verify header structure
    assert_eq!(lines[0], "ply");
    assert_eq!(lines[1], "format ascii 1.0");
    
    // Count vertices and faces from data section
    let data_lines = &lines[header_end + 1..];
    let vertex_count = 4;
    let face_count = 2;
    
    assert_eq!(data_lines.len(), vertex_count + face_count, "Should have correct number of data lines");
    
    // Verify vertex data
    let first_vertex: Vec<&str> = data_lines[0].split_whitespace().collect();
    assert_eq!(first_vertex.len(), 3, "Vertex should have 3 coordinates");
    
    // Verify face data
    let first_face: Vec<&str> = data_lines[vertex_count].split_whitespace().collect();
    assert_eq!(first_face[0], "3", "Triangle should have 3 vertices");
    assert_eq!(first_face.len(), 4, "Face line should have count + 3 indices");
}
