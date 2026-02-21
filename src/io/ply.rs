//! Polygon File Format (PLY) mesh export

use crate::mesh::TriangleMesh;
use crate::Result;
use std::fs::File;
use std::io::Write;

/// Write a triangle mesh to PLY (Polygon File Format) in ASCII format
///
/// The PLY format is a simple text format for storing 3D mesh data.
/// This implementation exports in ASCII format with vertices and face indices.
///
/// # Format
/// ```text
/// ply
/// format ascii 1.0
/// element vertex N
/// property float x
/// property float y
/// property float z
/// element face M
/// property list uchar int vertex_indices
/// end_header
/// ```
///
/// # Arguments
/// * `mesh` - The triangle mesh to export
/// * `path` - Output file path
///
/// # Example
/// ```rust,no_run
/// use cascade::mesh::TriangleMesh;
/// use cascade::io::write_ply;
///
/// let mesh = TriangleMesh {
///     vertices: vec![
///         [0.0, 0.0, 0.0],
///         [1.0, 0.0, 0.0],
///         [0.0, 1.0, 0.0],
///     ],
///     normals: vec![],
///     triangles: vec![[0, 1, 2]],
/// };
///
/// write_ply(&mesh, "output.ply").unwrap();
/// ```
pub fn write_ply(mesh: &TriangleMesh, path: &str) -> Result<()> {
    let mut file = File::create(path)?;
    
    let vertex_count = mesh.vertices.len();
    let face_count = mesh.triangles.len();
    
    // Write PLY header
    writeln!(file, "ply")?;
    writeln!(file, "format ascii 1.0")?;
    writeln!(file, "element vertex {}", vertex_count)?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    writeln!(file, "element face {}", face_count)?;
    writeln!(file, "property list uchar int vertex_indices")?;
    writeln!(file, "end_header")?;
    
    // Write vertices
    for vertex in &mesh.vertices {
        writeln!(file, "{:.6} {:.6} {:.6}", vertex[0], vertex[1], vertex[2])?;
    }
    
    // Write faces
    for triangle in &mesh.triangles {
        let [i, j, k] = triangle;
        // Each triangle has 3 vertices
        writeln!(file, "3 {} {} {}", i, j, k)?;
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_write_ply_basic() {
        // Create a simple triangle mesh (single triangle)
        let mesh = TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            normals: vec![],
            triangles: vec![[0, 1, 2]],
        };
        
        let output_path = "/tmp/test_basic.ply";
        write_ply(&mesh, output_path).expect("Failed to write PLY");
        
        let content = std::fs::read_to_string(output_path).expect("Failed to read PLY");
        
        // Check PLY header
        assert!(content.contains("ply"));
        assert!(content.contains("format ascii 1.0"));
        assert!(content.contains("element vertex 3"));
        assert!(content.contains("element face 1"));
        assert!(content.contains("property float x"));
        assert!(content.contains("property float y"));
        assert!(content.contains("property float z"));
        assert!(content.contains("property list uchar int vertex_indices"));
        assert!(content.contains("end_header"));
        
        // Check vertices
        assert!(content.contains("0.000000 0.000000 0.000000"));
        assert!(content.contains("1.000000 0.000000 0.000000"));
        assert!(content.contains("0.000000 1.000000 0.000000"));
        
        // Check face
        assert!(content.contains("3 0 1 2"));
    }
    
    #[test]
    fn test_write_ply_multiple_faces() {
        // Create a mesh with multiple triangles (e.g., two triangles forming a square)
        let mesh = TriangleMesh {
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
        
        let output_path = "/tmp/test_multiple_faces.ply";
        write_ply(&mesh, output_path).expect("Failed to write PLY");
        
        let content = std::fs::read_to_string(output_path).expect("Failed to read PLY");
        
        // Check header counts
        assert!(content.contains("element vertex 4"));
        assert!(content.contains("element face 2"));
        
        // Check vertices
        assert!(content.contains("0.000000 0.000000 0.000000"));
        assert!(content.contains("1.000000 0.000000 0.000000"));
        assert!(content.contains("1.000000 1.000000 0.000000"));
        assert!(content.contains("0.000000 1.000000 0.000000"));
        
        // Check faces
        assert!(content.contains("3 0 1 2"));
        assert!(content.contains("3 0 2 3"));
    }
    
    #[test]
    fn test_write_ply_empty_mesh() {
        let mesh = TriangleMesh {
            vertices: vec![],
            normals: vec![],
            triangles: vec![],
        };
        
        let output_path = "/tmp/test_empty.ply";
        write_ply(&mesh, output_path).expect("Failed to write PLY");
        
        let content = std::fs::read_to_string(output_path).expect("Failed to read PLY");
        
        // Check that header is valid
        assert!(content.contains("ply"));
        assert!(content.contains("format ascii 1.0"));
        assert!(content.contains("element vertex 0"));
        assert!(content.contains("element face 0"));
        assert!(content.contains("end_header"));
    }
}
