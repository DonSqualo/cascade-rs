//! VRML (Virtual Reality Modeling Language) mesh export

use crate::mesh::TriangleMesh;
use crate::Result;
use std::fs::File;
use std::io::Write;

/// Write a triangle mesh to VRML97 format (.wrl)
///
/// VRML97 is a standardized text format for representing 3D geometry.
/// This implementation exports a mesh as an IndexedFaceSet within a Shape node,
/// optionally including normals if present in the mesh.
///
/// # Format
/// The output follows VRML97 (ISO/IEC 14772-1:1997) with:
/// - Shape node containing geometry
/// - IndexedFaceSet with Coordinate points
/// - coordIndex for face definitions
/// - Normal node if normals are available
/// - normalIndex if normals are present
///
/// # Arguments
/// * `mesh` - The triangle mesh to export
/// * `path` - Output file path (.wrl extension recommended)
///
/// # Example
/// ```rust,no_run
/// use cascade::mesh::TriangleMesh;
/// use cascade::io::write_vrml;
///
/// let mesh = TriangleMesh {
///     vertices: vec![
///         [0.0, 0.0, 0.0],
///         [1.0, 0.0, 0.0],
///         [0.0, 1.0, 0.0],
///     ],
///     normals: vec![
///         [0.0, 0.0, 1.0],
///         [0.0, 0.0, 1.0],
///         [0.0, 0.0, 1.0],
///     ],
///     triangles: vec![[0, 1, 2]],
/// };
///
/// write_vrml(&mesh, "output.wrl").unwrap();
/// ```
pub fn write_vrml(mesh: &TriangleMesh, path: &str) -> Result<()> {
    let mut file = File::create(path)?;
    
    // Write VRML header
    writeln!(file, "#VRML V2.0 utf8")?;
    writeln!(file, "")?;
    writeln!(file, "# VRML97 file exported by cascade-rs")?;
    writeln!(file, "# Vertices: {}", mesh.vertices.len())?;
    writeln!(file, "# Triangles: {}", mesh.triangles.len())?;
    writeln!(file, "")?;
    
    // Write Shape node
    writeln!(file, "Shape {{")?;
    writeln!(file, "  geometry IndexedFaceSet {{")?;
    
    // Check if we have normals
    let has_normals = !mesh.normals.is_empty() && mesh.normals.len() == mesh.vertices.len();
    
    // Write coordinate points
    writeln!(file, "    coord Coordinate {{")?;
    writeln!(file, "      point [")?;
    for (i, vertex) in mesh.vertices.iter().enumerate() {
        if i == mesh.vertices.len() - 1 {
            // Last vertex without trailing comma
            writeln!(file, "        {} {} {}", vertex[0], vertex[1], vertex[2])?;
        } else {
            writeln!(file, "        {} {} {},", vertex[0], vertex[1], vertex[2])?;
        }
    }
    writeln!(file, "      ]")?;
    writeln!(file, "    }}")?;
    
    // Write coordinate indices (faces)
    writeln!(file, "    coordIndex [")?;
    for (i, triangle) in mesh.triangles.iter().enumerate() {
        let [a, b, c] = triangle;
        if i == mesh.triangles.len() - 1 {
            // Last face without trailing comma
            writeln!(file, "      {}, {}, {}, -1", a, b, c)?;
        } else {
            writeln!(file, "      {}, {}, {}, -1,", a, b, c)?;
        }
    }
    writeln!(file, "    ]")?;
    
    // Write normals if present
    if has_normals {
        writeln!(file, "    normal Normal {{")?;
        writeln!(file, "      vector [")?;
        for (i, normal) in mesh.normals.iter().enumerate() {
            if i == mesh.normals.len() - 1 {
                // Last normal without trailing comma
                writeln!(file, "        {} {} {}", normal[0], normal[1], normal[2])?;
            } else {
                writeln!(file, "        {} {} {},", normal[0], normal[1], normal[2])?;
            }
        }
        writeln!(file, "      ]")?;
        writeln!(file, "    }}")?;
        
        // Write normal indices (same as coord indices)
        writeln!(file, "    normalIndex [")?;
        for (i, triangle) in mesh.triangles.iter().enumerate() {
            let [a, b, c] = triangle;
            if i == mesh.triangles.len() - 1 {
                // Last face without trailing comma
                writeln!(file, "      {}, {}, {}, -1", a, b, c)?;
            } else {
                writeln!(file, "      {}, {}, {}, -1,", a, b, c)?;
            }
        }
        writeln!(file, "    ]")?;
    }
    
    // Close geometry and shape
    writeln!(file, "  }}")?;
    writeln!(file, "}}")?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_write_vrml_basic() {
        // Create a simple triangle mesh (single triangle)
        let mesh = TriangleMesh {
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
        
        let output_path = "/tmp/test_basic.wrl";
        write_vrml(&mesh, output_path).expect("Failed to write VRML");
        
        let content = std::fs::read_to_string(output_path).expect("Failed to read VRML");
        
        // Check VRML header
        assert!(content.contains("#VRML V2.0 utf8"));
        assert!(content.contains("Shape {"));
        assert!(content.contains("geometry IndexedFaceSet {"));
        
        // Check coordinate node
        assert!(content.contains("coord Coordinate {"));
        assert!(content.contains("point ["));
        assert!(content.contains("0 0 0"));
        assert!(content.contains("1 0 0"));
        assert!(content.contains("0 1 0"));
        
        // Check coordIndex
        assert!(content.contains("coordIndex ["));
        assert!(content.contains("0, 1, 2, -1"));
        
        // Check normal node
        assert!(content.contains("normal Normal {"));
        assert!(content.contains("vector ["));
        
        // Check normalIndex
        assert!(content.contains("normalIndex ["));
    }
    
    #[test]
    fn test_write_vrml_no_normals() {
        let mesh = TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            normals: vec![], // No normals
            triangles: vec![[0, 1, 2]],
        };
        
        let output_path = "/tmp/test_no_normals.wrl";
        write_vrml(&mesh, output_path).expect("Failed to write VRML");
        
        let content = std::fs::read_to_string(output_path).expect("Failed to read VRML");
        
        // Check that normals are NOT included
        assert!(!content.contains("normal Normal"));
        assert!(!content.contains("normalIndex"));
        
        // But coordinates should be there
        assert!(content.contains("coord Coordinate {"));
        assert!(content.contains("coordIndex ["));
    }
    
    #[test]
    fn test_write_vrml_multiple_faces() {
        // Create a mesh with multiple triangles
        let mesh = TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            normals: vec![
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            triangles: vec![
                [0, 1, 2],
                [0, 2, 3],
            ],
        };
        
        let output_path = "/tmp/test_multiple_faces.wrl";
        write_vrml(&mesh, output_path).expect("Failed to write VRML");
        
        let content = std::fs::read_to_string(output_path).expect("Failed to read VRML");
        
        // Check vertices
        assert!(content.contains("0 0 0"));
        assert!(content.contains("1 0 0"));
        assert!(content.contains("1 1 0"));
        assert!(content.contains("0 1 0"));
        
        // Check both faces in coordIndex
        assert!(content.contains("0, 1, 2, -1"));
        assert!(content.contains("0, 2, 3, -1"));
        
        // Check it's valid VRML
        assert!(content.contains("#VRML V2.0 utf8"));
    }
    
    #[test]
    fn test_write_vrml_empty_mesh() {
        let mesh = TriangleMesh {
            vertices: vec![],
            normals: vec![],
            triangles: vec![],
        };
        
        let output_path = "/tmp/test_empty.wrl";
        write_vrml(&mesh, output_path).expect("Failed to write VRML");
        
        let content = std::fs::read_to_string(output_path).expect("Failed to read VRML");
        
        // Check header is still valid
        assert!(content.contains("#VRML V2.0 utf8"));
        assert!(content.contains("Shape {"));
        assert!(content.contains("IndexedFaceSet {"));
    }
}
