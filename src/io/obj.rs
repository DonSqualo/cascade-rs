//! Wavefront OBJ mesh export

use crate::mesh::TriangleMesh;
use crate::Result;
use std::fs::File;
use std::io::Write;

/// Write a triangle mesh to Wavefront OBJ format
///
/// The OBJ format uses:
/// - `v x y z` for vertex positions
/// - `vn nx ny nz` for vertex normals (if present)
/// - `f i j k` for triangle faces (1-indexed)
///
/// # Arguments
/// * `mesh` - The triangle mesh to export
/// * `path` - Output file path
///
/// # Example
/// ```rust,no_run
/// use cascade::mesh::{TriangleMesh, triangulate};
/// use cascade::io::write_obj;
/// use cascade::make_box;
///
/// let solid = make_box(10.0, 10.0, 10.0).unwrap();
/// let mesh = triangulate(&solid, 1.0).unwrap();
/// write_obj(&mesh, "output.obj").unwrap();
/// ```
pub fn write_obj(mesh: &TriangleMesh, path: &str) -> Result<()> {
    let mut file = File::create(path)?;
    
    // Write header comment
    writeln!(file, "# Wavefront OBJ exported by cascade-rs")?;
    writeln!(file, "# Vertices: {}", mesh.vertices.len())?;
    writeln!(file, "# Triangles: {}", mesh.triangles.len())?;
    writeln!(file)?;
    
    // Write vertices
    for vertex in &mesh.vertices {
        writeln!(file, "v {:.6} {:.6} {:.6}", vertex[0], vertex[1], vertex[2])?;
    }
    
    // Write normals if present
    let has_normals = !mesh.normals.is_empty() && mesh.normals.len() == mesh.vertices.len();
    if has_normals {
        writeln!(file)?;
        for normal in &mesh.normals {
            writeln!(file, "vn {:.6} {:.6} {:.6}", normal[0], normal[1], normal[2])?;
        }
    }
    
    // Write faces (OBJ uses 1-indexed vertices)
    writeln!(file)?;
    for triangle in &mesh.triangles {
        let [i, j, k] = triangle;
        // Convert from 0-indexed to 1-indexed
        let i1 = i + 1;
        let j1 = j + 1;
        let k1 = k + 1;
        
        if has_normals {
            // Format: f v1//vn1 v2//vn2 v3//vn3 (vertex//normal)
            writeln!(file, "f {}//{} {}//{} {}//{}", i1, i1, j1, j1, k1, k1)?;
        } else {
            // Format: f v1 v2 v3
            writeln!(file, "f {} {} {}", i1, j1, k1)?;
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_write_obj_basic() {
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
        
        let output_path = "/tmp/test_basic.obj";
        write_obj(&mesh, output_path).expect("Failed to write OBJ");
        
        let content = std::fs::read_to_string(output_path).expect("Failed to read OBJ");
        
        // Check vertices
        assert!(content.contains("v 0.000000 0.000000 0.000000"));
        assert!(content.contains("v 1.000000 0.000000 0.000000"));
        assert!(content.contains("v 0.000000 1.000000 0.000000"));
        
        // Check normals
        assert!(content.contains("vn 0.000000 0.000000 1.000000"));
        
        // Check face (1-indexed with normals)
        assert!(content.contains("f 1//1 2//2 3//3"));
    }
    
    #[test]
    fn test_write_obj_no_normals() {
        let mesh = TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            normals: vec![], // No normals
            triangles: vec![[0, 1, 2]],
        };
        
        let output_path = "/tmp/test_no_normals.obj";
        write_obj(&mesh, output_path).expect("Failed to write OBJ");
        
        let content = std::fs::read_to_string(output_path).expect("Failed to read OBJ");
        
        // Check face without normal references
        assert!(content.contains("f 1 2 3"));
        assert!(!content.contains("vn")); // No normals
    }
}
