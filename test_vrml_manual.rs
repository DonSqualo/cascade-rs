use cascade::mesh::TriangleMesh;
use cascade::io::write_vrml;
use std::fs;

fn main() {
    // Create a simple triangle mesh
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
    
    // Write VRML
    let path = "/tmp/test_vrml_output.wrl";
    match write_vrml(&mesh, path) {
        Ok(_) => println!("✓ VRML file written successfully to {}", path),
        Err(e) => {
            eprintln!("✗ Failed to write VRML: {:?}", e);
            return;
        }
    }
    
    // Read and verify the output
    match fs::read_to_string(path) {
        Ok(content) => {
            println!("\n=== VRML File Content ===");
            println!("{}", content);
            println!("=== Verification Checks ===");
            
            let checks = vec![
                ("#VRML V2.0 utf8", "VRML header"),
                ("Shape {", "Shape node"),
                ("IndexedFaceSet {", "IndexedFaceSet geometry"),
                ("coord Coordinate {", "Coordinate node"),
                ("point [", "Points array"),
                ("coordIndex [", "Coordinate indices"),
                ("normal Normal {", "Normal node"),
                ("normalIndex [", "Normal indices"),
                ("0, 1, 2, -1", "Face definition with proper indexing"),
            ];
            
            for (check, desc) in checks {
                if content.contains(check) {
                    println!("✓ {}: found '{}'", desc, check);
                } else {
                    println!("✗ {}: NOT found '{}'", desc, check);
                }
            }
        }
        Err(e) => {
            eprintln!("✗ Failed to read VRML file: {:?}", e);
        }
    }
}
