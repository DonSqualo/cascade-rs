use cascade::{make_sphere, brep::topology};

fn main() {
    match make_sphere(1.0) {
        Ok(solid) => {
            println!("Sphere structure:");
            println!("  Outer shell: {} faces", solid.outer_shell.faces.len());
            
            // Check edge consistency
            let all_faces = topology::get_solid_faces_internal(&solid);
            println!("  Total faces (including inner shells): {}", all_faces.len());
            
            // Count edges
            let mut total_edges = 0;
            for (idx, face) in all_faces.iter().enumerate() {
                let edge_count = face.outer_wire.edges.len();
                println!("    Face {}: {} edges", idx, edge_count);
                total_edges += edge_count;
            }
            println!("  Total edges: {}", total_edges);
        }
        Err(e) => println!("Error: {}", e),
    }
}
