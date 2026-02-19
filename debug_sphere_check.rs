use cascade::{check_valid, check_self_intersection, Shape, make_sphere};

fn main() {
    match make_sphere(1.0) {
        Ok(solid) => {
            let shape = Shape::Solid(solid.clone());
            
            println!("Checking sphere validity...");
            match check_valid(&shape) {
                Ok(errors) => {
                    println!("Found {} errors:", errors.len());
                    for err in &errors {
                        println!("  - {}", err);
                    }
                }
                Err(e) => println!("Error: {}", e),
            }
            
            println!("\nChecking self-intersection...");
            let no_self_intersection = check_self_intersection(&solid);
            println!("No self-intersection: {}", no_self_intersection);
            
            // Debug sphere structure
            println!("\nSphere structure:");
            println!("  Outer shell: {} faces", solid.outer_shell.faces.len());
            for (idx, face) in solid.outer_shell.faces.iter().enumerate() {
                println!("    Face {}: {} edges, {} holes", 
                    idx, 
                    face.outer_wire.edges.len(), 
                    face.inner_wires.len());
            }
        }
        Err(e) => println!("Error creating sphere: {}", e),
    }
}
