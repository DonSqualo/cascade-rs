use cascade::make_cylinder;

fn main() {
    // Test basic cylinder creation
    match make_cylinder(1.0, 2.0) {
        Ok(solid) => {
            println!("✓ Cylinder created successfully");
            println!("  Outer shell faces: {}", solid.outer_shell.faces.len());
            println!("  Solid closed: {}", solid.outer_shell.closed);
            if solid.outer_shell.faces.len() == 3 {
                println!("✓ Correct number of faces (3): bottom cap + top cap + side");
            }
        }
        Err(e) => {
            eprintln!("✗ Failed to create cylinder: {}", e);
            std::process::exit(1);
        }
    }
    
    // Test invalid dimensions
    match make_cylinder(-1.0, 1.0) {
        Ok(_) => {
            eprintln!("✗ Should have rejected negative radius");
            std::process::exit(1);
        }
        Err(_) => {
            println!("✓ Correctly rejected negative radius");
        }
    }
    
    match make_cylinder(1.0, 0.0) {
        Ok(_) => {
            eprintln!("✗ Should have rejected zero height");
            std::process::exit(1);
        }
        Err(_) => {
            println!("✓ Correctly rejected zero height");
        }
    }
    
    println!("\n✓ All cylinder tests passed!");
}
