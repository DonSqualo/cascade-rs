fn main() {
    match cascade::make_cylinder(1.0, 2.0) {
        Ok(solid) => {
            println!("✓ Cylinder created!");
            println!("  Faces: {}", solid.outer_shell.faces.len());
            println!("  Closed: {}", solid.outer_shell.closed);
            if solid.outer_shell.faces.len() == 3 {
                println!("✓ PASS: correct face count");
            } else {
                println!("✗ FAIL: expected 3 faces, got {}", solid.outer_shell.faces.len());
            }
        }
        Err(e) => {
            eprintln!("✗ FAIL: {}", e);
            std::process::exit(1);
        }
    }
}
