use cascade_rs::{make_box, fuse_many, cut_many, common_many};

fn main() {
    println!("Testing fuse_many...");
    let box1 = make_box(1.0, 1.0, 1.0).expect("Failed to create box1");
    let box2 = make_box(1.0, 1.0, 1.0).expect("Failed to create box2");
    let box3 = make_box(1.0, 1.0, 1.0).expect("Failed to create box3");
    
    match fuse_many(&[box1.clone(), box2.clone(), box3.clone()]) {
        Ok(result) => {
            println!("✓ fuse_many succeeded");
            println!("  Result has {} faces", result.outer_shell.faces.len());
        }
        Err(e) => println!("✗ fuse_many failed: {}", e),
    }
    
    println!("\nTesting cut_many...");
    let base = make_box(3.0, 3.0, 3.0).expect("Failed to create base");
    let tool1 = make_box(1.0, 1.0, 1.0).expect("Failed to create tool1");
    let tool2 = make_box(1.0, 1.0, 1.0).expect("Failed to create tool2");
    
    match cut_many(&base, &[tool1, tool2]) {
        Ok(result) => {
            println!("✓ cut_many succeeded");
            println!("  Result has {} faces", result.outer_shell.faces.len());
        }
        Err(e) => println!("✗ cut_many failed: {}", e),
    }
    
    println!("\nTesting common_many...");
    let box1 = make_box(2.0, 2.0, 2.0).expect("Failed to create box1");
    let box2 = make_box(2.0, 2.0, 2.0).expect("Failed to create box2");
    let box3 = make_box(2.0, 2.0, 2.0).expect("Failed to create box3");
    
    match common_many(&[box1, box2, box3]) {
        Ok(result) => {
            println!("✓ common_many succeeded");
            println!("  Result has {} faces", result.outer_shell.faces.len());
        }
        Err(e) => println!("✗ common_many failed: {}", e),
    }
    
    println!("\nTesting empty cases...");
    match fuse_many(&[]) {
        Ok(_) => println!("✗ fuse_many empty should fail"),
        Err(_) => println!("✓ fuse_many empty correctly fails"),
    }
    
    match common_many(&[]) {
        Ok(_) => println!("✗ common_many empty should fail"),
        Err(_) => println!("✓ common_many empty correctly fails"),
    }
    
    println!("\nAll manual tests completed!");
}
