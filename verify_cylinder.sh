#!/bin/bash
cd ~/projects/cascade-rs
cat > test_cyl_quick.rs << 'RUST'
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
RUST

# Try to use rustc directly if library is available
if [ -f target/debug/libcascade.rlib ]; then
    rustc --edition 2021 -L target/debug/deps --extern cascade=target/debug/libcascade.rlib test_cyl_quick.rs && ./test_cyl_quick
else
    echo "Library not yet built, will need to wait for full compile"
fi
