//! Test STEP file reading against real CAD files

use cascade::{io::read_step, Shape};
use std::env;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        // Default: test all files in test_corpus
        let corpus_dir = Path::new("test_corpus");
        if corpus_dir.exists() {
            for entry in std::fs::read_dir(corpus_dir).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "step" || e == "stp") {
                    test_file(&path);
                }
            }
        }
    } else {
        // Test specific file
        test_file(Path::new(&args[1]));
    }
}

fn test_file(path: &Path) {
    println!("\n=== Testing: {} ===", path.display());
    
    match read_step(path.to_str().unwrap()) {
        Ok(shape) => {
            println!("✓ Successfully parsed STEP file");
            describe_shape(&shape, 0);
        }
        Err(e) => {
            println!("✗ Failed to parse: {:?}", e);
        }
    }
}

fn describe_shape(shape: &Shape, indent: usize) {
    let prefix = "  ".repeat(indent);
    match shape {
        Shape::Vertex(v) => {
            println!("{}Vertex at {:?}", prefix, v.point);
        }
        Shape::Edge(e) => {
            println!("{}Edge: {:?}", prefix, e.curve_type);
        }
        Shape::Wire(w) => {
            println!("{}Wire with {} edges", prefix, w.edges.len());
        }
        Shape::Face(f) => {
            println!("{}Face: {:?}", prefix, f.surface_type);
            println!("{}  outer_wire: {} edges", prefix, f.outer_wire.edges.len());
            println!("{}  inner_wires: {}", prefix, f.inner_wires.len());
        }
        Shape::Shell(s) => {
            println!("{}Shell with {} faces, closed={}", prefix, s.faces.len(), s.closed);
            for (i, face) in s.faces.iter().enumerate() {
                println!("{}  Face {}: {:?}", prefix, i, face.surface_type);
            }
        }
        Shape::Solid(s) => {
            println!("{}Solid:", prefix);
            println!("{}  outer_shell: {} faces", prefix, s.outer_shell.faces.len());
            println!("{}  inner_shells: {}", prefix, s.inner_shells.len());
        }
        Shape::Compound(c) => {
            println!("{}Compound with {} solids", prefix, c.solids.len());
            for solid in &c.solids {
                describe_shape(&Shape::Solid(solid.clone()), indent + 1);
            }
        }
    }
}
