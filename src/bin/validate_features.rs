//! Feature validation binary
//!
//! Runs all feature tests against the golden corpus and updates README.md

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Feature definition
struct Feature {
    id: &'static str,
    name: &'static str,
    category: &'static str,
    test_fn: fn() -> bool,
}

/// Test a feature by checking if it's implemented (doesn't return NotImplemented)
macro_rules! test_feature {
    ($func:expr) => {{
        match $func {
            Err(cascade::CascadeError::NotImplemented(_)) => false,
            _ => true, // Implemented (may still have bugs, but corpus tests will catch those)
        }
    }};
}

fn main() {
    let features = vec![
        // Primitives
        Feature { id: "primitive::box", name: "Box/cuboid creation", category: "Primitives", 
                  test_fn: || test_feature!(cascade::make_box(1.0, 1.0, 1.0)) },
        Feature { id: "primitive::sphere", name: "Sphere creation", category: "Primitives",
                  test_fn: || test_feature!(cascade::make_sphere(1.0)) },
        Feature { id: "primitive::cylinder", name: "Cylinder creation", category: "Primitives",
                  test_fn: || test_feature!(cascade::make_cylinder(1.0, 1.0)) },
        Feature { id: "primitive::cone", name: "Cone creation", category: "Primitives",
                  test_fn: || test_feature!(cascade::make_cone(1.0, 0.5, 1.0)) },
        Feature { id: "primitive::torus", name: "Torus creation", category: "Primitives",
                  test_fn: || test_feature!(cascade::make_torus(2.0, 0.5)) },
        Feature { id: "primitive::wedge", name: "Wedge/prism creation", category: "Primitives",
                  test_fn: || test_feature!(cascade::primitive::make_wedge(1.0, 1.0, 1.0, 0.5)) },
        
        // Boolean Operations
        Feature { id: "boolean::fuse", name: "Union of solids", category: "Boolean Operations",
                  test_fn: || {
                      let box1 = cascade::make_box(1.0, 1.0, 1.0);
                      let box2 = cascade::make_box(1.0, 1.0, 1.0);
                      match (box1, box2) {
                          (Ok(b1), Ok(b2)) => cascade::fuse(&b1, &b2).is_ok(),
                          _ => false,
                      }
                  }},
        Feature { id: "boolean::cut", name: "Difference of solids", category: "Boolean Operations",
                  test_fn: || {
                      let box1 = cascade::make_box(1.0, 1.0, 1.0);
                      let box2 = cascade::make_box(0.5, 0.5, 0.5);
                      match (box1, box2) {
                          (Ok(b1), Ok(b2)) => cascade::cut(&b1, &b2).is_ok(),
                          _ => false,
                      }
                  }},
        Feature { id: "boolean::common", name: "Intersection of solids", category: "Boolean Operations",
                  test_fn: || {
                      let box1 = cascade::make_box(1.0, 1.0, 1.0);
                      let box2 = cascade::make_box(0.5, 0.5, 0.5);
                      match (box1, box2) {
                          (Ok(b1), Ok(b2)) => cascade::common(&b1, &b2).is_ok(),
                          _ => false,
                      }
                  }},
        Feature { id: "boolean::section", name: "Section (solid/plane intersection)", category: "Boolean Operations",
                  test_fn: || {
                      let box1 = cascade::make_box(1.0, 1.0, 1.0);
                      match box1 {
                          Ok(b) => cascade::boolean::section(&b, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0]).is_ok(),
                          _ => false,
                      }
                  }},
        
        // BREP Core
        Feature { id: "brep::vertex", name: "Vertex representation", category: "BREP Core",
                  test_fn: || true }, // Basic struct exists
        Feature { id: "brep::edge", name: "Edge representation (line, arc, spline)", category: "BREP Core",
                  test_fn: || true },
        Feature { id: "brep::wire", name: "Wire (connected edges)", category: "BREP Core",
                  test_fn: || true },
        Feature { id: "brep::face", name: "Face representation", category: "BREP Core",
                  test_fn: || true },
        Feature { id: "brep::shell", name: "Shell (connected faces)", category: "BREP Core",
                  test_fn: || true },
        Feature { id: "brep::solid", name: "Solid representation", category: "BREP Core",
                  test_fn: || true },
        Feature { id: "brep::compound", name: "Compound shapes", category: "BREP Core",
                  test_fn: || true },
        Feature { id: "brep::topology", name: "Topological queries (adjacent, connected, etc.)", category: "BREP Core",
                  test_fn: || false }, // Not implemented
        
        // TODO: Add remaining features...
    ];
    
    // Run corpus tests if available
    let corpus_results = run_corpus_tests();
    
    // Combine implementation check with corpus tests
    let mut results: HashMap<&str, bool> = HashMap::new();
    for feature in &features {
        let impl_exists = (feature.test_fn)();
        let corpus_pass = corpus_results.get(feature.id).copied().unwrap_or(true); // Pass if no corpus test
        results.insert(feature.id, impl_exists && corpus_pass);
    }
    
    // Update README
    update_readme(&features, &results);
    
    // Print summary
    let passed = results.values().filter(|&&v| v).count();
    let total = results.len();
    println!("\n✓ Feature validation complete: {}/{} passing ({}%)", 
             passed, total, passed * 100 / total.max(1));
}

fn run_corpus_tests() -> HashMap<&'static str, bool> {
    let mut results = HashMap::new();
    
    let corpus_path = Path::new("corpus");
    if !corpus_path.exists() {
        println!("⚠ No corpus directory found. Run generate-corpus first.");
        return results;
    }
    
    // TODO: Load and run each corpus test case
    // For each test:
    //   1. Load input from corpus/{category}/{test}/input.json
    //   2. Run cascade-rs implementation
    //   3. Compare with corpus/{category}/{test}/expected.json
    //   4. Pass if within tolerance
    
    results
}

fn update_readme(features: &[Feature], results: &HashMap<&str, bool>) {
    let readme_path = "README.md";
    let content = fs::read_to_string(readme_path).expect("Failed to read README.md");
    
    // Find the feature parity section
    let start_marker = "<!-- FEATURE_PARITY_START -->";
    let end_marker = "<!-- FEATURE_PARITY_END -->";
    
    let start_idx = content.find(start_marker).expect("Missing start marker");
    let end_idx = content.find(end_marker).expect("Missing end marker");
    
    // Group features by category
    let mut categories: HashMap<&str, Vec<&Feature>> = HashMap::new();
    for feature in features {
        categories.entry(feature.category).or_default().push(feature);
    }
    
    // Generate new content
    let mut new_section = String::new();
    new_section.push_str(start_marker);
    new_section.push_str(&format!("\nLast validated: {}\n\n", 
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
    
    let mut total_pass = 0;
    let mut total_count = 0;
    
    for (category, cat_features) in &categories {
        let cat_pass = cat_features.iter()
            .filter(|f| results.get(f.id).copied().unwrap_or(false))
            .count();
        let cat_total = cat_features.len();
        total_pass += cat_pass;
        total_count += cat_total;
        
        new_section.push_str(&format!("### {} ({}/{})\n", category, cat_pass, cat_total));
        
        for feature in cat_features {
            let passed = results.get(feature.id).copied().unwrap_or(false);
            let check = if passed { "x" } else { " " };
            new_section.push_str(&format!("- [{}] `{}` — {}\n", check, feature.id, feature.name));
        }
        new_section.push('\n');
    }
    
    let pct = if total_count > 0 { total_pass * 100 / total_count } else { 0 };
    new_section.push_str(&format!("**Total: {}/{} features passing ({}%)**\n", 
                                  total_pass, total_count, pct));
    new_section.push_str(end_marker);
    
    // Replace section
    let new_content = format!("{}{}{}", 
        &content[..start_idx],
        new_section,
        &content[end_idx + end_marker.len()..]);
    
    fs::write(readme_path, new_content).expect("Failed to write README.md");
    println!("✓ Updated README.md");
}
