#[cfg(test)]
mod tests {
    use cascade::{make_box, io};
    use std::fs;
    
    #[test]
    fn test_write_step_box() {
        // Create a simple box
        let solid = make_box(10.0, 20.0, 30.0).expect("Failed to create box");
        let shape = cascade::Shape::Solid(solid);
        
        // Write to STEP file
        let path = "/tmp/test_box.step";
        io::write_step(&shape, path).expect("Failed to write STEP file");
        
        // Verify the file was created
        assert!(fs::metadata(path).is_ok(), "STEP file not created");
        
        let contents = fs::read_to_string(path).expect("Failed to read file");
        
        // Check for STEP file headers
        assert!(contents.contains("ISO-10303-21;"), "Missing ISO header");
        assert!(contents.contains("FILE_DESCRIPTION"), "Missing FILE_DESCRIPTION");
        assert!(contents.contains("FILE_NAME"), "Missing FILE_NAME");
        assert!(contents.contains("FILE_SCHEMA"), "Missing FILE_SCHEMA");
        assert!(contents.contains("MANIFOLD_SOLID_BREP"), "Missing MANIFOLD_SOLID_BREP");
        assert!(contents.contains("CARTESIAN_POINT"), "Missing CARTESIAN_POINT");
        assert!(contents.contains("CLOSED_SHELL"), "Missing CLOSED_SHELL");
        assert!(contents.contains("ENDSEC"), "Missing ENDSEC");
        assert!(contents.contains("END-ISO-10303-21;"), "Missing END marker");
    }
    
    #[test]
    fn test_write_step_cylinder() {
        // Create a cylinder
        let solid = cascade::make_cylinder(5.0, 10.0).expect("Failed to create cylinder");
        let shape = cascade::Shape::Solid(solid);
        
        // Write to STEP file
        let path = "/tmp/test_cylinder.step";
        io::write_step(&shape, path).expect("Failed to write STEP file");
        
        // Verify the file was created
        assert!(fs::metadata(path).is_ok(), "STEP file not created");
        
        let contents = fs::read_to_string(path).expect("Failed to read file");
        assert!(contents.contains("ISO-10303-21;"), "Missing ISO header");
        assert!(contents.contains("CYLINDRICAL_SURFACE"), "Missing CYLINDRICAL_SURFACE");
    }
    
    #[test]
    fn test_write_step_sphere() {
        // Create a sphere
        let solid = cascade::make_sphere(5.0).expect("Failed to create sphere");
        let shape = cascade::Shape::Solid(solid);
        
        // Write to STEP file
        let path = "/tmp/test_sphere.step";
        io::write_step(&shape, path).expect("Failed to write STEP file");
        
        // Verify the file was created
        assert!(fs::metadata(path).is_ok(), "STEP file not created");
        
        let contents = fs::read_to_string(path).expect("Failed to read file");
        assert!(contents.contains("SPHERICAL_SURFACE"), "Missing SPHERICAL_SURFACE");
    }
}
