#[cfg(test)]
mod tests {
    use cascade::{make_box, io};
    use std::fs;
    
    // Assembly tests require write_step_assembly() which hasn't been implemented yet
    // These tests are placeholders for future assembly support
    
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
    
    #[test]
    fn test_write_step_ap203_compliance() {
        // Create a simple box
        let solid = make_box(10.0, 20.0, 30.0).expect("Failed to create box");
        let shape = cascade::Shape::Solid(solid);
        
        // Write to STEP file
        let path = "/tmp/test_ap203.step";
        io::write_step(&shape, path).expect("Failed to write STEP file");
        
        // Verify the file was created
        assert!(fs::metadata(path).is_ok(), "STEP file not created");
        
        let contents = fs::read_to_string(path).expect("Failed to read file");
        
        // Check for AP203 compliance: FILE_SCHEMA must specify CONFIG_CONTROL_DESIGN
        assert!(contents.contains("FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'))"), 
                "Missing AP203 FILE_SCHEMA('CONFIG_CONTROL_DESIGN')");
        
        // Check for required AP203 entities
        assert!(contents.contains("APPLICATION_CONTEXT"), 
                "Missing APPLICATION_CONTEXT (AP203 requirement)");
        assert!(contents.contains("APPLICATION_PROTOCOL_DEFINITION"), 
                "Missing APPLICATION_PROTOCOL_DEFINITION (AP203 requirement)");
        assert!(contents.contains("PRODUCT_DEFINITION_CONTEXT"), 
                "Missing PRODUCT_DEFINITION_CONTEXT (AP203 requirement)");
        assert!(contents.contains("PRODUCT_DEFINITION_FORMATION"), 
                "Missing PRODUCT_DEFINITION_FORMATION (AP203 requirement)");
        assert!(contents.contains("PRODUCT"), 
                "Missing PRODUCT (AP203 requirement)");
        assert!(contents.contains("PRODUCT_DEFINITION"), 
                "Missing PRODUCT_DEFINITION (AP203 requirement)");
        
        // Verify config_control_design is mentioned in application protocol
        assert!(contents.contains("'config_control_design'"), 
                "Missing 'config_control_design' protocol identifier");
        assert!(contents.contains("'configuration controlled 3D designs"), 
                "Missing configuration controlled design context");
    }
    
    #[test]
    fn test_write_step_ap203_export() {
        // Test the dedicated write_step_ap203 function
        let solid = make_box(10.0, 20.0, 30.0).expect("Failed to create box");
        
        // Write using the AP203-specific function
        let path = "/tmp/test_ap203_export.step";
        io::write_step_ap203(&solid, path).expect("Failed to write AP203 STEP file");
        
        // Verify the file was created
        assert!(fs::metadata(path).is_ok(), "AP203 STEP file not created");
        
        let contents = fs::read_to_string(path).expect("Failed to read file");
        
        // Verify AP203 schema
        assert!(contents.contains("FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'))"), 
                "Missing AP203 schema in write_step_ap203()");
        
        // Check all required AP203 entities
        assert!(contents.contains("APPLICATION_CONTEXT"), "Missing APPLICATION_CONTEXT");
        assert!(contents.contains("APPLICATION_PROTOCOL_DEFINITION"), "Missing APPLICATION_PROTOCOL_DEFINITION");
        assert!(contents.contains("PRODUCT_DEFINITION_CONTEXT"), "Missing PRODUCT_DEFINITION_CONTEXT");
        assert!(contents.contains("PRODUCT_DEFINITION_FORMATION"), "Missing PRODUCT_DEFINITION_FORMATION");
        assert!(contents.contains("PRODUCT"), "Missing PRODUCT");
        assert!(contents.contains("PRODUCT_DEFINITION"), "Missing PRODUCT_DEFINITION");
        assert!(contents.contains("MANIFOLD_SOLID_BREP"), "Missing MANIFOLD_SOLID_BREP geometry");
        assert!(contents.contains("CLOSED_SHELL"), "Missing CLOSED_SHELL");
        assert!(contents.contains("CARTESIAN_POINT"), "Missing CARTESIAN_POINT");
        
        // Verify the file structure is complete
        assert!(contents.contains("ISO-10303-21;"), "Missing ISO header");
        assert!(contents.contains("HEADER;"), "Missing HEADER section");
        assert!(contents.contains("FILE_DESCRIPTION"), "Missing FILE_DESCRIPTION");
        assert!(contents.contains("FILE_NAME"), "Missing FILE_NAME");
        assert!(contents.contains("DATA;"), "Missing DATA section");
        assert!(contents.contains("ENDSEC;"), "Missing ENDSEC");
        assert!(contents.contains("END-ISO-10303-21;"), "Missing END marker");
    }
}
