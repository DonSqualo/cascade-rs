use cascade::io::read_step;
use cascade::Shape;

#[test]
fn test_read_step_box() {
    let result = read_step("test_data/box.step");
    if let Err(ref e) = result {
        eprintln!("Error: {:?}", e);
    }
    assert!(result.is_ok(), "Failed to parse STEP file");
    
    let shape = result.unwrap();
    match shape {
        Shape::Solid(solid) => {
            assert!(solid.outer_shell.faces.len() == 6, "Box should have 6 faces");
            assert!(solid.outer_shell.closed, "Outer shell should be closed");
        },
        _ => panic!("Expected a Solid"),
    }
}
