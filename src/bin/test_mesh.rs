use cascade::{make_box, mesh::triangulate, mesh::export_stl};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating a box...");
    let solid = make_box(10.0, 10.0, 10.0)?;
    
    println!("Triangulating...");
    let mesh = triangulate(&solid, 1.0)?;
    
    println!("Mesh Statistics:");
    println!("  Vertices: {}", mesh.vertices.len());
    println!("  Triangles: {}", mesh.triangles.len());
    println!("  Normals: {}", mesh.normals.len());
    
    println!("\nExporting to STL...");
    export_stl(&mesh, "/tmp/test_box.stl")?;
    
    println!("STL file exported to /tmp/test_box.stl");
    
    // Check the file
    let content = std::fs::read_to_string("/tmp/test_box.stl")?;
    println!("File size: {} bytes", content.len());
    println!("First 200 chars:\n{}", &content[..200.min(content.len())]);
    
    Ok(())
}
