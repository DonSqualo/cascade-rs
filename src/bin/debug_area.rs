use cascade::primitive::make_box;

fn main() {
    let solid = make_box(1.0, 1.0, 1.0).unwrap();
    
    println!("Number of faces: {}", solid.outer_shell.faces.len());
    
    for (i, face) in solid.outer_shell.faces.iter().enumerate() {
        let vertices: Vec<_> = face.outer_wire.edges.iter()
            .map(|e| e.start.point)
            .collect();
        println!("Face {}: {} vertices", i, vertices.len());
        for (j, v) in vertices.iter().enumerate() {
            println!("  v{}: ({:.2}, {:.2}, {:.2})", j, v[0], v[1], v[2]);
        }
    }
}
