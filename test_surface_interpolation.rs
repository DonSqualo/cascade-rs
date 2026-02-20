use cascade_rs::interpolate_surface;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple 3x3 grid of points
    let points = vec![
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        vec![[0.0, 1.0, 0.5], [1.0, 1.0, 0.7], [2.0, 1.0, 0.5]],
        vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
    ];

    // Interpolate a surface with degree 2 in both directions
    let surface = interpolate_surface(&points, 2, 2)?;
    
    println!("Surface created successfully: {:?}", surface);
    
    Ok(())
}
