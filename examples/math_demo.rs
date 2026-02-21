//! Demo of the foundation math module

use cascade::{Matrix3x3, Matrix4x4, Vec3, Pnt, Dir, solve_linear_system_3x3, eigenvalues_3x3, VectorOps};
use std::f64::consts::PI;

fn main() {
    println!("=== CASCADE-RS Foundation Math Demo ===\n");

    // Matrix3x3 operations
    println!("1. Matrix3x3 Operations:");
    let m_rot = Matrix3x3::rotation_z(PI / 4.0);
    let v = Vec3::new(1.0, 0.0, 0.0);
    let v_rotated = m_rot.multiply_vec3(&v);
    println!("  Rotated vector: ({:.3}, {:.3}, {:.3})", v_rotated.x, v_rotated.y, v_rotated.z);
    println!("  Determinant: {:.3}", m_rot.determinant());
    
    // Matrix4x4 transformations
    println!("\n2. Matrix4x4 Transformations:");
    let scale = Matrix4x4::scale(2.0, 2.0, 2.0);
    let trans = Matrix4x4::translation(5.0, 10.0, 15.0);
    let transform = trans.multiply(&scale);
    let p = Pnt::new(1.0, 1.0, 1.0);
    let p_transformed = transform.multiply_pnt(&p);
    println!("  Transformed point: ({:.1}, {:.1}, {:.1})", p_transformed.x, p_transformed.y, p_transformed.z);

    // Vector operations
    println!("\n3. Vector Operations:");
    let v1 = Vec3::new(1.0, 0.0, 0.0);
    let v2 = Vec3::new(0.0, 1.0, 0.0);
    println!("  Dot product: {:.3}", v1.dot(&v2));
    let cross = v1.cross(&v2);
    println!("  Cross product: ({:.3}, {:.3}, {:.3})", cross.x, cross.y, cross.z);
    println!("  Angle between: {:.3} rad ({:.1}°)", v1.angle_between(&v2), v1.angle_between(&v2).to_degrees());

    // Linear system solving
    println!("\n4. Linear System Solving:");
    let a = Matrix3x3::new(
        2.0, 1.0, 0.0,
        1.0, 3.0, 1.0,
        0.0, 1.0, 2.0,
    );
    let b = Vec3::new(4.0, 9.0, 5.0);
    if let Some(x) = solve_linear_system_3x3(&a, &b) {
        println!("  Solution to Ax = b: ({:.3}, {:.3}, {:.3})", x.x, x.y, x.z);
        // Verify
        let result = a.multiply_vec3(&x);
        println!("  Verification (Ax): ({:.3}, {:.3}, {:.3})", result.x, result.y, result.z);
    }

    // Eigenvalues
    println!("\n5. Eigenvalues & Eigenvectors:");
    let sym_matrix = Matrix3x3::new(
        4.0, 1.0, 0.0,
        1.0, 3.0, 1.0,
        0.0, 1.0, 2.0,
    );
    if let Some((evals, _evecs)) = eigenvalues_3x3(&sym_matrix) {
        println!("  Eigenvalues: {:.3}, {:.3}, {:.3}", evals[0], evals[1], evals[2]);
        println!("  Trace (sum): {:.3}", evals[0] + evals[1] + evals[2]);
        println!("  Matrix trace: {:.3}", sym_matrix.trace());
    }

    println!("\n=== All tests completed successfully! ===");
}
