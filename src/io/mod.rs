//! File I/O for CAD formats

use crate::brep::{Shape, Solid};
use crate::{Result, CascadeError};

pub fn read_step(path: &str) -> Result<Shape> {
    Err(CascadeError::NotImplemented("io::step_read".into()))
}

pub fn write_step(shape: &Shape, path: &str) -> Result<()> {
    Err(CascadeError::NotImplemented("io::step_write".into()))
}

pub fn read_brep(path: &str) -> Result<Shape> {
    Err(CascadeError::NotImplemented("io::brep_read".into()))
}

pub fn write_brep(shape: &Shape, path: &str) -> Result<()> {
    Err(CascadeError::NotImplemented("io::brep_write".into()))
}
