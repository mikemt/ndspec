use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::prelude as ndspec;

/// Return the version of the Rust crate
///
/// This function is a part of Python module `ndspec`.
///
/// Returns
/// -------
/// str
///    The version of the Rust crate
///
/// Raises
/// ------
/// None
///   This function does not raise any exceptions
///
#[pyfunction]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[pyfunction]
pub fn interp1(x_i: f64, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> f64 {
    ndspec::linear_interp(x_i, x.as_array(), y.as_array())
}

#[pyfunction]
pub fn interp2(
    x_i: f64,
    y_i: f64,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    z: PyReadonlyArray2<f64>,
) -> f64 {
    ndspec::bilinear_interp(x_i, y_i, x.as_array(), y.as_array(), z.as_array())
}
