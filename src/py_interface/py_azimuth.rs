use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::{pyfunction, PyResult, Python};

use crate::prelude::azimuth;

#[pyfunction]
pub fn convert_branch_180(theta: f64) -> PyResult<f64> {
    Ok(azimuth::convert_branch_180(theta))
}

#[pyfunction]
pub fn convert_branch_360(theta: f64) -> PyResult<f64> {
    Ok(azimuth::convert_branch_360(theta))
}

#[pyfunction]
pub fn convert_branch_180_vec<'a>(
    py: Python<'a>,
    theta: PyReadonlyArray1<f64>,
) -> &'a PyArray1<f64> {
    azimuth::convert_branch_180_vec(theta.as_array()).into_pyarray(py)
}

#[pyfunction]
pub fn theta_to_global(heading: f64) -> PyResult<f64> {
    Ok(azimuth::theta_to_global(heading))
}

#[pyfunction]
pub fn rotation_matrix_2d<'a>(py: Python<'a>, theta: f64) -> &'a PyArray2<f64> {
    azimuth::rotation_matrix_2d(theta).into_pyarray(py)
}
