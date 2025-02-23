//! # ndspec
//!
//! `ndspec` is a crate for working with energy density spectra with a focus on ocean waves, wind, and related response spectra.
//!
//! The crate is organised into the following modules:
//!
//! * `core` - Core functionality for working with energy density spectra.
//! * `waves` - Functions for calculating energy density spectra for ocean waves.
//! * `wind` - Functions for calculating energy density spectra for wind.
//! * `spectrum` - provides the Types `Spectrum1` and 'Spectrum2' for one-dimensional and two-dimensional spectra respectively.
//!
//! All wave and wind spectra can be converted into either a `Spectrum1` or
//! `Spectrum2` Type. These types provide various traits for working
//! with and evaluating energy density spectra.
//!
//! The crate is designed to be used in conjunction with the `ndarray`
//! crate. The `ndarray::Array1` and `ndarray::Array2` types are
//! underlying data structures adopted throughout.
//!
//! The crate also provides a Python extension that can be compiled and installed seperately.
//!
//! ## Installation
//!
//! Add the following to your `Cargo.toml` file:
//!
//! ```toml
//! [dependencies]
//! ndspec = "0.1.0"
//! ```
//!
//! ## Usage
//!
//! The following example demonstrates how to calculate the energy density spectrum for a Bretschneider wave spectrum:
//!
//! ```
//! use ndspec::prelude::*;
//!
//! let hs = 1.5;
//! let tp = 18.0;
//! let mut bretschneider = Bretschneider::new(hs, tp);
//! let omega = Array1::linspace(0.1, PI, 100);
//! let energy = bretschneider.set_omega(omega).energy();
//! ```
//!
//! ## Python Extension
//!
//! The crate provides a Python extension that can be compiled and
//! installed seperately. To build the extension, `maturin` is
//! required and the `python-extension` feature must be enabled:
//!
//! ```bash
//! maturin build --release --features python-extension
//! ```
//!
//! and install with `pip`.
//!
//! ### Examples
//!
//! Define a Jonswap energy density spectrum from only Hs and Tp,
//! convert to a `Spectrum1D` type, and then evaluate the most probable
//! maximum amplitude over a 3 hour (10,800 s) time window:
//! ```python
//! import ndspec
//! S = ndspec.Jonswap(hs=1.5, tp=10.0).to_spec1d()
//! print(S.Ampm(10_800))
//! ```
//!
//! Print out the help for the Jonswap class in Python::
//! ```python
//! import ndspec
//! help(ndspec.Jonswap)
//! ```
//!
#![cfg_attr(
    feature = "doc-images",
    cfg_attr(all(),
             doc = ::embed_doc_image::embed_image!("label_matrix", "./assets/matrix.png")))
]
//!
//! ![energy density data structure][label_matrix]
//!

//#![warn(missing_docs)]
//#![warn(missing_doc_code_examples)]

//
pub mod azimuth;
pub mod core;
pub mod spectrums;
pub mod waves;
pub mod wind;

//use embed_doc_image::embed_doc_image;

#[cfg(feature = "python-extension")]
use pyo3::prelude::*;

/// ## The prelude
///
/// The purpose of this module is to:
/// * define the namespace
/// * keep all imports in a central location
/// * help avoid the direct imports of traits defined by this crate in the parent module
/// * provide a convenient and consistent way to import the main types, traits, and functions with a single import
/// * provide access to useful external crate types and traits from a single import
///
/// ## Example:
///
/// ```
/// use ndspec::prelude::*;
/// ```
pub mod prelude {

    // constants
    pub use crate::core::constants::{GRAVITY, KNOT, PI, RHO_AIR, RHO_SEA_WATER, TWO_PI};

    // re-export of the core modules
    pub use crate::core::ndarray_ext;
    pub use crate::core::*;
    pub use crate::spectrums::*;
    pub use crate::waves::jonswap::{
        convert_t1_to_tp, convert_tp_to_t1, convert_tp_to_tz, convert_tz_to_tp, lewis_allos, maxhs,
        Jonswap,
    };
    pub use crate::waves::spreading::{spread_cos_2s, spread_cos_n, Spreading};
    pub use crate::waves::{bretschneider, gaussian::gaussian_spectrum, jonswap, PiersonMoskowitz};
    pub use crate::wind::*;

    // re-export of external crate types and traits for convenience
    pub use ndarray::{Array1, Array2};
    pub use ndarray_stats::QuantileExt;

    pub use crate::azimuth;
}

// convenience re-export of the prelude
#[doc(hidden)]
pub use crate::prelude::*;

#[cfg(feature = "python-extension")]
mod py_interface;

#[cfg(feature = "python-extension")]
#[pymodule]
#[pyo3(name = "_libndspec")]
fn ndspec(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_interface::version, m)?)?;

    m.add_function(wrap_pyfunction!(py_interface::interp1, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::interp2, m)?)?;

    // py_azimuth.rs
    m.add_function(wrap_pyfunction!(py_interface::convert_branch_180, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::convert_branch_360, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::rotation_matrix_2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::convert_branch_180_vec, m)?)?;

    // py_spectrums.rs
    m.add_function(wrap_pyfunction!(py_interface::maxhs, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::lewis_allos, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::bretschneider, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::jonswap, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::convert_tp_to_tz, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::convert_tz_to_tp, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::convert_tp_to_t1, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::convert_t1_to_tp, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::spread_cos_n, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::spread_cos_2s, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::froya_10min, m)?)?;
    m.add_function(wrap_pyfunction!(py_interface::froya_1hr, m)?)?;

    m.add_class::<py_interface::Jonswap>()?;
    m.add_class::<py_interface::Spreading>()?;
    m.add_class::<py_interface::FrequencyResponse>()?;
    m.add_class::<py_interface::Spectrum1D>()?;
    m.add_class::<py_interface::Spectrum2D>()?;
    Ok(())
}
