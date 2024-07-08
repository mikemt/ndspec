//! This module provides Types with various traits for working with one-dimensional and two-dimensional spectra.
//!
//! All wave and wind spectra can be converted into either a
//! `Spectrum1` or `Spectrum2` Type. These types provide various
//! traits for working with and evaluating energy density spectra.
//!

pub mod spectrum1d;
pub mod spectrum2d;

pub use spectrum1d::{FrequencyResponse, Spectrum1D};
pub use spectrum2d::Spectrum2D;
