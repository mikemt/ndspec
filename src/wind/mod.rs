//! wind module
//!
//! Functions for calculating energy density spectra for wind.
//!

mod froya;
mod wind;

pub use froya::{froya_10min, froya_1hr};
pub use wind::Wind;
