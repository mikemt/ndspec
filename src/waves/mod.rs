//! ocean wave energy density spectral formulation
//!
//! # Summary
//!
//! The `waves` module provides ocean wave energy density spectral formulations that are commonly used
//! in oceanography, meteorology, and engineering design. The module provides the following spectral formulations:
//!
//! * `bretschneider` - Bretschneider wave spectrum
//! * `gaussian` - Gaussian wave spectrum
//! * `jonswap` - JONSWAP wave spectrum
//! * `ochi` - Ochi-Hubble wave spectrum
//!
//! In addition, the module also provides commonly used spreading functions used to generate two-dimensional
//! wave spectra from one-dimensional wave spectra. These are:
//!
//! *

pub mod bretschneider;
pub mod gaussian;
pub mod jonswap;
pub mod ochi;
pub mod spreading;
