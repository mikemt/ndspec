//! Ocean wave energy density spectral formulations
//!
//! # Summary
//!
//! The `waves` module provides ocean wave energy density spectral
//! formulations that are commonly used in oceanography, meteorology,
//! and engineering design. The module provides the following spectral
//! formulations:
//!
//! * `Phillips1958`- the original formulation by Phillips[^1] that
//! captures the high-frequency behaviour of the spectral energy for
//! wind-generated ocean waves
//! * `Phillips` - Phillips' formulation re-cast into wave spectrum[^2] as a function of wind speed $U$  
//! * `bretschneider` - the Bretschneider[^3] wave spectrum
//! * `gaussian` - Gaussian wave spectrum
//! * `jonswap` - JONSWAP wave spectrum
//! * `ochi` - Ochi-Hubble wave spectrum
//!
//! In addition, the module also provides spreading functions commonly
//! used to generate two-dimensional wave spectra (short-crested
//! waves) from one-dimensional wave spectra (long-crested). These
//! are:
//!
//! # References
//!
//! [^1]: Phillips, O. M. (1958). The equilibrium range in the
//! spectrum of wind-generated waves. *Journal of Fluid Mechanics*,
//! 4(4): 426-434.
//! [^2]: M. J. Tucker and E. G. Pitt, *Waves in Ocean Engineering*, Volume 5. Elsevier Science, 2001.
//! [^3]: Bretschneider, C. L. (1958). Wave Variability and Wave
//! Spectra for Wind-Generated Gravity Waves. Technical Report 1,
//! U.S. Army Coastal Engineering Research Center, Fort Belvoir, VA.
//!

mod bretschneider;
mod common;
pub mod gaussian;
pub mod jonswap;
mod ochi;
mod phillips;
mod pierson_moskowitz;
pub mod spreading;

pub use bretschneider::Bretschneider;
pub use common::SpectralCommon;
pub use jonswap::Jonswap1973;
pub use ochi::OchiHubble;
pub use phillips::{Phillips, Phillips1958};
pub use pierson_moskowitz::PiersonMoskowitz;
