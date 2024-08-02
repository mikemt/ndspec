use ndarray::{Array1, ArrayView1};

use crate::prelude::*;
use crate::waves::common::SpectralCommon;

const N_FREQ: usize = 256;

/// The single parameter Pierson-Moskowitz spectrum
///
/// The Pierson-Moskowitz spectrum is a single parameter wave spectrum
/// used to describe the wave energy distribution across a narrow
/// frequency band for fully developed wind-generated ocean waves. The
/// implementation adopted here follows the parametric form[^1]:
///
/// $$
/// S(\omega) = \frac{\alpha g^2}{\omega^5} \exp \left[ -C (g/H_s)^2 \omega^{-4} \right]
/// $$
///
/// where $g$ is the acceleration due to gravity, $\alpha=8.1 \times
/// 10^{-3}$ -- the Phillips constant, $C=0.032$, and $H_s$ denotes the significant wave height.
///
/// ## Fields
/// * `hs` - significant wave height \[m\]
/// * `omega` - frequency range \[rad/s\]
///
/// ## Methods
/// * `energy` - calculate the energy density spectrum
/// * `set_hs` - set the significant wave height
/// * `wp` - peak spectral frequency \[rad/s\]
/// * `fp` - peak spectral frequency \[Hz\]
/// * `tp` - peak spectral wave period \[s\]
/// * `wind_speed` - wind speed at a height of 19.5 m above the sea surface
/// * `set_omega` - set the frequency range
///
/// ## Default values
/// * `hs` - 1.0
/// * `omega` - 256 linearly spaced frequencies from 0.1 to $\pi$ rad/s
///
/// ## Examples
///
/// ```
/// use ndspec::PiersonMoskowitz;
///
/// let mut pm = PiersonMoskowitz::new(1.5);
/// let energy = pm.energy();
/// ```
///
/// ## References
/// [^1]: Pierson, W. J., & Moskowitz, L. (1964). A proposed spectral
/// form for fully developed wind seas based on the similarity theory
/// of S. A. Kitaigorodskii. Journal of Geophysical Research, 69(24),
/// 5181-5190.
/// [^2]: Ochi, M. K. (1998). Ocean waves: the stochastic
/// approach. Cambridge University Press.
///
pub struct PiersonMoskowitz {
    pub hs: f64,
    pub omega: Array1<f64>,
}

pub fn energy(omega: ArrayView1<f64>, hs: f64) -> Array1<f64> {
    let a = 8.1e-3;

    omega
        .iter()
        .map(|w| {
            a * GRAVITY.powf(2.)
                * w.powf(-5.)
                * (-0.032 * (GRAVITY / hs).powf(2.) * w.powf(-4.)).exp()
        })
        .collect()
}

impl Default for PiersonMoskowitz {
    fn default() -> Self {
        PiersonMoskowitz {
            hs: 1.0,
            omega: Array1::<f64>::linspace(0.1, PI, N_FREQ),
        }
    }
}

impl SpectralCommon for PiersonMoskowitz {
    fn hs(&self) -> Result<f64, String> {
        Ok(self.hs)
    }

    // unique implementation for Pierson-Moskowitz
    fn tp(&self) -> Result<f64, String> {
        match self.wp() {
            Ok(wp) => Ok(TWO_PI / wp),
            Err(e) => Err(e),
        }
    }

    // unique implementation for Pierson-Moskowitz
    fn wp(&self) -> Result<f64, String> {
        match self.hs {
            0.0 => Err("hs is zero".to_string()),
            _ => Ok(0.4 * (GRAVITY / self.hs).sqrt()),
        }
    }

    fn omega(&self) -> &Array1<f64> {
        &self.omega
    }

    fn energy(&self) -> Array1<f64> {
        energy(self.omega.view(), self.hs)
    }
}

impl PiersonMoskowitz {
    /// Create a new Pierson-Moskowitz spectrum
    pub fn new(hs: f64) -> Self {
        PiersonMoskowitz {
            hs,
            ..Default::default()
        }
    }

    /// Set the significant wave height
    pub fn set_hs(&mut self, hs: f64) -> &mut Self {
        self.hs = hs;
        self
    }

    /// Return the corresponding wind speed in \[m/s\]
    ///
    /// # Remarks
    /// The wind speed is calculated using the following relationship for a fully developed sea:
    /// $$
    /// U_{19.5} = \sqrt{g  H_s / 0.21}
    /// $$
    /// and is defined at a height of 19.5 m -- see Ochi[^2].
    ///
    pub fn wind_speed(&self) -> f64 {
        (GRAVITY * self.hs / 0.21).sqrt()
    }

    /// Set the frequency space in units of rad/s
    ///
    /// # Arguments
    /// * `omega` - frequency space vector of units rad/s
    pub fn set_omega(&mut self, omega: Array1<f64>) -> &mut Self {
        self.omega = omega;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 128;

    #[test]
    fn test_energy() {
        let hs = 1.5;
        let mut pm = PiersonMoskowitz::new(hs);
        let omega = Array1::linspace(0.1, PI, N);
        let energy = pm.set_omega(omega).energy();

        assert_eq!(energy.len(), N);
    }

    #[test]
    fn test_abs_error() {
        let hs = 1.5;
        let mut pm = PiersonMoskowitz::new(hs);
        let omega = Array1::linspace(0.1, PI, N);
        let abs_error = pm.set_omega(omega).abs_error();

        assert!(abs_error < Ok(0.1));
    }
}
