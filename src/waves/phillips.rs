use ndarray::Array1;

use crate::prelude::*;
use crate::waves::common::SpectralCommon;

const N_FREQ: usize = 256;

/// The Phillips high-frequency tail
///
/// The Phillips spectrum is an early generational wave spectrum that
/// principally captures the energy distribution in the high-frequency
/// spectral tail. Its canonical form[^1] is:
///
/// $$
/// S(\omega) = \frac{\alpha g^2}{\omega^5}
/// $$
///
/// where $\alpha=0.0081$ denotes the Phillips constant and $g$ is the
/// acceleration due to gravity.
///
/// ## Fields
/// * `alpha` - the Phillips constant (default: 0.0081)
/// * `omega` - frequency range \[rad/s\]
///
/// ## Methods
/// * `energy` - calculate the energy density spectrum
/// * `f_hz` - return the frequency space in \[Hz\]
/// * `to_spec1d` - convert to a `Spectrum1` type
/// * `to_spec2d` - convert to a `Spectrum2` type for a given `Spreading` type
///  
/// ## References
/// [^1]: O. M. Phillips, The equilibrium range in the spectrum of
/// wind-generated waves, Journal of Fluid Mechanics, vol. 4, no. 4,
/// pp. 426 434, 1958, doi: 10.1017/S0022112058000550.
///
pub struct Phillips1958 {
    pub alpha: f64,
    pub omega: Array1<f64>,
}

/// The Phillips spectrum
///
/// The Phillips spectrum is an early generational wave spectrum that
/// principally captures the energy distribution in the high-frequency
/// spectral tail. Its canonical form[^1] is:
///
/// $$
/// S(\omega) = \frac{\alpha g^2}{\omega^5}
/// $$
///
/// where $\alpha=0.0081$ denotes the Phillips constant and $g$ is the
/// acceleration due to gravity.
///
/// ## Fields
/// * `alpha` - the Phillips constant (default: 0.0081)
/// * `omega` - frequency range \[rad/s\]
///
/// ## Methods
/// * `energy` - calculate the energy density spectrum
/// * `f_hz` - return the frequency space in \[Hz\]
/// * `to_spec1d` - convert to a `Spectrum1` type
/// * `to_spec2d` - convert to a `Spectrum2` type for a given `Spreading` type
///  
/// ## References
/// [^1]: O. M. Phillips, The equilibrium range in the spectrum of
/// wind-generated waves, Journal of Fluid Mechanics, vol. 4, no. 4,
/// pp. 426 434, 1958, doi: 10.1017/S0022112058000550.
///
pub struct Phillips {
    pub u: f64,
    pub alpha: f64,
    pub omega: Array1<f64>,
}

// general form of the energy distribution
pub fn energy(omega: &f64, alpha: f64) -> f64 {
    alpha * GRAVITY.powi(2) * omega.powi(-5)
}

// ////////////////////////////////////////////////////////////////////////////////
// Phillips 1958 implementation
// ////////////////////////////////////////////////////////////////////////////////

impl Default for Phillips1958 {
    fn default() -> Self {
        Phillips1958 {
            alpha: 0.0081,
            omega: Array1::<f64>::linspace(0.1, PI, N_FREQ),
        }
    }
}

impl Phillips1958 {
    /// create a new Phillips spectrum
    pub fn new() -> Self {
        Phillips1958 {
            ..Default::default()
        }
    }
}

impl SpectralCommon for Phillips1958 {
    fn hs(&self) -> Result<f64, String> {
        Err("hs is left undefined for the spectral tail".to_string())
    }

    fn tp(&self) -> Result<f64, String> {
        Err("tp is left undefined for the spectral tail".to_string())
    }

    fn omega(&self) -> &Array1<f64> {
        &self.omega
    }

    fn energy(&self) -> Array1<f64> {
        self.omega.iter().map(|w| energy(w, self.alpha)).collect()
    }
}

// ////////////////////////////////////////////////////////////////////////////////
// Phillips spectrum implementation
// ////////////////////////////////////////////////////////////////////////////////

impl Default for Phillips {
    fn default() -> Self {
        Phillips {
            u: 10.0,
            alpha: 0.0081,
            omega: Array1::<f64>::linspace(0.1, PI, N_FREQ),
        }
    }
}

impl SpectralCommon for Phillips {
    fn hs(&self) -> Result<f64, String> {
        Ok(2. * self.alpha.sqrt() * GRAVITY / self.wp()?.powi(2))
    }

    fn tp(&self) -> Result<f64, String> {
        Ok(TWO_PI / self.wp()?)
    }

    fn wp(&self) -> Result<f64, String> {
        Ok(GRAVITY / self.u)
    }

    fn omega(&self) -> &Array1<f64> {
        &self.omega
    }

    fn energy(&self) -> Array1<f64> {
        let wp = self.wp().unwrap_or(0.0);

        self.omega
            .iter()
            .map(|w| match w {
                w if w <= &wp => 0.0,
                _ => energy(w, self.alpha),
            })
            .collect()
    }
}

impl Phillips {
    /// create a new Phillips spectrum
    pub fn new(u: f64) -> Self {
        Phillips {
            u,
            ..Default::default()
        }
    }

    /// Analytic form for the spectral moments $m_n$
    /// where $n=0,1,2,3,4$
    ///
    /// # Desciption
    ///
    /// The spectral moments $m_n$ are defined:
    /// $$
    /// m_n = \int_0^\infty S(f) f^n \,df; \qquad \mathrm{for} \quad n=0,1,2,3,4
    /// $$
    ///
    pub fn m_n(&self, n: usize) -> f64 {
        self.alpha
            * GRAVITY.powi(2)
            * (TWO_PI).powi(-4)
            * self.fp().unwrap_or(0.0).powi((n as i32) - 4)
            / (4. - n as f64)
    }

    /// Analytic form for the spectral moments $M_n$
    /// where $n=0,1,2,3,4$
    ///
    /// # Desciption
    ///
    /// The spectral moments $M_n$ are defined:
    /// $$
    /// M_n = \int_0^\infty S(\omega) \omega^n \,d\omega; \qquad \mathrm{for} \quad  n=0,1,2,3,4
    /// $$
    ///
    #[allow(non_snake_case)]
    pub fn M_n(&self, n: usize) -> f64 {
        self.alpha
            * GRAVITY.powi(2)
            * (1. / TWO_PI).powi(n as i32)
            * self.wp().unwrap_or(0.0).powi((n as i32) - 4)
            / (4. - n as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phillips_energy() {
        let phillips = Phillips1958::new();
        let energy = phillips.energy();

        assert_eq!(phillips.alpha, 0.0081);
        assert_eq!(phillips.omega.len(), N_FREQ);
        assert_eq!(energy.len(), N_FREQ);
    }

    #[test]
    fn test_raises() {
        let phillips = Phillips1958::new();
        assert!(phillips.hs().is_err());
        assert!(phillips.tp().is_err());
    }
}
