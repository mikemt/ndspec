use ndarray::{Array1, ArrayView1};
use std::f64::consts::PI;

use crate::core::integration::trapz;
use crate::spectrums::{Spectrum1D, Spectrum2D};
use crate::waves::spreading::Spreading;

const N_FREQ: usize = 256;

/// # Bretschneider spectrum
///
/// A struct for the Bretschneider energy density spectrum.
///
/// ## Fields
/// * `hs` - significant wave height \[m\]
/// * `tp` - peak wave period \[s\]
/// * `omega` - frequency range \[rad/s\]
///
/// ## Methods
/// * `energy` - calculate the energy density spectrum
/// * `set_hs` - set the significant wave height
/// * `set_tp` - set the peak wave period
/// * `set_omega` - set the frequency range
/// * `set_parameters` - set the significant wave height and peak wave period
/// * `abs_error` - calculate the absolute error between the significant wave height and the integrated energy
/// * `to_spec1d` - convert the spectrum to a 1D spectrum
/// * `to_spec2d` - convert the spectrum to a 2D spectrum
///
/// ## Examples
///
/// ```
/// use spectrums as ndspec;
///
/// let mut spec = ndspec::Bretschneider::default();
/// spec.set_hs(1.5).set_tp(18.0);
/// let energy = spec.energy();
/// ```
///
/// ## Description
///
/// The Bretschneider energy density spectrum is defined[^1][^2]:
///
/// $$
///  S(\omega) = \frac{5}{16} H_s^2 \omega_p^4 \omega^{-5} \exp\left(-\frac{5}{4}\left(\frac{\omega_p}{\omega}\right)^4\right)
/// $$
///
/// ## References
/// [^1]: C. L. Bretschneider, "Revisions in wave forecasting: deep and
/// shallow water," Int. Conf. Coastal. Eng., no. 6, p. 3, Jan. 1957,
/// doi: 10.9753/icce.v6.3.
/// [^2]: DNV, "DNV-RP-C205: Environmental conditions and environmental loads," 2021.

///
pub struct Bretschneider {
    pub hs: f64,
    pub tp: f64,
    pub omega: Array1<f64>,
}

pub fn energy(omega: ArrayView1<f64>, hs: f64, tp: f64) -> Array1<f64> {
    let wp = 2. * PI / tp;
    let a = 5. / 16. * hs.powf(2f64) * wp.powf(4f64);

    let energy: Array1<f64> = omega
        .iter()
        .map(|w_i| a * w_i.powf(-5.) * (-5. / 4. * (wp / w_i).powf(4.)).exp())
        .collect();

    energy
}

impl Default for Bretschneider {
    fn default() -> Self {
        Bretschneider {
            hs: 1.0,
            tp: 10.0,
            omega: Array1::<f64>::linspace(0.1, PI, N_FREQ),
        }
    }
}

impl Bretschneider {
    pub fn new(hs: f64, tp: f64) -> Self {
        Bretschneider {
            hs,
            tp,
            ..Default::default()
        }
    }

    pub fn energy(&self) -> Array1<f64> {
        energy(self.omega.view(), self.hs, self.tp)
    }

    pub fn set_hs(&mut self, hs: f64) -> &mut Self {
        self.hs = hs;
        self
    }

    pub fn set_tp(&mut self, tp: f64) -> &mut Self {
        self.tp = tp;
        self
    }

    pub fn set_omega(&mut self, omega: Array1<f64>) -> &mut Self {
        self.omega = omega;
        self
    }

    pub fn set_parameters(&mut self, hs: f64, tp: f64) -> &mut Self {
        self.set_hs(hs).set_tp(tp)
    }

    pub fn abs_error(&self) -> f64 {
        let area = trapz(self.energy().view(), self.omega.view());
        (self.hs - 4.0 * area.sqrt()).abs()
    }

    pub fn to_spec1d(&self) -> Spectrum1D {
        Spectrum1D::new(self.omega.to_owned(), self.energy())
    }

    pub fn to_spec2d(&self, spreading: &Spreading) -> Spectrum2D {
        Spectrum2D::from_spec1d(&self.to_spec1d(), spreading)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let bretschneider = Bretschneider::default();
        assert_eq!(bretschneider.hs, 1.0);
        assert_eq!(bretschneider.tp, 10.0);
    }

    #[test]
    fn test_new() {
        let hs = 1.5;
        let tp = 18.0;
        let spec = Bretschneider::new(hs, tp);
        assert_eq!(spec.hs, hs);
        assert_eq!(spec.tp, tp);
    }

    #[test]
    fn test_energy() {
        let omega = Array1::linspace(0.1, PI, 100);
        let hs = 1.0;
        let tp = 10.0;

        let energy = energy(omega.view(), hs, tp);

        assert_eq!(energy.len(), omega.len());
    }

    #[test]
    fn test_set() {
        let hs = 1.5;
        let mut spec = Bretschneider::default();
        spec.set_hs(hs);
        assert_eq!(spec.hs, hs);

        let tp = 18.0;
        spec.set_tp(tp);
        assert_eq!(spec.tp, tp);

        let hs = 2.0;
        spec.set_hs(hs);
        assert_eq!(spec.hs, hs);
    }

    #[test]
    fn test_bretschneider() {
        let omega = Array1::linspace(0.1, PI, 100);
        let hs = 1.0;
        let tp = 10.0;

        let mut spec = Bretschneider::new(hs, tp);
        let energy = spec.set_omega(omega).energy();

        assert_eq!(energy.len(), 100);
    }
}
