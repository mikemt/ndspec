use ndarray::{arr1, Array1, ArrayView1};
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::prelude::*;

use crate::core::constants::GRAVITY;
use crate::core::integration::trapz;
use crate::core::interpolation;
use crate::spectrums::{Spectrum1D, Spectrum2D};
use crate::waves::bretschneider;
use crate::waves::spreading::Spreading;

use crate::waves::SpectralCommon;

#[derive(Clone)]
pub struct Jonswap {
    pub hs: f64,
    pub tp: f64,
    pub gamma: f64,
    pub sigma_a: f64,
    pub sigma_b: f64,
    pub omega: Array1<f64>,
}

impl Default for Jonswap {
    fn default() -> Self {
        Jonswap {
            hs: 0.001,
            tp: 10.0,
            gamma: 3.3,
            sigma_a: 0.07,
            sigma_b: 0.09,
            omega: Array1::<f64>::linspace(0.1, PI, 512),
        }
    }
}

impl Jonswap {
    /// Jonswap spectrum from Hs and Tp using the Lewis
    /// and Allos (1993) parmaeterization to determine the remaining
    /// parameters.
    pub fn new(hs: f64, tp: f64) -> Self {
        Jonswap::from_2p(hs, tp)
    }

    /// Create a Jonswap spectrum from Hs and Tp using the Lewis
    /// and Allos (1993) parmaeterization to determine the remaining
    /// parameters.
    pub fn from_2p(hs: f64, tp: f64) -> Self {
        let (_, gamma, sigma_a, sigma_b) = lewis_allos(hs, tp);
        Jonswap {
            hs,
            tp,
            gamma,
            sigma_a,
            sigma_b,
            ..Default::default()
        }
    }

    /// Create a Jonswap spectrum from Hs, Tp and gamma.
    pub fn from_3p(hs: f64, tp: f64, gamma: f64) -> Self {
        Jonswap {
            hs,
            tp,
            gamma,
            ..Default::default()
        }
    }

    /// Create a Jonswap spectrum from Hs, Tp, gamma, sigma_a and sigma_b.
    pub fn from_5p(hs: f64, tp: f64, gamma: f64, sigma_a: f64, sigma_b: f64) -> Self {
        Jonswap {
            hs,
            tp,
            gamma,
            sigma_a,
            sigma_b,
            ..Default::default()
        }
    }

    pub fn set_hs(&mut self, hs: f64) -> &mut Self {
        self.hs = hs;
        self
    }

    pub fn set_tp(&mut self, tp: f64) -> &mut Self {
        self.tp = tp;
        self
    }

    pub fn set_gamma(&mut self, gamma: f64) -> &mut Self {
        self.gamma = gamma;
        self
    }

    pub fn set_sigma_a(&mut self, sigma_a: f64) -> &mut Self {
        self.sigma_a = sigma_a;
        self
    }

    pub fn set_sigma_b(&mut self, sigma_b: f64) -> &mut Self {
        self.sigma_b = sigma_b;
        self
    }

    pub fn set_omega(&mut self, omega: Array1<f64>) -> &mut Self {
        self.omega = omega;
        self
    }

    /// Returns a summary of the defining parameters as a HashMap.
    pub fn get_parameters(&self) -> HashMap<String, f64> {
        [
            ("hs", self.hs),
            ("tp", self.tp),
            ("gamma", self.gamma),
            ("sigma_a", self.sigma_a),
            ("sigma_b", self.sigma_b),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect()
    }

    pub fn tz(&self) -> f64 {
        convert_tp_to_tz(self.tp, self.gamma)
    }

    pub fn t1(&self) -> f64 {
        convert_tp_to_t1(self.tp, self.gamma)
    }

    pub fn set_tp_from_tz(&mut self, tz: f64) -> &mut Self {
        self.tp = convert_tz_to_tp(tz, self.gamma);
        self
    }

    pub fn set_tp_from_t1(&mut self, t1: f64) -> &mut Self {
        self.tp = convert_t1_to_tp(t1, self.gamma);
        self
    }

    pub fn energy(&self) -> Array1<f64> {
        energy(
            self.omega.view(),
            self.hs,
            self.tp,
            self.gamma,
            self.sigma_a,
            self.sigma_b,
        )
    }

    pub fn abs_error(&self) -> f64 {
        let area = trapz(self.energy().view(), self.omega.view());
        (self.hs - 4.0 * area.sqrt()).abs()
    }

    pub fn to_spec1d(&self) -> Spectrum1D {
        Spectrum1D::new(self.omega.to_owned(), self.energy().to_owned()).to_owned()
    }

    pub fn to_spec2d(&self, spreading: &Spreading) -> Spectrum2D {
        Spectrum2D::from_spec1d(&self.to_spec1d(), spreading)
    }
}

pub fn energy(
    omega: ArrayView1<f64>,
    hs: f64,
    tp: f64,
    gamma: f64,
    sigma_a: f64,
    sigma_b: f64,
) -> Array1<f64> {
    let mut spectrum = Array1::<f64>::zeros(omega.len());
    let spectrum_0 = bretschneider::energy(omega, hs, tp);

    let wp = 2. * PI / tp;
    let a = 1.0 - 0.287 * gamma.ln();
    let mut s: f64;

    for (i, w_i) in omega.iter().enumerate() {
        if w_i <= &wp {
            s = sigma_a;
        } else {
            s = sigma_b;
        }

        let b = (-0.5 * ((w_i - wp) / (s * wp)).powf(2f64)).exp();
        spectrum[i] = a * spectrum_0[i] * gamma.powf(b);
    }

    spectrum
}

pub fn maxhs(tp: f64) -> f64 {
    let sp = if tp <= 8.0 {
        1. / 15.
    } else if tp >= 15.0 {
        1. / 25.
    } else {
        interpolation::linear_interp(
            tp,
            arr1(&[8.0, 15.0]).view(),
            arr1(&[1. / 15., 1. / 25.]).view(),
        )
    };

    GRAVITY * sp * tp.powi(2) / (2.0 * PI)
}

pub fn lewis_allos(hs: f64, tp: f64) -> (f64, f64, f64, f64) {
    let g: f64 = 9.80665;
    let m0: f64 = hs.powi(2) / 16.0;

    let alpha = 103.39 * m0.powf(0.687) * g.powf(-1.375) * tp.powf(-2.75);
    let gamma = (1.0f64).max(221400.0 * m0.powf(0.887) * g.powf(-1.774) * tp.powf(-3.55));
    let sigma_a = 0.001071 * m0.powf(-0.331) * g.powf(0.662) * tp.powf(1.325);
    let sigma_b = 0.01104 * m0.powf(-0.165) * g.powf(0.33) * tp.powf(0.66);

    (alpha, gamma, sigma_a, sigma_b)
}

pub fn convert_tp_to_tz(tp: f64, gamma: f64) -> f64 {
    tp * (0.6673 + 0.05037 * gamma - 0.006230 * gamma.powf(2f64) + 0.0003341 * gamma.powf(3f64))
}

pub fn convert_tz_to_tp(tz: f64, gamma: f64) -> f64 {
    tz / (0.6673 + 0.05037 * gamma - 0.006230 * gamma.powf(2.) + 0.0003341 * gamma.powf(3.))
}

pub fn convert_tp_to_t1(tp: f64, gamma: f64) -> f64 {
    tp * (0.7303 + 0.04936 * gamma - 0.006556 * gamma.powf(2f64) + 0.0003610 * gamma.powf(3f64))
}

pub fn convert_t1_to_tp(t1: f64, gamma: f64) -> f64 {
    t1 / (0.7303 + 0.04936 * gamma - 0.006556 * gamma.powf(2f64) + 0.0003610 * gamma.powf(3f64))
}

/// The original JONSWAP spectral formulation from the joint North Sea wave project.
///
/// # Description
///
/// Implementation of the original JONSWAP spectral formulation as
/// described in Hasselmann et al. (1973)[^1]. The form of which is
/// dependent upon the wind speed $U_{10}$ defined at 10$~\mathrm{m}$ above the
/// sea surface and the fetch length $x$ defined in $\mathrm{m}$. The
/// one-dimensional energy density, as a function of $\omega$, with five
/// free parameters $\omega_p$, $\alpha$, $\gamma$, $\sigma_a$ and
/// $\sigma_b$ is written
/// (Hasselmann *et. al.*[^1], Eq. 2.4.1):
/// $$
/// S(\omega) = \alpha g^{2} \omega^{-5}
/// \exp \left[-\frac{5}{4} \left(\frac{\omega_p}{\omega}\right)^4 \right]
/// \gamma^G,
/// $$
/// $$
/// G = \exp \left[-\frac{(\omega-\omega_p)^2}{2 \sigma^{2} \omega_{p}^{2}}\right],
/// $$
/// The free parameters are defined as:
/// $$
/// \alpha = 0.0076 (U_{10}^{2} / gF)^{0.22},
/// $$
/// $$
/// \omega_{p} = 2 \pi 3.5 (g^{2} / U_{10}F)^{1/3},
/// $$
/// $$
/// \gamma=3.3,
/// $$
/// and
/// $$
/// \sigma =
/// \begin{cases}
/// 0.07,&\quad \omega \le \omega_p,\\\\
/// 0.09,&\quad \omega >   \omega_p.\\
/// \end{cases}
/// $$
///
/// Where we have the following definitions:
/// - $U_{10}$ is the wind speed at a height of $10~\mathrm{m}$ above the free-surface ($\mathrm{m/s}$)
/// - $F$ is the fetch length measured in $\mathrm{m}$
/// - $\omega_p$ is the peak spectral wave period (s)
/// - $\gamma$ denotes the peak shape parameter  (-)
/// - $g$ is the gravitational acceleration constant ($\mathrm{m^2/s}$)
///
/// The significant wave height $H_s$ can be recovered under the assumption of a narrow-banded process[^2]:
/// $$
/// H_{s} \equiv 0.0163 X^{0.5} U_{10}
/// $$
/// where $X$ is the fetch length in $\mathrm{km}$.
///
/// # Fields
///
/// * `u` - wind speed at 10$\mathrm{~m} above the sea surface in $\mathrm{m/s}$
/// * `x` - fetch length in $\mathrm{m}$
/// * `omega` - frequency space in $\mathrm{rad/s}$
///
/// # Methods
///
/// * `hs` - significant wave height in $\mathrm{m}$
/// * `tp` - peak spectral wave period in $\mathrm{s}$
/// * `wp` - peak spectral wave frequency in $\mathrm{rad/s}$
/// * `energy` - the energy density in $\mathrm{m^2 s/rad}$
/// * `abs_error` - calculate the absolute error between the
/// significant wave height and the corresponding sigmificant wave
/// height calculated through the integrated energy density
/// * `to_spec1d` - convert the one dimensional spectrum to a `Spectrum1` Type
/// * `to_spec2d` - convert the one-dimensional spectrum to a `Spectrum2` Type for a given `Spreading` Type
///
/// # Default values
///
/// * `u` - 10.0 m/s
/// * `x` - 1000.0 m
/// * `omega` - 512 linearly spaced points between 0.1 and $\pi$ rad/s
///
/// # Example
///
/// ```
/// use oceanwaves::waves::Jonswap1973;
///
/// let mut jonswap = Jonswap1973::default();
/// let hs = jonswap.hs().unwrap();
/// let tp = jonswap.tp().unwrap();
/// let wp = jonswap.wp().unwrap();
/// let energy = jonswap.energy();
/// let abs_error = jonswap.abs_error().unwrap();
/// ```
///
/// # References
///
/// [^1]: Hasselmann, K. *et al.* Measurements of wind-wave growth and
/// swell decay during the joint North Sea wave project
/// (JONSWAP). *Deutschen Hydrographischen Zeitschrift* **12**,
/// (1973).
/// [^2]: Hasselmann, K., Sell, W., Ross, D. B. & M$\text{\\\"{u}}$ller, P. A
/// Parametric Wave Prediction Model. *Journal of Physical
/// Oceanography* **6**, 200 228 (1976).
///
pub struct Jonswap1973 {
    pub u: f64,
    pub x: f64,
    pub omega: Array1<f64>,
}

fn energy_jonswap1973(omega: ArrayView1<f64>, u: f64, x: f64) -> Array1<f64> {
    // default constants as per JONSWAP 1973
    let gamma: f64 = 3.3;
    let sigma_a: f64 = 0.07;
    let sigma_b: f64 = 0.09;

    let alpha = 0.076 * (u.powi(2) / (x * GRAVITY));
    let omega_p = TWO_PI * 3.5 * (GRAVITY.powi(2) / (u * x)).powf(1. / 3.);

    omega
        .iter()
        .map(|w| {
            let sigma = match w {
                w if w <= &omega_p => sigma_a,
                _ => sigma_b,
            };
            let shape_factor = (-0.5 * ((w - omega_p) / (sigma * omega_p)).powi(2)).exp();
            let a = alpha * GRAVITY.powi(2) * w.powi(-5) * (-1.25 * (omega_p / w).powi(4)).exp();
            a * gamma.powf(shape_factor)
        })
        .collect()
}

impl Default for Jonswap1973 {
    fn default() -> Self {
        Jonswap1973 {
            u: 10.0,
            x: 1000.0,
            omega: Array1::<f64>::linspace(0.1, PI, 512),
        }
    }
}

impl SpectralCommon for Jonswap1973 {
    fn omega(&self) -> &Array1<f64> {
        &self.omega
    }

    fn hs(&self) -> Result<f64, String> {
        Ok(0.0163 * (self.x / 1000.).sqrt() * self.u)
    }

    fn tp(&self) -> Result<f64, String> {
        Ok(TWO_PI / self.wp()?)
    }

    fn wp(&self) -> Result<f64, String> {
        Ok(TWO_PI * 3.5 * (GRAVITY.powi(2) / (self.u * self.x)).powf(1. / 3.))
    }

    fn energy(&self) -> Array1<f64> {
        energy_jonswap1973(self.omega.view(), self.u, self.x)
    }

    fn abs_error(&self) -> Result<f64, String> {
        let area = trapz(self.energy().view(), self.omega().view());
        Ok((self.hs()? - 4.0 * area.sqrt()).abs())
    }

    fn to_spec1d(&self) -> Spectrum1D {
        Spectrum1D::new(self.omega().to_owned(), self.energy().to_owned()).to_owned()
    }

    fn to_spec2d(&self, spreading: &Spreading) -> Spectrum2D {
        Spectrum2D::from_spec1d(&self.to_spec1d(), spreading)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::integration::trapz2d;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use ndarray::arr1;

    #[test]
    fn test_jonswap() {
        let hs = 2.0;
        let tp = 10.0;
        let gamma = 1.0;
        let sigma_a = 0.16235213912552415;
        let sigma_b = 0.13474442172471993;

        let spec = Jonswap::new(hs, tp);
        assert_relative_eq!(spec.hs, hs);
        assert_relative_eq!(spec.tp, tp);
        assert_relative_eq!(spec.gamma, gamma);
        assert_relative_eq!(spec.sigma_a, sigma_a);
        assert_relative_eq!(spec.sigma_b, sigma_b);
    }

    #[test]
    fn test_jonswap_from_2p() {
        let hs = 2.0;
        let tp = 10.0;

        let spec = Jonswap::from_2p(hs, tp);
        assert_relative_eq!(spec.hs, hs);
        assert_relative_eq!(spec.tp, tp);
        assert_relative_eq!(spec.gamma, 3.3);
        assert_relative_eq!(spec.sigma_a, 0.07);
        assert_relative_eq!(spec.sigma_b, 0.09);
    }

    #[test]
    fn test_jonswap_from_3p() {
        let hs = 2.0;
        let tp = 10.0;
        let gamma = 3.3;

        let spec = Jonswap::from_3p(hs, tp, gamma);
        assert_relative_eq!(spec.hs, hs);
        assert_relative_eq!(spec.tp, tp);
        assert_relative_eq!(spec.gamma, gamma);
        assert_relative_eq!(spec.sigma_a, 0.07);
        assert_relative_eq!(spec.sigma_b, 0.09);
    }

    #[test]
    fn test_jonswap_from_5p() {
        let hs = 2.0;
        let tp = 10.0;
        let gamma = 3.3;
        let sigma_a = 0.1;
        let sigma_b = 0.2;

        let spec = Jonswap::from_5p(hs, tp, gamma, sigma_a, sigma_b);
        assert_relative_eq!(spec.hs, hs);
        assert_relative_eq!(spec.tp, tp);
        assert_relative_eq!(spec.gamma, gamma);
        assert_relative_eq!(spec.sigma_a, sigma_a);
        assert_relative_eq!(spec.sigma_b, sigma_b);
    }

    #[test]
    fn test_energy() {
        // Test case values
        let hs = 2.0;
        let tp = 10.0;
        let gamma = 3.3;

        let mut spec = Jonswap::from_3p(hs, tp, gamma);

        let omega = Array1::<f64>::linspace(0.1, PI, 10);
        let energy = spec.set_omega(omega).energy();

        // Expected results
        let energy_expected = arr1(&[
            0., 0.03985516, 0.27677771, 0.06581066, 0.01900272, 0.00684205, 0.0029088, 0.00139781,
            0.00073696, 0.00041764,
        ]);

        assert_abs_diff_eq!(
            energy.as_slice().unwrap(),
            energy_expected.as_slice().unwrap(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_set() {
        let mut jonswap = Jonswap::default();

        let spec = jonswap
            .set_hs(3.0)
            .set_tp(12.0)
            .set_gamma(1.0)
            .set_sigma_a(0.1)
            .set_sigma_b(0.2);

        assert_relative_eq!(spec.hs, 3.0);
        assert_relative_eq!(spec.tp, 12.0);
        assert_relative_eq!(spec.gamma, 1.0);
        assert_relative_eq!(spec.sigma_a, 0.1);
        assert_relative_eq!(spec.sigma_b, 0.2);
    }

    #[test]
    fn test_abs_error() {
        let hs = 2.0;
        let tp = 10.0;
        let gamma = 3.3;

        let spec = Jonswap::from_3p(hs, tp, gamma);

        let abs_error = spec.abs_error();
        assert_abs_diff_eq!(abs_error, 0.001, epsilon = 1e-3);
    }

    #[test]
    fn test_spec2d() {
        let hs = 2.0;
        let tp = 10.0;
        let gamma = 3.3;

        let spectrum = Jonswap::from_3p(hs, tp, gamma);
        let spreading = Spreading::new(0.0, 2.0);

        let spec2d = spectrum.to_spec2d(&spreading);

        let theta_r = &spreading.theta.mapv(|angle| angle.to_radians());
        let m0 = trapz2d(spec2d.energy.view(), spectrum.omega.view(), theta_r.view());

        assert_abs_diff_eq!(4.0 * m0.sqrt(), hs, epsilon = 1e-2);
    }

    struct TestCase {
        hs: f64,
        tp: f64,
        expected_alpha: f64,
        expected_gamma: f64,
        expected_sigma_a: f64,
        expected_sigma_b: f64,
    }

    #[test]
    fn test_lewis_allos() {
        // Test case values and expected results
        let test_cases = vec![
            TestCase {
                hs: 10.0,
                tp: 15.0,
                expected_alpha: 0.00919721239454214,
                expected_gamma: 1.309307162905551,
                expected_sigma_a: 0.09573280643573788,
                expected_sigma_b: 0.10353112291143472,
            },
            TestCase {
                hs: 1.0,
                tp: 10.0,
                expected_alpha: 0.0011855153411190384,
                expected_gamma: 1.0,
                expected_sigma_a: 0.25688566637890375,
                expected_sigma_b: 0.1693755402501826,
            },
            TestCase {
                hs: 9.0,
                tp: 12.0,
                expected_alpha: 0.014699005745666294,
                expected_gamma: 2.3982712526645193,
                expected_sigma_a: 0.07637416622388807,
                expected_sigma_b: 0.09251460262369318,
            },
        ];

        for test_case in test_cases {
            let (alpha, gamma, sigma_a, sigma_b) = lewis_allos(test_case.hs, test_case.tp);

            assert_abs_diff_eq!(alpha, test_case.expected_alpha, epsilon = f64::EPSILON);
            assert_abs_diff_eq!(gamma, test_case.expected_gamma, epsilon = f64::EPSILON);
            assert_abs_diff_eq!(sigma_a, test_case.expected_sigma_a, epsilon = f64::EPSILON);
            assert_abs_diff_eq!(sigma_b, test_case.expected_sigma_b, epsilon = f64::EPSILON);
        }
    }
}
