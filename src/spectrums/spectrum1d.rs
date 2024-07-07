use crate::core::integration::trapz;

use ndarray::{array, Array1, ArrayView1};
use ndarray_stats::QuantileExt;
use std::f64::{consts::PI, EPSILON};

fn spectral_moment(omega: ArrayView1<f64>, energy: ArrayView1<f64>, n: u8) -> f64 {
    let mut omega_n = omega.to_owned();
    omega_n.mapv_inplace(|x| x.powi(n.into()));
    trapz((&energy * omega_n).view(), omega)
}

#[derive(Default, Clone, Copy, Debug)]
#[allow(non_snake_case)]
pub struct FrequencyResponse {
    pub std_dev: f64,
    pub Tm02: f64,
    pub As: f64,
    pub Ampm: f64,
}

impl FrequencyResponse {
    pub fn to_degrees(&self) -> Self {
        Self {
            std_dev: self.std_dev.to_degrees(),
            Tm02: self.Tm02,
            As: self.As.to_degrees(),
            Ampm: self.Ampm.to_degrees(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Spectrum1D {
    pub omega: Array1<f64>,
    pub energy: Array1<f64>,
}

impl Default for Spectrum1D {
    fn default() -> Self {
        Self {
            omega: array![0.1, 1.0],
            energy: Array1::from_elem(2, EPSILON),
        }
    }
}

impl Spectrum1D {
    pub fn new(omega: Array1<f64>, energy: Array1<f64>) -> Self {
        Self { omega, energy }
    }
    pub fn set_omega(&mut self, omega: Array1<f64>) -> &mut Self {
        self.omega = omega;
        self
    }

    pub fn set_energy(&mut self, energy: Array1<f64>) -> &mut Self {
        self.energy = energy;
        self
    }

    pub fn area(&self) -> f64 {
        trapz(self.energy.view(), self.omega.view())
    }

    pub fn spectral_moment(&self, n: u8) -> f64 {
        spectral_moment(self.omega.view(), self.energy.view(), n)
    }

    #[allow(non_snake_case)]
    pub fn M0(&self) -> f64 {
        spectral_moment(self.omega.view(), self.energy.view(), 0)
    }

    #[allow(non_snake_case)]
    pub fn M1(&self) -> f64 {
        spectral_moment(self.omega.view(), self.energy.view(), 1)
    }

    #[allow(non_snake_case)]
    pub fn M2(&self) -> f64 {
        spectral_moment(self.omega.view(), self.energy.view(), 2)
    }

    #[allow(non_snake_case)]
    pub fn M3(&self) -> f64 {
        spectral_moment(self.omega.view(), self.energy.view(), 3)
    }

    #[allow(non_snake_case)]
    pub fn M4(&self) -> f64 {
        spectral_moment(self.omega.view(), self.energy.view(), 4)
    }

    pub fn std_dev(&self) -> f64 {
        self.M0().sqrt()
    }

    #[allow(non_snake_case)]
    pub fn As(&self) -> f64 {
        2. * self.std_dev()
    }

    #[allow(non_snake_case)]
    pub fn Xs(&self) -> f64 {
        4. * self.std_dev()
    }

    /// Peak period, T_p
    #[allow(non_snake_case)]
    pub fn Tp(&self) -> f64 {
        2.0 * PI / self.omega[self.energy.argmax().unwrap()]
    }

    /// mean period, T_1, T_m
    #[allow(non_snake_case)]
    pub fn Tm01(&self) -> f64 {
        2. * PI * self.M0() / self.M1()
    }

    /// Zero-up-crossing period
    #[allow(non_snake_case)]
    pub fn Tm02(&self) -> f64 {
        2. * PI * (self.M0() / self.M2()).sqrt()
    }

    /// Most probable maximum amplitude value
    #[allow(non_snake_case)]
    pub fn Ampm(&self, time_window: f64) -> f64 {
        (2. * self.M0() * (time_window / self.Tm02()).ln()).sqrt()
    }

    /// Most probable maximum double amplitude value
    #[allow(non_snake_case)]
    pub fn Xmpm(&self, time_window: f64) -> f64 {
        self.Xs() * (0.5 * (time_window / self.Tm02()).ln()).sqrt()
    }

    pub fn response(&self, duration: f64) -> FrequencyResponse {
        FrequencyResponse {
            std_dev: self.std_dev(),
            Tm02: self.Tm02(),
            As: self.As(),
            Ampm: self.Ampm(duration),
        }
    }

    /// Non-exceedance level for a given time window and probability of exceedance, fractile
    pub fn fractile_extreme(&self, fractile: f64, time_window: f64) -> f64 {
        let n = time_window / self.Tm02();
        let f = (0.5 * n.ln()).sqrt() * (1. - (-fractile.ln()).ln() / n.ln()).sqrt();
        self.As() * f
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Jonswap;
    use ndarray::arr1;

    use approx::assert_relative_eq;

    const EPSILON: f64 = 0.01;

    #[test]
    fn test_spectrum_1d() {
        let omega = arr1(&[0.0, 1.0, 2.0, 3.0]);
        let energy = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let spectrum = Spectrum1D::new(omega, energy);

        let result = spectrum.area();
        assert_eq!(result, 7.5);
    }

    #[test]
    fn test_response() {
        let spec = Jonswap::from_3p(2.0, 10.0, 1.0).to_spec1d();
        let response = spec.response(10800.0);

        assert_relative_eq!(response.std_dev, 2.0 * 0.24975, epsilon = EPSILON);
        assert_relative_eq!(response.As, 1.0, epsilon = EPSILON);
        assert_relative_eq!(response.Tm02, 7.2826, epsilon = EPSILON);
        assert_relative_eq!(response.Ampm, 3.8276 / 2.0, epsilon = EPSILON);
    }
}
