use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;
use std::f64::consts::PI;

use crate::core::integration::trapz2d;
use crate::core::ndarray_ext::Angles;
use crate::spectrum1d::Spectrum1D;
use crate::waves::spreading::Spreading;

pub struct Spectrum2D {
    pub omega: Array1<f64>,
    pub theta: Array1<f64>,
    pub energy: Array2<f64>,
}

impl Default for Spectrum2D {
    fn default() -> Self {
        Self {
            omega: Array1::default(2),
            theta: Array1::default(2),
            energy: Array2::default((2, 2)),
        }
    }
}
impl Spectrum2D {
    pub fn new(omega: Array1<f64>, theta: Array1<f64>, energy: Array2<f64>) -> Self {
        Self {
            omega,
            theta,
            energy,
        }
    }

    pub fn get_direction(&self) -> f64 {
        self.theta[self.energy.argmax().unwrap().0]
    }

    pub fn from_spec1d(spectrum: &Spectrum1D, spreading: &Spreading) -> Self {
        let n_omega = spectrum.omega.len();
        let n_theta = spreading.theta.len();

        // broadcast spectral energy to 2d
        let s_ij = spectrum
            .energy
            .broadcast((n_theta, n_omega))
            .unwrap()
            .to_owned();

        // broadcast spreading to 2d
        let d_ij = spreading
            .distribution()
            .into_shape((n_theta, 1))
            .unwrap()
            .broadcast((n_theta, n_omega))
            .unwrap()
            .to_owned();

        Self {
            omega: spectrum.omega.to_owned(),
            theta: spreading.theta.to_owned(),
            energy: s_ij * d_ij,
        }
    }

    pub fn area(&self) -> f64 {
        trapz2d(
            self.energy.view(),
            self.omega.view(),
            self.theta.to_radians().view(),
        )
    }

    pub fn spectral_moment(&self, n: i32) -> f64 {
        let mut omega_n = self.omega.to_owned();
        let theta_radians = self.theta.mapv(|angle| angle.to_radians());

        omega_n.mapv_inplace(|x| x.powi(n));

        trapz2d(
            (&self.energy * omega_n).view(),
            self.omega.view(),
            theta_radians.view(),
        )
    }

    #[allow(non_snake_case)]
    pub fn M0(&self) -> f64 {
        self.spectral_moment(0)
    }

    #[allow(non_snake_case)]
    pub fn M1(&self) -> f64 {
        self.spectral_moment(1)
    }

    #[allow(non_snake_case)]
    pub fn M2(&self) -> f64 {
        self.spectral_moment(2)
    }

    #[allow(non_snake_case)]
    pub fn M3(&self) -> f64 {
        self.spectral_moment(3)
    }

    #[allow(non_snake_case)]
    pub fn M4(&self) -> f64 {
        self.spectral_moment(4)
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
        2.0 * PI / self.omega[self.energy.argmax().unwrap().1]
    }

    #[allow(non_snake_case)]
    pub fn theta_p(&self) -> f64 {
        self.theta[self.energy.argmax().unwrap().0]
    }

    #[allow(non_snake_case)]
    pub fn T_mean(&self) -> f64 {
        2. * PI * self.M0() / self.M1()
    }

    #[allow(non_snake_case)]
    pub fn Tz(&self) -> f64 {
        2. * PI * self.M0() / self.M2().sqrt()
    }

    #[allow(non_snake_case)]
    /// Most probable maximum amplitude value
    pub fn Ampm(&self, time_window: f64) -> f64 {
        (2. * self.M0() * (time_window / self.Tz()).ln()).sqrt()
    }

    /// Most probable maximum double amplitude value
    #[allow(non_snake_case)]
    pub fn Xmpm(&self, time_window: f64) -> f64 {
        self.Xs() * (0.5 * (time_window / self.Tz()).ln()).sqrt()
    }

    /// Non-exceedance level for a given time window and probability of exceedance, fractile
    pub fn fractile_extreme(&self, fractile: f64, time_window: f64) -> f64 {
        let n = time_window / self.Tz();
        let f = (0.5 * n.ln()).sqrt() * (1. - (-fractile.ln()).ln() / n.ln()).sqrt();
        self.As() * f
    }

    #[allow(non_snake_case)]
    pub fn X50(&self, time_window: f64) -> f64 {
        self.fractile_extreme(0.5, time_window)
    }

    #[allow(non_snake_case)]
    pub fn X90(&self, time_window: f64) -> f64 {
        self.fractile_extreme(0.9, time_window)
    }

    #[allow(non_snake_case)]
    pub fn X95(&self, time_window: f64) -> f64 {
        self.fractile_extreme(0.95, time_window)
    }

    // Rotate the directional spectra by an angle theta in degrees
    //pub fn rotate_check(&self, theta: f64) -> Array1<f64> {
    //    let theta_new = self.theta.mapv(|angle| (angle + theta).rem_euclid(360.));
    //    theta_new
    //}

    // note that there is an off by one error here because theta \in [0,360] and not [0, 359]
    // therefore the first and last values are repeated causing the `argsort` to contain repeated values
    // this error is also in the original vresponse and wvaning Fortran code
    //
    pub fn rotate(&mut self, theta: f64) -> &mut Self {
        let theta_i = self.theta.mapv(|angle| (angle + theta).rem_euclid(360.0));
        let idx = argsort(theta_i.as_slice().unwrap());
        self.energy = self.energy.select(ndarray::Axis(0), &idx);

        self
    }
}

fn argsort(vector: &[f64]) -> Vec<usize> {
    let mut indexed: Vec<_> = vector
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    indexed.into_iter().map(|(index, _)| index).collect()
}
