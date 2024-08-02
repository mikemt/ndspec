use ndarray::{Array1, ArrayView1};
use special::Gamma;
use std::f64::consts::{FRAC_PI_2, PI};

use crate::core::integration::trapz;
use crate::core::ndarray_ext::Angles;

#[derive(Clone, Debug)]
pub struct Spreading {
    pub theta_p: f64,
    pub n: f64,
    pub s: f64,
    pub theta: Array1<f64>,
}

impl Default for Spreading {
    fn default() -> Self {
        let n = 0.0;

        Spreading {
            theta_p: 0.0,
            n,
            s: s_from_n(n),
            theta: Array1::linspace(0.0, 360.0, 361),
        }
    }
}

impl Spreading {
    pub fn new(theta_p: f64, n: f64) -> Self {
        Self {
            theta_p,
            n,
            s: s_from_n(n),
            ..Default::default()
        }
    }

    pub fn from_d1_form(theta_p: f64, n: f64) -> Self {
        Self {
            theta_p,
            n,
            s: s_from_n(n),
            ..Default::default()
        }
    }

    pub fn from_d2_form(theta_p: f64, s: f64) -> Self {
        Self {
            theta_p,
            n: n_from_s(s),
            s,
            ..Default::default()
        }
    }

    pub fn set_theta(&mut self, theta: Array1<f64>) -> &mut Self {
        self.theta = theta;
        self
    }

    pub fn set_theta_p(&mut self, theta_p: f64) -> &mut Self {
        self.theta_p = theta_p;
        self
    }

    pub fn set_n(&mut self, n: f64) -> &mut Self {
        self.n = n;
        self.s = s_from_n(n);
        self
    }

    pub fn set_s(&mut self, s: f64) -> &mut Self {
        self.s = s;
        self.n = n_from_s(s);
        self
    }

    pub fn distribution(&self) -> Array1<f64> {
        spread_cos_n(self.theta.view(), self.theta_p, self.n)
    }

    pub fn d1_form(&self) -> Array1<f64> {
        spread_cos_n(self.theta.view(), self.theta_p, self.n)
    }

    pub fn d2_form(&self) -> Array1<f64> {
        spread_cos_2s(self.theta.view(), self.theta_p, self.s)
    }

    pub fn area(&self) -> f64 {
        trapz(self.distribution().view(), self.theta.to_radians().view())
    }

    pub fn abs_error(&self) -> f64 {
        (1.0 - self.area()).abs()
    }
}

/// The so-called PNJ spreading function (type 0) for a wind generated wave
///
/// # Description
///
/// A simple cosine spreading function after Pierson, Neumann and
/// James[^1] for a wind generated wave travelling in the direction
/// $\theta_p$. It comprises the following form:
///
/// $$
/// D_0(\theta) = \frac{2}{\pi} \cos^2(\theta - \theta_p)
/// $$
///
/// where $\theta$ is the wave direction and $\theta_p$ is the peak
/// spectral wave direction. Both are defined in degrees.
///
/// # Arguments
///
/// * `theta` - the wave direction in degrees
/// * `theta_p` - the peak spectral wave direction in degrees
///
/// # Returns
///
/// The directional spreading function in the form of a 1D array.
///
/// # References
///
/// [^1]: Pierson, W. J., Neuman, G. & James, R. W. *Practical Methods
/// for Observing and Forecasting Ocean Waves by Means of Wave Spectra
/// and Statistics*. (Reprinted 1971). U.S. Naval Hydrographic Office,
/// Washington D.C., H.O. Pub 603, (1955) doi:
/// <http://dx.doi.org/10.25607/OBP-985>.
///
pub fn type_0(theta: ArrayView1<f64>, theta_p: f64) -> Array1<f64> {
    let theta_r = (theta.to_owned() - theta_p).to_radians().mapv(|angle| {
        if angle < -PI {
            angle + 2.0 * PI
        } else if angle > PI {
            angle - 2.0 * PI
        } else {
            angle
        }
    });

    theta_r.mapv(|angle| 2.0 / PI * angle.cos().powi(2))
}

/// Calculates the directional spreading function in D_1 form.
pub fn spread_cos_n(theta: ArrayView1<f64>, theta_p: f64, n: f64) -> Array1<f64> {
    let theta_p = theta_p.to_radians();
    let theta_rel = theta.to_owned().to_radians() - theta_p;

    // ensure all angles are in the range [-pi, pi]
    let theta = theta_rel.mapv(|angle| {
        if angle < -PI {
            angle + 2.0 * PI
        } else if angle > PI {
            angle - 2.0 * PI
        } else {
            angle
        }
    });

    let c = |n: f64| Gamma::gamma(0.5 * n + 1.0) / (PI.sqrt() * Gamma::gamma(0.5 * n + 0.5));

    // compute the spreading function
    theta.mapv_into(|th| {
        if th.abs() < FRAC_PI_2 {
            c(n) * (th.cos()).powf(n)
        } else {
            0.0
        }
    })
}

/// Calculates the directional spreading function in D_2 form.
pub fn spread_cos_2s(theta: ArrayView1<f64>, theta_p: f64, s: f64) -> Array1<f64> {
    let theta_p = theta_p.to_radians();
    let theta_r = theta.to_owned().to_radians() - theta_p;

    let c = |s: f64| Gamma::gamma(s + 1.0) / (2.0 * PI.sqrt() * Gamma::gamma(s + 0.5));

    // ensure all angles are in the range [-pi, pi]
    let theta_r = theta_r.mapv(|angle| {
        if angle < -PI {
            angle + 2.0 * PI
        } else if angle > PI {
            angle - 2.0 * PI
        } else {
            angle
        }
    });

    // compute the spreading function
    theta_r.mapv_into(|th| {
        if th.abs() < PI {
            c(s) * (th / 2.0).cos().powf(2. * s)
        } else {
            0.0
        }
    })
}

fn s_from_n(n: f64) -> f64 {
    2.0 * n + 1.0
}

fn n_from_s(s: f64) -> f64 {
    (s - 1.0) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use ndarray_stats::QuantileExt;
    use std::f64;

    use crate::core::trapz;

    #[test]
    fn test_spreading_default() {
        let spreading = Spreading::default();
        assert_eq!(spreading.theta_p, 0.0);
        assert_eq!(spreading.n, 0.0);
        assert_eq!(spreading.s, 1.0);
        assert_eq!(spreading.theta.len(), 361);
    }

    #[test]
    fn test_spreading_new() {
        let spreading = Spreading::new(45.0, 2.0);
        assert_eq!(spreading.theta_p, 45.0);
        assert_eq!(spreading.n, 2.0);
        assert_eq!(spreading.s, 5.0);
        assert_eq!(spreading.theta.len(), 361);

        println!("{:?}", spreading.distribution());

        let i = spreading.distribution().argmax().unwrap();

        println!("theta: {:?}", spreading.theta[i]);
        println!(
            "value: {:?}",
            trapz(
                spreading.distribution().view(),
                spreading.theta.to_radians().view()
            )
        );

        let spreading = Spreading::new(180.0, 2.0);
        assert_eq!(spreading.theta_p, 180.0);
        assert_eq!(spreading.n, 2.0);
        assert_abs_diff_eq!(spreading.area(), 1.0, epsilon = 1E-3);
    }

    #[test]
    fn test_spreading_from_d1_form() {
        let theta_p = 45.0;

        let spreading = Spreading::from_d1_form(theta_p, 10.0);

        assert_eq!(spreading.theta_p, theta_p);
        assert_eq!(spreading.n, 10.0);
        assert_eq!(spreading.s, 21.0);
        assert_eq!(spreading.theta.len(), 361);

        let d1 = spread_cos_n(spreading.theta.view(), spreading.theta_p, spreading.n);
        let d2 = spread_cos_2s(spreading.theta.view(), spreading.theta_p, spreading.s);

        assert_abs_diff_eq!(
            spreading.distribution().as_slice().unwrap(),
            d1.as_slice().unwrap(),
            epsilon = f64::EPSILON
        );

        assert_abs_diff_eq!(
            d1.as_slice().unwrap(),
            d2.as_slice().unwrap(),
            epsilon = 0.1
        );

        assert_abs_diff_eq!(spreading.abs_error(), 0.0, epsilon = 1E-2);
    }

    #[test]
    fn test_spreading_from_d2_form() {
        let spreading = Spreading::from_d2_form(45.0, 401.0);
        assert_eq!(spreading.theta_p, 45.0);
        assert_eq!(spreading.n, 200.0);
        assert_eq!(spreading.s, 401.0);
        assert_eq!(spreading.theta.len(), 361);
    }

    #[test]
    fn test_spread_cos_n() {
        let theta = array![0.0, 45.0, 90.0, 135.0, 180.0];
        let theta_p = 90.0;
        let n = 2.0;

        let result = spread_cos_n(theta.view(), theta_p, n);

        let expected = array![
            0.,
            f64::consts::FRAC_1_PI,
            f64::consts::FRAC_2_PI,
            f64::consts::FRAC_1_PI,
            0.
        ];

        assert_abs_diff_eq!(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn test_spread_cos_2s() {
        let theta = array![0.0, 45.0, 90.0, 135.0, 180.0];
        let theta_p = 90.0;
        let s = 2.0;

        let result = spread_cos_2s(theta.view(), theta_p, s);

        // values derived externally from an equivalent Python implementation
        let expected = array![0.1061033, 0.30920766, 0.42441318, 0.30920766, 0.1061033];

        assert_abs_diff_eq!(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            epsilon = 1e-6
        );

        let spreading = Spreading::new(90.0, 10.0);
        let dist = spread_cos_2s(spreading.theta.view(), spreading.theta_p, spreading.s);

        let theta_r = spreading.theta.mapv(|angle| angle.to_radians());
        let area = trapz(dist.view(), theta_r.view());

        assert_abs_diff_eq!(area, 1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_energy() {
        let spreading = Spreading::new(90.0, 10.0);
        let dist = spreading.distribution();
        let max_value = dist.max().unwrap().to_owned();

        // value derived externally from an equivalent Python implementation
        assert_abs_diff_eq!(max_value, 1.293449696238895, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_error() {
        let spreading = Spreading::new(90.0, 10.0);
        let error = spreading.abs_error();

        assert_abs_diff_eq!(error, 0.0, epsilon = f64::EPSILON);
    }
}
