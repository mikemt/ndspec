//! Froya wind spectrum functions

use ndarray::{Array1, ArrayView1};
use std::f64::consts::PI;

/// Froya wind spectrum with an averaging window of 10 minutes
///
/// # Arguments
/// * `omega` - Angular frequency \[rad/s\]
/// * `wind_speed` - Wind speed at 10m above sea level \[m/s\]
///
/// # Returns
/// * `Array1<f64>` - energy density spectrum [m^2 s]
///
pub fn froya_10min(omega: ArrayView1<f64>, wind_speed: f64) -> Array1<f64> {
    let mut spectrum = Array1::<f64>::zeros(omega.len());
    let elevation: f64 = 10.0;
    let t0: f64 = 60. * 60.;
    let t10: f64 = 10. * 60.;

    let a = -1. * 0.043 * 0.41 * 0.06 * (t10 / t0).ln();
    let b = 1. - 0.41 * 0.06 * (t10 / t0).ln();
    let c = -1. * wind_speed;
    let u_0 = 0.5 / a * (-b + (b.powf(2.) - 4. * a * c).sqrt());

    let u1 = u_0 / 10.0;
    let z1 = elevation / 10.0;

    for (i, w_i) in omega.iter().enumerate() {
        let f_i = w_i / (2. * PI);
        let f_bar = 172.0 * f_i * z1.powf(0.6667) * u1.powf(-0.75);
        spectrum[i] = 320.0 * u1.powf(2f64) * z1.powf(0.45) / (1. + f_bar.powf(0.468)).powf(3.561);
        spectrum[i] /= 2. * PI;
    }

    spectrum
}

/// Froya wind spectrum with an averaging window of 1 hour
///
/// # Arguments
/// * `omega` - Angular frequency \[rad/s\]
/// * `wind_speed` - Wind speed at 10m above sea level \[m/s\]
/// * `elevation` - Elevation above sea level \[m\]
///
/// # Returns
/// * `Array1<f64>` - energy density spectrum \[m^2 s\]
///
pub fn froya_1hr(omega: ArrayView1<f64>, wind_speed: f64, elevation: f64) -> Array1<f64> {
    let mut spectrum = Array1::<f64>::zeros(omega.len());
    let u1 = wind_speed / 10.0;
    let z1 = elevation / 10.0;

    for (i, w_i) in omega.iter().enumerate() {
        let f_i = w_i / (2. * PI);
        let f_bar = 172.0 * f_i * z1.powf(0.6667) * u1.powf(-0.75);
        spectrum[i] = 320.0 * u1.powf(2f64) * z1.powf(0.45) / (1. + f_bar.powf(0.468)).powf(3.561);
        spectrum[i] /= 2. * PI;
    }

    spectrum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn froya_1hr_test() {
        let omega = Array1::<f64>::from_vec(vec![0.1, 0.5]);
        let s0 = Array1::<f64>::from_vec(vec![1.6905358974436757, 0.2598720226634548]);

        let s1 = froya_1hr(omega.view(), 10., 10.);

        assert_eq!(s0, s1);
    }
}
