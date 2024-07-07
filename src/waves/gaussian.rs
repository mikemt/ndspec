use ndarray::{Array1, ArrayView1};
use std::f64::consts::PI;

#[allow(dead_code)]
/// Gaussian spectrum as per [1, page 62]
///
/// References:
/// ISO 19901-1:2015
pub fn gaussian_spectrum(omega: ArrayView1<f64>, hs: f64, tp: f64, sigma_g: f64) -> Array1<f64> {
    let f = &omega / (2. * PI);
    let fp = 1. / tp;

    let energy: Array1<f64> = f
        .iter()
        .map(|f_i| {
            let a = 1. / (sigma_g * (2. * PI).sqrt());
            let b = (-0.5 * ((f_i - fp) / sigma_g).powf(2.)).exp();
            a * b * hs.powf(2.) / 16.
        })
        .collect();

    energy / (2. * PI)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_gaussian_spectrum() {
        let omega: Array1<f64> = array![0.4, 0.5, 0.6, 0.7, 0.8];
        let hs = 1.0;
        let tp = 10.0;
        let sigma_g = 0.01;

        let expected_energy: Array1<f64> =
            array![0.00053862, 0.0493098, 0.35850921, 0.20700777, 0.00949275];

        let energy = gaussian_spectrum(omega.view(), hs, tp, sigma_g);

        assert_abs_diff_eq!(
            energy.as_slice().unwrap(),
            expected_energy.as_slice().unwrap(),
            epsilon = EPSILON
        );
    }
}
