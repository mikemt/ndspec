//! Functions for handling branch angles and conversions.

use ndarray::{Array1, ArrayView1};

pub fn convert_branch_180_vec(theta: ArrayView1<f64>) -> Array1<f64> {
    let theta_new: Array1<f64> = theta
        .iter()
        .map(|&theta_i| convert_branch_180(theta_i))
        .collect();

    theta_new
}

pub fn convert_branch_180(theta: f64) -> f64 {
    match theta {
        theta if theta % 360.0 > 180.0 => theta % 360.0 - 360.0,
        theta if theta % 360.0 < -180.0 => theta % 360.0 + 360.0,
        theta if theta % 360.0 == 0.0 => 0.0,
        _ => theta % 360.0,
    }
}

pub fn convert_branch_360(theta: f64) -> f64 {
    match theta {
        theta if theta % 360.0 < 0.0 => theta % 360.0 + 360.0,
        _ => theta % 360.0,
    }
}

pub fn theta_add_180(theta: f64) -> f64 {
    (theta + 180.0) % 360.0
}

#[cfg(test)]
mod test_azimuth_conversions {
    use super::*;

    #[test]
    fn test_convert_branch_180() {
        assert_eq!(convert_branch_180(0.0), 0.0);
        assert_eq!(convert_branch_180(180.0), 180.0);
        assert_eq!(convert_branch_180(181.0), -179.0);
        assert_eq!(convert_branch_180(-180.0), -180.0);
        assert_eq!(convert_branch_180(270.0), -90.0);
        assert_eq!(convert_branch_180(-270.0), 90.0);
        assert_eq!(convert_branch_180(360.0), 0.0);
        assert_eq!(convert_branch_180(-360.0), 0.0);
        assert_eq!(convert_branch_180(370.0), 10.0);
        assert_eq!(convert_branch_180(-370.0), -10.0);
        assert_eq!(convert_branch_180(380.0), 20.0);
        assert_eq!(convert_branch_180(-380.0), -20.0);
        assert_eq!(convert_branch_180(90.0), 90.0);
        assert_eq!(convert_branch_180(-90.0), -90.0);
        assert_eq!(convert_branch_180(45.0), 45.0);
        assert_eq!(convert_branch_180(-45.0), -45.0);
        assert_eq!(convert_branch_180(135.0), 135.0);
        assert_eq!(convert_branch_180(-135.0), -135.0);
        assert_eq!(convert_branch_180(225.0), -135.0);
        assert_eq!(convert_branch_180(-225.0), 135.0);
        //
        assert_eq!(convert_branch_180(420.0), 60.0);
        assert_eq!(convert_branch_180(-420.0), -60.0);
        assert_eq!(convert_branch_180(720.0), 0.0);
        assert_eq!(convert_branch_180(-720.0), 0.0);
        assert_eq!(convert_branch_180(730.0), 10.0);
        assert_eq!(convert_branch_180(-730.0), -10.0);
        assert_eq!(convert_branch_180(721.0), 1.0);
        assert_eq!(convert_branch_180(-721.0), -1.0);
    }

    #[test]
    fn test_convert_branch_360() {
        assert_eq!(convert_branch_360(0.0), 0.0);
        assert_eq!(convert_branch_360(180.0), 180.0);
        assert_eq!(convert_branch_360(-180.0), 180.0);
        assert_eq!(convert_branch_360(270.0), 270.0);
        assert_eq!(convert_branch_360(-270.0), 90.0);
        assert_eq!(convert_branch_360(360.0), 0.0);
        assert_eq!(convert_branch_360(-360.0), 0.0);
        assert_eq!(convert_branch_360(370.0), 10.0);
        assert_eq!(convert_branch_360(-370.0), 350.0);
    }

    #[test]
    fn test_convert_theta_vec() {
        let theta = Array1::from(vec![0.0, 45.0, 180.0, 190.0, -180.0, 360.0, -360.0, -370.0]);
        let theta_new = convert_branch_180_vec(theta.view());

        let theta_expected = Array1::from(vec![0.0, 45.0, 180.0, -170.0, -180.0, 0.0, 0.0, -10.0]);

        assert_eq!(theta_new, theta_expected);
    }
}
