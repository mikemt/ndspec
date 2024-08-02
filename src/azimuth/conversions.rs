use ndarray::Array2;

use crate::azimuth::branch::convert_branch_180;

pub fn rotation_matrix_2d(theta: f64) -> Array2<f64> {
    let theta_rad = theta.to_radians();
    let cos_theta = theta_rad.cos();
    let sin_theta = theta_rad.sin();

    Array2::from(vec![[cos_theta, -sin_theta], [sin_theta, cos_theta]])
}

pub fn theta_to_global(theta: f64) -> f64 {
    ((90.0 - theta + 180.0) % 360.0 + 360.0) % 360.0
}

pub fn theta_to_encounter_angle(theta: f64) -> f64 {
    convert_branch_180(theta + 180.0)
}

#[cfg(test)]
mod test_azimuth_conversions {
    use super::*;

    #[test]
    fn test_theta_to_global() {
        assert_eq!(theta_to_global(0.0), 90.0);
        assert_eq!(theta_to_global(90.0), 0.0);
        assert_eq!(theta_to_global(180.0), 270.0);
        assert_eq!(theta_to_global(270.0), 180.0);
        assert_eq!(theta_to_global(360.0), 90.0);
        assert_eq!(theta_to_global(-90.0), 180.0);
        assert_eq!(theta_to_global(-180.0), 270.0);
        assert_eq!(theta_to_global(-270.0), 0.0);
        assert_eq!(theta_to_global(-360.0), 90.0);
    }
}
