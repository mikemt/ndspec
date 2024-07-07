use ndarray::{Array, Array1, ArrayView1, ArrayView2};

/// Linear interpolation
pub fn linear_interp(x_val: f64, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    if x_val <= x[0] {
        return y[0];
    }

    if x_val >= x[x.len() - 1] {
        return y[x.len() - 1];
    }

    for i in 0..x.len() - 1 {
        if x[i] <= x_val && x_val <= x[i + 1] {
            return y[i] + (y[i + 1] - y[i]) * (x_val - x[i]) / (x[i + 1] - x[i]);
        }
    }
    0.0
}

#[allow(dead_code)]
pub fn linear_interp_vec(
    x_vals: ArrayView1<f64>,
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
) -> Array1<f64> {
    let mut result = Array::zeros(x_vals.len());
    for (j, x_val) in x_vals.iter().enumerate() {
        if x_val <= &x[0] {
            result[j] = y[0];
        } else if x_val >= &x[x.len() - 1] {
            result[j] = y[x.len() - 1];
        } else {
            for i in 0..x.len() - 1 {
                if x[i] <= *x_val && x_val <= &x[i + 1] {
                    result[j] = y[i] + (y[i + 1] - y[i]) * (x_val - x[i]) / (x[i + 1] - x[i]);
                    break;
                }
            }
        }
    }
    result
}

/// Cubic interpolation
#[allow(dead_code)]
pub fn cubic_interp(x_val: f64, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    if x_val <= x[0] {
        return y[0];
    }

    if x_val >= x[x.len() - 1] {
        return y[x.len() - 1];
    }

    for i in 0..x.len() - 1 {
        if x[i] <= x_val && x_val <= x[i + 1] {
            let x0 = x[i];
            let x1 = x[i + 1];
            let y0 = y[i];
            let y1 = y[i + 1];

            if i + 2 < x.len() {
                let dx = x1 - x0;
                let a = (y1 - y0) / dx - (y[i + 2] - y0) / (x[i + 2] - x0) + (y[i + 2] - y1) / dx;
                let b = (y[i + 2] - y0) / (x[i + 2] - x0) - 2.0 * (y1 - y0) / dx;
                let c = (y1 - y0) / dx;
                return y0 + (x_val - x0) * (c + (x_val - x0) * (b + (x_val - x0) * a));
            } else {
                let dx = x1 - x0;
                let a = (y1 - y0) / dx;
                let b = 0.0;
                let c = 0.0;
                return y0 + (x_val - x0) * (c + (x_val - x0) * (b + (x_val - x0) * a));
            }
        }
    }
    0.0
}

pub fn bilinear_interp(
    x_val: f64,
    y_val: f64,
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    z: ArrayView2<f64>,
) -> f64 {
    // Check if x_val and y_val are outside the range of x and y
    if x_val < x[0] || x_val > x[x.len() - 1] || y_val < y[0] || y_val > y[y.len() - 1] {
        return 0.0;
    }

    // Find the indices i and j such that x[i] <= x_val <= x[i + 1] and y[j] <= y_val <= y[j + 1]
    let i = if x_val == x[x.len() - 1] {
        x.len() - 2
    } else {
        x.iter()
            .enumerate()
            .filter(|&(_, &x_el)| x_el <= x_val)
            .map(|(i, _)| i)
            .last()
            .unwrap()
    };

    let j = if y_val == y[y.len() - 1] {
        y.len() - 2
    } else {
        y.iter()
            .enumerate()
            .filter(|&(_, &y_el)| y_el <= y_val)
            .map(|(j, _)| j)
            .last()
            .unwrap()
    };

    // Interpolate the values of the function at the four surrounding points
    let dx = x[i + 1] - x[i];
    let dy = y[j + 1] - y[j];

    let w11 = ((x[i + 1] - x_val) / dx) * ((y[j + 1] - y_val) / dy);
    let w12 = ((x[i + 1] - x_val) / dx) * ((y_val - y[j]) / dy);
    let w21 = ((x_val - x[i]) / dx) * ((y[j + 1] - y_val) / dy);
    let w22 = ((x_val - x[i]) / dx) * ((y_val - y[j]) / dy);

    let q11 = z[(j, i)];
    let q21 = z[(j, i + 1)];
    let q12 = z[(j + 1, i)];
    let q22 = z[(j + 1, i + 1)];

    w11 * q11 + w21 * q21 + w12 * q12 + w22 * q22
}

#[cfg(test)]
mod test_interpolation {
    use super::*;

    use assert_approx_eq::assert_approx_eq;
    use ndarray::{array, Array1, Array2};
    use ndarray_stats::QuantileExt;

    #[test]
    fn test_linear_interp() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0];

        assert_eq!(linear_interp(0.0, x.view(), y.view()), 0.0);
        assert_eq!(linear_interp(0.5, x.view(), y.view()), 0.5);
        assert_eq!(linear_interp(1.0, x.view(), y.view()), 1.0);
        assert_eq!(linear_interp(1.5, x.view(), y.view()), 1.5);
        assert_eq!(linear_interp(2.0, x.view(), y.view()), 2.0);
        assert_eq!(linear_interp(2.5, x.view(), y.view()), 2.5);
        assert_eq!(linear_interp(3.0, x.view(), y.view()), 3.0);
        assert_eq!(linear_interp(3.5, x.view(), y.view()), 3.5);
        assert_eq!(linear_interp(4.0, x.view(), y.view()), 4.0);
    }

    #[test]
    fn test_bilinear_interp() {
        let x = Array1::linspace(-5.01, 5.01, 41);
        let y = Array1::linspace(-5.01, 5.01, 41);

        let z = Array2::from_shape_fn((x.len(), y.len()), |(i, j)| {
            let x_val: f64 = x[i];
            let y_val: f64 = y[j];
            x_val.powi(2) + y_val.powi(2)
        });

        assert_approx_eq!(*z.max().unwrap(), 50.2002, 1e-6);

        // Python code to generate the test data
        //  from scipy import interpolate
        //  import numpy as np
        //  x = np.linspace(-5.01, 5.01, 41)
        //  y = np.linspace(-5.01, 5.01, 41)
        //  xx, yy = np.meshgrid(x, y)
        //  z = xx**2 + yy**2
        //  f = interpolate.RegularGridInterpolator((x, y), z)
        // f([0.0, 0.0])
        // f([2.5, 2.5])
        // f([2.5, 4.0])

        assert_eq!(bilinear_interp(0.0, 0.0, x.view(), y.view(), z.view()), 0.0);

        assert_eq!(
            bilinear_interp(10.0, 10.0, x.view(), y.view(), z.view()),
            0.0
        );
        assert_eq!(
            bilinear_interp(10.0, -10.0, x.view(), y.view(), z.view()),
            0.0
        );
        assert_eq!(
            bilinear_interp(-10.0, 10.0, x.view(), y.view(), z.view()),
            0.0
        );
        assert_eq!(
            bilinear_interp(-10.0, -10.0, x.view(), y.view(), z.view()),
            0.0
        );

        assert_approx_eq!(
            bilinear_interp(2.5, 2.5, x.view(), y.view(), z.view()),
            12.502455,
            1e-5
        );

        assert_approx_eq!(
            bilinear_interp(2.5, 4.0, x.view(), y.view(), z.view()),
            22.2531675,
            1e-5
        );

        assert_approx_eq!(
            bilinear_interp(-5.01, 1.0, x.view(), y.view(), z.view()),
            26.100597,
            1e-5
        );

        assert_approx_eq!(
            bilinear_interp(-5.0101, 1.0, x.view(), y.view(), z.view()),
            0.0,
            1e-5
        );

        assert_approx_eq!(
            bilinear_interp(-5.01, -5.01, x.view(), y.view(), z.view()),
            50.2002,
            1e-5
        );

        assert_approx_eq!(
            bilinear_interp(5.01, 5.01, x.view(), y.view(), z.view()),
            50.2002,
            1e-5
        );

        assert_approx_eq!(
            bilinear_interp(5.0101, 5.01, x.view(), y.view(), z.view()),
            0.0,
            1e-5
        );

        assert_approx_eq!(
            bilinear_interp(5.0101, -5.01, x.view(), y.view(), z.view()),
            0.0,
            1e-5
        );

        assert_approx_eq!(
            bilinear_interp(5.01, -5.01, x.view(), y.view(), z.view()),
            50.2002,
            1e-5
        );

        assert_approx_eq!(
            bilinear_interp(4.8, 4.9, x.view(), y.view(), z.view()),
            47.07396,
            1e-5
        );
    }
}
