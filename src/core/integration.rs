use ndarray::{ArrayView1, ArrayView2};
use rayon::prelude::*;

// sequential trapezoidal integration
pub fn trapz(y: ArrayView1<f64>, x: ArrayView1<f64>) -> f64 {
    (0..(x.len() - 1))
        .map(|i| (x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2.0)
        .sum()
}

// parallel trapezoidal integration
#[allow(dead_code)]
pub fn trapzp(y: ArrayView1<f64>, x: ArrayView1<f64>) -> f64 {
    (0..x.len() - 1)
        .into_par_iter()
        .map(|i| (x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2.0)
        .sum()
}

// parallel 2D trapezoidal integration
pub fn trapz2d(f: ArrayView2<f64>, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    let nx = x.len();
    let ny = y.len();

    (0..ny - 1)
        .into_par_iter()
        .map(|j| {
            let dy = (y[j + 1] - y[j]) / 2.0;
            (0..nx - 1)
                //                .into_par_iter()
                .map(|i| {
                    let dx = (x[i + 1] - x[i]) / 2.0;
                    (f[[j, i]] + f[[j + 1, i]] + f[[j, i + 1]] + f[[j + 1, i + 1]]) * dx * dy
                })
                .sum::<f64>()
        })
        .sum()
}

#[cfg(test)]
mod tests_integration {
    use super::*;
    use ndarray::arr1;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_trapz() {
        // Test case 1
        let y1 = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let x1 = arr1(&[0.0, 1.0, 2.0, 3.0]);
        let result1 = trapz(y1.view(), x1.view());
        assert_eq!(result1, 7.5);

        // Test case 2
        let y2 = arr1(&[0.0, 0.5, 1.0]);
        let x2 = arr1(&[0.0, 1.0, 2.0]);
        let result2 = trapz(y2.view(), x2.view());
        assert_eq!(result2, 1.0);

        // Add more test cases as needed
    }

    #[test]
    fn test_trapz2d() {
        let x = Array1::from(vec![0., 1., 2.]);
        let y = Array1::from(vec![0., 1., 2.]);
        let z = Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 0.0, 4.0, 16.0])
            .unwrap();
        assert_eq!(trapz2d(z.view(), x.view(), y.view()), 9.0);
    }
}
