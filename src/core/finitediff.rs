#![allow(dead_code)]

use ndarray::Array1;
fn second_order_central_finite_difference(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut res = vec![0.0; n];

    for i in 2..n - 2 {
        res[i] = (v[i - 2] - 4.0 * v[i - 1] + 6.0 * v[i] - 4.0 * v[i + 1] + v[i + 2]) / 12.0;
    }

    res
}

pub fn gradient(y: &[f64], x: &[f64]) -> Array1<f64> {
    let n = x.len();
    let mut df = vec![0.0; n];

    // Forward point
    df[0] = (y[1] - y[0]) / (x[1] - x[0]);

    // Central points
    for i in 1..n - 1 {
        df[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
    }

    // Backward point
    df[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);

    Array1::from(df)
}
