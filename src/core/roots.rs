use crate::core::finitediff::gradient;
use crate::core::interpolation::linear_interp as interp;
use ndarray::ArrayView1;

/// Find a root using the Newton-Raphson algorithm.
#[allow(dead_code)]
pub fn find_root_fn(f: fn(f64) -> f64, df: fn(f64) -> f64, guess: f64, iterations: i32) -> f64 {
    let mut result = guess;

    let iteration =
        |f: fn(f64) -> f64, df: fn(f64) -> f64, guess: f64| -> f64 { guess - f(guess) / df(guess) };

    for _ in 0..iterations {
        result = iteration(f, df, result);
    }
    result
}

pub fn zero_crossings(y: &Vec<f64>, x: &Vec<f64>) -> Vec<f64> {
    let mut roots = Vec::new();
    for i in 0..y.len() - 1 {
        if y[i] * y[i + 1] <= 0.0 {
            roots.push(x[i] + y[i] * (x[i] - x[i + 1]) / (y[i] - y[i + 1]));
        }
    }
    roots
}

pub fn find_root<F, DF>(f: F, df: DF, guess: f64, error_tolerance: f64) -> f64
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    let mut result = guess;
    let mut prev_result;

    loop {
        prev_result = result;
        result = result - f(result) / df(result);

        if (result - prev_result).abs() <= error_tolerance {
            break;
        }
    }

    result
}

pub fn roots_newton_raphson(y: &Vec<f64>, x: &Vec<f64>) -> Vec<f64> {
    let x0 = zero_crossings(y, x);
    let dy = gradient(y, x);

    let f = |x_i| interp(x_i, ArrayView1::from(x), ArrayView1::from(y));
    let df = |x_i| interp(x_i, ArrayView1::from(x), ArrayView1::from(&dy));

    let mut roots = Vec::new();
    for i in 0..x0.len() {
        roots.push(find_root(f, df, x0[i], 10e-6));
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_newton_raphson() {
        fn f(x: f64) -> f64 {
            x * x
        }
        fn df(x: f64) -> f64 {
            2.0 * x
        }
        let root = find_root_fn(f, df, 10.0, 100);
        assert_approx_eq!(root, 0.0, 1e-10);
        println!("ROOT = {}", root);
    }

    #[test]
    fn test_roots_newton_raphson() {
        let x = (0..=100)
            .map(|i| i as f64 / 100.0 * 2.0 * std::f64::consts::PI)
            .collect::<Vec<_>>();
        let y = x.iter().map(|&x_i| (2.0 * x_i).cos()).collect::<Vec<_>>();

        let roots = roots_newton_raphson(&y, &x);

        for root in roots {
            assert!((2.0 * root).cos().abs() < 1e-6);
        }
    }
}
