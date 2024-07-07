use ndarray::{Array, Array1, Array2, Dimension, Ix1, Ix2};
use num::complex::Complex;
use num::traits::Float;

/// Trait for converting angles between radians and degrees.
///
/// This trait is implemented for both 1D and 2D arrays.
/// - `to_radians` converts angles from degrees to radians.
/// - `to_degrees` converts angles from radians to degrees.
pub trait Angles {
    fn to_radians(&self) -> Self;
    fn to_degrees(&self) -> Self;
}

/// Implement the `Angles` trait for 1D arrays of floating-point numbers.
impl<T: Float + ndarray::ScalarOperand> Angles for Array1<T> {
    fn to_radians(&self) -> Self {
        self.mapv(|angle| angle.to_radians())
    }

    fn to_degrees(&self) -> Self {
        self.mapv(|angle| angle.to_degrees())
    }
}

/// Implement the `Angles` trait for 2D arrays of floating-point numbers.
impl<T: Float + ndarray::ScalarOperand> Angles for Array2<T> {
    fn to_radians(&self) -> Self {
        self.mapv(|angle| angle.to_radians())
    }

    fn to_degrees(&self) -> Self {
        self.mapv(|angle| angle.to_degrees())
    }
}

/// Trait for converting between real and complex numbers.
///
/// This trait is implemented for both 1D and 2D arrays.
pub trait ComplexExt<T, D: Dimension> {
    fn abs(&self) -> Array<T, D>;
    fn imag(&self) -> Array<T, D>;
    fn real(&self) -> Array<T, D>;
    fn phase(&self) -> Array<T, D>;
    fn from_real_imag(real: Array<T, D>, imag: Array<T, D>) -> Array<Complex<T>, D>;
}

/// Implement the `ComplexExt` trait for 1D arrays.
impl<T: Float> ComplexExt<T, Ix1> for Array1<Complex<T>> {
    fn abs(&self) -> Array<T, Ix1> {
        self.map(|c| c.norm())
    }

    fn imag(&self) -> Array<T, Ix1> {
        self.map(|c| c.im)
    }

    fn real(&self) -> Array<T, Ix1> {
        self.map(|c| c.re)
    }

    fn phase(&self) -> Array<T, Ix1> {
        self.map(|c| c.arg())
    }

    fn from_real_imag(real: Array<T, Ix1>, imag: Array<T, Ix1>) -> Array<Complex<T>, Ix1> {
        let vec: Vec<_> = real
            .iter()
            .zip(imag.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        Array::from_shape_vec(real.raw_dim(), vec).unwrap()
    }
}

/// Implement the `ComplexExt` trait for 2D arrays.
impl<T: Float> ComplexExt<T, Ix2> for Array2<Complex<T>> {
    fn abs(&self) -> Array2<T> {
        self.map(|c| c.norm())
    }

    fn imag(&self) -> Array2<T> {
        self.map(|c| c.im)
    }

    fn real(&self) -> Array2<T> {
        self.map(|c| c.re)
    }

    fn phase(&self) -> Array2<T> {
        self.map(|c| c.arg())
    }

    fn from_real_imag(real: Array<T, Ix2>, imag: Array<T, Ix2>) -> Array2<Complex<T>> {
        let vec: Vec<_> = real
            .iter()
            .zip(imag.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        Array::from_shape_vec(real.raw_dim(), vec).unwrap()
    }
}

/// `FromVecVec` is a trait for converting a vector of vectors (2D vector) into a 2D array.
///
/// This trait is generic over a type `T` that implements `Clone`.
/// The `from_vec_vec` method takes a vector of vectors (`Vec<Vec<T>>`) and
/// returns a 2D array (`Array2<T>`).
///
/// # Panics
///
/// The `from_vec_vec` method will panic if the provided vector of vectors
/// cannot be shaped into a 2D array.
///
pub trait FromVecVec<T: Clone> {
    fn from_vec_vec(vec_vec: Vec<Vec<T>>) -> Array2<T> {
        let rows = vec_vec.len();
        let cols = vec_vec[0].len();

        Array2::from_shape_vec((rows, cols), vec_vec.into_iter().flatten().collect())
            .expect("Failed to create Array2, incorrect shape")
    }
}

/// Implementation of `FromVecVec` for `Array2<f32>`.
impl FromVecVec<f32> for Array2<f32> {}

/// Implementation of `FromVecVec` for `Array2<f64>`.
impl FromVecVec<f64> for Array2<f64> {}

/// Implementation of `FromVecVec` for `Array2<Complex<f32>>`.
impl FromVecVec<Complex<f32>> for Array2<Complex<f32>> {}

/// Implementation of `FromVecVec` for `Array2<Complex<f64>>`.
impl FromVecVec<Complex<f64>> for Array2<Complex<f64>> {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use num_complex::Complex64;

    #[test]
    fn test_from_real_imag_1x3() {
        let real = arr1(&[1.0, 2.0, 3.0]);
        let imag = arr1(&[4.0, 5.0, 6.0]);
        let expected = arr1(&[
            Complex64::new(1.0, 4.0),
            Complex64::new(2.0, 5.0),
            Complex64::new(3.0, 6.0),
        ]);
        assert_eq!(Array1::from_real_imag(real, imag), expected);
    }

    #[test]
    fn test_from_real_imag_2x2() {
        let real = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let imag = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let expected = arr2(&[
            [Complex64::new(1.0, 5.0), Complex64::new(2.0, 6.0)],
            [Complex64::new(3.0, 7.0), Complex64::new(4.0, 8.0)],
        ]);
        assert_eq!(Array2::from_real_imag(real, imag), expected);
    }

    #[test]
    fn test_from_real_imag_2x3() {
        let real = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let imag = arr2(&[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]);
        let expected = arr2(&[
            [
                Complex64::new(1.0, 7.0),
                Complex64::new(2.0, 8.0),
                Complex64::new(3.0, 9.0),
            ],
            [
                Complex64::new(4.0, 10.0),
                Complex64::new(5.0, 11.0),
                Complex64::new(6.0, 12.0),
            ],
        ]);
        assert_eq!(Array2::from_real_imag(real, imag), expected);
    }
}
