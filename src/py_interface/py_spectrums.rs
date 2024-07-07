use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::{pyclass, pymethods, PyResult, Python};

use crate::prelude as spec;

#[pyclass]
#[derive(Clone)]
pub struct FrequencyResponse(pub spec::FrequencyResponse);

#[pymethods]
impl FrequencyResponse {
    pub fn to_dict(&self, py: Python) -> PyObject {
        [
            ("std_dev", self.0.std_dev),
            ("Tm02", self.0.Tm02),
            ("Xs", self.0.As),
            ("Xmpm", self.0.Ampm),
        ]
        .into_py_dict(py)
        .to_object(py)
    }

    #[getter]
    fn std_dev(&self) -> PyResult<f64> {
        Ok(self.0.std_dev)
    }

    #[getter]
    #[pyo3(name = "Tm02")]
    fn tm02(&self) -> PyResult<f64> {
        Ok(self.0.Tm02)
    }

    #[getter]
    #[pyo3(name = "As")]
    fn a_s(&self) -> PyResult<f64> {
        Ok(self.0.As)
    }

    #[getter]
    #[pyo3(name = "Ampm")]
    fn a_mpm(&self) -> PyResult<f64> {
        Ok(self.0.Ampm)
    }

    fn __repr__(&self) -> String {
        format!(
            "FrequencyResponse(
                std_dev: {},
                Tm02: {},
                As: {},
                Ampm: {}
            )",
            self.0.std_dev, self.0.Tm02, self.0.As, self.0.Ampm
        )
    }

    fn __dict__(&self, py: Python) -> PyObject {
        [
            ("std_dev", self.0.std_dev),
            ("Tm02", self.0.Tm02),
            ("As", self.0.As),
            ("Ampm", self.0.Ampm),
        ]
        .into_py_dict(py)
        .to_object(py)
    }
}

#[pyfunction]
pub fn maxhs(tp: f64) -> f64 {
    spec::maxhs(tp)
}

/// calculates the Bretschneider (PM) energy density spectrum
///
/// Parameters
/// ----------
/// omega : array_like
///     Angular frequency [rad/s]
/// hs : float
///     Significant wave height [m]
/// tp : float
///     Peak spectral wave period [s]
///
/// Returns
/// -------
/// energy : ndarray
///     Energy density spectrum [m^2 s / rad]
///
#[pyfunction]
pub fn bretschneider<'py>(
    py: Python<'py>,
    omega: PyReadonlyArray1<f64>,
    hs: f64,
    tp: f64,
) -> &'py PyArray1<f64> {
    spec::bretschneider::energy(omega.as_array(), hs, tp).into_pyarray(py)
}

/// calculates the Jonswap energy density spectrum
#[pyfunction]
pub fn jonswap<'py>(
    py: Python<'py>,
    omega: PyReadonlyArray1<f64>,
    hs: f64,
    tp: f64,
    gamma: f64,
    sigma_a: f64,
    sigma_b: f64,
) -> &'py PyArray1<f64> {
    spec::jonswap::energy(omega.as_array(), hs, tp, gamma, sigma_a, sigma_b).into_pyarray(py)
}

/// calculates the Gaussian energy density spectrum
#[pyfunction]
pub fn gaussian<'a>(
    py: Python<'a>,
    omega: PyReadonlyArray1<f64>,
    hs: f64,
    tp: f64,
    sigma_g: f64,
) -> &'a PyArray1<f64> {
    spec::gaussian_spectrum(omega.as_array(), hs, tp, sigma_g).into_pyarray(py)
}

#[pyfunction]
pub fn lewis_allos(hs: f64, tp: f64) -> PyResult<(f64, f64, f64, f64)> {
    Ok(spec::lewis_allos(hs, tp))
}

#[pyfunction]
pub fn spread_cos_n<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<f64>,
    theta_p: f64,
    n: f64,
) -> &'py PyArray1<f64> {
    spec::spread_cos_n(theta.as_array(), theta_p, n).into_pyarray(py)
}

#[pyfunction]
pub fn spread_cos_2s<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<f64>,
    theta_p: f64,
    s: f64,
) -> &'py PyArray1<f64> {
    spec::spread_cos_2s(theta.as_array(), theta_p, s).into_pyarray(py)
}

/// This function converts Tp to Tz for a given value of gamma
///
/// Tp: peak wave period
/// Tz: mean zero crossing wave period
///    
/// For \gamma=3.3:
///    T_p = 1.2859 T_z
/// For \gamma = 1.0 (PM spectrum):
///    T_p = 1.4049 T_z
/// References
/// ----------
/// DNV RP C205 \S 3.5.5.4#
#[pyfunction]
#[pyo3(name = "convert_Tp_to_Tz")]
#[pyo3(signature=(tp, gamma), text_signature = "(Tp: float, gamma: float) -> float")]
pub fn convert_tp_to_tz(tp: f64, gamma: f64) -> f64 {
    spec::convert_tp_to_tz(tp, gamma)
}

/// This function converts Tz to Tp for a given value of gamma
///
/// Tp: peak wave period
/// Tz: mean zero crossing wave period
///    
/// For \gamma=3.3:
///    T_p = 1.2859 T_z
/// For \gamma = 1.0 (PM spectrum):
///    T_p = 1.4049 T_z
/// References
/// ----------
/// DNV RP C205 \S 3.5.5.4#
#[pyfunction]
#[pyo3(name = "convert_Tz_to_Tp")]
#[pyo3(signature=(tz, gamma), text_signature = "(Tz: float, gamma: float) -> float")]
pub fn convert_tz_to_tp(tz: f64, gamma: f64) -> f64 {
    spec::convert_tz_to_tp(tz, gamma)
}

/// This function converts Tp to Tm (T1) for a given value of gamma
#[pyfunction]
#[pyo3(name = "convert_Tp_to_Tm")]
#[pyo3(signature=(tp, gamma), text_signature = "(Tp: float, gamma: float) -> float")]
pub fn convert_tp_to_t1(tp: f64, gamma: f64) -> f64 {
    spec::convert_tp_to_t1(tp, gamma)
}

/// This function converts Tm (T1) to Tp for a given value of gamma
#[pyfunction]
#[pyo3(name = "convert_Tm_to_Tp")]
#[pyo3(signature=(t1, gamma), text_signature = "(Tm: float, gamma: float) -> float")]
pub fn convert_t1_to_tp(t1: f64, gamma: f64) -> f64 {
    spec::convert_t1_to_tp(t1, gamma)
}

#[pyfunction]
pub fn froya_10min<'a>(
    py: Python<'a>,
    omega: PyReadonlyArray1<f64>,
    wind_speed: f64,
) -> &'a PyArray1<f64> {
    spec::froya_10min(omega.as_array(), wind_speed).into_pyarray(py)
}

#[pyfunction]
pub fn froya_1hr<'a>(
    py: Python<'a>,
    omega: PyReadonlyArray1<f64>,
    wind_speed: f64,
    elevation: f64,
) -> &'a PyArray1<f64> {
    spec::froya_1hr(omega.as_array(), wind_speed, elevation).into_pyarray(py)
}

/// Jonswap spectrum class
///
/// Parameters
/// ----------
/// hs : float
///     Significant wave height [m]
/// tp : float
///     Peak spectral wave period [s]
///
/// Returns
/// -------
/// Jonswap
///     Jonswap object
///
/// Note
/// ----
/// Spectral parameters are calculated by default using the Lewis-Allos spectral
/// parameterization.
///
/// Methods
/// -------
/// set_hs(hs: float)
///     Set the significant wave height [m]
/// set_tp(tp: float)
///     Set the peak spectral wave period [s]
/// set_gamma(gamma: float)
///     Set the peak shape factor [-]
/// set_sigma_a(sigma_a: float)
///     Set the lower bound bandwidth parameter [-]
/// set_sigma_b(sigma_b: float)
///     Set the upper bound bandwidth parameter [-]
/// set_omega(omega: array_like)
///     Set the angular frequency [rad/s]
/// tz() -> float
///     Calculate the zero-crossing period [s]
/// t1() -> float
///     Calculate the mean spectral period [s]
/// set_tp_from_tz(tz: float)
///     Set the peak spectral wave period from the zero-crossing period [s]
/// set_tp_from_t1(t1: float)
///     Set the peak spectral wave period from the mean spectral period [s]
/// energy(omega: array_like) -> ndarray
///     Calculate the energy density spectrum [m^2 s / rad]
/// abs_error() -> float
///     Calculate the absolute error of the spectrum
/// to_spec1d() -> Spectrum1D
///     Convert the Jonswap spectrum to a Spectrum1D object
/// to_spec2d(spreading: Spreading) -> Spectrum2D
///     Convert the Jonswap spectrum to a Spectrum2D object
///
#[pyclass]
pub struct Jonswap(spec::Jonswap);

#[pymethods]
impl Jonswap {
    #[new]
    fn new(hs: f64, tp: f64) -> Self {
        Jonswap(spec::Jonswap::new(hs, tp))
    }

    /// Create a Jonswap spectrum from 3 parameters
    ///
    /// Parameters
    /// ----------
    /// hs : float
    ///     Significant wave height [m]
    /// tp : float
    ///     Peak spectral wave period [s]
    /// gamma : float
    ///     Peak shape factor
    ///
    /// Returns
    /// -------
    /// Jonswap
    ///     Jonswap object
    ///
    #[staticmethod]
    fn from_3p(hs: f64, tp: f64, gamma: f64) -> Self {
        Jonswap(spec::Jonswap::from_3p(hs, tp, gamma))
    }

    /// Create a Jonswap spectrum from 5 parameters
    ///
    /// Parameters
    /// ----------
    /// hs : float
    ///     Significant wave height [m]
    /// tp : float
    ///     Peak spectral wave period [s]
    /// gamma : float
    ///     Peak shape factor [-]
    /// sigma_a : float
    ///     lower bound bandwidth parameter [-]
    /// sigma_b : float
    ///     upper bound bandwidth parameter [-]
    ///
    /// Returns
    /// -------
    /// Jonswap
    ///     Jonswap object
    ///
    #[staticmethod]
    fn from_5p(hs: f64, tp: f64, gamma: f64, sigma_a: f64, sigma_b: f64) -> Self {
        Jonswap(spec::Jonswap::from_5p(hs, tp, gamma, sigma_a, sigma_b))
    }

    fn __repr__(&self) -> PyResult<String> {
        let n = self.0.omega.len();
        Ok(format!(
            "Jonswap(hs: {:?}, tp: {:?}, gamma: {:?}, sigma_a: {:?}, sigma_b: {:?}), omega: [{:?},...,{:?}](len={:?})",
            self.0.hs, self.0.tp, self.0.gamma, self.0.sigma_a, self.0.sigma_b,
            self.0.omega[0], self.0.omega[n-1], n))
    }

    #[getter]
    fn hs(&self) -> f64 {
        self.0.hs
    }

    #[getter]
    fn tp(&self) -> f64 {
        self.0.tp
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.0.gamma
    }

    #[getter]
    fn sigma_a(&self) -> f64 {
        self.0.sigma_a
    }

    #[getter]
    fn sigma_b(&self) -> f64 {
        self.0.sigma_b
    }

    #[getter]
    fn omega<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.0.omega.clone().into_pyarray(py)
    }

    fn set_hs(&mut self, hs: f64) {
        self.0.hs = hs;
    }

    fn set_tp(&mut self, tp: f64) {
        self.0.tp = tp;
    }

    fn set_gamma(&mut self, gamma: f64) {
        self.0.gamma = gamma;
    }

    fn set_sigma_a(&mut self, sigma_a: f64) {
        self.0.sigma_a = sigma_a;
    }

    fn set_sigma_b(&mut self, sigma_b: f64) {
        self.0.sigma_b = sigma_b;
    }

    fn set_omega(&mut self, omega: Vec<f64>) {
        self.0.omega = omega.into();
    }

    /// calculate the zero-crossing period
    ///
    /// Returns
    /// -------
    /// Tz : float
    ///     Zero-crossing period [s]
    ///
    fn tz(&self) -> PyResult<f64> {
        Ok(self.0.tz())
    }

    /// calculate the mean period
    ///
    /// Returns
    /// -------
    /// T1 : float
    ///     Mean spectral period [s]
    ///
    fn t1(&self) -> PyResult<f64> {
        Ok(self.0.t1())
    }

    fn set_tp_from_tz(&mut self, tz: f64) {
        self.0.set_tp_from_tz(tz);
    }

    fn set_tp_from_t1(&mut self, t1: f64) {
        self.0.set_tp_from_t1(t1);
    }

    /// Calculates the energy density spectrum
    ///
    /// Parameters
    /// ----------
    /// omega : array_like
    ///     Angular frequency [rad/s]
    ///
    /// Returns
    /// -------
    /// energy : ndarray
    ///     Energy density spectrum [m^2 s / rad]
    ///
    fn energy<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.0.energy().into_pyarray(py)
    }

    fn abs_error(&self) -> PyResult<f64> {
        Ok(self.0.abs_error())
    }

    /// Converts the Jonswap spectrum to a Spectrum1D object
    ///
    /// Parameters
    /// ----------
    /// direction : float, optional, default=0.0
    ///     Direction of the waves [deg]. Measured clockwise from the North
    ///     (North = 0, East = 90, South = 180, West = 270)
    ///
    /// Returns
    /// -------
    /// spec : Spectrum1D
    ///     Spectrum1D object
    ///
    fn to_spec1d(&self) -> Spectrum1D {
        let spec = self.0.to_spec1d();
        Spectrum1D::new(spec.omega.to_vec(), spec.energy.to_vec())
    }

    /// Converts the Jonswap spectrum to a Spectrum2D object
    ///
    /// Parameters
    /// ----------
    /// spreading : Spreading
    ///     Spreading object
    ///
    /// Returns
    /// -------
    /// spec : Spectrum2D
    ///     Spectrum2D object
    ///
    /// Notes
    /// -----
    /// - see also the `Spreading` class
    ///
    fn to_spec2d(&self, py: Python, spreading: &Spreading) -> Spectrum2D {
        let spec = self.0.to_spec2d(&spreading.0);
        Spectrum2D::new(
            spec.omega.to_vec(),
            spec.theta.to_vec(),
            spec.energy.into_pyarray(py).readonly(),
        )
    }
}

/// # Spreading class
///
/// ## Parameters
/// - `theta_p` : float
///     - Peak spreading angle [deg]
/// - `n` : float
///     - Spreading parameter
///
/// ## Returns
/// - `Spreading`
///     - Spreading object
///
/// ## Methods
/// - `set_theta` : ndarray
///     - Set the spreading angles [deg]
/// - `set_theta_p` : float
///     - Set the peak spectral wave direction [deg]
/// - `set_n` : float
///     - Set the spreading parameter [-]
/// - `distribution` : ndarray
///     - Spreading function [-]
/// - `d1_form` : ndarray
///     - Spreading function in D_1 form
/// - `d2_form` : ndarray
///     - Spreading function in D_2 form
/// - `abs_error` : float
///     - Absolute error of the spreading function
///
#[pyclass]
pub struct Spreading(pub spec::Spreading);

#[pymethods]
impl Spreading {
    #[new]
    pub fn new(theta_p: f64, n: f64) -> Self {
        Self(spec::Spreading::new(theta_p, n))
    }

    fn __repr__(&self) -> PyResult<String> {
        let n = self.0.theta.len();
        Ok(format!(
            "Spreading(theta_p: {:?}, n: {:?}, s: {:?}), theta: [{:?},...,{:?}](len={:?})",
            self.0.theta_p,
            self.0.n,
            self.0.s,
            self.0.theta[0],
            self.0.theta[n - 1],
            n
        ))
    }

    #[getter]
    fn theta_p(&self) -> f64 {
        self.0.theta_p
    }

    #[getter]
    fn n(&self) -> f64 {
        self.0.n
    }

    #[getter]
    fn s(&self) -> f64 {
        self.0.s
    }

    #[getter]
    fn theta<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.0.theta.clone().into_pyarray(py)
    }

    fn set_theta(&mut self, theta: Vec<f64>) {
        self.0.theta = theta.into();
    }

    fn set_n(&mut self, n: f64) {
        self.0.set_n(n);
    }

    fn set_theta_p(&mut self, theta_p: f64) {
        self.0.set_theta_p(theta_p);
    }

    fn set_s(&mut self, s: f64) {
        self.0.set_s(s);
    }

    /// Calculates the spreading function
    ///
    /// Returns
    /// -------
    /// spreading : ndarray
    ///     Spreading function
    ///
    /// Notes
    /// -----
    /// - The spreading function is calculated in D_1 form by default and is dependent on `n`.
    /// - use the `d1_form` method to calculate the spreading function in D_1 form (`n` dependent).
    /// - use the `d2_form` method to calculate the spreading function in D_2 form (`s` dependent).
    ///
    fn distribution<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.0.distribution().into_pyarray(py)
    }

    /// Calculates the directional spreading function in D_1 form.
    ///
    /// Returns
    /// -------
    /// spreading : ndarray
    ///     Spreading function
    ///
    fn d1_form<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.0.d1_form().into_pyarray(py)
    }
    /// Calculates the directional spreading function in D_2 form.
    ///
    /// Returns
    /// -------
    /// spreading : ndarray
    ///     Spreading function
    ///
    fn d2_form<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.0.d2_form().into_pyarray(py)
    }

    /// Calculates the absolute error of the spreading function
    ///
    /// Returns
    /// -------
    /// error : float
    ///     Absolute error
    ///
    fn abs_error(&self) -> f64 {
        self.0.abs_error()
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Spectrum1D(pub spec::Spectrum1D);

#[pymethods]
impl Spectrum1D {
    #[pyo3(signature = (omega, energy))]
    #[new]
    pub fn new(omega: Vec<f64>, energy: Vec<f64>) -> Self {
        Spectrum1D(spec::Spectrum1D {
            omega: Array1::from(omega),
            energy: Array1::from(energy),
        })
    }

    #[getter]
    fn omega<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.0.omega.clone().into_pyarray(py)
    }

    #[getter]
    fn energy<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.0.energy.clone().into_pyarray(py)
    }

    fn area(&self) -> PyResult<f64> {
        Ok(self.0.area())
    }

    fn spectral_moment(&self, n: u8) -> PyResult<f64> {
        Ok(self.0.spectral_moment(n))
    }

    #[allow(non_snake_case)]
    fn M0(&self) -> PyResult<f64> {
        Ok(self.0.M0())
    }

    #[allow(non_snake_case)]
    fn M1(&self) -> PyResult<f64> {
        Ok(self.0.M1())
    }

    #[allow(non_snake_case)]
    fn M2(&self) -> PyResult<f64> {
        Ok(self.0.M2())
    }

    #[allow(non_snake_case)]
    fn M4(&self) -> PyResult<f64> {
        Ok(self.0.M4())
    }

    fn std_dev(&self) -> PyResult<f64> {
        Ok(self.0.std_dev())
    }

    #[allow(non_snake_case)]
    fn As(&self) -> PyResult<f64> {
        Ok(self.0.As())
    }

    #[allow(non_snake_case)]
    fn Xs(&self) -> PyResult<f64> {
        Ok(self.0.Xs())
    }

    /// peak wave period, T_p
    #[allow(non_snake_case)]
    fn Tp(&self) -> PyResult<f64> {
        Ok(self.0.Tp())
    }

    /// mean period, T_1
    #[allow(non_snake_case)]
    fn T_mean(&self) -> PyResult<f64> {
        Ok(self.0.Tm01())
    }

    /// Zero-up-crossing period
    #[allow(non_snake_case)]
    fn Tm02(&self) -> PyResult<f64> {
        Ok(self.0.Tm02())
    }

    /// self.Ampm(time_window: float = 10800.0) -> float
    ///
    ///  
    /// Calculates the most probable maximum amplitude value, Ampm, for a
    /// given time window, time_window in seconds.
    ///
    /// Parameters
    /// ----------
    /// time_window : float, optional
    ///   Time window in seconds. Default is 10800.0 seconds (3 hours).
    ///
    /// Returns
    /// -------
    /// Ampm : float
    ///    Most probable maximum amplitude value
    ///
    ///
    #[allow(non_snake_case)]
    #[pyo3(signature = (time_window=10800.0, /), text_signature = "(time_window = 10800.0, /)")]
    fn Ampm(&self, time_window: f64) -> PyResult<f64> {
        Ok(self.0.Ampm(time_window))
    }

    /// self.Xmpm(time_window: float = 10800.0) -> float
    ///  
    /// Calculates the most probable maximum double amplitude value, Xmpm, for a
    /// given time window, time_window in seconds.
    ///
    /// Parameters
    /// ----------
    /// time_window : float, optional
    ///   Time window in seconds. Default is 10800.0 seconds (3 hours).
    ///
    /// Returns
    /// -------
    /// Xmpm : float
    ///    Most probable maximum double amplitude value
    ///
    ///
    #[allow(non_snake_case)]
    #[pyo3(signature = (time_window=10800.0, /),
    text_signature = "(time_window = 10800.0, /)")]
    fn Xmpm(&self, time_window: f64) -> PyResult<f64> {
        Ok(self.0.Xmpm(time_window))
    }

    #[pyo3(signature = (fractile=0.36787944117144233, time_window=10800.0, /),
    text_signature = "(fractile=np.exp(-1), time_window = 10800.0, /)")]
    fn fractile_extreme(&self, fractile: f64, time_window: f64) -> PyResult<f64> {
        Ok(self.0.fractile_extreme(fractile, time_window))
    }

    #[allow(non_snake_case)]
    #[pyo3(signature = (time_window=10800.0, /), text_signature = "(time_window = 10800.0, /)")]
    fn X50(&self, time_window: f64) -> PyResult<f64> {
        Ok(self.0.fractile_extreme(0.50, time_window))
    }

    #[allow(non_snake_case)]
    #[pyo3(signature = (time_window=10800.0, /), text_signature = "(time_window = 10800.0, /)")]
    fn X90(&self, time_window: f64) -> PyResult<f64> {
        Ok(self.0.fractile_extreme(0.90, time_window))
    }

    #[allow(non_snake_case)]
    #[pyo3(signature = (time_window=10800.0, /), text_signature = "(time_window = 10800.0, /)")]
    fn X95(&self, time_window: f64) -> PyResult<f64> {
        Ok(self.0.fractile_extreme(0.95, time_window))
    }

    #[pyo3(signature = (time_window=10800.0, /), text_signature = "(time_window = 10800.0, /)")]
    fn response(&self, time_window: f64) -> FrequencyResponse {
        FrequencyResponse(spec::FrequencyResponse {
            std_dev: self.0.std_dev(),
            Tm02: self.0.Tm02(),
            As: self.0.As(),
            Ampm: self.0.Ampm(time_window),
        })
    }
}

#[pyclass]
pub struct Spectrum2D(pub spec::Spectrum2D);

#[pymethods]
impl Spectrum2D {
    #[new]
    pub fn new(omega: Vec<f64>, theta: Vec<f64>, energy: PyReadonlyArray2<f64>) -> Self {
        Spectrum2D(spec::Spectrum2D {
            omega: Array1::from(omega),
            theta: Array1::from(theta),
            energy: energy.as_array().to_owned(),
        })
    }

    #[getter]
    fn omega<'a>(&self, py: Python<'a>) -> &'a PyArray1<f64> {
        PyArray1::from_array(py, &self.0.omega)
    }

    #[getter]
    fn theta<'a>(&self, py: Python<'a>) -> &'a PyArray1<f64> {
        PyArray1::from_array(py, &self.0.theta)
    }

    #[getter]
    fn energy<'a>(&self, py: Python<'a>) -> &'a PyArray2<f64> {
        PyArray2::from_array(py, &self.0.energy)
    }

    fn area(&self) -> f64 {
        self.0.area()
    }

    #[allow(non_snake_case)]
    fn M0(&self) -> PyResult<f64> {
        Ok(self.0.M0())
    }

    #[allow(non_snake_case)]
    fn M1(&self) -> PyResult<f64> {
        Ok(self.0.M1())
    }

    #[allow(non_snake_case)]
    fn M2(&self) -> PyResult<f64> {
        Ok(self.0.M2())
    }

    #[allow(non_snake_case)]
    fn M4(&self) -> PyResult<f64> {
        Ok(self.0.M4())
    }

    fn std_dev(&self) -> PyResult<f64> {
        Ok(self.0.std_dev())
    }

    #[allow(non_snake_case)]
    fn As(&self) -> PyResult<f64> {
        Ok(self.0.As())
    }

    #[allow(non_snake_case)]
    fn Xs(&self) -> PyResult<f64> {
        Ok(self.0.Xs())
    }

    #[allow(non_snake_case)]
    fn Tp(&self) -> PyResult<f64> {
        Ok(self.0.Tp())
    }

    fn theta_p(&self) -> PyResult<f64> {
        Ok(self.0.theta_p())
    }

    #[allow(non_snake_case)]
    fn T_mean(&self) -> PyResult<f64> {
        Ok(self.0.T_mean())
    }

    #[allow(non_snake_case)]
    fn Tz(&self) -> PyResult<f64> {
        Ok(self.0.Tz())
    }

    #[allow(non_snake_case)]
    fn Ampm(&self, time_window: f64) -> PyResult<f64> {
        Ok(self.0.Ampm(time_window))
    }

    #[allow(non_snake_case)]
    fn Xmpm(&self, time_window: f64) -> PyResult<f64> {
        Ok(self.0.Xmpm(time_window))
    }

    #[allow(non_snake_case)]
    fn X50(&self, time_window: f64) -> PyResult<f64> {
        Ok(self.0.X50(time_window))
    }

    #[allow(non_snake_case)]
    fn X90(&self, time_window: f64) -> PyResult<f64> {
        Ok(self.0.X90(time_window))
    }

    #[allow(non_snake_case)]
    fn X95(&self, time_window: f64) -> PyResult<f64> {
        Ok(self.0.X95(time_window))
    }

    fn fractile_extreme(&self, fractile: f64, time_window: f64) -> PyResult<f64> {
        Ok(self.0.fractile_extreme(fractile, time_window))
    }

    //    fn rotate_check<'a>(&self, py: Python<'a>, angle: f64) -> &'a PyArray2<f64> {
    //        PyArray2::from_array(py, &self.0.rotate_check(angle))
    //    }

    fn rotate(&mut self, angle: f64) -> PyResult<()> {
        self.0.rotate(angle);
        Ok(())
    }
}
