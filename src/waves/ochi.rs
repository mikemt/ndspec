/// The Ochi-Hubble [1] energy density spectrum is defined [1]:
///
/// ```latex
/// ```
///
use ndarray::Array1;
use std::f64::consts::PI;

const N_FREQ: usize = 256;

pub struct OchiHubble {
    pub hs: [f64; 2],
    pub tp: [f64; 2],
    pub omega: Array1<f64>,
}

impl Default for OchiHubble {
    fn default() -> Self {
        Self {
            hs: [1.0, 0.5],
            tp: [20.0, 8.0],
            omega: Array1::<f64>::linspace(0.1, PI, N_FREQ),
        }
    }
}
