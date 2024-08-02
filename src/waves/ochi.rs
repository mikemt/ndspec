//! The Ochi-Hubble energy density spectru:
//!
//!
//!
use crate::prelude::*;
use ndarray::{Array1, ArrayView1};
use special::Gamma;

use crate::waves::common::SpectralCommon;

const N_FREQ: usize = 256;

pub struct OchiHubble {
    pub hs: [f64; 2],
    pub tp: [f64; 2],
    pub lambda: [f64; 2],
    pub omega: Array1<f64>,
}

pub fn energy(omega: ArrayView1<f64>, hs: [f64; 2], tp: [f64; 2], lambda: [f64; 2]) -> Array1<f64> {
    let a = |w: f64, l: f64| w.powf(-4. * (l + 0.25)) * (-w.powf(-4.) * (l + 0.25)).exp();
    let b = |l: f64| 4. * (l + 0.25).powf(l) / Gamma::gamma(l);
    let c = |h: f64, t: f64| h.powf(2.) * t / (32. * PI);

    let energy: Array1<f64> = omega
        .iter()
        .map(|w_i| {
            let w_n = [w_i * tp[0] / TWO_PI, w_i * tp[1] / TWO_PI];
            let mut sum = 0.0;
            for i in 0..2 {
                sum += a(w_n[i], lambda[i]) * b(lambda[i]) * c(hs[i], tp[i]);
            }
            sum
        })
        .collect();

    energy
}

impl Default for OchiHubble {
    fn default() -> Self {
        Self {
            hs: [1.0, 0.5],
            tp: [20.0, 8.0],
            lambda: [0.5, 0.5],
            omega: Array1::<f64>::linspace(0.1, PI, N_FREQ),
        }
    }
}

impl SpectralCommon for OchiHubble {
    /// rms significant wave height approximation
    fn hs(&self) -> Result<f64, String> {
        match self.hs.len() {
            2 => Ok((self.hs[0].powf(2.) + self.hs[1].powf(2.)).sqrt()),
            _ => Err("hs is not length 2".to_string()),
        }
    }

    /// rms peak spectral wave period approximation
    fn tp(&self) -> Result<f64, String> {
        match self.tp.len() {
            2 => Ok((self.tp[0].powf(2.) + self.tp[1].powf(2.)).sqrt()),
            _ => Err("tp is not length 2".to_string()),
        }
    }

    fn omega(&self) -> &Array1<f64> {
        &self.omega
    }

    fn energy(&self) -> Array1<f64> {
        energy(self.omega.view(), self.hs, self.tp, self.lambda)
    }
}

impl OchiHubble {
    /// Create a new Ochi-Hubble spectrum
    pub fn new(hs: [f64; 2], tp: [f64; 2], lambda: [f64; 2]) -> Self {
        Self {
            hs,
            tp,
            lambda,
            ..Default::default()
        }
    }

    /// Set the significant wave height for each component
    pub fn set_hs(&mut self, hs: [f64; 2]) -> &mut Self {
        self.hs = hs;
        self
    }

    /// Set the peak spectral wave period for each component
    pub fn set_tp(&mut self, tp: [f64; 2]) -> &mut Self {
        self.tp = tp;
        self
    }

    /// Set the shape factor for each component
    pub fn set_lambda(&mut self, lambda: [f64; 2]) -> &mut Self {
        self.lambda = lambda;
        self
    }

    /// Return the peak spectral wave frequency for each component
    pub fn get_wp(&self) -> Result<[f64; 2], String> {
        match self.tp.len() {
            2 => Ok([self.tp[0] / TWO_PI, self.tp[1] / TWO_PI]),
            _ => Err("tp is not length 2".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ochi_hubble() {
        let ochi = OchiHubble::default();
        let energy = ochi.energy();
        assert_eq!(energy.len(), N_FREQ);
    }

    #[test]
    fn test_ochi_hubble_hs() {
        let ochi = OchiHubble::default();
        let hs = ochi.hs().unwrap();

        assert_eq!(ochi.hs[0], 1.0_f64);
        assert_eq!(ochi.hs[1], 0.5_f64);
        assert_eq!(hs, (1.25_f64).sqrt());
    }

    #[test]
    fn test_ochi_hubble_tp() {
        let ochi = OchiHubble::default();
        let tp = ochi.tp().unwrap();

        assert_eq!(ochi.tp[0], 20.0_f64);
        assert_eq!(ochi.tp[1], 8.0_f64);
        assert_eq!(tp, (20.0_f64.powf(2.) + 8.0_f64.powf(2.)).sqrt());
    }
}
