use crate::prelude::*;

pub trait SpectralCommon {
    /// significant wave height in \[m\]
    fn hs(&self) -> Result<f64, String>;

    /// peak spectral wave period in \[s\]
    fn tp(&self) -> Result<f64, String>;

    /// frequency space in \[rad/s\]
    fn omega(&self) -> &Array1<f64>;

    /// return the frequency space in \[Hz\]
    fn f_hz(&self) -> Array1<f64> {
        self.omega() / TWO_PI
    }

    /// return the peak spectral frequency in \[rad/s\]
    fn wp(&self) -> Result<f64, String> {
        match self.tp() {
            Ok(tp) => Ok(TWO_PI / tp),
            Err(e) => Err(e),
        }
    }

    /// return the peak spectral frequency in \[Hz\]
    fn fp(&self) -> Result<f64, String> {
        match self.wp() {
            Ok(wp) => Ok(wp / TWO_PI),
            Err(e) => Err(e),
        }
    }

    /// calculate the energy density spectrum
    fn energy(&self) -> Array1<f64>;

    /// calculate the absolute error between the significant wave height and the integrated energy
    fn abs_error(&self) -> Result<f64, String> {
        let area = trapz(self.energy().view(), self.omega().view());

        match self.hs() {
            Ok(hs) => Ok((hs - 4.0 * area.sqrt()).abs()),
            Err(e) => Err(e),
        }
    }

    /// convert the spectrum to a 1D spectrum Type
    fn to_spec1d(&self) -> Spectrum1D {
        Spectrum1D::new(self.omega().to_owned(), self.energy())
    }

    /// convert the spectrum to a 2D spectrum Type for a given spreading Type
    fn to_spec2d(&self, spreading: &Spreading) -> Spectrum2D {
        Spectrum2D::from_spec1d(&self.to_spec1d(), spreading)
    }
}
