use crate::prelude::*;

pub const GRAVITY: f64 = 9.80665;
pub const PI: f64 = std::f64::consts::PI;
pub const TWO_PI: f64 = 2.0 * PI;
pub const RHO_SEA_WATER: f64 = 1025.0;
pub const RHO_AIR: f64 = 1.225;
pub const KNOT: f64 = 0.5144444444444444;

pub trait VelocityConversions {
    fn speed(&self) -> f64;

    /// Return meters per second (mps) in knots
    fn knots(&self) -> f64 {
        self.speed() / KNOT
    }

    /// Return knots in meters per second (mps)
    fn mps(&self) -> f64 {
        self.speed() * KNOT
    }
}

//pub trait VelocityConversions {
//    fn knots(&self) -> f64;
//    fn mps(&self) -> f64;
//}
