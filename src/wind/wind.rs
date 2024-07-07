use crate::core::constants::VelocityConversions;
use std::default::Default;

#[derive(Clone)]
pub struct Wind {
    pub speed: f64,
    pub direction: f64,
    pub interval: Option<f64>,
    pub elevation: Option<f64>,
}

impl Default for Wind {
    fn default() -> Self {
        Self {
            speed: 0.0,
            direction: 0.0,
            interval: Some(10.0 * 60.0),
            elevation: Some(10.0),
        }
    }
}

impl VelocityConversions for Wind {
    fn speed(&self) -> f64 {
        self.speed
    }
}

impl Wind {
    pub fn new(speed: f64, direction: f64) -> Self {
        Wind {
            speed,
            direction,
            ..Default::default()
        }
    }

    #[allow(dead_code)]
    fn set_elevation(&mut self, elevation: f64) {
        self.elevation = Some(elevation);
    }

    #[allow(dead_code)]
    fn set_interval(&mut self, interval: f64) {
        self.interval = Some(interval);
    }

    #[allow(dead_code)]
    fn change_interval(&mut self, _interval: f64) {}

    #[allow(dead_code)]
    fn speed_10min(&self) {}
}

#[cfg(test)]
mod test_wind {
    use super::*;

    #[test]
    fn test_wind() {
        let wind = Wind::new(10.0, 180.0);
        assert_eq!(wind.speed, 10.0);
        assert_eq!(wind.direction, 180.0);
        assert_eq!(wind.interval, Some(10.0 * 60.0));
        assert_eq!(wind.elevation, Some(10.0));
    }
}
