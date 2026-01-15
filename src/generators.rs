use nalgebra::{Rotation3, UnitQuaternion, Vector3};
use rand::Rng;

pub fn random_euler(rng: &mut impl Rng) -> Vector3<f64> {
    Vector3::new(
        rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
        rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
        rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
    )
}

pub fn random_quat(rng: &mut impl Rng) -> UnitQuaternion<f64> {
    UnitQuaternion::from_euler_angles(
        rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
        rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
        rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
    )
}

pub fn random_matrix(rng: &mut impl Rng) -> Rotation3<f64> {
    Rotation3::from_euler_angles(
        rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
        rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
        rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
    )
}
