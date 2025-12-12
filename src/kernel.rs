use crate::math::MathError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Kernel {
    RBF { gamma: f32 },
}

pub fn rbf_kernel(x: &[f32], y: &[f32], gamma: f32) -> Result<f32, MathError> {
    if x.len() != y.len() {
        return Err(MathError::DimensionMismatch {
            left: x.len(),
            right: y.len(),
        });
    }

    let sq_dist: f32 = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum();

    Ok((-gamma * sq_dist).exp())
}

pub fn apply_kernel<F>(a: &[f32], mut f: F) -> Vec<f32>
where
    F: FnMut(f32) -> f32,
{
    a.iter().copied().map(|x| f(x)).collect()
}

pub fn apply_kernel2<F>(a: &[f32], b: &[f32], mut f: F) -> Result<Vec<f32>, MathError>
where
    F: FnMut(f32, f32) -> f32,
{
    if a.len() != b.len() {
        return Err(MathError::DimensionMismatch {
            left: a.len(),
            right: b.len(),
        });
    }

    Ok(a.iter()
        .copied()
        .zip(b.iter().copied())
        .map(|(x, y)| f(x, y))
        .collect())
}

pub fn apply_kernel_indexed<F>(a: &[f32], mut f: F) -> Vec<f32>
where
    F: FnMut(usize, f32) -> f32,
{
    a.iter()
        .copied()
        .enumerate()
        .map(|(i, x)| f(i, x))
        .collect()
}

impl Kernel {
    pub fn apply(&self, x: &[f32], y: &[f32]) -> Result<f32, MathError> {
        match self {
            Kernel::RBF { gamma } => rbf_kernel(x, y, *gamma),
        }
    }
}
