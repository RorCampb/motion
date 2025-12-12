use thiserror::Error;
use serde::{Serialize, Deserialize};

#[derive(Debug, Error)]
pub enum MathError {
    #[error("dimension mismatch: left = {left}, right = {right}")]
    DimensionMismatch { left: usize, right: usize },

    #[error("zero-length vector: cannot normalize")]
    ZeroNorm,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VecN {
    pub data: Vec<f32>,
    pub norm: Option<f32>,
    pub normalized: Option<Vec<f32>>,
}

impl VecN {
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data,
            norm: None,
            normalized: None,
        }
    }

    pub fn from_slice(slice: &[f32]) -> Self {
        Self::new(slice.to_vec())
    }

    pub fn dim(&self) -> usize {
        self.data.len()
    }
    
    pub fn clear_cache(&mut self) {
        self.norm = None;
        self.normalized = None;
    }

    pub fn set_data(&mut self, data: Vec<f32>) {
        self.data = data;
        self.clear_cache();
    }

    pub fn norm(&mut self) -> f32 {
        if let Some(n) = self.norm {
            return n;
        }
        let n = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        self.norm = Some(n);
        n
    }

    pub fn normalize(&mut self) -> Result<&[f32], MathError> {
        if let Some(ref x) = self.normalized {
            return Ok(x);
        }

        let n = self.norm();
        if n == 0.0 {
            return Err(MathError::ZeroNorm);
        }

        let inv = 1.0 / n;
        let x: Vec<f32> = self.data.iter().map(|v| v * inv).collect();

        self.normalized = Some(x);
        Ok(self.normalized.as_ref().unwrap())
    }
}

pub fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, MathError> {
    if a.len() != b.len() {
        return Err(MathError::DimensionMismatch {
            left: a.len(),
            right: b.len(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

pub fn sub(a: &[f32], b: &[f32]) -> Result<Vec<f32>, MathError> {
    if a.len() != b.len() {
        return Err(MathError::DimensionMismatch {
            left: a.len(),
            right: b.len(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x - y).collect())
}


pub fn dot(a: &[f32], b: &[f32]) -> Result<f32, MathError> {
    if a.len() != b.len() {
        return Err(MathError::DimensionMismatch {
            left: a.len(),
            right: b.len(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

pub fn scale(a: &[f32], s: f32) -> Vec<f32> {
    a.iter().map(|x| x * s).collect()
}

pub fn normalize_slice(a: &[f32]) -> Result<VecN, MathError> {
    let mut v = VecN::from_slice(a);
    v.normalize()?; // populate cache / validate non-zero norm
    Ok(v)
}


