use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::math::{apply_kernel2, MathError, VecN};

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("user not found for id: {user_id}")]
    UserNotFound { user_id: String },
    
    #[error("post not found for id: {post_id}")]
    PostNotFound { post_id: String },

    #[error("math error: {0}")]
    Math(#[from] MathError), 
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MotionUser {
    pub id: String,
    pub coord: VecN,
    pub motion: f32,
}

impl MotionUser {
    pub fn new(id: String, coord: VecN) -> Self {
        let mut c = coord.clone();
        let m = c.norm();
        Self { id, coord, motion: m }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MotionPost {
    pub id: String,
    pub coord: VecN,
    pub features: Vec<VecN>,
}

impl MotionPost {
    pub fn new(id: String, coord: VecN) -> Self {
        Self {
            id,
            coord,
            features: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum MotionEntry {
    User(MotionUser),
    Post(MotionPost),
}

impl MotionEntry {
    pub fn id(&self) -> &str {
        match self {
            MotionEntry::User(u) => &u.id,
            MotionEntry::Post(p) => &p.id,
        }
    }

    pub fn coord(&mut self) -> &mut VecN {
        match self {
            MotionEntry::User(u) => &mut u.coord,
            MotionEntry::Post(p) => &mut p.coord,
        }
    }
}


#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MotionSpace {
    pub dim: usize,
    pub entries: Vec<MotionEntry>,
}

impl MotionSpace {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entries: Vec::new(),
        }
    }

    pub fn enter(&mut self, entry: MotionEntry) {
        self.entries.push(entry);
    }
    
    pub fn apply_post_to_user(
        &mut self,
        user_id: &str,
        post_id: &str,
        alpha: f32,
    ) -> Result<(), CoreError> {
        let user_idx = self
            .entries
            .iter()
            .position(|e| matches!(e, MotionEntry::User(u) if u.id == user_id))
            .ok_or_else(|| CoreError::UserNotFound { user_id: user_id.to_string() })?;

        let post_idx = self
            .entries
            .iter()
            .position(|e| matches!(e, MotionEntry::Post(p) if p.id == post_id))
            .ok_or_else(|| CoreError::PostNotFound { post_id: post_id.to_string() })?;

        let (user_data, post_data) = {
            let u = match &self.entries[user_idx] {
                MotionEntry::User(u) => u,
                _ => unreachable!("user_idx must point to a user"),
            };
            let p = match &self.entries[post_idx] {
                MotionEntry::Post(p) => p,
                _ => unreachable!("post_idx must point to a post"),
            };
            (u.coord.data.clone(), p.coord.data.clone())
        };

        let new_data =
            apply_kernel2(&user_data, &post_data, |u, p| u * (1.0 - alpha) + p * alpha)?;

        let mut new_coord = VecN::new(new_data);
        let new_motion = new_coord.norm();

        if let MotionEntry::User(u) = &mut self.entries[user_idx] {
            u.coord = new_coord;
            u.motion = new_motion;
        }

        Ok(())
    }
}
