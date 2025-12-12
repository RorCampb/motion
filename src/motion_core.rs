use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{Sender, Receiver};
use thiserror::Error;

use crate::embedding::embed_post;
use crate::math::{MathError, VecN};
use crate::kernel::{apply_kernel2, Kernel};
use crate::motion_input::{MotionInput};


#[derive(Debug, Error)]
pub enum CoreError {
    #[error("user not found for id: {user_id}")]
    UserNotFound { user_id: String },
    
    #[error("post not found for id: {post_id}")]
    PostNotFound { post_id: String },

    #[error("math error: {0}")]
    Math(#[from] MathError), 
   
    #[error("channel closed while sending motion entry")]
    ChannelError
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MotionUser {
    pub id: String,
    pub coord: VecN,
    pub motion: f32,
}

impl MotionUser {
    pub fn new(id: impl Into<String>, dim: usize) -> Self {
        let coord = VecN::new(vec![0.0; dim]);
        let motion = 0.0;
        Self {
            id: id.into(),
            coord,
            motion,
        }
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
    pub kernel: Kernel,
}

impl MotionSpace {
    pub fn new(dim: usize) -> Self {
        let default_kernel = Kernel::RBF { gamma: 0.5 };
        Self::new_with_kernel(dim, default_kernel)
    }

    pub fn new_with_kernel(dim: usize, kernel: Kernel) -> Self {
        Self {
            dim,
            entries: Vec::new(),
            kernel,
        }
    }

    pub fn enter(&mut self, entry: MotionEntry) {
        self.entries.push(entry);
    }
    
    pub async fn core_loop(&mut self, mut rx: Receiver<MotionInput>, tx: Sender<MotionEntry>) -> Result<(), CoreError> {
        while let Some(input) = rx.recv().await {
            match input {
                MotionInput::Post(post) => {
                    let embedding: VecN = embed_post(&post.text);

                    let motion_post = MotionPost::new(
                        post.id.clone(),
                        embedding,
                    );

                    let entry = MotionEntry::Post(motion_post);

                    self.enter(entry.clone());
                    match self.apply_post_to_user(&post.user_id, &post.id, 0.5) {
                        Ok(()) => println!("post application successful for {:?}", entry),
                        Err(e) => {
                            eprintln!("post application error: {}", e);
                            return Err(e);
                        }
                    }

                    tx.send(entry)
                        .await
                        .map_err(|_| CoreError::ChannelError)?;
                }
                MotionInput::User(user) => {
                    let motion_user = MotionUser::new(&user.id, self.dim);

                    let entry = MotionEntry::User(motion_user);

                    self.enter(entry.clone());
                    tx.send(entry)
                        .await
                        .map_err(|_| CoreError::ChannelError)?;
                }
            }
        }
        Ok(())
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

        let similarity = self.kernel.apply(&user_data, &post_data)?;
        let weight = (alpha * similarity).clamp(0.0, 1.0);

        let new_data = apply_kernel2(&user_data, &post_data, |u, p| {
            u * (1.0 - weight) + p * weight
        })?;

        let mut new_coord = VecN::new(new_data);
        let new_motion = new_coord.norm();

        if let MotionEntry::User(u) = &mut self.entries[user_idx] {
            u.coord = new_coord;
            u.motion = new_motion;
        }

        Ok(())
    }
}
