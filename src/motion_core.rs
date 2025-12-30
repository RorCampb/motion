use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{Sender, Receiver};
use thiserror::Error;

use crate::embedding::embed_post;
use crate::math::{MathError, VecN};
use crate::kernel::{apply_kernel2, Kernel};
use crate::motion_input::{MotionInput, Interaction, InteractionType};


#[derive(Debug, Error)]
pub enum CoreError {
    #[error("user not found for id: {user_id}")]
    UserNotFound { user_id: String },
    
    #[error("post not found for id: {post_id}")]
    PostNotFound { post_id: String },

    #[error("no coord loaded for user id: {user_id}")]
    CoordNotLoaded { user_id: String },

    #[error("math error: {0}")]
    Math(#[from] MathError), 
   
    #[error("channel closed while sending motion entry")]
    ChannelError
}



#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MotionUser {
    pub id: String,
    pub coord: Option<VecN>,
    pub motion: f32,
}

impl MotionUser {
    pub fn new(id: impl Into<String>, _dim: usize) -> Self {
        let motion = 0.0;
        Self {
            id: id.into(),
            coord: None,
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
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InteractionResult {
    pub src_id: String,
    pub dst_id: String,
    pub weight: f32,
    pub similarity: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum MotionOutput {
    Entered(MotionEntry),
    Updated(MotionEntry),
    InteractionApplied(InteractionResult) 
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MotionSpace {
    pub dim: usize,
    pub entries: Vec<MotionEntry>,
    pub kernel: Kernel,
}

impl MotionSpace {
    pub fn new(dim: usize) -> Self {
        let kernel = Kernel::RBF { gamma: 2.0 };
        Self {
            dim,
            entries: Vec::new(),
            kernel,
        }
    }

    pub fn enter(&mut self, entry: MotionEntry) {
        self.entries.push(entry);
    }
   

    pub fn apply_user_to_user(
        &mut self,
        actor_id: &str,
        target_id: &str,
        alpha: f32,
    ) -> Result<InteractionResult, CoreError> {
        let actor_idx = self
            .entries
            .iter()
            .position(|e| matches!(e, MotionEntry::User(u) if u.id == actor_id))
            .ok_or_else(|| CoreError::UserNotFound { user_id: actor_id.to_string() })?;

        let target_idx = self
            .entries
            .iter()
            .position(|e| matches!(e, MotionEntry::User(u) if u.id == target_id))
            .ok_or_else(|| CoreError::UserNotFound { user_id: target_id.to_string() })?;
        
        let (actor_data, target_data, actor_motion, target_motion) = {
            let a = match &self.entries[actor_idx] {
                MotionEntry::User(u) => u,
                _ => unreachable!("actor idx must point to a user"),
            };
            let t = match &self.entries[target_idx] {
                MotionEntry::User(u) => u,
                _ => unreachable!("target idx must point to a user"),
            };
            let actor_coord = a.coord.as_ref().ok_or_else(|| CoreError::CoordNotLoaded {
                user_id: actor_id.to_string(),
            })?;
            let target_coord = t.coord.as_ref().ok_or_else(|| CoreError::CoordNotLoaded {
                user_id: target_id.to_string(),
            })?;
            (
                actor_coord.data.clone(),
                target_coord.data.clone(),
                a.motion,
                t.motion,
            )
        };
        
        let similarity = self.kernel.apply(&actor_data, &target_data)?;
        let weight = 1.0 - (-alpha * similarity).exp();
        
        let step = 0.5 * weight;
        
        let new_actor_data = apply_kernel2(&actor_data, &target_data, |a, t| a * (1.0 - step) + t * step)?;
        let new_target_data = apply_kernel2(&target_data, &actor_data, |t, a| t * (1.0 - step) + a * step)?;

        let mut new_actor_coord = VecN::new(new_actor_data);
        let mut new_target_coord = VecN::new(new_target_data);
        let _ = new_actor_coord.normalize();
        let _ = new_target_coord.normalize();

        let decay = 0.02;
        let gain_target = 1.0;
        let gain_actor = 0.5;
        
        let new_target_motion = (1.0 - decay) * target_motion + gain_target * weight;
        let new_actor_motion = (1.0 - decay) * actor_motion + gain_actor * weight;
        
        if let MotionEntry::User(u) = &mut self.entries[target_idx] {
            u.coord = Some(new_target_coord);
            u.motion = new_target_motion;
        }
        if let MotionEntry::User(u) = &mut self.entries[actor_idx] {
            u.coord = Some(new_actor_coord);
            u.motion = new_actor_motion;
        }
        
        println!("sim={:.4} weight={:.4} actor motion={:.4} target motion={:.4}", similarity, weight, new_actor_motion, new_target_motion);
        Ok(InteractionResult {
            src_id: actor_id.to_string(),
            dst_id: target_id.to_string(),
            weight,
            similarity,
        })
    }

    pub fn apply_post_to_user(
        &mut self,
        user_id: &str,
        post_id: &str,
        alpha: f32,
    ) -> Result<InteractionResult, CoreError> {
        let post_idx = self
            .entries
            .iter()
            .position(|e| matches!(e, MotionEntry::Post(p) if p.id == post_id))
            .ok_or_else(|| CoreError::PostNotFound { post_id: post_id.to_string() })?;

        let post_coord = match &self.entries[post_idx] {
            MotionEntry::Post(p) => p.coord.clone(),
            _ => unreachable!("post idx must point to a post"),
        };
        let user_idx = match self
            .entries
            .iter()
            .position(|e| matches!(e, MotionEntry::User(u) if u.id == user_id))
        {
            Some(idx) => idx,
            None => {
                let motion_user = MotionUser::new(user_id, self.dim);
                self.entries.push(MotionEntry::User(motion_user));
                self.entries.len() - 1
            }
        };
        let user_coord = match &mut self.entries[user_idx] {
            MotionEntry::User(u) => u.coord.get_or_insert_with(|| post_coord.clone()),
            _ => unreachable!("user idx must point to a user"),
        };
        let user_data = user_coord.data.clone();
        let post_data = post_coord.data.clone();

        let similarity = self.kernel.apply(&user_data, &post_data)?;
        let weight = 1.0 - (-alpha * similarity).exp();
         
        
        let new_data = apply_kernel2(&user_data, &post_data, |u, p| {
            u * (1.0 - weight) + p * weight
        })?;

        let mut new_coord = VecN::new(new_data);
        let _ = new_coord.normalize();
       
        let decay = 0.02;
        let gain = 1.0;

        
        if let MotionEntry::User(u) = &mut self.entries[user_idx] {
            let new_motion = (1.0 - decay) * u.motion + gain * weight;

            u.coord = Some(new_coord);
            u.motion = new_motion;
            println!("sim={:.4} weight={:.4} motion={:.4}", similarity, weight, u.motion);
        }

        Ok(InteractionResult {
            src_id: post_id.to_string(),
            dst_id: user_id.to_string(),
            weight,
            similarity,
        })
    }

    pub fn apply_interaction(&mut self, interaction: Interaction) -> Result<InteractionResult, CoreError> {
        match interaction.interaction_type {
            InteractionType::PostToUser => {
                self.apply_post_to_user(&interaction.dst_id, &interaction.src_id, interaction.alpha)
            },
            InteractionType::UserToUser => {
                self.apply_user_to_user(&interaction.src_id, &interaction.dst_id, interaction.alpha)
            },
        }
    }

    pub async fn core_loop(&mut self, mut rx: Receiver<MotionInput>, tx: Sender<MotionOutput>) -> Result<(), CoreError> {
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
                    tx.send(MotionOutput::Entered(entry))
                        .await
                        .map_err(|_| CoreError::ChannelError)?;
                   
                    if self.entries.iter().all(|e| !matches!(e, MotionEntry::User(u) if u.id == post.user_id)) {
                        let motion_user = MotionUser::new(&post.user_id, self.dim);
                        let user_entry = MotionEntry::User(motion_user);
                        self.enter(user_entry.clone());
                        tx.send(MotionOutput::Entered(user_entry))
                            .await
                            .map_err(|_| CoreError::ChannelError)?;
                    }

                    let interaction = Interaction {
                        interaction_type: InteractionType::PostToUser, 
                        src_id: post.id.clone(),
                        dst_id: post.user_id.clone(),
                        alpha: 0.5,
                    };
                    let res = self.apply_interaction(interaction)?;
                    tx.send(MotionOutput::InteractionApplied(res))
                        .await
                        .map_err(|_| CoreError::ChannelError)?;
                }
                MotionInput::User(user) => {
                    let motion_user = MotionUser::new(&user.id, self.dim);

                    let entry = MotionEntry::User(motion_user);

                    self.enter(entry.clone());
                    tx.send(MotionOutput::Entered(entry))
                        .await
                        .map_err(|_| CoreError::ChannelError)?;
                }
                MotionInput::Interaction(interaction) => {
                    let res = self.apply_interaction(interaction)?; 
                    tx.send(MotionOutput::InteractionApplied(res))
                        .await
                        .map_err(|_| CoreError::ChannelError)?; 
                }
            }
        }
        Ok(())
    }    
}
