use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;
use thiserror::Error;
use chrono::Utc;
use std::collections::HashSet;

#[derive(Debug, Error)]
pub enum InputError {
    #[error("input is not valid")]
    InvalidInput,
    #[error("channel closed while sending post input")]
    ChannelError,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PostInput {
    pub id: String,
    pub user_id: String,
    pub text: String
}
impl PostInput {
    pub fn new(id: impl Into<String>, user_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            user_id: user_id.into(),
            text: text.into()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserInput {
    pub id: String,
}
impl UserInput {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum InteractionType {
    PostToUser,
    UserToUser,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Interaction {
    pub interaction_type: InteractionType, 
    pub src_id: String,
    pub dst_id: String,
    pub alpha: f32,
}

pub enum MotionInput {
    User(UserInput),
    Post(PostInput),
    Interaction(Interaction),
}

impl MotionInput {
    pub async fn input_loop(tx: Sender<MotionInput>) -> Result<(), InputError> {
        use tokio::io::{self, AsyncBufReadExt};
                 
        let stdin = io::BufReader::new(io::stdin());
        let mut lines = stdin.lines();
        
        let mut current_user: Option<String> = None;
        let mut known_users: HashSet<String> = HashSet::new();

        async fn ensure_user(
            tx: &Sender<MotionInput>,
            known: &mut HashSet<String>,
            user_id: &str,
        ) -> Result<(), InputError> {
            if known.insert(user_id.to_string()) {
                let user = UserInput::new(user_id);
                tx.send(MotionInput::User(user))
                    .await
                    .map_err(|_| InputError::ChannelError)?;
            }
            Ok(())
        }

        async fn send_post(
            tx: &Sender<MotionInput>,
            user_id: &str,
            text: &str,
        ) -> Result<(), InputError> {
            let post_id = format!("post-{}", Utc::now().timestamp_millis());
            let post = PostInput::new(post_id, user_id, text);
            tx.send(MotionInput::Post(post))
                .await
                .map_err(|_| InputError::ChannelError)?;
            Ok(())
        }

        println!("Commands: u <id>, s <id>, p <text>, i post <post_id> <user_id> [alpha], i user <src_id> <dst_id> [alpha], q");

        loop {
            print!("> ");

            let Some(line) = lines
                .next_line()
                .await
                .map_err(|_| InputError::InvalidInput)?
            else {
                break;
            };

            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line.eq_ignore_ascii_case("q") {
                break;
            }

            let mut parts = line.split_whitespace();
            let cmd = parts.next().unwrap_or("");

            match cmd {
                "u" | "s" => {
                    let Some(id) = parts.next() else {
                        println!("Usage: {} <id>", cmd);
                        continue;
                    };
                    if id.is_empty() {
                        println!("Cannot accept empty user id");
                        continue;
                    }
                    ensure_user(&tx, &mut known_users, id).await?;
                    current_user = Some(id.to_string());
                    println!("Current user: {}", id);
                }
                "p" => {
                    let text = line.strip_prefix("p").unwrap_or("").trim();
                    if text.is_empty() {
                        println!("Usage: p <text>");
                        continue;
                    }
                    let Some(user_id) = current_user.as_ref() else {
                        println!("No current user...");
                        continue;
                    };
                    ensure_user(&tx, &mut known_users, user_id).await?;
                    send_post(&tx, user_id, text).await?;
                    println!("Posted as user: {}", user_id);
                }
                "i" => {
                    let Some(kind) = parts.next() else {
                        println!("Usage: i post|user ...");
                        continue;
                    };
                    match kind {
                        "post" => {
                            let Some(post_id) = parts.next() else {
                                println!("Usage: i post <post_id> <user_id> [alpha]");
                                continue;
                            };
                            let Some(user_id) = parts.next() else {
                                println!("Usage: i post <post_id> <user_id> [alpha]");
                                continue;
                            };
                            let alpha = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0.5);
                            ensure_user(&tx, &mut known_users, user_id).await?;
                            tx.send(MotionInput::Interaction(Interaction {
                                interaction_type: InteractionType::PostToUser,
                                src_id: post_id.to_string(),
                                dst_id: user_id.to_string(),
                                alpha,
                            }))
                            .await
                            .map_err(|_| InputError::ChannelError)?;
                        }
                        "user" => {
                            let Some(src_id) = parts.next() else {
                                println!("Usage: i user <src_id> <dst_id> [alpha]");
                                continue;
                            };
                            let Some(dst_id) = parts.next() else {
                                println!("Usage: i user <src_id> <dst_id> [alpha]");
                                continue;
                            };
                            let alpha = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0.5);
                            ensure_user(&tx, &mut known_users, src_id).await?;
                            ensure_user(&tx, &mut known_users, dst_id).await?;
                            tx.send(MotionInput::Interaction(Interaction {
                                interaction_type: InteractionType::UserToUser,
                                src_id: src_id.to_string(),
                                dst_id: dst_id.to_string(),
                                alpha,
                            }))
                            .await
                            .map_err(|_| InputError::ChannelError)?;
                        }
                        _ => {
                            println!("Usage: i post|user ...");
                        }
                    }
                }
                "?" | "help" => {
                    println!("Commands: u <id>, s <id>, p <text>, i post <post_id> <user_id> [alpha], i user <src_id> <dst_id> [alpha], q");
                }
                _ => {
                    if let Some((user_id, text)) = line.split_once(':') {
                        let user_id = user_id.trim();
                        let text = text.trim();
                        if user_id.is_empty() || text.is_empty() {
                            println!("Usage: <user_id>: <text>");
                            continue;
                        }
                        ensure_user(&tx, &mut known_users, user_id).await?;
                        send_post(&tx, user_id, text).await?;
                        println!("Posted as user: {}", user_id);
                    } else {
                        let Some(user_id) = current_user.as_ref() else {
                            println!("No current user...");
                            continue;
                        };
                        ensure_user(&tx, &mut known_users, user_id).await?;
                        send_post(&tx, user_id, line).await?;
                        println!("Posted as user: {}", user_id);
                    }
                }
            }
        }
        Ok(())
    }
}
