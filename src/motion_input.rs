use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;
use thiserror::Error;
use chrono::Utc;

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

pub enum MotionInput {
    User(UserInput),
    Post(PostInput),
}
impl MotionInput {
    pub async fn input_loop(tx: Sender<MotionInput>) -> Result<(), InputError> {
        use tokio::io::{self, AsyncBufReadExt};
                 
        let stdin = io::BufReader::new(io::stdin());
        let mut lines = stdin.lines();
        
        let mut current_user: Option<String> = None;

        loop {
            println!();
            println!("=== Motion Input ===");
            println!("1. New User");
            println!("2. New Post (for current user)");
            println!("3. Switch Current User");
            println!("q. Quit");
            print!("> ");

            let Some(choice) = lines
                .next_line()
                .await
                .map_err(|_| InputError::InvalidInput)?
            else {
                break;
            };

            let choice = choice.trim();

            match choice {
                "1" => {
                    println!("Enter new user id:");
                    let Some(id) = lines
                        .next_line()
                        .await
                        .map_err(|_| InputError::InvalidInput)?
                    else {
                        break;
                    };
                    let id = id.trim().to_string();
                    if id.is_empty() {
                        println!("Cannot accept empty user id");
                        continue;
                    }

                    let user = UserInput::new(id.clone());
                    current_user = Some(id.clone());

                    tx.send(MotionInput::User(user))
                        .await
                        .map_err(|_| InputError::ChannelError)?;
                    println!("Created and switched to user: {}", id);
                }

                "2" => {
                    let user_id = match &current_user {
                        Some(id) => id.clone(),
                        None => {
                            println!("No current user...");
                            continue;
                        }
                    };

                    println!("Write a post: ");
                    let Some(text) = lines
                        .next_line()
                        .await
                        .map_err(|_| InputError::InvalidInput)?
                    else {
                        break;
                    };
                    let text = text.trim().to_string();
                    if text.is_empty() {
                        println!("Post cannot be empty");
                        continue;
                    }
                    let post_id = format!("post-{}", Utc::now().timestamp_millis());
                    let post = PostInput::new(
                        post_id,
                        user_id.clone(),
                        text
                    );
                    tx.send(MotionInput::Post(post))
                        .await
                        .map_err(|_| InputError::ChannelError)?;
                    println!("Posted as user: {}", user_id);
                }
                "3" => {
                    println!("Enter user id to switch to:");
                    let Some(id) = lines
                        .next_line()
                        .await
                        .map_err(|_| InputError::InvalidInput)?
                    else {
                        break;
                    };
                    let id = id.trim().to_string();
                    if id.is_empty() {
                        println!("Cannot accept empty user id");
                        continue;
                    }
                    current_user = Some(id.clone());
                    println!("Switched to user: {}", id);
                }
                "q" | "Q" => break,
                _ => {
                    println!("Unrecognized choice.");
                }
            }
        }
        while let Some(line) = lines.next_line().await.map_err(|_| InputError::InvalidInput)? {
            let line = line.trim();
            if line.is_empty() {
                continue; 
            }
            
            let (user_id, text) = match line.split_once(':') {
                Some((id, msg)) => (id.trim(), msg.trim()),
                None => ("default", line),
            };
            
            let post_id = format!("post-{}", Utc::now().timestamp_millis());

            let post = PostInput::new(post_id, user_id, text);
            tx.send(MotionInput::Post(post))
                .await
                .map_err(|_| InputError::ChannelError)?;
        }
        Ok(())
    }
}
