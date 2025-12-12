use std::error::Error;

mod embedding;
mod kernel;
mod math;
mod motion_core;
mod motion_input;

use tokio::sync::mpsc;

use crate::embedding::EMBEDDING_DIM;
use crate::motion_core::{MotionEntry, MotionSpace};
use crate::motion_input::MotionInput;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Channel from stdin loop -> core loop
    let (input_tx, input_rx) = mpsc::channel::<MotionInput>(64);
    // Channel from core loop -> logger
    let (entry_tx, mut entry_rx) = mpsc::channel::<MotionEntry>(64);

    // Spawn the input loop (stdin driven)
    let input_handle = tokio::spawn(async move {
        if let Err(e) = MotionInput::input_loop(input_tx).await {
            eprintln!("input loop error: {}", e);
        }
    });

    // Spawn the core loop that processes inputs into motion space updates
    let core_handle = tokio::spawn(async move {
        let mut space = MotionSpace::new(EMBEDDING_DIM);
        if let Err(e) = space.core_loop(input_rx, entry_tx).await {
            eprintln!("core loop error: {}", e);
        }
    });

    // Log entries as they are produced
    while let Some(entry) = entry_rx.recv().await {
        log_entry(&entry);
    }

    // Ensure tasks complete (they may already be done if channels closed)
    let _ = input_handle.await;
    let _ = core_handle.await;

    Ok(())
}

fn log_entry(entry: &MotionEntry) {
    match entry {
        MotionEntry::User(u) => {
            println!("User [{}] motion {:.4} coord {:?}", u.id, u.motion, u.coord.data);
        }
        MotionEntry::Post(p) => {
            println!("Post [{}] coord {:?}", p.id, p.coord.data);
        }
    }
}
