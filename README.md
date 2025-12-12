# Motion Core

A Rust-based system for tracking users and content in a dynamic multi-dimensional embedding space.

## What is this?

Motion Core is an experimental system that models users and their posts as points in a high-dimensional vector space. As users create posts, their position in this "motion space" shifts based on the semantic content of what they write, creating a dynamic representation of user interests and behavior over time.

## Key Concepts

### Motion Space
The core concept is a **motion space** - a 128-dimensional vector space where:
- **Users** are represented as points with coordinates that evolve over time
- **Posts** are embedded as static points based on their text content
- User positions update dynamically as they create posts, moving toward the semantic content they engage with

### Motion Metric
Each user has a **motion** value that represents the magnitude of their position vector in the space - essentially how far they've "moved" from the origin. This can be interpreted as a measure of their activity or engagement level.

## Components

The codebase is organized into several key modules:

### Core Modules

- **`motion_core.rs`**: The heart of the system
  - `MotionSpace`: Manages the entire embedding space
  - `MotionUser`: Represents users with coordinates and motion values
  - `MotionPost`: Represents posts with their embeddings
  - `apply_post_to_user()`: Updates user positions based on kernel similarity

- **`motion_input.rs`**: Interactive CLI for user input
  - Provides a menu-driven interface for creating users and posts
  - Handles stdin-based interaction with the system
  - Manages the current user context for posting

- **`embedding.rs`**: Text-to-vector embedding
  - Converts post text into 128-dimensional vectors
  - Uses a simple but effective feature extraction:
    - Token-level bag-of-words (hashed to dimensions)
    - Character 3-gram features
  - Normalizes embeddings to unit vectors

- **`kernel.rs`**: Similarity computation
  - Implements RBF (Radial Basis Function) kernel for measuring similarity
  - Provides utility functions for element-wise operations on vectors
  - Used to compute how "similar" a user's current position is to a post

- **`math.rs`**: Vector mathematics
  - `VecN`: N-dimensional vector with caching for performance
  - Standard operations: norm, normalize, dot product, add, subtract, scale
  - Error handling for dimension mismatches and zero-norm cases

## How It Works

1. **User Creation**: A user enters the space at the origin (all zeros) with motion = 0

2. **Post Creation**: When a user creates a post:
   - The post text is embedded into a 128-dimensional vector using feature extraction
   - The post is added to the motion space at its embedded coordinates

3. **User Position Update**: After posting, the user's position is updated:
   - Compute similarity between user's current position and the post using RBF kernel
   - Calculate a weighted interpolation: `new_position = (1-w) * user_pos + w * post_pos`
   - The weight `w = alpha * similarity` (with alpha = 0.5 by default)
   - Update the user's motion metric based on their new position

4. **Continuous Tracking**: Users can create multiple posts, and their position evolves continuously, creating a trajectory through the motion space

## Architecture

The system uses an asynchronous architecture built on Tokio:

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│ Input Loop   │────────▶│  Core Loop   │────────▶│ Logger       │
│ (stdin)      │ channel │ (MotionSpace)│ channel │ (stdout)     │
└──────────────┘         └──────────────┘         └──────────────┘
```

- **Input Loop**: Reads user commands from stdin and sends `MotionInput` events
- **Core Loop**: Processes inputs, updates the motion space, emits `MotionEntry` events
- **Logger**: Receives entries and logs user positions and motion values

## Technical Details

- **Language**: Rust (Edition 2024)
- **Async Runtime**: Tokio with full features
- **Serialization**: Serde with JSON support
- **Dependencies**: 
  - `tokio` - Async runtime
  - `serde` & `serde_json` - Serialization
  - `thiserror` - Error handling
  - `chrono` - Timestamps for post IDs

## Running the System

```bash
cargo run
```

This starts the interactive CLI where you can:
1. Create new users
2. Post content as the current user
3. Switch between users
4. Observe how user positions evolve in real-time

## Use Cases

While this is an experimental system, potential applications include:

- **User Interest Modeling**: Track how user interests evolve over time
- **Content Recommendation**: Recommend content based on proximity in motion space
- **Community Detection**: Identify user clusters based on their trajectories
- **Engagement Metrics**: Motion values as a measure of user engagement
- **Anomaly Detection**: Detect unusual posting patterns via sudden position shifts

## Future Enhancements

Potential areas for expansion:

- Multiple kernel functions (Linear, Polynomial, etc.)
- Persistence layer for saving/loading motion spaces
- Visualization of user trajectories in reduced dimensions
- More sophisticated embedding models (e.g., transformer-based)
- User-to-user interactions and influence modeling
- Temporal decay of motion values
