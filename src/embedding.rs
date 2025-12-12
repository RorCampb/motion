use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::motion_core::{MotionEntry, MotionPost, MotionUser};
use crate::math::VecN;

pub const EMBEDDING_DIM: usize = 128;

pub fn embed_post(text: &str) -> VecN {
    let mut data = vec![0.0_f32; EMBEDDING_DIM];
    
    add_text_features(&mut data, text);
    let mut v = VecN::new(data);

    if v.norm() > 0.0 {
        let _ = v.normalize();
    }

    v
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    // simple FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn hash_str(s: &str) -> u64 {
    hash_bytes(s.as_bytes())
}

fn add_text_features(bucket: &mut [f32], text: &str) {
    if bucket.is_empty() {
        return;
    }
    let dim = bucket.len() as u64;
    let lower = text.to_lowercase();

    // 1) token-level bag of words
    for token in lower.split_whitespace() {
        let h = hash_str(token);
        let idx = (h % dim) as usize;
        bucket[idx] += 1.0;
    }

    // 2) character 3-grams
    let chars: Vec<char> = lower.chars().collect();
    for window in chars.windows(3) {
        let mut buf = [0u8; 12];
        let mut len = 0;

        for &ch in window {
            let mut tmp = [0u8; 4];
            let encoded = ch.encode_utf8(&mut tmp);
            let bytes = encoded.as_bytes();
            let end = (len + bytes.len()).min(buf.len());
            buf[len..end].copy_from_slice(&bytes[..(end - len)]);
            len = end;
        }

        let h = hash_bytes(&buf[..len]);
        let idx = (h % dim) as usize;
        bucket[idx] += 0.3;
    }
}

