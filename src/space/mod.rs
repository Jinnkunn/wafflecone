use crate::models::TokenOperators;
use super::Token;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

pub mod global_space_generator;
pub mod subspace_generator;

pub trait SpaceOperator {
    fn new<T: TokenOperators>(tokens: T, words_of_interests: Option<Vec<String>>) -> Self;
    fn find(&self, words_of_interests: Vec<String>) -> Vec<Token>;
    fn get_center(&self) -> Vec<f64>;
    fn get_random_tokens(&self, num: i64, random_seed: i64) -> Vec<Token>;
    fn print_summary(&self);
}

pub(crate) fn get_center(tokens: Vec<Token>) -> Vec<f64> {
    let mut sum_of_embeddings: Vec<f64> = Vec::new();
    for token in tokens.clone() {
        if sum_of_embeddings.len() == 0 {
            for embedding in token.embedding.clone() {
                sum_of_embeddings.push(embedding);
            }
        } else {
            for i in 0..sum_of_embeddings.len() {
                sum_of_embeddings[i] += token.embedding[i];
            }
        }
    }
    let mut center: Vec<f64> = Vec::new();
    for embedding in sum_of_embeddings {
        center.push(embedding / tokens.len() as f64);
    }
    center
}

pub(crate) fn find(tokens: Vec<Token>, target_words: Vec<String>) -> Vec<Token> {
    let mut find_tokens:Vec<Token> = Vec::new();
    for token in tokens.clone() {
        if target_words.contains(&token.word) {
            find_tokens.push(token);
        }
    }
    find_tokens
}

pub(crate) fn get_random_tokens(tokens: Vec<Token>, num: i64, random_seed: i64) -> Vec<Token> {
    let total_tokens = tokens.len();
    let mut random_tokens: Vec<Token> = Vec::new();
    // set random seed, get random tokens without replacement
    let mut rng = ChaCha8Rng::seed_from_u64(random_seed as u64);
    let mut random_indices: Vec<usize> = (0..total_tokens).collect();
    random_indices.shuffle(&mut rng);
    for i in 0..num {
        random_tokens.push(tokens[random_indices[i as usize]].clone());
    }

    random_tokens
}