use crate::models::TokenOperators;
use super::Token;
pub mod space_generator;
// pub mod subspace_generator;

pub trait SpaceOperator {
    fn new<T: TokenOperators>(tokens: T, words_of_interests: Option<Vec<String>>) -> Self;
    fn find(&self, words_of_interests: &Vec<String>) -> Vec<Token>;
    fn get_center(&self) -> Vec<f64>;
    fn get_random_tokens(&self, num: i64, random_seed: i64) -> Vec<Token>;
    fn print_summary(&self);
}

pub(crate) fn cos_similarity(center1: &Vec<f64>, center2: &Vec<f64>) -> f64 {
    // calculate the cosine similarity between two vectors
    let mut dot_product: f64 = 0.0;
    let mut norm1: f64 = 0.0;
    let mut norm2: f64 = 0.0;
    for i in 0..center1.len() {
        dot_product += center1[i] * center2[i];
        norm1 += center1[i] * center1[i];
        norm2 += center2[i] * center2[i];
    }
    dot_product / (norm1.sqrt() * norm2.sqrt())
}