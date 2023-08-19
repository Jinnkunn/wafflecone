use crate::models::TokenOperators;
use super::Token;
pub mod space_generator;
// pub mod subspace_generator;

pub trait SpaceOperator {
    fn new<T: TokenOperators>(tokens: T, words_of_interests: Option<Vec<String>>) -> Self;
    fn find(&self, words_of_interests: Vec<String>) -> Vec<Token>;
    fn get_center(&self) -> Vec<f64>;
    fn get_random_tokens(&self, num: i64, random_seed: i64) -> Vec<Token>;
    fn print_summary(&self);
}