pub mod seeds;
pub mod space_generator;

use super::Token;
use crate::embedding::models::TokenOperators;
use seeds::SubspaceSeeds;

pub trait SpaceGenerator {
    fn new<T: TokenOperators>(items: T, words_of_interests: Option<SubspaceSeeds>, pca_dimension: Option<usize>) -> Self;
    fn set_space_name(&mut self, name: String);
    fn find(&self, subspace_seed: &SubspaceSeeds) -> Vec<Token>;
    fn get_center(&self) -> Vec<f64>;
    fn get_std(&self) -> Vec<f64>;
    fn get_neutral_tokens(&self, exclude: Vec<String>) -> Vec<Token>;
    fn print_summary(&self);
}
