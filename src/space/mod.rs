pub mod space_generator;
pub mod space_calculator;
pub mod seeds;

use std::collections::HashMap;
use super::Token;
use crate::embedding::models::TokenOperators;
use crate::space::space_generator::Space;
use seeds::SubspaceSeeds;

pub trait SpaceGenerator {
    fn new<T: TokenOperators>(items: T, words_of_interests: Option<SubspaceSeeds>, pca_dimension: Option<usize>) -> Space;
    fn set_space_name(&mut self, name: String);
    fn find(&self, words_of_interests: &Vec<String>) -> Vec<Token>;
    fn get_center(&self) -> Vec<f64>;
    fn get_std(&self) -> Vec<f64>;
    fn get_random_tokens(&self, num: i64, random_seed: i64, exclude: Option<Vec<String>>) -> Vec<Token>;
    fn print_summary(&self);
}

pub trait SpaceCalculator {
    fn new(random_space: Space, compare_space: Vec<Space>) -> Self;
    fn get_bias_result(&self, normalized: Option<bool>) -> HashMap<String, f64>;
    fn get_normalized_similarity_summary(&self) -> HashMap<String, f64>;
    fn get_similarity_summary(&self) -> HashMap<String, f64>;
    fn print(&self);
}