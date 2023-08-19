use std::collections::HashMap;
use crate::models::TokenOperators;
use crate::space::space_generator::Space;
use super::Token;
pub mod space_generator;
pub mod space_calculator;

pub trait SpaceGenerator {
    fn new<T: TokenOperators>(tokens: T, words_of_interests: Option<Vec<String>>) -> Self;
    fn set_space_name(&mut self, name: String);
    fn find(&self, words_of_interests: &Vec<String>) -> Vec<Token>;
    fn get_center(&self) -> Vec<f64>;
    fn get_random_tokens(&self, num: i64, random_seed: i64) -> Vec<Token>;
    fn print_summary(&self);
}

pub trait SpaceCalculator {
    fn new() -> Self;
    fn bias_calculate(&self, random_space: Space, compare_space: Vec<Space>) -> HashMap<String, f64>;
}