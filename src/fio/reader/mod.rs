pub mod conceptx;

use crate::embedding::models::Line;

pub trait Reader {
    fn new() -> Self;
    fn read(&self, path: &str, user_friendly: bool) -> Vec<Line>;
}