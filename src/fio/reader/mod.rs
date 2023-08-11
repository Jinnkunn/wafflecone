pub mod conceptx;

use crate::models::Line;
use crate::models::Token;

pub trait Reader {
    fn new() -> Self;
    fn read(&self, path: &str) -> Vec<Line>;
}