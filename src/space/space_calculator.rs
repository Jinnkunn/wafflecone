use std::collections::HashMap;
use crate::space::space_generator::Space;
use crate::space::{SpaceCalculator, SpaceGenerator};

pub struct Calculator{
    similarities: HashMap<String, HashMap<String, f64>>,
}

impl SpaceCalculator for Calculator {
    fn new(random_space: Space, compare_space: Vec<Space>) -> Calculator {
        let mut bias_dict: HashMap<String, HashMap<String, f64>> = HashMap::new();

        for one_compare_space in compare_space {
            let one_compare_space_center = one_compare_space.get_center();
            // calculate the cosine similarity between one_compare_space_center and all
            // tokens in random_space

            let relationship = random_space.tokens.iter().map(|token| {
                let similarity = cos_similarity(&token.embedding, &one_compare_space_center);
                (token.word.clone(), similarity)
            }).collect::<HashMap<String, f64>>();

            println!("relationship: {:?}", relationship);

            bias_dict.insert(one_compare_space.space_name, relationship);
        }

        Calculator{
            similarities: bias_dict,
        }
    }

    fn print(&self){
        // create a dictionary <String, f64> to store the bias
        println!("bias_dict: {:?}", self.similarities);
    }
}

pub fn cos_similarity(center1: &Vec<f64>, center2: &Vec<f64>) -> f64 {
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