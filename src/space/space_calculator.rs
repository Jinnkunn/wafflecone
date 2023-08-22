use std::collections::HashMap;
use crate::space::space_generator::Space;
use crate::space::{SpaceCalculator, SpaceGenerator};

pub struct Calculator{
    similarities: HashMap<String, HashMap<String, f64>>,
    ideal_similarity: Vec<f64>,
}

impl SpaceCalculator for Calculator {
    fn new(random_space: Space, compare_space: Vec<Space>) -> Calculator {
        let mut bias_dict: HashMap<String, HashMap<String, f64>> = HashMap::new();

        for one_compare_space in compare_space.clone() {
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

        // calculate the mean of all the center of compare_space with the size of compare_space[0].len
        let mut ideal_similarity: Vec<f64> = vec![0.0; compare_space[0].get_center().len()];
        for one_compare_space in compare_space.clone() {
            let one_compare_space_center = one_compare_space.get_center();
            ideal_similarity = ideal_similarity.iter().zip(one_compare_space_center.iter()).map(|(a, b)| a + b).collect();
        }
        ideal_similarity = ideal_similarity.iter().map(|a| a / compare_space.len() as f64).collect();

        assert_eq!(ideal_similarity.len(), compare_space[0].get_center().len());

        Calculator{
            similarities: bias_dict,
            ideal_similarity,
        }
    }

    fn bias_asb_sum_average(&self) -> f64 {
        todo!()
    }

    fn bias_sum_average(&self) -> f64 {
        todo!()
    }

    fn print(&self){
        // create a dictionary <String, f64> to store the bias
        println!("bias_dict: {:?}", self.similarities);
        println!("ideal_similarity: {:?}", self.ideal_similarity)
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