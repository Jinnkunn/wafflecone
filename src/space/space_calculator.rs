use std::collections::HashMap;
use crate::space::space_generator::Space;
use crate::space::{SpaceCalculator, SpaceGenerator};

pub struct Calculator{
    similarities: HashMap<String, HashMap<String, f64>>,
    ideal_similarity: f64,
}

impl SpaceCalculator for Calculator {
    fn new(random_space: Space, compare_space: Vec<Space>) -> Calculator {
        assert!(compare_space.len() > 0, "compare_space should have at least one space");

        // Calculate the ideal center and ideal similarity
        let mut ideal_center: Vec<f64> = vec![0.0; compare_space[0].get_center().len()];
        for one_compare_space in compare_space.clone() {
            let one_compare_space_center = one_compare_space.get_center();
            ideal_center = ideal_center.iter().zip(one_compare_space_center.iter()).map(|(a, b)| a + b).collect();
        }
        ideal_center = ideal_center.iter().map(|a| a / compare_space.len() as f64).collect();

        assert_eq!(ideal_center.len(), compare_space[0].get_center().len());

        let ideal_similarity = cos_similarity(&compare_space[0].get_center(), &ideal_center);

        // calculate the normalized cosine similarity between one_compare_space_center and all
        let mut bias_dict: HashMap<String, HashMap<String, f64>> = HashMap::new();

        for one_compare_space in compare_space.clone() {
            let one_compare_space_center = one_compare_space.get_center();
            // calculate the cosine similarity between one_compare_space_center and all
            // tokens in random_space

            let relationship = random_space.tokens.iter().map(|token| {
                let similarity = cos_similarity(&token.embedding, &one_compare_space_center) / ideal_similarity;
                (token.word.clone(), similarity)
            }).collect::<HashMap<String, f64>>();

            bias_dict.insert(one_compare_space.space_name, relationship);
        }

        Calculator{
            similarities: bias_dict,
            ideal_similarity,
        }
    }

    fn bias_sum_average(&self) -> HashMap<String, f64> {
        let mut bias_sum_average: HashMap<String, f64> = HashMap::new();
        for (space_name, relationship) in self.similarities.iter() {
            let sum_average = relationship.iter().map(|(_, similarity)| similarity).sum::<f64>() / relationship.len() as f64;
            bias_sum_average.insert(space_name.clone(), sum_average);
        }
        bias_sum_average
    }

    fn bias_asb_sum_average(&self) -> HashMap<String, f64> {
        let mut bias_asb_sum_average: HashMap<String, f64> = HashMap::new();
        for (space_name, relationship) in self.similarities.iter() {
            let asb_sum_average = relationship.iter().map(|(_, similarity)| similarity.abs()).sum::<f64>() / relationship.len() as f64;
            bias_asb_sum_average.insert(space_name.clone(), asb_sum_average);
        }
        bias_asb_sum_average
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