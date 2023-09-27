use std::collections::HashMap;
use pyo3::{pyclass, pymethods};
use crate::fio::writer::WriterOperator;
use crate::space::space_generator::Space;
use crate::space::{SpaceCalculator, SpaceGenerator};

#[pyclass]
pub struct Calculator{
    pub(crate) bias: HashMap<String, HashMap<String, f64>>,
    pub(crate) ideal_similarity: HashMap<String, f64>, // space name: ideal similarity
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

        let mut ideal_similarities:HashMap<String, f64>  = HashMap::new();

        for one_compare_space in compare_space.clone() {
            let one_compare_space_center = one_compare_space.get_center();
            let ideal_similarity = cos_similarity(&one_compare_space_center, &ideal_center);
            ideal_similarities.insert(one_compare_space.space_name, ideal_similarity);
        }

        // calculate the normalized cosine similarity between one_compare_space_center and all
        let mut bias_dict: HashMap<String, HashMap<String, f64>> = HashMap::new();

        for one_compare_space in compare_space.clone() {
            let one_compare_space_center = one_compare_space.get_center();

            // find the ideal similarity from ideal_similarities, which the space is one_compare_space
            let target_similarity = ideal_similarities.get(&one_compare_space.space_name).unwrap();

            let relationship = random_space.tokens.iter().map(|token| {
                let bias = cos_similarity(&token.embedding, &one_compare_space_center) / target_similarity;
                (token.word.clone(), bias)
            }).collect::<HashMap<String, f64>>();

            bias_dict.insert(one_compare_space.space_name, relationship);
        }

        Calculator{
            bias: bias_dict,
            ideal_similarity: ideal_similarities,
        }
    }


    fn bias_sum_average_calculator(&self) -> HashMap<String, f64> {
        let mut bias_sum_average: HashMap<String, f64> = HashMap::new();
        for (space_name, relationship) in self.bias.iter() {
            // use each similarity - ideal_similarity
            // then add them together
            let sum_average = relationship.iter().map(|(_, similarity)| similarity).sum::<f64>() / relationship.len() as f64;
            bias_sum_average.insert(space_name.clone(), sum_average);
        }
        bias_sum_average
    }

    fn bias_asb_sum_average_calculator(&self) -> HashMap<String, f64> {
        let mut bias_asb_sum_average: HashMap<String, f64> = HashMap::new();
        for (space_name, relationship) in self.bias.iter() {
            // use each similarity - ideal_similarity, get the absolute value
            // then add them together
            let asb_sum_average = relationship.iter().map(|(_, similarity)| similarity.abs()).sum::<f64>() / relationship.len() as f64;
            bias_asb_sum_average.insert(space_name.clone(), asb_sum_average);
        }
        bias_asb_sum_average
    }

    fn print(&self){
        // create a dictionary <String, f64> to store the bias
        println!("bias_dict: {:?}", self.bias);
        println!("ideal_similarity: {:?}", self.ideal_similarity)
    }
}

fn cos_similarity(center1: &Vec<f64>, center2: &Vec<f64>) -> f64 {
    assert_eq!(center1.len(), center2.len(), "center1 and center2 should have the same length");
    // calculate the cosine similarity between two vectors
    let dot_product_result = dot_product(center1, center2);
    let center1_norm = dot_product(center1, center1).sqrt();
    let center2_norm = dot_product(center2, center2).sqrt();
    dot_product_result / (center1_norm * center2_norm)

}

fn dot_product(center1: &Vec<f64>, center2: &Vec<f64>) -> f64 {
    // calculate the dot product between two vectors
    let mut dot_product: f64 = 0.0;
    for i in 0..center1.len() {
        dot_product += center1[i] * center2[i];
    }
    dot_product
}


// Expose to Python
#[pymethods]
impl Calculator {
    pub(crate) fn bias_sum_average(&self) -> HashMap<String, f64> {
        self.bias_sum_average_calculator()
    }

    pub(crate) fn bias_asb_sum_average(&self) -> HashMap<String, f64> {
        self.bias_asb_sum_average_calculator()
    }

    pub(crate) fn save_summary(&self, path: Option<&str>) {
        self.write(path.unwrap_or("./"), false);
    }
}