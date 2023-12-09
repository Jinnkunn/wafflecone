use std::collections::HashMap;
use pyo3::{pyclass, pymethods};
use crate::fio::writer::WriterOperator;
use crate::space::space_generator::Space;
use crate::space::{SpaceCalculator, SpaceGenerator};

#[pyclass]
pub struct Calculator{
    pub(crate) bias: HashMap<String, HashMap<String, f64>>,
    pub(crate) bias_normalized: HashMap<String, HashMap<String, f64>>,
    pub(crate) ideal_similarity: HashMap<String, f64>, // space name: ideal similarity
    pub(crate) average_similarity: f64, // average similarity of all spaces, will be used to normalize the bias
}

impl SpaceCalculator for Calculator {
    fn new(random_space: Space, compare_space: Vec<Space>, whole_space: Space) -> Calculator {
        // e.g., random space is the space without all the gender words
        // compare space is a list of space which only contains the gender words
        assert!(compare_space.len() > 0, "compare_space should have at least one space");

        let ideal_center = find_ideal_center(&compare_space);
        let ideal_similarities = find_ideal_similarity(&compare_space, &ideal_center);
        let average_similarity = find_average_similarity(&ideal_center, whole_space);

        // calculate the normalized cosine similarity between one_compare_space_center and all
        let mut bias_dict: HashMap<String, HashMap<String, f64>> = HashMap::new();
        let mut bias_normalized_dict: HashMap<String, HashMap<String, f64>> = HashMap::new();

        for one_compare_space in compare_space.clone() {
            let one_compare_space_center = one_compare_space.get_center();

            // find the ideal similarity from ideal_similarities, which the space is one_compare_space
            let target_similarity = ideal_similarities.get(&one_compare_space.space_name).unwrap();

            let mut relationship: HashMap<String, f64> = HashMap::new();
            let mut relationship_normalized: HashMap<String, f64> = HashMap::new();

            for one_random_token in random_space.tokens.iter() {
                let one_random_token_embedding = &one_random_token.embedding;
                let similarity = cos_similarity(&one_compare_space_center, one_random_token_embedding);
                relationship.insert(format!("{}:{}:{}", one_random_token.word.clone(), one_random_token.position, one_random_token.line_num), similarity - target_similarity);
                relationship_normalized.insert(format!("{}:{}:{}", one_random_token.word.clone(), one_random_token.position, one_random_token.line_num), (similarity - target_similarity) / average_similarity);
            }

            bias_dict.insert(one_compare_space.space_name.clone(), relationship);
            bias_normalized_dict.insert(one_compare_space.space_name, relationship_normalized);
        }

        Calculator{
            bias: bias_dict,
            bias_normalized: bias_normalized_dict,
            ideal_similarity: ideal_similarities,
            average_similarity: average_similarity,
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

    fn bias_normalized_sum_average_calculator(&self) -> HashMap<String, f64> {
        let mut bias_sum_average: HashMap<String, f64> = HashMap::new();
        for (space_name, relationship_normalized) in self.bias_normalized.iter() {
            // use each similarity - ideal_similarity
            // then add them together
            let sum_average = relationship_normalized.iter().map(|(_, similarity)| similarity).sum::<f64>() / relationship_normalized.len() as f64;
            bias_sum_average.insert(space_name.clone(), sum_average);
        }
        bias_sum_average
    }

    fn bias_normalized_asb_sum_average_calculator(&self) -> HashMap<String, f64> {
        let mut bias_asb_sum_average: HashMap<String, f64> = HashMap::new();
        for (space_name, relationship_normalized) in self.bias_normalized.iter() {
            // use each similarity - ideal_similarity, get the absolute value
            // then add them together
            let asb_sum_average = relationship_normalized.iter().map(|(_, similarity)| similarity.abs()).sum::<f64>() / relationship_normalized.len() as f64;
            bias_asb_sum_average.insert(space_name.clone(), asb_sum_average);
        }
        bias_asb_sum_average
    }

    fn print(&self){
        // create a dictionary <String, f64> to store the bias
        println!("bias_dict: {:?}", self.bias);
        println!("bias_normalized_dict: {:?}", self.bias_normalized);
        println!("ideal_similarity: {:?}", self.ideal_similarity);
        println!("average_similarity: {:?}", self.average_similarity);
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

fn find_ideal_center(compare_space: &Vec<Space>) -> Vec<f64> {
    let mut ideal_center: Vec<f64> = vec![0.0; compare_space[0].get_center().len()];

    for one_compare_space in compare_space{
        let one_compare_space_center = one_compare_space.get_center();
        ideal_center = ideal_center.iter().zip(one_compare_space_center.iter()).map(|(a, b)| a + b).collect();
    }
    ideal_center = ideal_center.iter().map(|a| a / compare_space.len() as f64).collect();

    assert_eq!(ideal_center.len(), compare_space[0].get_center().len());
    ideal_center
}

fn find_ideal_similarity(compare_space: &Vec<Space>, ideal_center: &Vec<f64>) -> HashMap<String, f64>{
    let mut ideal_similarities:HashMap<String, f64>  = HashMap::new();

    for one_compare_space in compare_space {
        let one_compare_space_center = one_compare_space.get_center();
        let ideal_similarity = cos_similarity(&one_compare_space_center, ideal_center);
        ideal_similarities.insert(one_compare_space.space_name.clone(), ideal_similarity);
    }

    println!("ideal_similarity: {:?}", ideal_similarities);

    ideal_similarities
}

fn find_average_similarity(ideal_center: &Vec<f64>, whole_space: Space) -> f64 {
    // find all the similarity between ideal_center and all the tokens in whole_space
    let mut similarities: Vec<f64> = Vec::new();
    for one_token in whole_space.tokens.iter() {
        let one_token = &one_token.embedding;
        let similarity = cos_similarity(&ideal_center, one_token);
        similarities.push(similarity);
    }

    // find the average similarity
    let average_similarity = similarities.iter().sum::<f64>() / similarities.len() as f64;
    println!("average_similarity: {:?}", average_similarity);
    average_similarity

}

// Expose to Python
#[pymethods]
impl Calculator {
    pub(crate) fn avg_bias(&self, normalize: Option<bool>) -> HashMap<String, f64> {
        // if normalize is true, or normalize is None, then use normalized bias
        if normalize.unwrap_or(true) {
            println!("use normalized bias");
            self.bias_normalized_sum_average_calculator()
        } else {
            println!("use bias");
            self.bias_sum_average_calculator()
        }
    }

    pub(crate) fn avg_asb_bias(&self, normalize: Option<bool>) -> HashMap<String, f64> {
        // if normalize is true, or normalize is None, then use normalized bias
        if normalize.unwrap_or(true) {
            println!("use normalized bias");
            self.bias_normalized_asb_sum_average_calculator()
        } else {
            println!("use bias");
            self.bias_asb_sum_average_calculator()
        }
    }

     pub(crate) fn get_all_bias(&self) -> Vec<HashMap<String, HashMap<String, f64>>> {
        let mut bias: Vec<HashMap<String, HashMap<String, f64>>> = Vec::new();
        bias.push(self.bias.clone());
        bias.push(self.bias_normalized.clone());

        bias
    }

    pub(crate) fn get_all_bias_value(&self) -> Vec<HashMap<String, Vec<f64>>> {
        let mut bias: Vec<HashMap<String, Vec<f64>>> = Vec::new();
        let mut bias_dict: HashMap<String, Vec<f64>> = HashMap::new();
        let mut bias_normalized_dict: HashMap<String, Vec<f64>> = HashMap::new();

        for (space_name, bias_value) in self.bias.iter() {
            let mut bias_value_vec: Vec<f64> = Vec::new();
            for (_, value) in bias_value.iter() {
                bias_value_vec.push(*value);
            }
            bias_dict.insert(space_name.clone(), bias_value_vec);
        }

        for (space_name, bias_normalized_value) in self.bias_normalized.iter() {
            let mut bias_normalized_value_vec: Vec<f64> = Vec::new();
            for (_, value) in bias_normalized_value.iter() {
                bias_normalized_value_vec.push(*value);
            }
            bias_normalized_dict.insert(space_name.clone(), bias_normalized_value_vec);
        }

        bias.push(bias_dict);
        bias.push(bias_normalized_dict);

        bias

    }

    pub(crate) fn save_summary(&self, path: Option<&str>) {
        self.write(path.unwrap_or("./"), false);
    }
}