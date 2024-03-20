use std::collections::HashMap;
use pyo3::{pyclass, pymethods};
use crate::fio::writer::WriterOperator;
use crate::space::space_generator::Space;
use crate::space::{SpaceCalculator, SpaceGenerator};

#[pyclass]
pub struct Calculator{
    pub(crate) similarity: HashMap<String, HashMap<String, f64>>,
    pub(crate) similarity_softmax : HashMap<String, HashMap<String, f64>>,
}

impl SpaceCalculator for Calculator {
    fn new(random_space: Space, compare_space: Vec<Space>) -> Calculator {
        // e.g., random space is the space without all the gender words
        // compare space is a list of space which only contains the gender words
        assert!(compare_space.len() > 0, "compare_space should have at least one space");

        let mut similarity_dict: HashMap<String, HashMap<String, f64>> = HashMap::new();

        for one_compare_space in compare_space.clone() {
            let one_compare_space_center = one_compare_space.get_center();

            // find the ideal similarity from ideal_similarities, which the space is one_compare_space
            let mut relationship: HashMap<String, f64> = HashMap::new();

            for one_random_token in random_space.tokens.iter() {
                let one_random_token_embedding = &one_random_token.embedding;
                let similarity = cos_similarity(&one_compare_space_center, one_random_token_embedding);
                relationship.insert(one_random_token.token_id.clone(), similarity);
            }

            similarity_dict.insert(one_compare_space.space_name.clone(), relationship);
        }

        Calculator{
            similarity: similarity_dict.clone(),
            similarity_softmax: get_similarity_softmax(&similarity_dict),
        }
    }

    fn get_bias_result(&self, normalized: Option<bool>) -> HashMap<String, f64> {
        if normalized.unwrap_or(true) {
            get_entropy(&self.similarity_softmax)
        } else {
            get_entropy(&self.similarity)
        }
    }

    // One space, one similarity
    fn get_normalized_similarity_summary(&self) -> HashMap<String, f64> {
        let mut similarity: HashMap<String, f64> = HashMap::new();
        for (space_name, relationship) in self.similarity.iter() {
            let mean = relationship.iter().map(|(_, similarity)| similarity.abs()).sum::<f64>() / relationship.len() as f64;
            similarity.insert(space_name.clone(), mean);
        }
        similarity
    }

    fn get_similarity_summary(&self) -> HashMap<String, f64> {
        let mut similarity: HashMap<String, f64> = HashMap::new();
        for (space_name, relationship) in self.similarity.iter() {
            let mean = relationship.iter().map(|(_, similarity)| similarity.abs()).sum::<f64>() / relationship.len() as f64;
            similarity.insert(space_name.clone(), mean);
        }
        similarity
    }

    fn print(&self){
        println!("similarity_dict: {:?}", self.similarity);
    }
}

fn get_similarity_softmax(similarity_dict: &HashMap<String, HashMap<String, f64>>) -> HashMap<String, HashMap<String, f64>> {
    let mut similarity_softmax: HashMap<String, HashMap<String, f64>> = HashMap::new();
    let exp_sum: f64 = similarity_dict.iter().map(|(_, relationship)| relationship.iter().map(|(_, similarity)| similarity.exp()).sum::<f64>()).sum();

    for (space_name, relationship) in similarity_dict.iter() {
        let mut relationship_softmax: HashMap<String, f64> = HashMap::new();
        for (token, similarity) in relationship.iter() {
            relationship_softmax.insert(token.clone(), similarity.exp() / exp_sum);
        }
        similarity_softmax.insert(space_name.clone(), relationship_softmax);
    }
    similarity_softmax
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

fn get_entropy(similarity_dict: &HashMap<String, HashMap<String, f64>>) -> HashMap<String, f64> {
    let mut entropy: HashMap<String, f64> = HashMap::new();
    for (space_name, relationship) in similarity_dict.iter() {
        let mut entropy_sum: f64 = 0.0;
        for (_, similarity) in relationship.iter() {
            entropy_sum += similarity * similarity.log2();
        }
        entropy.insert(space_name.clone(), -entropy_sum);
    }
    entropy
}

// Expose to Python
#[pymethods]
impl Calculator {
    pub(crate) fn get_bias(&self, normalize: Option<bool>) -> HashMap<String, f64> {
        // if normalize is true, or normalize is None, then use normalized similarity
        if normalize.unwrap_or(true) {
            println!("use normalized similarity");
            self.get_bias_result(Option::from(true))
        } else {
            println!("use similarity");
            self.get_bias_result(Option::from(false))
        }
    }

    pub(crate) fn save_summary(&self, path: Option<&str>) {
        self.write(path.unwrap_or("./"), false);
    }
}