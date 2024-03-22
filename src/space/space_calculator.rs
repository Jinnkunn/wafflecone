use std::collections::HashMap;
use pyo3::{pyclass, pymethods};
use crate::fio::writer::WriterOperator;
use crate::space::space_generator::Space;
use crate::space::SpaceCalculator;

#[pyclass]
pub struct Calculator{
    pub(crate) similarity: HashMap<String, HashMap<String, f64>>,
    pub(crate) similarity_softmax : HashMap<String, HashMap<String, f64>>,
}

impl SpaceCalculator for Calculator {
    fn new(bias_free_token_space: Space, bias_group_spaces: Vec<Space>) -> Calculator {
        // e.g., random space is the space without all the gender words
        // compare space is a list of space which only contains the gender words
        assert!(bias_group_spaces.len() > 0, "compare_space should have at least one space");

        // Word: <Group: similarity>
        let mut similarity_dict: HashMap<String, HashMap<String, f64>> = HashMap::new();

        for one_bias_free_token in &bias_free_token_space.tokens {

            // find the ideal similarity from ideal_similarities, which the space is one_compare_space
            let mut relationship: HashMap<String, f64> = HashMap::new();

            for one_bias_group_space in &bias_group_spaces {
                let similarity = cos_similarity(&one_bias_free_token.embedding, &one_bias_group_space.space_center);
                relationship.insert(one_bias_group_space.space_name.clone(), similarity);
            }

            similarity_dict.insert(one_bias_free_token.token_id.clone(), relationship);
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
    let max_similarity = similarity_dict.iter().map(|(_, relationship)| relationship.iter().map(|(_, similarity)| similarity).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let exp_sum: f64 = similarity_dict.iter().map(|(_, relationship)| relationship.iter().map(|(_, similarity)| (similarity - max_similarity).exp()).sum::<f64>()).sum::<f64>();
    for (space_name, relationship) in similarity_dict.iter() {
        let mut relationship_softmax: HashMap<String, f64> = HashMap::new();
        for (token, similarity) in relationship.iter() {
            relationship_softmax.insert(token.clone(), (similarity - max_similarity).exp() / exp_sum);
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

// fn get_entropy(similarity_dict: &HashMap<String, HashMap<String, f64>>) -> HashMap<String, f64> {
//     let mut entropy: HashMap<String, f64> = HashMap::new();
//     for (space_name, relationship) in similarity_dict.iter() {
//         let mut entropy_sum: f64 = 0.0;
//         for (_, similarity) in relationship.iter() {
//             entropy_sum += similarity * similarity.log2();
//         }
//         entropy.insert(space_name.clone(), entropy_sum);
//     }
//     entropy
// }

fn get_entropy(similarity_dict: &HashMap<String, HashMap<String, f64>>) -> HashMap<String, f64> {
    let mut entropy: HashMap<String, f64> = HashMap::new();
    for (space_name, relationship) in similarity_dict.iter() {
        let mut entropy_sum: f64 = 0.0;
        for (_, similarity) in relationship.iter() {
            if *similarity > 0.0 {
                // Avoids taking log2(0), directly addresses potential -inf result
                let product = similarity * similarity.log2();
                if product.is_finite() {
                    entropy_sum += product;
                } else {
                    // Handle non-finite product case here (could log, panic, etc.)
                    eprintln!("Non-finite product encountered for similarity value: {}", similarity);
                }
            }
            // Optionally, handle the similarity == 0 case explicitly if needed
        }
        if entropy_sum.is_finite() {
            entropy.insert(space_name.clone(), -entropy_sum);
        } else {
            // Handle non-finite entropy_sum case here
            eprintln!("Non-finite entropy sum encountered for space: {}", space_name);
        }
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