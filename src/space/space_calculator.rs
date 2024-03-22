use pyo3::{pyclass, pymethods};
use crate::fio::writer::WriterOperator;
use crate::space::space_generator::Space;
use crate::space::SpaceCalculator;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum SimilarityType {
    TokenToGroup,
    GroupToToken
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SimilarityItem {
    pub(crate) name: String,
    pub(crate) value: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Similarity {
    pub(crate) name: String,
    pub(crate) similarity_type: SimilarityType,
    pub(crate) similarity: Vec<SimilarityItem>,
    pub(crate) softmax: Vec<SimilarityItem>,
}

#[pyclass]
#[derive(Debug, Deserialize, Serialize)]
pub struct Bias {
    pub(crate) name: String,
    pub(crate) bias: f64,
}

#[pyclass]
pub struct Calculator{
    pub(crate) similarity_token_to_group: Vec<Similarity>,
    pub(crate) similarity_group_to_token: Vec<Similarity>,
}

impl SpaceCalculator for Calculator {
    fn new(bias_free_token_space: Space, bias_group_spaces: Vec<Space>) -> Calculator {
        // e.g., random space is the space without all the gender words
        // compare space is a list of space which only contains the gender words
        assert!(bias_group_spaces.len() > 0, "compare_space should have at least one space");

        let mut token_to_group_dict: Vec<Similarity> = Vec::new();
        let mut group_to_token_dict: Vec<Similarity> = Vec::new();

        let mut group_to_token_map: HashMap<String, Vec<SimilarityItem>> = HashMap::new();
        for one_bias_group in &bias_group_spaces {
            let one_bias_group_name = one_bias_group.space_name.clone();
            group_to_token_map.insert(one_bias_group_name, Vec::new());
        }

        for one_bias_free_token in &bias_free_token_space.tokens {
            // find the ideal similarity from ideal_similarities, which the space is one_compare_space
            let mut relationship_token_to_group: Vec::<SimilarityItem> = Vec::new();
            for one_bias_group_space in &bias_group_spaces {
                let similarity = cos_similarity(&one_bias_free_token.embedding, &one_bias_group_space.space_center);
                relationship_token_to_group.push(
                    SimilarityItem{
                        name: one_bias_group_space.space_name.clone(),
                        value: similarity
                    }
                );

                group_to_token_map.get_mut(&one_bias_group_space.space_name).unwrap().push(
                    SimilarityItem{
                        name: one_bias_free_token.word.clone(),
                        value: similarity
                    }
                );
            }

            token_to_group_dict.push(
                Similarity{
                    name: one_bias_free_token.word.clone(),
                    similarity_type: SimilarityType::TokenToGroup,
                    softmax: get_similarity_softmax(&relationship_token_to_group),
                    similarity: relationship_token_to_group
                }
            );
        }

        // group_to_token_map to vectors
        for (group_name, similarity_items) in group_to_token_map.iter() {
            group_to_token_dict.push(
                Similarity{
                    name: group_name.clone(),
                    similarity_type: SimilarityType::GroupToToken,
                    softmax: get_similarity_softmax(similarity_items),
                    similarity: similarity_items.clone()
                }
            );
        }

        Calculator{
            similarity_token_to_group: token_to_group_dict.clone(),
            similarity_group_to_token: group_to_token_dict.clone(),
        }
    }

}

fn get_similarity_softmax(similarity_dict: &Vec<SimilarityItem>) -> Vec<SimilarityItem> {
    let mut similarity_softmax: Vec<SimilarityItem> = Vec::new();
    let max_similarity = similarity_dict.iter().map(|one_similarity| one_similarity.value).fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = similarity_dict.iter().map(|one_similarity| (one_similarity.value - max_similarity).exp()).sum();

    for one_similarity in similarity_dict.iter() {
        let mut new_one_similarity = one_similarity.clone();
        new_one_similarity.value = (one_similarity.value - max_similarity).exp() / exp_sum;
        similarity_softmax.push(new_one_similarity);
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

fn get_entropy(similarity_dict: &Vec<Similarity>) -> Vec<Bias> {
    let mut entropy: Vec<Bias> = Vec::new();
    for one_similarity in similarity_dict.iter() {
        let mut entropy_sum: f64 = 0.0;
        for one_similarity_item in one_similarity.similarity.iter() {
            entropy_sum += one_similarity_item.value * one_similarity_item.value.log2();
        }
        entropy.push(
            Bias{
                name: one_similarity.name.clone(),
                bias: -entropy_sum
            }
        );
    }
    entropy
}

// Expose to Python
#[pymethods]
impl Calculator {
    fn get_bias_per_token(&self) -> HashMap<String, f64> {
        get_entropy(&self.similarity_token_to_group).iter().map(|bias| (bias.name.clone(), bias.bias)).collect()
    }

    fn get_bias_per_group(&self) -> HashMap<String, f64> {
        let mut bias_group: HashMap<String, f64> = HashMap::new();
        for one_similarity in &self.similarity_group_to_token {
            let mut total: f64 = 0.0;
            for one_similarity_item in &one_similarity.similarity {
                total += total + one_similarity_item.value;
            }
            bias_group.insert(one_similarity.name.clone(), total / one_similarity.similarity.len() as f64);
        }
        bias_group
    }

    fn get_report_per_token(&self) -> HashMap<String, HashMap<String, f64>> {
        let mut bias_token_to_group: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for one_similarity in &self.similarity_token_to_group {
            let mut bias_token_to_group_inner: HashMap<String, f64> = HashMap::new();
            for one_similarity_item in &one_similarity.similarity {
                bias_token_to_group_inner.insert(one_similarity_item.name.clone(), one_similarity_item.value);
            }
            bias_token_to_group.insert(one_similarity.name.clone(), bias_token_to_group_inner);
        }
        bias_token_to_group
    }

    fn get_report_per_group(&self) -> HashMap<String, HashMap<String, f64>> {
        // &self.similarity_group_to_token to HashMap
        let mut bias_group_to_token: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for one_similarity in &self.similarity_group_to_token {
            let mut bias_group_to_token_inner: HashMap<String, f64> = HashMap::new();
            for one_similarity_item in &one_similarity.similarity {
                bias_group_to_token_inner.insert(one_similarity_item.name.clone(), one_similarity_item.value);
            }
            bias_group_to_token.insert(one_similarity.name.clone(), bias_group_to_token_inner);
        }
        bias_group_to_token
    }

    fn get_bias(&self) -> f64 {
        get_entropy(&self.similarity_token_to_group).iter().map(|bias| bias.bias).sum::<f64>() / self.similarity_token_to_group.len() as f64
    }

    pub(crate) fn save_summary(&self, path: Option<&str>) {
        self.write(path.unwrap_or("./"), false);
    }
}