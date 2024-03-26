use crate::analyizer::SpaceCalculator;
use crate::fio::writer::WriterOperator;
use crate::space::space_generator::Space;
use pyo3::{pyclass, pymethods, FromPyObject, PyAny, PyErr, PyResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum SimilarityType {
    TokenToGroup,
    GroupToToken,
}

impl<'a> FromPyObject<'a> for SimilarityType {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        if let Ok(string) = obj.extract::<&str>() {
            match string {
                "TokenToGroup" => Ok(SimilarityType::TokenToGroup),
                "GroupToToken" => Ok(SimilarityType::GroupToToken),
                _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid enum variant: {}",
                    string
                ))),
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Invalid type for enum conversion",
            ))
        }
    }
}

#[derive(Debug, Clone, FromPyObject)]
pub struct SimilarityItem {
    pub(crate) name: String,
    pub(crate) value: f64,
}

#[derive(Debug, Clone, FromPyObject)]
pub struct Similarity {
    pub(crate) name: String,
    pub(crate) similarity_type: SimilarityType,
    pub(crate) similarity: Vec<SimilarityItem>,
    pub(crate) softmax: Vec<SimilarityItem>,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Bias {
    pub(crate) name: String,
    pub(crate) bias: f64,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Calculator {
    pub(crate) model_name: String,
    pub(crate) similarity_per_token: Vec<Similarity>,
    pub(crate) entropy_per_token: HashMap<String, Vec<Bias>>,
}

impl SpaceCalculator for Calculator {
    fn new(
        model_name: String,
        bias_free_token_space: Space,
        bias_group_spaces: Vec<Space>,
    ) -> Self {
        // e.g., random space is the space without all the gender words
        // compare space is a list of space which only contains the gender words
        assert!(
            !bias_group_spaces.is_empty(),
            "compare_space should have at least one space"
        );

        let mut token_to_group_dict: Vec<Similarity> = Vec::new();

        for one_bias_free_token in &bias_free_token_space.tokens {
            // find the ideal similarity from ideal_similarities, which the space is one_compare_space
            let mut relationship_token_to_group: Vec<SimilarityItem> = Vec::new();
            for one_bias_group_space in &bias_group_spaces {
                let similarity = cos_similarity(
                    &one_bias_free_token.embedding,
                    &one_bias_group_space.space_center,
                );
                relationship_token_to_group.push(SimilarityItem {
                    name: one_bias_group_space.space_name.clone(),
                    value: similarity,
                });
            }

            token_to_group_dict.push(Similarity {
                name: one_bias_free_token.word.clone(),
                similarity_type: SimilarityType::TokenToGroup,
                softmax: get_similarity_softmax(&relationship_token_to_group),
                similarity: relationship_token_to_group,
            });
        }

        Calculator {
            model_name,
            similarity_per_token: token_to_group_dict.clone(),
            entropy_per_token: get_entropy_map(&token_to_group_dict),
        }
    }
}

fn get_entropy_map(similarity_per_token: &Vec<Similarity>) -> HashMap<String, Vec<Bias>> {
    let mut entropy_per_token: HashMap<String, Vec<Bias>> = HashMap::new();
    for one_similarity in similarity_per_token.iter() {
        let mut entropy_per_token_inner: Vec<Bias> = Vec::new();
        for one_similarity_item in one_similarity.softmax.iter() {
            entropy_per_token_inner.push(Bias {
                name: one_similarity_item.name.clone(),
                bias: -(one_similarity_item.value * one_similarity_item.value.log2()),
            });
        }
        entropy_per_token.insert(one_similarity.name.clone(), entropy_per_token_inner);
    }
    entropy_per_token
}

fn get_similarity_softmax(similarity_dict: &Vec<SimilarityItem>) -> Vec<SimilarityItem> {
    let mut similarity_softmax: Vec<SimilarityItem> = Vec::new();
    let max_similarity = similarity_dict
        .iter()
        .map(|one_similarity| one_similarity.value)
        .fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = similarity_dict
        .iter()
        .map(|one_similarity| (one_similarity.value - max_similarity).exp())
        .sum();

    for one_similarity in similarity_dict.iter() {
        let mut new_one_similarity = one_similarity.clone();
        new_one_similarity.value = (one_similarity.value - max_similarity).exp() / exp_sum;
        similarity_softmax.push(new_one_similarity);
    }
    similarity_softmax
}

fn cos_similarity(center1: &Vec<f64>, center2: &Vec<f64>) -> f64 {
    assert_eq!(
        center1.len(),
        center2.len(),
        "center1 and center2 should have the same length"
    );
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
    fn get_bias_per_token(&self) -> HashMap<String, f64> {
        self.entropy_per_token
            .iter()
            .map(|(token_name, bias)| {
                (
                    token_name.clone(),
                    bias.iter().map(|bias| bias.bias).sum::<f64>(),
                )
            })
            .collect()
    }

    fn get_bias_per_group(&self) -> HashMap<String, f64> {
        let mut bias_per_group: HashMap<String, f64> = HashMap::new();
        for one_similarity in &self.entropy_per_token {
            for one_similarity_item in one_similarity.1 {
                *bias_per_group
                    .entry(one_similarity_item.name.clone())
                    .or_insert(0.0) += one_similarity_item.bias;
            }
        }
        bias_per_group
            .iter()
            .map(|(group_name, bias)| {
                (
                    group_name.clone(),
                    *bias / self.entropy_per_token.len() as f64,
                )
            })
            .collect()
    }

    fn get_report_per_token(&self) -> HashMap<String, HashMap<String, f64>> {
        self.entropy_per_token
            .iter()
            .map(|(token_name, entropy_per_token_inner)| {
                let bias_token_to_group_inner: HashMap<String, f64> = entropy_per_token_inner
                    .iter()
                    .map(|one_similarity_item| {
                        (one_similarity_item.name.clone(), one_similarity_item.bias)
                    })
                    .collect();
                (token_name.clone(), bias_token_to_group_inner)
            })
            .collect()
    }

    fn get_report_per_group(&self) -> HashMap<String, HashMap<String, f64>> {
        // &self.similarity_per_group to HashMap
        let mut bias_group_to_token: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for one_similarity in &self.entropy_per_token {
            for one_similarity_item in one_similarity.1 {
                let bias = bias_group_to_token
                    .entry(one_similarity_item.name.clone())
                    .or_default();
                bias.insert(one_similarity.0.clone(), one_similarity_item.bias);
            }
        }
        bias_group_to_token
    }

    fn get_similarity_report(&self) -> HashMap<String, Vec<HashMap<String, f64>>> {
        let mut similarity: HashMap<String, Vec<HashMap<String, f64>>> = HashMap::new();
        for one_similarity in &self.similarity_per_token {
            let mut similarity_inner: Vec<HashMap<String, f64>> = Vec::new();
            for one_similarity_item in &one_similarity.similarity {
                let mut similarity_item: HashMap<String, f64> = HashMap::new();
                similarity_item.insert(one_similarity_item.name.clone(), one_similarity_item.value);
                similarity_inner.push(similarity_item);
            }
            similarity.insert(one_similarity.name.clone(), similarity_inner);
        }
        similarity
    }

    pub(crate) fn get_bias(&self) -> f64 {
        // average bias per token
        (self.get_bias_per_token().values().sum::<f64>() - self.get_bias_per_token().len() as f64)
            .abs()
    }

    pub(crate) fn get_model_name(&self) -> String {
        self.model_name.clone()
    }

    pub(crate) fn save_summary(&self, path: Option<&str>) {
        self.write(path.unwrap_or("./"), false);
    }
}
