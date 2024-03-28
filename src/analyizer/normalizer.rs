use pyo3::{prelude::*, pyclass};

use crate::analyizer::calculator::Calculator;

use crate::util::normalization::z_score_noramlization;

#[pyclass]
#[derive(Debug)]
pub struct BiasNormalized {
    #[pyo3(get)]
    pub model_name: String,
    #[pyo3(get)]
    pub bias: f64,
    #[pyo3(get)]
    pub bias_raw: f64,
    #[pyo3(get)]
    pub index: i64,
}

#[pyfunction]
#[pyo3(name = "result_normalize")]
pub fn bias_normalize(calculators: Vec<Calculator>) -> Vec<BiasNormalized> {
    let mut result: Vec<BiasNormalized> = Vec::new();
    for calculator in calculators {
        let mut bias_raw: Vec<f64> = Vec::new();
        for similarity in calculator.similarity_per_token.clone() {
            for item in similarity.similarity {
                bias_raw.push(item.value);
            }
        }
        let bias_normalized = z_score_noramlization(bias_raw);
        for (index, similarity) in calculator.similarity_per_token.iter().enumerate() {
            for item in similarity.similarity.iter() {
                result.push(BiasNormalized {
                    model_name: similarity.name.clone(),
                    bias: bias_normalized[index],
                    bias_raw: item.value,
                    index: index as i64,
                });
            }
        }
    }
    result
}
