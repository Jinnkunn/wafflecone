use std::collections::HashMap;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::calculator::Calculator;

#[pyfunction]
#[pyo3(name = "result_normalize")]
pub fn bias_normalize(calculators: Vec<Calculator>) -> Vec<HashMap<String, f64>> {
    #[derive(Serialize, Deserialize, Debug)]
    struct BiasNormalize {
        name: String,
        bias: f64,
    }

    let biases: Vec<BiasNormalize> = calculators
        .iter()
        .map(|calculator| BiasNormalize {
            name: calculator.get_model_name(),
            bias: calculator.get_bias(),
        })
        .collect();

    let sum: f64 = biases.iter().map(|bias| bias.bias).sum();
    let mean = sum / biases.len() as f64;

    let variance: f64 = biases
        .iter()
        .map(|bias| (bias.bias - mean).powi(2))
        .sum();
    let std_dev = (variance / biases.len() as f64).sqrt();

    let result: Vec<HashMap<String, f64>> = biases
        .iter()
        .map(|bias| {
            let mut one_result: HashMap<String, f64> = HashMap::new();
            one_result.insert(
                bias.name.clone(),
                (bias.bias - mean) / std_dev,
            );
            one_result
        })
        .collect();

    result
}
