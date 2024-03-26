use std::collections::HashMap;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::calculator::{Bias, Calculator, Similarity, SimilarityItem, SimilarityType};

#[pyfunction]
#[pyo3(name = "result_normalize")]
pub fn bias_normalize(calculators: Vec<Calculator>) -> Vec<HashMap<String, f64>> {
    #[derive(Serialize, Deserialize, Debug)]
    struct BiasNormalize {
        name: String,
        bias: f64,
    }
    let mut biases: Vec<BiasNormalize> = Vec::new();
    for calculator in calculators {
        let one_bias_item = BiasNormalize {
            name: calculator.get_model_name(),
            bias: calculator.get_bias(),
        };
        biases.push(one_bias_item);
    }
    println!("{:?}", biases);
    let min_value = biases
        .iter()
        .min_by(|a, b| a.bias.partial_cmp(&b.bias).unwrap())
        .unwrap()
        .bias;
    let max_value = biases
        .iter()
        .max_by(|a, b| a.bias.partial_cmp(&b.bias).unwrap())
        .unwrap()
        .bias;
    let mut result: Vec<HashMap<String, f64>> = Vec::new();
    for item in biases {
        let normalized = (item.bias - min_value) / (max_value - min_value);
        let mut map = HashMap::new();
        map.insert(item.name, normalized);
        result.push(map);
    }
    // sort the result by the bias value
    result.sort_by(|a, b| {
        a.values()
            .next()
            .unwrap()
            .partial_cmp(b.values().next().unwrap())
            .unwrap()
    });
    result
}
