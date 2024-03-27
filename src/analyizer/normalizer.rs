use pyo3::{prelude::*, pyclass};

use super::calculator::Calculator;

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
    let mean = calculators
        .iter()
        .map(|calculator| calculator.entropy_per_token.len())
        .sum::<usize>() as f64
        / calculators.len() as f64;

    let std_dev = (calculators
        .iter()
        .map(|calculator| (calculator.entropy_per_token.len() as f64 - mean).powi(2))
        .sum::<f64>()
        / calculators.len() as f64)
        .sqrt();

    let mut normalized_biases: Vec<BiasNormalized> = calculators
        .iter()
        .map(|calculator| {
            let bias = calculator.get_bias();
            let normalized_bias = (bias - mean) / std_dev;
            BiasNormalized {
                model_name: calculator.get_model_name(),
                bias: normalized_bias,
                bias_raw: bias,
                index: -1,
            }
        })
        .collect();

    normalized_biases.sort_by(|a, b| a.bias.partial_cmp(&b.bias).unwrap());
    normalized_biases
        .iter_mut()
        .enumerate()
        .for_each(|(i, item)| item.index = i as i64);

    normalized_biases
}
