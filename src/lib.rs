use pyo3::prelude::*;

mod analyizer;
mod embedding;
mod fio;
mod space;
mod util;
mod web;

use embedding::models::Token;

use fio::reader::conceptx::ConceptXReader;
use fio::reader::Reader;

use crate::analyizer::calculator::Calculator;
use crate::analyizer::SpaceCalculator;
use crate::space::seeds::SubspaceSeeds;
use crate::space::space_generator::Space;
use space::SpaceGenerator;

#[pyfunction]
fn version() {
    println!("Wafflecone version: {}", env!("CARGO_PKG_VERSION"));
}

#[allow(non_snake_case)]
#[pyfunction]
fn new_subspace_seeds(name: String, seeds: Vec<String>) -> SubspaceSeeds {
    SubspaceSeeds::new(name, seeds)
}

#[pyfunction]
fn calculator(
    path: &str,
    subspace_seeds: Vec<SubspaceSeeds>,
    exclude_words: Option<Vec<String>>, // words to exclude from random tokens
    user_friendly: Option<bool>,
    pca_dimension: Option<usize>,
    model_name: Option<String>,
) -> Calculator {
    let data = ConceptXReader::new().read(path, user_friendly.unwrap_or(false));

    let mut num_of_tokens = 0;
    for line in &data {
        num_of_tokens += line.tokens.len();
    }
    println!("Total number of tokens: {}", num_of_tokens);

    // Build the global space
    let space = Space::new(data, None, pca_dimension);

    // all words need to be excluded: exclude_words + subspace_seeds
    let mut exclude_words = exclude_words.unwrap_or_default();
    for subspace_seed in &subspace_seeds {
        exclude_words.extend(subspace_seed.seeds.clone());
    }
    let neutral_tokens = space.get_neutral_tokens(exclude_words);
    let neutral_space = Space::new(neutral_tokens, None, pca_dimension);

    // build subspaces with the tokens of interests. e.g., male or female
    let mut sub_spaces: Vec<Space> = Vec::new();

    for subspace_seed in subspace_seeds {
        let sub_space = Space::new(space.find(&subspace_seed), Some(subspace_seed), None);
        sub_spaces.push(sub_space);
    }

    // compute the bias of the random subspace
    Calculator::new(
        match model_name {
            Some(name) => name,
            None => path.to_string(),
        },
        neutral_space,
        sub_spaces,
    )
}

#[pyfunction]
fn visualize(port: Option<u16>) {
    let web = web::run::Web::new(port.unwrap_or(8000));
    web.run();
}

#[pymodule]
fn wafflecone(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(calculator, m)?)?;
    m.add_function(wrap_pyfunction!(visualize, m)?)?;
    m.add_function(wrap_pyfunction!(new_subspace_seeds, m)?)?;
    m.add_function(wrap_pyfunction!(analyizer::normalizer::bias_normalize, m)?)?;
    m.add_class::<SubspaceSeeds>()?;
    Ok(())
}
