use pyo3::prelude::*;

mod fio;
mod space;
mod embedding;
mod web;
mod util;

use embedding::models::Token;

use fio::reader::conceptx::ConceptXReader;
use fio::reader::Reader;

use space::SpaceGenerator;
use crate::fio::writer::WriterOperator;
use crate::space::space_generator::Space;
use crate::space::space_calculator::Calculator;
use crate::space::SpaceCalculator;

#[pyfunction]
fn version() {
    println!("Wafflecone version: {}", env!("CARGO_PKG_VERSION"));
}

#[pyfunction]
fn calculator(path: &str,
              subspace_seeds: Vec<Vec<String>>,
              random_token_num: Option<f64>, // number of random tokens
              random_token_seed: Option<i64>, // random seed
              subspace_folder_path: Option<&str>, // folder path to save subspaces
              exclude_words: Option<Vec<String>>, // words to exclude from random tokens
              user_friendly: Option<bool>,
              pca_dimension: Option<usize>,
) -> Calculator {
    let data = ConceptXReader::new().read(path, user_friendly.unwrap_or(false));

    let mut num_of_tokens = 0;
    for line in &data {
        num_of_tokens += line.tokens.len();
    }
    println!("Total number of tokens: {}", num_of_tokens);

    // Build the global space
    let space = Space::new(data.clone(), None, pca_dimension);

    // all words need to be excluded: exclude_words + subspace_seeds
    let mut exclude_words = exclude_words.unwrap_or(Vec::new());
    for subspace_seed in &subspace_seeds {
        exclude_words.extend(subspace_seed.clone());
    }

    // select random tokens from the global space
    // then build a subspace with the random tokens
    let random_token = space.get_random_tokens(
        (num_of_tokens as f64 * random_token_num.unwrap_or(0.8)) as i64,
        random_token_seed.unwrap_or(1),
        Some(exclude_words)
    );

    let subspace_folder = subspace_folder_path.unwrap_or("./");

    // save the random tokens to a file
    let random_sub_space = Space::new(random_token.clone(), None, None);
    random_sub_space.write(format!("{}/random_subspace.txt", subspace_folder).as_str(), user_friendly.unwrap_or(false));

    // build subspaces with the tokens of interests. e.g., male or female
    let mut sub_spaces: Vec<Space> = Vec::new();
    for subspace_seed in subspace_seeds {
        let sub_space = Space::new(
            space.find(&subspace_seed),
            Option::from(subspace_seed),
            None
        );
        sub_spaces.push(sub_space);
    }

    // compute the bias of the random subspace
    let calculator = Calculator::new(random_sub_space, sub_spaces, space);

    calculator

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
    Ok(())
}
