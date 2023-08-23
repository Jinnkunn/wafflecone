use std::collections::HashMap;
use pyo3::prelude::*;

mod fio;
mod space;
mod models;

use models::Token;

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
fn calculator(path: &str, subspace_seeds: Vec<Vec<String>>) -> Calculator {
    let data = ConceptXReader::new().read(path);
    let space = Space::new(data, None);

    let random_token = space.get_random_tokens(10, 1);

    let random_sub_space = Space::new(random_token.clone(), None);
    random_sub_space.write("./random_subspace.txt");

    // let male_words = vec![
    //     String::from("male"),
    //     String::from("he"),
    //     String::from("boy")
    // ];
    // let male_sub_space = Space::new(space.find(&male_words), Option::from(male_words));
    // // male_sub_space.print_summary();
    // let _male_center = male_sub_space.get_center();
    //
    // let female_words = vec![
    //     String::from("female"),
    //     String::from("she"),
    //     String::from("girl")
    // ];
    // let female_sub_space = Space::new(space.find(&female_words), Option::from(female_words));
    // // female_sub_space.print_summary();
    // let _female_center = female_sub_space.get_center();

    let mut sub_spaces: Vec<Space> = Vec::new();
    for subspace_seed in subspace_seeds {
        let sub_space = Space::new(space.find(&subspace_seed), Option::from(subspace_seed));
        sub_spaces.push(sub_space);
    }

    let calculator = Calculator::new(random_sub_space, sub_spaces);

    calculator

    // calculator.bias_asb_sum_average()

}

#[pymodule]
fn wafflecone(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(calculator, m)?)?;
    Ok(())
}
