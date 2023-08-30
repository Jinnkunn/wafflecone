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

fn main() {
    let data = ConceptXReader::new().read("./test_data/layer12.json", true);
    let space = Space::new(data, None);

    let random_token = space.get_random_tokens(10, 1, None);

    let random_sub_space = Space::new(random_token.clone(), None);
    random_sub_space.write("./random_subspace.txt", true);

    let male_words = vec![
        String::from("male"),
        String::from("he"),
        String::from("boy")
    ];
    let male_sub_space = Space::new(space.find(&male_words), Option::from(male_words));
    male_sub_space.print_summary();
    let male_center = male_sub_space.get_center();

    let female_words = vec![
        String::from("female"),
        String::from("she"),
        String::from("girl")
    ];
    let female_sub_space = Space::new(space.find(&female_words), Option::from(female_words));
    female_sub_space.print_summary();
    let female_center = female_sub_space.get_center();

    println!("male center: {:?}", male_center);
    println!("female center: {:?}", female_center);

    let calculator = Calculator::new(random_sub_space, vec![
        female_sub_space,
        male_sub_space,
    ]);

    calculator.print();

    let abs_sum_ave = calculator.bias_asb_sum_average();
    let sum_ave = calculator.bias_sum_average();

    println!("abs_sum_ave: {:?}", abs_sum_ave);
    println!("sum_ave: {:?}", sum_ave);

}
