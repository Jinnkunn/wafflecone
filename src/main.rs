mod fio;
mod space;
mod models;

use models::Token;

use fio::reader::conceptx::ConceptXReader;
use fio::reader::Reader;

use space::SpaceOperator;
use crate::fio::writer::WriterOperator;
use crate::space::space_generator::Space;

fn main() {
    let data = ConceptXReader::new().read("./test_data/conceptx.json");
    let space = Space::new(data, None);
    space.print_summary();

    let random_token = space.get_random_tokens(10, 1);

    let word_of_interest = Option::from(vec![
        String::from("male"),
        String::from("he"),
        String::from("boy")
    ]);
    let sub_space = Space::new(random_token, word_of_interest);
    sub_space.print_summary();

    sub_space.write("./test_subspace.txt")
}
