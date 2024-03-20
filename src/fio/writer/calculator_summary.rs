use std::fs::OpenOptions;
use std::io::Write;
use crate::fio::writer::WriterOperator;
use crate::Calculator;

impl WriterOperator for Calculator {
    fn write(&self, path: &str, _if_show: bool) {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(format!("{}/calculator_summary.txt", path))
            .unwrap();

        // todo: format the output
        file.write_all(format!("ideal_similarity: {:?}\n", self.get_bias(Option::from(true))).as_bytes()).unwrap();
    }
}