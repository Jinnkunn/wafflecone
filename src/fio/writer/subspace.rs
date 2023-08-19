use std::fs::OpenOptions;
use std::io::Write;
use crate::fio::writer::WriterOperator;
use crate::space::space_generator::Space;

impl WriterOperator for Space {

    fn write(&self, path: &str) {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(path)
            .unwrap();

        for token in &self.tokens {
            let mut line = String::new();
            line.push_str(&token.word);
            line.push_str(" ");
            line.push_str(&token.line_num.to_string());
            line.push_str(" ");
            line.push_str(&token.position.to_string());
            line.push_str(" ");
            for value in &token.embedding {
                line.push_str(&value.to_string());
                line.push_str(",");
            }
            line.push_str("\n");
            file.write_all(line.as_bytes()).unwrap();
        }
    }
}