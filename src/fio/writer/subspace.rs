use std::fs::OpenOptions;
use std::io::Write;
use crate::fio::writer::WriterOperator;
use crate::space::space_generator::Space;
use crate::util::constant;
use crate::util::progress_bar::ProgressBar;

impl WriterOperator for Space {

    fn write(&self, path: &str, if_show: bool) {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(path)
            .unwrap();

        let mut progress_bar = ProgressBar::new(self.tokens.len() as u64, constant::SPACE_GENERATING, if_show);

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
            progress_bar.inc(1);
        }
        progress_bar.finish();
    }
}