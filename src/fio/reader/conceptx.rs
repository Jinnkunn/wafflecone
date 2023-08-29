use super::Reader;
use crate::embedding::models::Line;
use crate::embedding::models::Token;
use crate::util::progress_bar::ProgressBar;
use std::io::BufRead;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
struct LineConceptX {
    pub linex_index: usize,
    pub features: Vec<FeatureConecptX>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FeatureConecptX {
    pub token: String,
    pub layers: Vec<TokenConecptX>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenConecptX {
    pub index: usize,
    pub values: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConceptXReader {}

impl Reader for ConceptXReader {
    fn new() -> Self {
        ConceptXReader {}
    }

    fn read(&self, path: &str, user_friendly: bool) -> Vec<Line> {
        let mut activations: Vec<LineConceptX> = Vec::new();

        let file = std::fs::File::open(path).unwrap();
        let all_lines = std::io::BufReader::new(file).lines();
        let mut bar = ProgressBar::new(all_lines.count() as u64, user_friendly);

        let file = std::fs::File::open(path).unwrap();
        let all_lines = std::io::BufReader::new(file).lines();

        // for loop to read line by line
        for line in all_lines {
            let line = match line {
                Ok(line) => line,
                Err(_) => {
                    println!("Error reading line");
                    return vec![];
                },
            };
            match serde_json::from_str::<LineConceptX>(&line) {
                Ok(mut activation) => {
                    activation.features.iter_mut().for_each(|x| {
                        x.token = x.token.replace("##", "");
                        // for roberta
                        x.token = x.token.replace("Ġ", "");
                    });
                    activations.push(activation);
                },
                Err(_) => {
                    println!("Error parsing line");
                    return vec![];
                },
            };
            bar.inc(1);
        }

        bar.finish();

        converter(activations)
    }
}

fn converter(activations: Vec<LineConceptX>) -> Vec<Line> {
    let mut lines: Vec<Line> = Vec::new();
    for line in activations {
        let mut tokens: Vec<Token> = Vec::new();
        for feature in line.features {
            for token in feature.layers {
                tokens.push(Token {
                    word: String::from(&feature.token),
                    line_num: line.linex_index,
                    position: token.index,
                    embedding: token.values
                });
            }
        }
        lines.push(Line {
            tokens: tokens,
            line_num: line.linex_index,
        });
    }
    lines
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_read_with_correct_number() {
        let reader = ConceptXReader::new();
        let lines = reader.read("./test_data/conceptx.json", false);
        assert_eq!(lines.len(), 10);
    }

    #[test]
    fn test_read_values() {
        let reader = ConceptXReader::new();
        let lines = reader.read("./test_data/conceptx.json", false);
        assert_eq!(lines.len(), 10);
        assert_eq!(lines[0].tokens[0].word, "[CLS]0");
        assert_eq!(lines[0].tokens[0].line_num, 0);
        assert_eq!(lines[0].tokens[0].position, 0);
    }

    #[test]
    fn test_read_values2() {
        let reader = ConceptXReader::new();
        let lines = reader.read("./test_data/conceptx.json", false);
        assert_eq!(lines.len(), 10);
        assert_eq!(lines[1].tokens[0].word, "[CLS]1");
        assert_eq!(lines[1].tokens[0].line_num, 1);
        assert_eq!(lines[1].tokens[0].position, 0);
    }
}