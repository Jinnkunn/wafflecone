use super::Reader;
use crate::embedding::models::Line;
use crate::embedding::models::Token;
use crate::util::progress_bar::ProgressBar;
use std::io::BufRead;
use serde::{Deserialize, Serialize};
use crate::util::constant;

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
        let file = std::fs::File::open(path).unwrap();
        let buf = std::io::BufReader::new(&file);

        let mut pb_readfile = ProgressBar::new(file.metadata().unwrap().len(), constant::FILE_READING, user_friendly);

        let activations = buf.lines()
            .map(|line| {
                pb_readfile.inc(line.as_ref().unwrap().len() as u64);
                line.unwrap()
            })
            .map(|line| {
                let mut activation = serde_json::from_str::<LineConceptX>(&line).unwrap();
                activation.features.iter_mut().for_each(|x| {
                    x.token = x.token.replace("##", "");
                    x.token = x.token.replace("Ġ", "");
                    x.token = x.token.replace("\\u0120", "");
                });
                activation
            })
            .collect::<Vec<LineConceptX>>();

        pb_readfile.finish();

        converter(activations)
    }
}

fn converter(activations: Vec<LineConceptX>) -> Vec<Line> {
    let mut lines: Vec<Line> = Vec::new();
    let mut pb_tokenlize = ProgressBar::new(activations.len() as u64, constant::TOKEN_GENERATING, true);
    for line in activations {
        let mut tokens: Vec<Token> = Vec::new();
        for feature in line.features {
            for token in feature.layers {
                tokens.push(Token {
                    word: String::from(&feature.token),
                    line_num: line.linex_index,
                    position: token.index,
                    embedding: token.values,
                });
            }
        }
        lines.push(Line {
            tokens,
            line_num: line.linex_index,
        });
        pb_tokenlize.inc(1);
    }
    pb_tokenlize.finish();
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
        assert_eq!(lines[0].tokens[1].word, "new");
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