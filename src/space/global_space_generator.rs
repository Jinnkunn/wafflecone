use crate::models::TokenOperators;
use crate::models::Token;
use super::SpaceOperator;
use super::get_center;
use super::find;
use super::get_random_tokens;

#[derive(Clone, Debug)]
pub struct GlobalSpace {
    pub tokens: Vec<Token>,
}

impl SpaceOperator for GlobalSpace {
    fn new<T: TokenOperators>(items: T, _words_of_interests: Option<Vec<String>>) -> GlobalSpace {
        let mut tokens: Vec<Token> = Vec::new();
        for item in items.get_all_tokens() {
            tokens.push(item);
        }
        GlobalSpace {
            tokens,
        }
    }

    /// Find the words of interest in the space
    fn find(&self, target_words: Vec<String>) -> Vec<Token> {
        find(self.tokens.clone(), target_words)
    }

    /// Calculate the center of the space
    fn get_center(&self) -> Vec<f64> {
        get_center(self.tokens.clone())
    }

    /// Get random tokens from the space, which will be used to generate subspaces
    fn get_random_tokens(&self, num: i64, random_seed: i64) -> Vec<Token> {
        get_random_tokens(self.tokens.clone(), num, random_seed)
    }

    /// Print the summary of the space
    fn print_summary(&self) {
        println!("--- Summary of Space ---");
        println!("type: global space");
        println!("number of tokens: {}", self.tokens.len());
        println!("dimensions: {}", self.tokens[0].embedding.len());
        println!("token of interest: None");
        println!("-----------------------");
    }
}

#[cfg(test)]
mod test {
    use crate::models::Line;
    use super::*;

    #[test]
    fn test_space_builder() {
        let space: GlobalSpace = GlobalSpace::new(vec![
            Token {
                word: String::from("test"),
                line_num: 0,
                position: 0,
                embedding: vec![1.0, 2.0, 3.0],
            },
            Token {
                word: String::from("new"),
                line_num: 0,
                position: 1,
                embedding: vec![2.0, 3.0, 4.0],
            },
            Token {
                word: String::from("run"),
                line_num: 1,
                position: 0,
                embedding: vec![3.0, 4.0, 5.0],
            }], None);

        assert_eq!(space.tokens.len(), 3);
        assert_eq!(space.tokens[0].word, String::from("test"));
        assert_eq!(space.tokens[0].line_num, 0);
        assert_eq!(space.tokens[0].position, 0);
        assert_eq!(space.tokens[0].embedding.len(), 3);
        assert_eq!(space.tokens[0].embedding[0], 1.0);
        assert_eq!(space.tokens[0].embedding[1], 2.0);
        assert_eq!(space.tokens[0].embedding[2], 3.0);
        assert_eq!(space.tokens[1].word, String::from("new"));
        assert_eq!(space.tokens[1].line_num, 0);
        assert_eq!(space.tokens[1].position, 1);
        assert_eq!(space.tokens[1].embedding.len(), 3);
        assert_eq!(space.tokens[1].embedding[0], 2.0);
        assert_eq!(space.tokens[1].embedding[1], 3.0);
        assert_eq!(space.tokens[1].embedding[2], 4.0);
        assert_eq!(space.tokens[2].word, String::from("run"));
        assert_eq!(space.tokens[2].line_num, 1);
        assert_eq!(space.tokens[2].position, 0);
        assert_eq!(space.tokens[2].embedding.len(), 3);
        assert_eq!(space.tokens[2].embedding[0], 3.0);
        assert_eq!(space.tokens[2].embedding[1], 4.0);
        assert_eq!(space.tokens[2].embedding[2], 5.0);
    }

    #[test]
    fn test_init_from_line() {
        let mut lines: Vec<Line> = Vec::new();

        let first_line: Vec<Token> = vec![
            Token {
                word: String::from("test"),
                line_num: 0,
                position: 0,
                embedding: vec![1.0, 2.0, 3.0],
            },
            Token {
                word: String::from("new"),
                line_num: 0,
                position: 1,
                embedding: vec![2.0, 3.0, 4.0],
            }
        ];

        let second_line: Vec<Token> = vec![
            Token {
                word: String::from("run"),
                line_num: 1,
                position: 0,
                embedding: vec![3.0, 4.0, 5.0],
            }
        ];

        lines.push(Line {
            tokens: first_line,
            line_num: 0,
        });

        lines.push(Line {
            tokens: second_line,
            line_num: 1,
        });

        let space: GlobalSpace = GlobalSpace::new(lines, None);
        assert_eq!(space.tokens.len(), 3);
        assert_eq!(space.tokens[0].word, String::from("test"));
        assert_eq!(space.tokens[0].line_num, 0);
        assert_eq!(space.tokens[0].position, 0);
        assert_eq!(space.tokens[0].embedding.len(), 3);
        assert_eq!(space.tokens[0].embedding[0], 1.0);
        assert_eq!(space.tokens[0].embedding[1], 2.0);
        assert_eq!(space.tokens[0].embedding[2], 3.0);
        assert_eq!(space.tokens[1].word, String::from("new"));
        assert_eq!(space.tokens[1].line_num, 0);
        assert_eq!(space.tokens[1].position, 1);
        assert_eq!(space.tokens[1].embedding.len(), 3);
        assert_eq!(space.tokens[1].embedding[0], 2.0);
        assert_eq!(space.tokens[1].embedding[1], 3.0);
        assert_eq!(space.tokens[1].embedding[2], 4.0);
        assert_eq!(space.tokens[2].word, String::from("run"));
        assert_eq!(space.tokens[2].line_num, 1);
        assert_eq!(space.tokens[2].position, 0);
        assert_eq!(space.tokens[2].embedding.len(), 3);
        assert_eq!(space.tokens[2].embedding[0], 3.0);
        assert_eq!(space.tokens[2].embedding[1], 4.0);
        assert_eq!(space.tokens[2].embedding[2], 5.0);
    }

    #[test]
    fn test_center_calculation() {
         let space: GlobalSpace = GlobalSpace::new(vec![
            Token {
                word: String::from("test"),
                line_num: 0,
                position: 0,
                embedding: vec![1.0, 2.0, 3.0],
            },
            Token {
                word: String::from("new"),
                line_num: 0,
                position: 1,
                embedding: vec![2.0, 3.0, 4.0],
            }], None);

        let center = space.get_center();
        assert_eq!(center.len(), 3);
        assert_eq!(center[0], 1.5);
        assert_eq!(center[1], 2.5);
        assert_eq!(center[2], 3.5);
    }

    #[test]
    fn test_get_random_tokens() {
        let space: GlobalSpace = GlobalSpace::new(vec![
            Token {
                word: String::from("test"),
                line_num: 0,
                position: 0,
                embedding: vec![1.0, 2.0, 3.0],
            },
            Token {
                word: String::from("new"),
                line_num: 0,
                position: 1,
                embedding: vec![2.0, 3.0, 4.0],
            },
            Token {
                word: String::from("run"),
                line_num: 1,
                position: 0,
                embedding: vec![3.0, 4.0, 5.0],
            }], None);

        let token = space.get_random_tokens(1, 1);
        assert_eq!(token.len(), 1);
    }
}