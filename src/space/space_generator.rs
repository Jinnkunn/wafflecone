use crate::embedding::models::TokenOperators;
use crate::embedding::models::Token;
use super::SpaceGenerator;

use rand_chacha::ChaCha8Rng;
use rand::prelude::*;


#[derive(Clone, Debug)]
pub struct Space {
    pub space_name: String,
    pub tokens: Vec<Token>,
    pub words_of_interests: Option<Vec<String>>,
}

impl SpaceGenerator for Space {
    fn new<T: TokenOperators>(items: T, words_of_interests: Option<Vec<String>>) -> Space {
        let mut tokens: Vec<Token> = Vec::new();
        for item in items.get_all_tokens() {
            tokens.push(item);
        }

        match tokens.len() > 0 {
            true => {
                Space {
                    space_name: match &words_of_interests {
                        None => {"Global Space".to_string()}
                        Some(x) => {format!("Space of {}", x[0])}
                    },
                    tokens,
                    words_of_interests,
                }
            },
            false => {
                panic!("The space is empty!");
            }
        }
    }

    fn set_space_name(&mut self, name: String) {
        self.space_name = name;
    }

    /// Find the words of interest in the space
    fn find(&self, target_words: &Vec<String>) -> Vec<Token> {
        find(&self.tokens.clone(), &target_words)
    }

    /// Calculate the center of the space
    fn get_center(&self) -> Vec<f64> {
        get_center(self.tokens.clone())
    }

    /// Get random tokens from the space, which will be used to generate subspaces
    fn get_random_tokens(&self, num: i64, random_seed: i64, exclude: Option<Vec<String>>) -> Vec<Token> {
        // combine exclude words and words of interests
        let mut exclude: Vec<String> = match exclude {
            None => {Vec::new()}
            Some(x) => {x}
        };
        match self.words_of_interests.clone() {
            None => {}
            Some(x) => {
                for word in x {
                    exclude.push(word);
                }
            }
        }
        get_random_tokens(self.tokens.clone(), num, random_seed, Some(exclude))
    }

    /// Print the summary of the space
    fn print_summary(&self) {
        println!("--- Summary of Space ---");
        println!("number of tokens: {}", self.tokens.len());
        println!("dimensions: {}", self.tokens[0].embedding.len());
        println!("token of interest: {}", self.words_of_interests.clone().unwrap_or(vec![]).join(", "));
        println!("-----------------------");
    }
}


fn get_center(tokens: Vec<Token>) -> Vec<f64> {
    let mut sum_of_embeddings: Vec<f64> = Vec::new();
    for token in tokens.clone() {
        if sum_of_embeddings.len() == 0 {
            for embedding in token.embedding.clone() {
                sum_of_embeddings.push(embedding);
            }
        } else {
            for i in 0..sum_of_embeddings.len() {
                sum_of_embeddings[i] += token.embedding[i];
            }
        }
    }
    let mut center: Vec<f64> = Vec::new();
    for embedding in sum_of_embeddings {
        center.push(embedding / tokens.len() as f64);
    }
    center
}

fn find(space_tokens: &Vec<Token>, passed_in_words: &Vec<String>) -> Vec<Token> {
    let mut find_tokens:Vec<Token> = Vec::new();
    for token in space_tokens {
        if passed_in_words.contains(&token.word) {
            find_tokens.push(token.clone())
        }
    }
    find_tokens
}

fn get_random_tokens(tokens: Vec<Token>, num: i64, random_seed: i64, exclude: Option<Vec<String>>) -> Vec<Token> {
    let total_tokens = tokens.len();

    let mut random_tokens: Vec<Token> = Vec::new();
    // set random seed, get random tokens without replacement
    let mut rng = ChaCha8Rng::seed_from_u64(random_seed as u64);
    let mut random_indices: Vec<usize> = (0..total_tokens).collect();
    random_indices.shuffle(&mut rng);

    let mut selected_num = 0;
    for i in 0..total_tokens {
        // if tokens[random_indices[i as usize]].word is not in exclude
        if exclude.is_some() {
            if exclude.clone().unwrap().contains(&tokens[random_indices[i as usize]].word) {
                continue;
            }
        }
        random_tokens.push(tokens[random_indices[i as usize]].clone());
        selected_num += 1;
        if selected_num == num {
            break;
        }
    }
    random_tokens
}

#[cfg(test)]
mod test {
    use crate::embedding::models::Line;
    use super::*;

    #[test]
    fn test_space_builder() {
        let space: Space = Space::new(vec![
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

        let word_of_interest: Vec<String> = vec![String::from("test"), String::from("run")];

        let space: Space = Space::new(lines, Some(word_of_interest));
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

        assert!(space.words_of_interests.is_some());
        assert_eq!(space.words_of_interests.clone().unwrap().len(), 2);
        assert_eq!(space.words_of_interests.clone().unwrap()[0], String::from("test"));
        assert_eq!(space.words_of_interests.clone().unwrap()[1], String::from("run"));
    }

    #[test]
    fn test_center_calculation() {
         let space: Space = Space::new(vec![
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
        let mut word_of_interest: Vec<String> = Vec::new();
        word_of_interest.push(String::from("test"));
        let space: Space = Space::new(vec![
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
            }], Some(word_of_interest));

        let token = space.get_random_tokens(1, 1, None);
        assert_eq!(token.len(), 1);
    }
}