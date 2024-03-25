use super::SpaceGenerator;
use crate::embedding::models::Token;
use crate::embedding::models::TokenOperators;
use crate::space::SubspaceSeeds;
use crate::util::pca::PCA;
use nalgebra::{DMatrix, RowDVector};

#[derive(Clone, Debug)]
pub struct Space {
    pub space_name: String,
    pub tokens: Vec<Token>,
    pub space_center: Vec<f64>,
    pub words_of_interests: Option<Vec<String>>,
}

impl SpaceGenerator for Space {
    fn new<T: TokenOperators>(
        items: T,
        words_of_interests: Option<SubspaceSeeds>,
        pca_dimension: Option<usize>,
    ) -> Space {
        // assert!((words_of_interests.iter().len() > 0 && parent_space.iter().len() > 0) || (words_of_interests.iter().len() == 0 && parent_space.iter().len() == 0));
        let mut tokens: Vec<Token> = Vec::new();
        for item in items.get_all_tokens() {
            tokens.push(item);
        }

        match tokens.len() > 0 {
            true => Space {
                space_name: match &words_of_interests {
                    None => "Global Space".to_string(),
                    Some(x) => {
                        format!("{}", x.name)
                    }
                },
                tokens: match &words_of_interests {
                    None => pca(tokens.clone(), pca_dimension),
                    Some(_) => tokens.clone(),
                },
                space_center: get_center(tokens.clone()),
                words_of_interests: match words_of_interests {
                    None => None,
                    Some(x) => Some(x.seeds),
                },
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
        find(&self.tokens.clone(), target_words)
    }

    /// Calculate the center of the space
    fn get_center(&self) -> Vec<f64> {
        get_center(self.tokens.clone())
    }

    fn get_std(&self) -> Vec<f64> {
        get_std(self.tokens.clone())
    }

    fn get_neutral_tokens(&self, exclude: Vec<String>) -> Vec<Token> {
        let mut neutral_tokens: Vec<Token> = Vec::new();
        for token in &self.tokens {
            if !exclude.contains(&token.word) {
                neutral_tokens.push(token.clone());
            }
        }
        neutral_tokens
    }

    /// Print the summary of the space
    fn print_summary(&self) {
        println!("--- Summary of Space ---");
        println!("number of tokens: {}", self.tokens.len());
        println!("dimensions: {}", self.tokens[0].embedding.len());
        println!(
            "token of interest: {}",
            match &self.words_of_interests {
                None => {
                    "None".to_string()
                }
                Some(x) => {
                    x.join(", ")
                }
            }
        );
        println!("-----------------------");
    }
}

fn pca(tokens: Vec<Token>, pca_dimension: Option<usize>) -> Vec<Token> {
    // use PCA to do the dimension reduction.
    // reduce the demension to 512 by default

    let n_components = match pca_dimension {
        None => return tokens,
        Some(x) => x,
    };

    println!("The number of embedding is {}", tokens[0].embedding.len());
    if tokens[0].embedding.len() <= n_components {
        println!(
            "The number of tokens is less than the number of components, so PCA is not applied."
        );
        return tokens;
    }

    let mut pca_model = PCA::new(n_components);

    let embeddings: DMatrix<f64> = DMatrix::from_rows(
        &tokens
            .iter()
            .map(|token| RowDVector::from_vec(token.embedding.clone()))
            .collect::<Vec<RowDVector<f64>>>(),
    );

    pca_model = pca_model.fit(embeddings.clone());

    let transformed_embeddings = pca_model.transform(embeddings);

    let mut updated_tokens: Vec<Token> = Vec::new();
    for (i, token) in tokens.iter().enumerate() {
        updated_tokens.push(Token::new(
            token.word.clone(),
            token.line_num,
            token.position,
            transformed_embeddings.row(i).iter().cloned().collect(),
        ));
    }

    updated_tokens
}

fn get_std(tokens: Vec<Token>) -> Vec<f64> {
    // calculate the stand deviation
    let mut std: Vec<f64> = Vec::new();
    for i in 0..tokens[0].embedding.len() {
        let mut sum = 0.0;
        for token in &tokens {
            sum += token.embedding[i];
        }
        let mean = sum / tokens.len() as f64;
        let mut sum_of_square = 0.0;
        for token in &tokens {
            sum_of_square += (token.embedding[i] - mean).powi(2);
        }
        std.push((sum_of_square / tokens.len() as f64).sqrt());
    }
    std
}

fn get_center(tokens: Vec<Token>) -> Vec<f64> {
    // sort the embeddings
    let mut embeddings: Vec<f64> = Vec::new();
    for token in &tokens {
        embeddings.push(token.embedding.iter().sum::<f64>());
    }
    embeddings.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // calculate the quartiles
    let q1 = embeddings[embeddings.len() / 4];
    let q3 = embeddings[embeddings.len() * 3 / 4];
    let iqr = q3 - q1;

    // filter out the outliers
    let mut filtered_embeddings: Vec<f64> = Vec::new();
    for embedding in embeddings {
        if embedding >= q1 - 1.5 * iqr && embedding <= q3 + 1.5 * iqr {
            filtered_embeddings.push(embedding);
        }
    }

    // calculate the center
    let mut center: Vec<f64> = Vec::new();
    for i in 0..tokens[0].embedding.len() {
        let mut sum = 0.0;
        for token in &tokens {
            sum += token.embedding[i];
        }
        center.push(sum / tokens.len() as f64);
    }
    center
}

fn find(space_tokens: &Vec<Token>, passed_in_words: &Vec<String>) -> Vec<Token> {
    let mut find_tokens: Vec<Token> = Vec::new();
    for token in space_tokens {
        if passed_in_words.contains(&token.word) {
            find_tokens.push(token.clone())
        }
    }
    println!(
        "{:?}, The number of tokens found is {}",
        passed_in_words,
        find_tokens.len()
    );
    find_tokens
}
