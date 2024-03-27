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
    pub subspace_seed_words: Option<Vec<String>>,
}

impl SpaceGenerator for Space {
    fn new<T: TokenOperators>(
        items: T,
        subspace_seeds: Option<SubspaceSeeds>,
        pca_dimension: Option<usize>,
    ) -> Self {
        let mut tokens: Vec<Token> = Vec::new();
        for item in items.get_all_tokens() {
            tokens.push(item);
        }

        if tokens.is_empty() {
            panic!("The space is empty!");
        }

        Space {
            space_name: subspace_seeds
                .as_ref()
                .map(|seeds| seeds.name.to_string())
                .unwrap_or_else(|| "Global".to_string()),
            tokens: subspace_seeds
                .as_ref()
                .map(|_| tokens.clone())
                .unwrap_or_else(|| pca(tokens.clone(), pca_dimension)),
            space_center: get_center(tokens.clone()),
            subspace_seed_words: subspace_seeds.map(|seeds| seeds.seeds),
        }
    }

    fn set_space_name(&mut self, name: String) {
        self.space_name = name;
    }

    /// Find the words of interest in the space
    fn find(&self, subspace_seed: &SubspaceSeeds) -> Vec<Token> {
        find(
            &self.tokens.clone(),
            &subspace_seed.seeds,
            subspace_seed.name.clone(),
        )
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
            self.subspace_seed_words
                .as_ref()
                .map(|x| x.join(", "))
                .unwrap_or_else(|| "None".to_string())
        );
        println!("-----------------------");
    }
}

fn pca(tokens: Vec<Token>, pca_dimension: Option<usize>) -> Vec<Token> {
    let n_components = match pca_dimension {
        None => return tokens,
        Some(x) => x,
    };

    if tokens[0].embedding.len() <= n_components {
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

    tokens
        .iter()
        .enumerate()
        .map(|(i, token)| {
            Token::new(
                token.word.clone(),
                token.line_num,
                token.position,
                transformed_embeddings.row(i).iter().cloned().collect(),
            )
        })
        .collect()
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
    let mut center: Vec<f64> = Vec::new();

    for i in 0..tokens[0].embedding.len() {
        let sum: f64 = tokens.iter().map(|token| token.embedding[i]).sum();
        center.push(sum / tokens.len() as f64);
    }

    center
}

fn find(
    space_tokens: &Vec<Token>,
    passed_in_words: &Vec<String>,
    subspace_name: String,
) -> Vec<Token> {
    let find_tokens: Vec<Token> = space_tokens
        .iter()
        .filter(|token| passed_in_words.contains(&token.word))
        .cloned()
        .collect();

    println!(
        "Subpace `{}`: The number of tokens found is {}",
        subspace_name,
        find_tokens.len()
    );

    find_tokens
}
