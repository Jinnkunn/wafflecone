#[derive(Clone, Debug)]
pub struct Line {
    pub tokens: Vec<Token>,
    pub line_num: usize,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub word: String,
    pub position: usize,
    pub line_num: usize,
    pub embedding: Vec<f64>,
    pub token_id: String,
}

impl Token {
    pub fn new(word: String, position: usize, line_num: usize, embedding: Vec<f64>) -> Self {
        Token {
            word: word.clone(),
            position: position,
            line_num: line_num,
            embedding: embedding,
            token_id: format!("{}:{}:{}", word, position, line_num),
        }
    }
}

pub trait TokenOperators {
    fn get_all_tokens(&self) -> Vec<Token>;
}

impl TokenOperators for Vec<Line> {
    fn get_all_tokens(&self) -> Vec<Token> {
        let mut tokens: Vec<Token> = Vec::new();
        for line in self {
            for token in &line.tokens {
                tokens.push(token.clone());
            }
        }
        tokens
    }
}

impl TokenOperators for Vec<Token> {
    fn get_all_tokens(&self) -> Vec<Token> {
        let mut tokens: Vec<Token> = Vec::new();
        for token in self {
            tokens.push(token.clone());
        }
        tokens
    }
}
