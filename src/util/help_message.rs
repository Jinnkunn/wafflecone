use crate::util::Message;

impl Message {
    pub fn calculator_info(model_name: Option<String>, path: &str, pca_dimension: Option<usize>) {
        println!(
            "🔮 Model: {}",
            model_name.unwrap_or_else(|| path.to_string())
        );
        println!("📚 Reading data from: {}", path);
        match pca_dimension {
            Some(dimension) => println!("📊 PCA Dimension: {}", dimension),
            None => println!("📊 PCA Dimension: None"),
        }
    }
}
