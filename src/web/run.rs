
pub struct Web {
    pub port: u16,
    pub version: String,
}

impl Web {
    fn new() -> Web {
        Web {
            port: 8080,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub fn run(&self) {
        todo!()
    }
}