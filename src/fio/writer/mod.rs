pub mod subspace;

pub trait WriterOperator {
    fn write(&self, path: &str);
}