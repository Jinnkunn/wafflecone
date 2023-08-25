pub mod subspace;
pub mod calculator_summary;

pub trait WriterOperator {
    fn write(&self, path: &str, if_show: bool);
}