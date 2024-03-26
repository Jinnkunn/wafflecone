pub mod calculator_summary;
pub mod subspace;

pub trait WriterOperator {
    fn write(&self, path: &str, if_show: bool);
}
