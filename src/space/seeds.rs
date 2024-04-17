use pyo3::pyclass;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubspaceSeeds {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub seeds: Vec<String>,
}

impl SubspaceSeeds {
    pub fn new(name: String, seeds: Vec<String>) -> SubspaceSeeds {
        SubspaceSeeds { name, seeds }
    }
}
