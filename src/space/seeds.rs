use pyo3::pyclass;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubspaceSeeds {
    pub name: String,
    pub seeds: Vec<String>,
}

impl SubspaceSeeds {
    pub fn new(name: String, seeds: Vec<String>) -> SubspaceSeeds {
        SubspaceSeeds { name, seeds }
    }
}
