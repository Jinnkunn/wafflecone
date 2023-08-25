use std::process::Command;
use std::thread;
use pyo3::{Py, PyAny, Python};
use pyo3::prelude::PyModule;

pub struct Web {
    pub port: u16,
    pub version: String,
}

impl Web {
    pub fn new(port: u16) -> Self {
        Web {
            port: port,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub fn run(&self) {
        // run two python processes
        let port = self.port;

        let path = get_lib_path();

        let handle = thread::spawn(move || {
            // wait for 1 second to make sure the server is up
            thread::sleep(std::time::Duration::from_secs(1));
            let _output = Command::new("python")
                .arg("-m")
                .arg("webbrowser")
                .arg("-t")
                .arg(format!("http://localhost:{}", port))
                .output()
                .expect("failed to execute process");
        });

        handle.join().unwrap();

        println!("Control-C to stop the server");

        let _output = Command::new("python")
                .arg("-m")
                .arg("http.server")
                .arg(format!("{}", self.port))
                .arg("--directory")
                .arg(path)
                .output()
                .expect("failed to execute process");

    }
}

fn get_lib_path() -> String {
    let result = Python::with_gil(|py| {
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            r#"
import os
import wafflecone
def get_dir():
    return os.path.dirname(wafflecone.__file__)
            "#,
            "",
            "",
        )
            .unwrap()
            .getattr("get_dir")
            .unwrap()
            .into();

        // call object without any arguments
        fun.call0(py).unwrap()
    });

    result.to_string()
}