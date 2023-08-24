use std::process::Command;
use std::thread;

pub struct Web {
    pub port: u16,
    pub version: String,
}

impl Web {
    pub fn new() -> Web {
        Web {
            port: 8081,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub fn run(&self) {
        // run two python processes
        let port = self.port;

        let handle = thread::spawn(move || {
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
                .output()
                .expect("failed to execute process");

    }
}