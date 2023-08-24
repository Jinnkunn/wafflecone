use std::process::Command;
use std::thread;

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
                .output()
                .expect("failed to execute process");

    }
}