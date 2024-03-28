fn main() {
    // macOS specific configuration
    // This is required to link the C++ standard library on macOS
    #[cfg(target_os = "macos")]
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,/Library/Developer/CommandLineTools/Library/Frameworks"
    );
}
