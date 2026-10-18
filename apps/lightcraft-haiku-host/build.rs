//! Builds the native host (`native/host.cpp`) against the Be API when the target is Haiku (air/OS
//! is a Haiku distribution). Other targets build an empty binding crate.

fn main() {
    println!("cargo:rerun-if-changed=native/host.cpp");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("haiku") {
        return;
    }
    cc::Build::new()
        .cpp(true)
        .file("native/host.cpp")
        .flag("-Wno-multichar")
        .flag("-Wno-unused-parameter")
        .opt_level(2)
        .compile("lightcraft_haiku_host");
    for lib in ["be", "tracker", "translation", "stdc++"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
