use std::{env, path::PathBuf};

#[allow(deprecated)]
use bindgen::CargoCallbacks;

use cmake::Config;

fn main() {
    let lib_path = Config::new("vendor/llama.cpp")
        .define("LLAMA_STATIC", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_CUBLAS", "ON")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .build();

    println!(
        "cargo:rustc-link-search=native={}/build",
        lib_path.display()
    );

    println!("cargo:rustc-link-lib=static=llama");

    let bindings = bindgen::builder()
        .header("wrapper.h")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .parse_callbacks(Box::new(CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
