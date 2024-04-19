mod masuku;

use std::fs;

fn main()  {
    let model = masuku::load_model("../models/best.onnx").expect("Failed to load the model");

    let entries = fs::read_dir("../images").expect("Failed to read directory");
    for entry_result in entries {
        match entry_result {
            Ok(entry) => {
                let img_path = entry.path().into_os_string().into_string().expect("Failed to convert path to string");
                let _ = masuku::infer(&img_path,&model);
            },
            _ => print!(""), // Fix: Replace empty print!() with a placeholder format string.
        }
    }
    

}



