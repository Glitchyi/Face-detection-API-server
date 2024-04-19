use std::mem;
use std::sync::{Arc, Mutex, Once};
use rayon::prelude::*;
use ort::{GraphOptimizationLevel, Session};
use ndarray::Array;
use image::{imageops::FilterType, GenericImageView}; 


pub fn load_model(model_path: &str) -> ort::Result<Session> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path);
    return model;
}

pub fn infer(img_path: &str, model: &Session) -> ort::Result<()> {
    let img = image::open(img_path).expect("Failed to open image");
    let img = img.resize_exact(640, 640, FilterType::CatmullRom);

    let input = Array::zeros((1, 3, 640, 640));

    let mut norm_table = [0.0; 256];
    for i in 0..=255 {
        norm_table[i] = i as f32 / 255.0;
    }


    let input = Arc::new(Mutex::new(input));
    let input_clone = Arc::clone(&input);

    let pixels: Vec<_> = img.pixels().collect();

    pixels.par_iter().for_each({
        let input = Arc::clone(&input);
        move |&pixel| {
            let x = pixel.0 as _;
            let y = pixel.1 as _;
            let [r, g, b, _] = pixel.2.0;
            let mut input = input.lock().unwrap();
            input[[0, 0, y, x]] = norm_table[r as usize];
            input[[0, 1, y, x]] = norm_table[g as usize];
            input[[0, 2, y, x]] = norm_table[b as usize];
        }
    });

    let mut input_out = None;
    let once = Once::new();
    once.call_once(|| {
        let mut input = input_clone.lock().unwrap();
        input_out = Some(mem::replace(&mut *input, Array::zeros((1, 3, 640, 640))));
    });

    let input = input_out.unwrap();

    let outputs = model.run(ort::inputs!["images" => input.view()]?)?;
    let predictions = outputs["output0"].try_extract_tensor::<f32>()?;

    // println!("{:?}",predictions);
    let (mut co, mut noco ) =(0.0,0.0);

    for i in 0..25200 {
        let objectness_score = predictions[[0, i, 4]];
        co += objectness_score * predictions[[0, i, 5]];
        noco += objectness_score * predictions[[0, i, 6]];
    }
    println!("Objectness Weighted probabilty for covered = {}",co/25200.0);
    println!("Objectness Weighted probabilty for covered = {}",noco/25200.0);
    Ok(())
}

