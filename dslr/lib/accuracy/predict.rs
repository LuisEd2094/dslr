use crate::accuracy::accuracy::calculate_accuracy;
use crate::log_reg::sigmoid::sigmoid;
use crate::structs::House;
use ndarray::{s, Array1, Array2};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

fn predict(x: &Array2<f64>, weights_dict: &HashMap<House, Array1<f64>>) -> Vec<House> {
    let mut predictions = Vec::new();
    for i in 0..x.nrows() {
        let row = x.slice(s![i, ..]).to_owned();

        let mut probs: Vec<(House, f64)> = weights_dict
            .iter()
            .map(|(house, weights)| {
                let z = row.dot(weights); // dot product between row and weights (weights should be 1D)
                let prob = sigmoid(&Array1::from(vec![z]))[0]; // Apply sigmoid to compute probability
                (*house, prob)
            })
            .collect();

        // Sort houses by probability and pick the house with the highest probability
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        predictions.push(probs[0].0); // The house with the highest probability
    }

    predictions
}

pub fn make_predictions(weights_dict: &HashMap<House, Array1<f64>>, x: &Array2<f64>) {
    let predictions = predict(&x, &weights_dict);
    let file = File::create("predictions.csv").unwrap();
    let mut writer = BufWriter::new(file);
    writeln!(writer, "Index,Hogwarts House").unwrap();

    for (i, pred) in predictions.iter().enumerate() {
        writeln!(writer, "{},{}", i, pred.to_string()).unwrap();
    }
}

pub fn check_accuracy(weights_dict: &HashMap<House, Array1<f64>>, x: &Array2<f64>) {
    let y: Array1<f64> = x.column(0).to_owned();
    // Shadows first x_train, but I need to slice it again to remove the first column
    let x: Array2<f64> = x.slice(s![.., 1..]).to_owned();
    let predictions = predict(&x, &weights_dict);

    println!(
        "Accuracy: {:.2}%",
        calculate_accuracy(&predictions, &y) * 100.0
    );
}
