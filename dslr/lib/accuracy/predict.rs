use crate::accuracy::accuracy::calculate_accuracy;
use crate::log_reg::sigmoid::sigmoid;
use crate::structs::{House, ColumnStats};
use crate::aux::{get_describe_from_json, read_csv, get_columns_to_keep};
use ndarray::{s, Array1, Array2, Axis};
use std::collections::HashMap;

fn predict(x: &Array2<f64>, weights_dict: &HashMap<House, Array1<f64>>) -> Vec<House> {
    let mut predictions = Vec::new();
    let x = x.slice(s![.., 1..]).to_owned();
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

pub fn check_accuracy(weights_dict: HashMap<House, Array1<f64>>, file_path: &str) {
    // Load the parsed data from the JSON file
    let parsed_data: HashMap<String, ColumnStats> = get_describe_from_json("output.json").unwrap();
    let x = read_csv(file_path, &get_columns_to_keep(file_path), &parsed_data).unwrap();
    let y = x.index_axis(Axis(1), 0).to_owned(); // Assuming the first column is the true label (house)

    // Get predictions using the trained models and the same training data (x)
    let predictions = predict(&x, &weights_dict);

    // Calculate accuracy
    println!(
        "Accuracy: {:.2}%",
        calculate_accuracy(&predictions, &y) * 100.0
    );

    // Output the predictions and accuracy
}
