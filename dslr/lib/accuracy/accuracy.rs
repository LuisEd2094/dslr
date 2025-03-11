use crate::structs::House;
use ndarray::Array1;

pub fn calculate_accuracy(predictions: &Vec<House>, actual: &Array1<f64>) -> f64 {
    let mut correct_predictions = 0;

    for (pred, actual_label) in predictions.iter().zip(actual.iter()) {
        // Convert actual_label to a House using from_index
        if let Some(actual_house) = House::from_index(*actual_label as i32) {
            // Compare predicted house with the actual house
            let pred_str = pred.to_string();
            if pred_str == actual_house {
                correct_predictions += 1;
            }
        }
    }

    // Return the accuracy
    correct_predictions as f64 / predictions.len() as f64
}