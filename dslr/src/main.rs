use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::vec::Vec;

// Sigmoid function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Function to read CSV and extract needed columns
fn read_csv(file_path: &str, column_name: &str) -> io::Result<Vec<(f64, i32)>> {
    let path: &Path = Path::new(file_path);
    let file: File = File::open(&path)?;
    let reader: io::BufReader<File> = io::BufReader::new(file);
    let mut column_index: Option<usize> = None;

    let mut data: Vec<(f64, i32)> = Vec::new();
    for (e, l) in reader.lines().enumerate() {
        // We can't set types in for loops, so we need to specify them here
        // We also unwrap the result of the parse method, if we know it will always succeed
        let (i, line): (usize, String) = (e, l?);
        let columns: Vec<&str> = line.split(',').collect();
        if i == 0 {
            // Find the index of the requested column
            column_index = columns.iter().position(|&col| col == column_name);
            if column_index.is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Column not found",
                ));
            }
            continue; // Skip header row
        }
        if let Some(idx) = column_index {
            let feature_value: f64 = columns[idx].parse().unwrap_or(0.0);
            let house = match columns[1] {
                "Gryffindor" => 0,
                "Hufflepuff" => 1,
                "Ravenclaw" => 2,
                "Slytherin" => 3,
                _ => continue,
            };
            data.push((feature_value, house));
        }
    }
    Ok(data)
}

// Compute the cost function
fn cost_function(weights: f64, data: &Vec<(f64, i32)>) -> f64 {
    let mut cost: f64 = 0.0;
    for (e, e1) in data {
        let (x, y): (&f64, &i32) = (e, e1);
        let y_hat: f64 = sigmoid(weights * x);
        let y_f: f64 = *y as f64;
        cost += -y_f * y_hat.ln() - (1.0 - y_f) * (1.0 - y_hat).ln();
    }
    cost / data.len() as f64
}

// Perform gradient descent
fn gradient_descent(
    mut weights: f64,
    data: &Vec<(f64, i32)>,
    learning_rate: f64,
    iterations: usize,
) -> (f64, f64) {
    let mut cost = cost_function(weights, data);
    for _ in 0..iterations {
        let mut gradient = 0.0;
        for (x, y) in data {
            let y_hat = sigmoid(weights * x);
            let y_f = *y as f64;
            gradient += (y_hat - y_f) * x;
        }
        gradient /= data.len() as f64;
        weights -= learning_rate * gradient;
        cost = cost_function(weights, data);
    }
    (weights, cost)
}

fn main() {
    let file_path = "../dataset_train.csv"; // Adjust path as needed
    match read_csv(file_path, "Potions") {
        Ok(data) => {
            let learning_rate: f64 = 0.01;
            let iterations: usize = 1000;
            let initial_weight: f64 = 0.0;
            let trained_weight: (f64, f64) =
                gradient_descent(initial_weight, &data, learning_rate, iterations);
            println!("Trained weight: {:?}", trained_weight);
        }
        Err(e) => eprintln!("Error reading file: {}", e),
    }
}
