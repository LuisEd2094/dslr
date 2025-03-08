use ndarray::s;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use serde_json::{self};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::Read;
use std::path::Path;
use std::vec::Vec;

// Sigmoid function (predictor) g(z) = 1 / (1 + e^−z)
fn sigmoid(z: &Array1<f64>) -> Array1<f64> {
    z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

fn train_model(x: &Array2<f64>, y: &Array1<f64>, alpha: f64, epochs: usize) -> Array1<f64> {
    let mut weights = Array1::<f64>::zeros(x.ncols());
    let m = y.len() as f64;

    for _ in 0..epochs {
        let z = x.dot(&weights);
        let h = sigmoid(&z);
        let gradient = x.t().dot(&(h - y)) / m;

        for (i, weight) in weights.iter_mut().enumerate() {
            *weight -= gradient[i] * alpha;
        }
    }
    weights
}

#[derive(Debug, Serialize, Deserialize)]
struct ColumnStats {
    count: Option<f64>,
    mean: Option<f64>,
    std: Option<f64>,
    #[serde(rename = "50%")]
    median: Option<f64>,
}

// Create House enum to better map house names
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum House {
    Gryffindor,
    Hufflepuff,
    Ravenclaw,
    Slytherin,
}

impl House {
    fn from_str(s: &str) -> Option<House> {
        match s {
            "Gryffindor" => Some(House::Gryffindor),
            "Hufflepuff" => Some(House::Hufflepuff),
            "Ravenclaw" => Some(House::Ravenclaw),
            "Slytherin" => Some(House::Slytherin),
            _ => None,
        }
    }

    fn from_index(index: i32) -> Option<&'static str> {
        match index {
            0 => Some("Gryffindor"),
            1 => Some("Hufflepuff"),
            2 => Some("Ravenclaw"),
            3 => Some("Slytherin"),
            _ => None,
        }
    }

    fn to_index(self) -> i32 {
        match self {
            House::Gryffindor => 0,
            House::Hufflepuff => 1,
            House::Ravenclaw => 2,
            House::Slytherin => 3,
        }
    }
}

use std::fmt;

impl fmt::Display for House {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let house_name = match *self {
            House::Gryffindor => "Gryffindor",
            House::Hufflepuff => "Hufflepuff",
            House::Ravenclaw => "Ravenclaw",
            House::Slytherin => "Slytherin",
        };
        write!(f, "{}", house_name)
    }
}

fn get_describe_from_json(file_path: &str) -> Result<HashMap<String, ColumnStats>, Box<dyn Error>> {
    let mut describe_data = File::open(file_path)?;
    let mut contents = String::new();
    describe_data.read_to_string(&mut contents)?;
    let parsed_data: HashMap<String, ColumnStats> = serde_json::from_str(&contents)?;
    Ok(parsed_data)
}

// Function to read CSV and extract needed columns
fn read_csv(
    file_path: &str,
    columns_to_keep: &[usize],
    stats: &HashMap<String, ColumnStats>,
) -> Result<Array2<f64>, Box<dyn Error>> {
    let path: &Path = Path::new(file_path);
    let file: File = File::open(&path)?;
    let reader: io::BufReader<File> = io::BufReader::new(file);

    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut headers: Vec<String> = Vec::new();

    // Read the file line by line
    for (i, line) in reader.lines().enumerate() {
        let line = line?; // Unwrap the line result
        let columns: Vec<&str> = line.split(',').collect();

        // First row contains the headers
        if i == 0 {
            headers = columns.iter().map(|&s| s.to_string()).collect();
            continue;
        }

        // Only keep specified columns (ignoring first column, House name)
        let mut values: Vec<f64> = Vec::new();
        for &idx in columns_to_keep {
            if let Some(col_name) = headers.get(idx) {
                // check if it's a house and transform it to its index,
                // if its a valid column check if its not empty, if it is use the median value
                if let Some(house) = House::from_str(columns[idx]) {
                    values.push(house.to_index() as f64);
                } else {
                    if let Some(stat) = stats.get(col_name) {
                        let mean = stat.mean.unwrap(); // Assuming mean is Some(f64)
                        let std = stat.std.unwrap(); // Assuming std is Some(f64)

                        let val; // Declare val before the conditional blocks

                        // Check if the column value can be parsed to a f64
                        if let Ok(parsed_value) = columns[idx].parse::<f64>() {
                            // Normalize the parsed value using the formula: (x - mean) / std
                            val = (parsed_value - mean) / std;
                        } else {
                            // If the column value can't be parsed, use the median
                            val = (stat.median.unwrap() - mean) / std; // Normalize using the median instead
                        }

                        // Add the computed value (normalized) to the values vector
                        values.push(val);
                    }
                }
            }
        }
        if !values.is_empty() {
            data.push(values);
        }
    }

    // Convert to ndarray Array2<f64>
    let data_array = Array2::from_shape_vec(
        (data.len(), data[0].len()),
        data.into_iter().flatten().collect(),
    )?;

    Ok(data_array)
}

fn get_columns_to_keep(csv_file_path: &str) -> Vec<usize> {
    // Columns to keep (ignoring first column, House name)
    let columns_to_drop = vec!["Index", "First Name", "Last Name", "Birthday", "Best Hand"];

    let file = File::open(csv_file_path).unwrap();
    let reader = io::BufReader::new(file);
    let headers: Vec<String> = reader
        .lines()
        .next()
        .unwrap()
        .unwrap()
        .split(',')
        .map(|s| s.to_string())
        .collect();

    let mut columns_to_keep = Vec::new();
    for (index, header) in headers.iter().enumerate() {
        // Only keep the columns that are not in the drop list
        if !columns_to_drop.contains(&header.as_str()) {
            columns_to_keep.push(index);
        }
    }
    columns_to_keep
}

fn run(file_path: &str) -> Result<HashMap<House, Array1<f64>>, Box<dyn Error>> {
    // Load your dataset here
    let parsed_data: HashMap<String, ColumnStats> = get_describe_from_json("output.json")?;

    // Read the data from CSV
    let x = read_csv(file_path, &get_columns_to_keep(file_path), &parsed_data)?;
    let y = x.index_axis(Axis(1), 0).to_owned(); // Assumes the first column is the target (house)
    let x = x.slice(s![.., 1..]).to_owned(); // Features are everything except the first column

    // Hyperparameters for training
    let alpha = 0.1; // Learning rate
    let epochs = 100;

    // Define the houses (as enum values)
    let houses = vec![
        House::Gryffindor,
        House::Hufflepuff,
        House::Ravenclaw,
        House::Slytherin,
    ];

    // Initialize a HashMap to store weights for each house
    let mut weights_dict: HashMap<House, Array1<f64>> = HashMap::new();

    // Train a separate logistic regression model for each house (1-vs-All)
    for (idx, house) in houses.iter().enumerate() {
        let y_binary: Array1<f64> = y.mapv(|y| if y == idx as f64 { 1.0 } else { 0.0 });

        // Train the model for the current house
        let weights = train_model(&x, &y_binary, alpha, epochs);

        // Store the weights for the current house
        weights_dict.insert(house.clone(), weights);
    }
    Ok(weights_dict)
}
#[derive(Serialize, Deserialize)]
struct Weights {
    weights: Vec<f64>,
}
// Prediction function using the trained models
fn predict(x: &Array2<f64>, weights_dict: &HashMap<House, Array1<f64>>) -> Vec<House> {
    let mut predictions = Vec::new();
    let x = x.slice(s![.., 1..]).to_owned();
    for i in 0..x.nrows() {
        let row = x.slice(s![i, ..]).to_owned();

        // Compute the probabilities for each house
        let mut probs: Vec<(House, f64)> = weights_dict
            .iter()
            .map(|(house, weights)| {
                let z = row.dot(weights); // dot product between row and weights (weights should be 1D)
                let prob = sigmoid(&Array1::from(vec![z]))[0]; // Apply sigmoid to compute probability
                (*house, prob)
            })
            .collect();

        // Sort houses by probability and pick the house with the highest probability
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Sort in descending order by probability
        predictions.push(probs[0].0); // The house with the highest probability
    }

    predictions
}
fn calculate_accuracy(predictions: &Vec<House>, actual: &Array1<f64>) -> f64 {
    let mut correct_predictions = 0;

    // Loop through predictions and actual values
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
fn main() {
    let file_path = "/home/luis/proyects/dslr/dataset_train.csv";
    match run(file_path) {
        Ok(weights_dict) => {
            // Load the parsed data from the JSON file
            let parsed_data: HashMap<String, ColumnStats> =
                get_describe_from_json("output.json").unwrap();
            let x = read_csv(file_path, &get_columns_to_keep(file_path), &parsed_data).unwrap();
            let y = x.index_axis(Axis(1), 0).to_owned(); // Assuming the first column is the true label (house)

            // Get predictions using the trained models and the same training data (x)
            let predictions = predict(&x, &weights_dict);

            // Calculate accuracy
            println!("Accuracy: {:.2}%", calculate_accuracy(&predictions, &y) * 100.0);

            // Output the predictions and accuracy
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
