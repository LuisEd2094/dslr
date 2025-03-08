use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::error::Error;
use std::f64::consts::E;
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

fn train_model(
    x: &Array2<f64>,
    y: &Array1<f64>,
    weights: &mut Array1<f64>,
    alpha: f64,
    epochs: usize,
) {
    let m = y.len() as f64;

    for _ in 0..epochs {
        let z = x.dot(weights);
        let h = sigmoid(&z);
        let gradient = x.t().dot(&(h - y)) / m;

        for (i, weight) in weights.iter_mut().enumerate() {
            *weight -= gradient[i] * alpha;
        }
    }
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

fn run(file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Open and read the JSON file
    let parsed_data: Result<HashMap<String, ColumnStats>, Box<dyn Error>> =
        get_describe_from_json("output.json");
    let x = read_csv(
        file_path,
        &get_columns_to_keep(file_path),
        &parsed_data.unwrap(),
    ).unwrap(); // Adjust path as needed
    // Initialize weights to small random values
    let mut weights = Array1::<f64>::zeros(x.ncols());
    // Hyperparameters
    let alpha = 0.1; // Learning rate
    let epochs = 1000;
    let y = x.index_axis(Axis(1), 0).to_owned();

    let weights = train_model(&x[1:], &y, &mut weights, alpha, epochs);
    Ok(weights)
}

fn main() {
    match run("/home/luis/proyects/dslr/dataset_train.csv") {
        Ok(_) => println!("Processing completed successfully."),
        Err(e) => eprintln!("Error: {}", e),
    }
}
