use dslr::accuracy::check_accuracy;
use dslr::aux::{get_columns_to_keep, get_describe_from_json, read_csv};
use dslr::log_reg::train;
use dslr::structs::{ColumnStats, House};

use ndarray::Array1;
use std::collections::HashMap;
use std::error::Error;

fn run(file_path: &str) -> Result<HashMap<House, Array1<f64>>, Box<dyn Error>> {
    // Load your dataset here
    let parsed_data: HashMap<String, ColumnStats> = get_describe_from_json("output.json")?;
    let x = read_csv(file_path, &get_columns_to_keep(file_path), &parsed_data)?;

    Ok(train(&x)?)
}

fn main() {
    let file_path = "/home/luis/proyects/dslr/dataset_train.csv";
    match run(file_path) {
        Ok(weights_dict) => {
            check_accuracy(weights_dict, file_path);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
