use dslr::accuracy::{check_accuracy, make_predictions};
use dslr::aux::{get_columns_to_keep, get_describe_from_json, read_csv};
use dslr::log_reg::train;
use dslr::structs::{ColumnStats, House};

use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::error::Error;

fn run(
    file_path: &str,
    parsed_data: &HashMap<String, ColumnStats>,
) -> Result<HashMap<House, Array1<f64>>, Box<dyn Error>> {
    // Load your dataset here
    let x = read_csv(file_path, &get_columns_to_keep(file_path), &parsed_data)?;
    Ok(train(&x)?)
}

fn main() {
    let training_file = "/home/luis/proyects/dslr/dataset_train.csv";
    let testing_file = "/home/luis/proyects/dslr/dataset_test.csv";
    match get_describe_from_json("output.json") {
        Ok(parsed_data) => match run(training_file, &parsed_data) {
            Ok(weights_dict) => {
                let x_train: Array2<f64> = read_csv(
                    training_file,
                    &get_columns_to_keep(training_file),
                    &parsed_data,
                )
                .unwrap();
                let x_test: Array2<f64> = read_csv(
                    testing_file,
                    &get_columns_to_keep(testing_file),
                    &parsed_data,
                )
                .unwrap();

                check_accuracy(&weights_dict, &x_train);
                make_predictions(&weights_dict, &x_test);
            }
            Err(e) => eprintln!("Error: {}", e),
        },
        Err(e) => eprintln!("Error: {}", e),
    }
}
