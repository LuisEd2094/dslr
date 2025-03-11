use dslr::accuracy::{check_accuracy, make_predictions};
use dslr::aux::{get_columns_to_keep, get_describe_from_json, read_csv};
use dslr::log_reg::{train, train_model_gradient, train_model_mbgd, train_model_sgd};
use dslr::structs::{ColumnStats, House};

use clap::Parser;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::error::Error;

/// Command-line argument parser
#[derive(Parser, Debug)]
#[command(name = "ML Trainer", about = "Train a model using SGD | GD | MBGD")]
struct Args {
    /// Choose optimization function (sgd | gd | mbgd)
    #[arg(long, default_value = "gd")]
    func: Option<String>,

    /// Training data file
    #[arg(long, default_value = "/home/luis/proyects/dslr/dataset_train.csv")]
    train: Option<String>,

    /// Testing data file
    #[arg(long, default_value = "/home/luis/proyects/dslr/dataset_test.csv")]
    test: Option<String>,

    /// JSON file with parsed data
    #[arg(long, default_value = "output.json")]
    json: Option<String>,

    /// Learning rate
    #[arg(long, default_value = "0.0000000000000001", allow_hyphen_values = true)]
    lr: Option<f64>,

    /// Number of epochs
    #[arg(long, default_value = "100", allow_hyphen_values = true)]
    epochs: Option<usize>,

    /// Output file
    #[arg(long, default_value = "predictions.csv")]
    output: Option<String>,
}

fn run(
    file_path: &str,
    parsed_data: &HashMap<String, ColumnStats>,
    learning_rate: f64,
    epochs: usize,
    train_func: fn(&Array2<f64>, &Array1<f64>, f64, usize) -> Array1<f64>,
) -> Result<HashMap<House, Array1<f64>>, Box<dyn Error>> {
    // Load your dataset here
    let x = read_csv(file_path, &get_columns_to_keep(file_path), &parsed_data)?;
    Ok(train(&x, learning_rate, epochs, train_func)?)
}

pub fn parse_and_validate_args() -> (
    String,
    String,
    String,
    String,
    f64,
    usize,
    fn(&Array2<f64>, &Array1<f64>, f64, usize) -> Array1<f64>,
) {
    let args = Args::parse();

    let training_file: String = args.train.unwrap();
    let testing_file: String = args.test.unwrap();
    let output_file: String = args.output.unwrap();
    let json: String = args.json.unwrap();
    let learning_rate: f64 = args.lr.unwrap();
    let epochs: usize = args.epochs.unwrap();
    let func: String = args.func.unwrap();

    if learning_rate <= 0.0 {
        panic!("Learning rate (--lr) must be greater than zero!");
    }

    if epochs == 0 {
        panic!("Epochs (--epochs) must be greater than zero!");
    }

    let train_func: fn(&Array2<f64>, &Array1<f64>, f64, usize) -> Array1<f64> = match func.as_str()
    {
        "sgd" => train_model_sgd,
        "gd" => train_model_gradient,
        "mbgd" => train_model_mbgd,
        _ => panic!("Invalid optimization function"),
    };

    (
        training_file,
        testing_file,
        output_file,
        json,
        learning_rate,
        epochs,
        train_func,
    )
}

fn main() {
    let (training_file, testing_file, output_file, json, learning_rate, epochs, train_func) =
        parse_and_validate_args();

    match get_describe_from_json(&json) {
        Ok(parsed_data) => match run(
            &training_file,
            &parsed_data,
            learning_rate,
            epochs,
            train_func,
        ) {
            Ok(weights_dict) => {
                let x_train: Array2<f64> = read_csv(
                    &training_file,
                    &get_columns_to_keep(&training_file),
                    &parsed_data,
                )
                .unwrap();

                let x_test: Array2<f64> = read_csv(
                    &testing_file,
                    &get_columns_to_keep(&testing_file),
                    &parsed_data,
                )
                .unwrap();

                check_accuracy(&weights_dict, &x_train);
                make_predictions(&weights_dict, &x_test, &output_file);
            }
            Err(e) => eprintln!("Error: {}", e),
        },
        Err(e) => eprintln!("Error: {}", e),
    }
}
