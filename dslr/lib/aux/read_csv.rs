use std::path::Path;
use std::fs::File;
use std::io::{self, BufRead};
use std::error::Error;
use std::collections::HashMap;
use ndarray::Array2;
use crate::structs::{ColumnStats, House};

// Function to read CSV and extract needed columns
pub fn read_csv(
    file_path: &str,
    columns_to_keep: &[usize],
    stats: &HashMap<String, ColumnStats>,
) -> Result<Array2<f64>, Box<dyn Error>> {
    let path: &Path = Path::new(file_path);
    let file: File = File::open(&path)?;
    let reader: io::BufReader<File> = io::BufReader::new(file);

    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut headers: Vec<String> = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line: String = line?; // Unwrap the line result
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
                        let mean: f64 = stat.mean.unwrap();
                        let std: f64 = stat.std.unwrap();

                        let val: f64;

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
    let data_array: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> = Array2::from_shape_vec(
        (data.len(), data[0].len()),
        data.into_iter().flatten().collect(),
    )?;

    Ok(data_array)
}