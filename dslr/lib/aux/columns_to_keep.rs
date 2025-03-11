use std::fs::File;
use std::io::{self, BufRead};

pub fn get_columns_to_keep(csv_file_path: &str) -> Vec<usize> {
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
        if !columns_to_drop.contains(&header.as_str()) {
            columns_to_keep.push(index);
        }
    }
    columns_to_keep
}
