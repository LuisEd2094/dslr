use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use crate::structs::ColumnStats;
use std::io::Read;

pub fn get_describe_from_json(file_path: &str) -> Result<HashMap<String, ColumnStats>, Box<dyn Error>> {
    let mut describe_data = File::open(file_path)?;
    let mut contents = String::new();
    describe_data.read_to_string(&mut contents)?;
    let parsed_data: HashMap<String, ColumnStats> = serde_json::from_str(&contents)?;
    Ok(parsed_data)
}