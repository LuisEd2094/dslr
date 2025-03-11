use crate::structs::House;
use ndarray::{s, Array1, Array2, Axis};
use std::collections::HashMap;
use std::error::Error;

pub fn train(
    x: &Array2<f64>,
    learning_rate: f64,
    epochs: usize,
    train_func: fn(&Array2<f64>, &Array1<f64>, f64, usize) -> Array1<f64>, // Function parameter
) -> Result<HashMap<House, Array1<f64>>, Box<dyn Error>> {
    // Assumes the first column is the target (house)
    let y = x.index_axis(Axis(1), 0).to_owned();
    // Features are everything except the first column
    let x = x.slice(s![.., 1..]).to_owned();

    // Define the houses (as enum values)
    let houses = vec![
        House::Gryffindor,
        House::Hufflepuff,
        House::Ravenclaw,
        House::Slytherin,
    ];

    let mut weights_dict: HashMap<House, Array1<f64>> = HashMap::new();

    for (idx, house) in houses.iter().enumerate() {
        let y_binary: Array1<f64> = y.mapv(|y| if y == idx as f64 { 1.0 } else { 0.0 });

        // Train the model for the current house
        let weights = train_func(&x, &y_binary, learning_rate, epochs);
        weights_dict.insert(house.clone(), weights);
    }
    Ok(weights_dict)
}
