use crate::log_reg::sigmoid;
use crate::structs::House;
use ndarray::{s, Array1, Array2, Axis};
use std::collections::HashMap;
use std::error::Error;

fn train_model(x: &Array2<f64>, y: &Array1<f64>, learning_rate: f64, epochs: usize) -> Array1<f64> {
    let mut weights = Array1::<f64>::zeros(x.ncols());
    let m = y.len() as f64;

    for _ in 0..epochs {
        let z = x.dot(&weights);
        let h = sigmoid(&z);
        let gradient = x.t().dot(&(h - y)) / m;

        for (i, weight) in weights.iter_mut().enumerate() {
            *weight -= gradient[i] * learning_rate;
        }
    }
    weights
}

pub fn train(
    x: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>,
) -> Result<HashMap<House, Array1<f64>>, Box<dyn Error>> {
    // Assumes the first column is the target (house)
    let y = x.index_axis(Axis(1), 0).to_owned();
    // Features are everything except the first column
    let x = x.slice(s![.., 1..]).to_owned();

    let learning_rate = 0.1; // Learning rate
    let epochs = 100;

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
        let weights = train_model(&x, &y_binary, learning_rate, epochs);
        weights_dict.insert(house.clone(), weights);
    }
    Ok(weights_dict)
}
