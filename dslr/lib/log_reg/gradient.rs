use crate::log_reg::sigmoid;
use ndarray::{Array1, Array2};

pub fn train_model_gradient(
    x: &Array2<f64>,
    y: &Array1<f64>,
    learning_rate: f64,
    epochs: usize,
) -> Array1<f64> {
    println!("Training using Batch Gradient Descent");
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
