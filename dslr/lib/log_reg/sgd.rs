use crate::log_reg::sigmoid::sigmoid_single;
use ndarray::{Array1, Array2, Axis};

pub fn train_model_sgd(
    x: &Array2<f64>,
    y: &Array1<f64>,
    learning_rate: f64,
    epochs: usize,
) -> Array1<f64> {
    println!("Training using Stochastic Gradient Descent");
    let mut weights = Array1::<f64>::zeros(x.ncols());

    for _ in 0..epochs {
        for i in 0..y.len() {
            let x_i = x.index_axis(Axis(0), i);
            let y_i = y[i];

            let z = x_i.dot(&weights);
            let h = sigmoid_single(z);
            let gradient = &x_i * (h - y_i);

            for (w, &g) in weights.iter_mut().zip(gradient.iter()) {
                *w -= learning_rate * g;
            }
        }
    }
    weights
}
