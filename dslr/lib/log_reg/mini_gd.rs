use crate::log_reg::sigmoid::sigmoid;
use ndarray::{Array1, Array2, Axis};

pub fn train_model_mbgd(
    x: &Array2<f64>,
    y: &Array1<f64>,
    learning_rate: f64,
    epochs: usize,
) -> Array1<f64> {
    println!("Training using Mini-Batch Gradient Descent");
    let mut weights = Array1::<f64>::zeros(x.ncols());
    let m = y.len();
    let batch_size = 100;
    let indices: Vec<usize> = (0..m).collect();

    for _ in 0..epochs {
        for batch in indices.chunks(batch_size) {
            let x_batch = x.select(Axis(0), batch);
            let y_batch = y.select(Axis(0), batch);
            let z = x_batch.dot(&weights);
            let h = sigmoid(&z);
            let gradient = x_batch.t().dot(&(h - &y_batch)) / batch.len() as f64;

            for (w, &g) in weights.iter_mut().zip(gradient.iter()) {
                *w -= learning_rate * g;
            }
        }
    }
    weights
}
