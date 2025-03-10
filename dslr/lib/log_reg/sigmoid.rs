use ndarray::Array1;

// Sigmoid function (predictor) g(z) = 1 / (1 + e^−z)
pub fn sigmoid(z: &Array1<f64>) -> Array1<f64> {
    z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}
