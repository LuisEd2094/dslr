pub mod sigmoid;
pub mod train;
pub mod gradient;
pub mod sgd;
pub mod mini_gd;

pub use sigmoid::{sigmoid, sigmoid_single};
pub use train::train;
pub use gradient::train_model_gradient;
pub use sgd::train_model_sgd;
pub use mini_gd::train_model_mbgd;