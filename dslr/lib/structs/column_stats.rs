use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ColumnStats {
    pub count: Option<f64>,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    #[serde(rename = "50%")]
    pub median: Option<f64>,
}
