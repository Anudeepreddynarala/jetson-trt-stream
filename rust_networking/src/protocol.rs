use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BoundingBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Detection {
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InferenceMetrics {
    pub preprocessing_ms: f64,
    pub inference_ms: f64,
    pub postprocessing_ms: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DetectionMessage {
    pub message_type: String,
    pub timestamp_ns: u64,
    pub frame_id: u64,
    pub detections: Vec<Detection>,
    pub metrics: InferenceMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "command")]
pub enum Command {
    #[serde(rename = "START")]
    Start,
    #[serde(rename = "STOP")]
    Stop,
    #[serde(rename = "STATS")]
    Stats,
    #[serde(rename = "GET_CONFIG")]
    GetConfig,
}

#[derive(Debug, Serialize)]
pub struct CommandResponse {
    pub status: String,
    pub data: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl CommandResponse {
    pub fn success(data: serde_json::Value) -> Self {
        Self {
            status: "success".to_string(),
            data,
            message: None,
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            status: "error".to_string(),
            data: serde_json::json!({}),
            message: Some(message),
        }
    }
}
