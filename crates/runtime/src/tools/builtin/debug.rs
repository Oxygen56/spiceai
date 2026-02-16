/*
Copyright 2024-2025 The Spice.ai OSS Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::borrow::Cow;
use tools::SpiceModelTool;

use crate::tools::utils::parameters;

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct DebugToolParams {
    /// The message to print.
    message: String,
}

pub struct DebugTool {
    name: String,
    description: String,
}

impl DebugTool {
    #[must_use]
    pub fn new(name: Option<&str>, description: Option<&str>) -> Self {
        Self {
            name: name.unwrap_or("debug").to_string(),
            description: description
                .unwrap_or("Print a debug message to the task history log")
                .to_string(),
        }
    }
}

#[async_trait]
impl SpiceModelTool for DebugTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<DebugToolParams>()
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::debug", tool = self.name().to_string(), input = arg);

        let params: DebugToolParams = serde_json::from_str(arg)?;

        tracing::info!(target: "task_history", parent: &span, captured_output = %params.message);

        Ok(json!({ "message": params.message }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_debug_tool() {
        let tool = DebugTool::new(None, None);
        let result = tool
            .call(r#"{"message": "hello world"}"#)
            .await
            .unwrap();
        assert_eq!(result["message"], "hello world");
    }

    #[test]
    fn test_default_name_and_description() {
        let tool = DebugTool::new(None, None);
        assert_eq!(tool.name(), "debug");
        assert!(tool.description().is_some());
    }

    #[test]
    fn test_has_parameters() {
        let tool = DebugTool::new(None, None);
        assert!(tool.parameters().is_some());
    }
}
