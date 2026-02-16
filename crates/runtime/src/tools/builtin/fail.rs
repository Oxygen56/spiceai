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
pub struct FailToolParams {
    /// The reason for failure.
    reason: String,
}

pub struct FailTool {
    name: String,
    description: String,
}

impl FailTool {
    #[must_use]
    pub fn new(name: Option<&str>, description: Option<&str>) -> Self {
        Self {
            name: name.unwrap_or("fail").to_string(),
            description: description
                .unwrap_or("Signal that this step cannot be completed. Use when the input is invalid, information is missing, or the task cannot proceed.")
                .to_string(),
        }
    }
}

#[async_trait]
impl SpiceModelTool for FailTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<FailToolParams>()
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::fail", tool = self.name().to_string(), input = arg);

        let params: FailToolParams = serde_json::from_str(arg)?;

        tracing::info!(target: "task_history", parent: &span, captured_output = %params.reason);

        Ok(json!({ "failed": true, "reason": params.reason }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fail_tool() {
        let tool = FailTool::new(None, None);
        let result = tool
            .call(r#"{"reason": "missing required input"}"#)
            .await
            .unwrap();
        assert_eq!(result["failed"], true);
        assert_eq!(result["reason"], "missing required input");
    }

    #[test]
    fn test_default_name_and_description() {
        let tool = FailTool::new(None, None);
        assert_eq!(tool.name(), "fail");
        assert!(tool.description().is_some());
    }

    #[test]
    fn test_has_parameters() {
        let tool = FailTool::new(None, None);
        assert!(tool.parameters().is_some());
    }
}
