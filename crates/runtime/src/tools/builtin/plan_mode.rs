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
use tools::{SpiceModelTool, ToolCapability};

use crate::tools::utils::parameters;

pub const ENTER_PLAN_MODE_TOOL_NAME: &str = "enter_plan_mode";
pub const EXIT_PLAN_MODE_TOOL_NAME: &str = "exit_plan_mode";

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct EnterPlanModeParams {
    /// The reason for entering plan mode.
    reason: String,
}

pub struct EnterPlanModeTool {
    name: String,
    description: String,
}

impl EnterPlanModeTool {
    #[must_use]
    pub fn new(name: Option<&str>, description: Option<&str>) -> Self {
        Self {
            name: name.unwrap_or(ENTER_PLAN_MODE_TOOL_NAME).to_string(),
            description: description
                .unwrap_or(
                    "Enter planning mode. In plan mode, only read-only tools (queries, searches, file reads) \
                     are available. Use this to research and gather information before taking action.",
                )
                .to_string(),
        }
    }
}

#[async_trait]
impl SpiceModelTool for EnterPlanModeTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<EnterPlanModeParams>()
    }

    fn capability(&self) -> ToolCapability {
        ToolCapability::ReadOnly
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::enter_plan_mode", tool = self.name().to_string(), input = arg);

        let params: EnterPlanModeParams = serde_json::from_str(arg)?;

        tracing::info!(target: "task_history", parent: &span, captured_output = %params.reason);

        Ok(json!({ "plan_mode": true, "reason": params.reason }))
    }
}

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct ExitPlanModeParams {
    /// The plan summary describing what actions will be taken.
    plan: String,
}

pub struct ExitPlanModeTool {
    name: String,
    description: String,
}

impl ExitPlanModeTool {
    #[must_use]
    pub fn new(name: Option<&str>, description: Option<&str>) -> Self {
        Self {
            name: name.unwrap_or(EXIT_PLAN_MODE_TOOL_NAME).to_string(),
            description: description
                .unwrap_or(
                    "Exit planning mode and unlock all tools. Provide a summary of your plan \
                     describing what actions you will take. After calling this, write tools become available.",
                )
                .to_string(),
        }
    }
}

#[async_trait]
impl SpiceModelTool for ExitPlanModeTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<ExitPlanModeParams>()
    }

    fn capability(&self) -> ToolCapability {
        ToolCapability::ReadOnly
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::exit_plan_mode", tool = self.name().to_string(), input = arg);

        let params: ExitPlanModeParams = serde_json::from_str(arg)?;

        tracing::info!(target: "task_history", parent: &span, captured_output = %params.plan);

        Ok(json!({ "plan_mode": false, "plan": params.plan }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_enter_plan_mode_tool() {
        let tool = EnterPlanModeTool::new(None, None);
        let result = tool
            .call(r#"{"reason": "Need to research before acting"}"#)
            .await
            .unwrap();
        assert_eq!(result["plan_mode"], true);
        assert_eq!(result["reason"], "Need to research before acting");
    }

    #[tokio::test]
    async fn test_exit_plan_mode_tool() {
        let tool = ExitPlanModeTool::new(None, None);
        let result = tool
            .call(r#"{"plan": "1. Query database 2. Update records"}"#)
            .await
            .unwrap();
        assert_eq!(result["plan_mode"], false);
        assert_eq!(result["plan"], "1. Query database 2. Update records");
    }

    #[test]
    fn test_default_names() {
        let enter = EnterPlanModeTool::new(None, None);
        assert_eq!(enter.name(), "enter_plan_mode");
        assert!(enter.description().is_some());
        assert!(enter.parameters().is_some());
        assert_eq!(enter.capability(), ToolCapability::ReadOnly);

        let exit = ExitPlanModeTool::new(None, None);
        assert_eq!(exit.name(), "exit_plan_mode");
        assert!(exit.description().is_some());
        assert!(exit.parameters().is_some());
        assert_eq!(exit.capability(), ToolCapability::ReadOnly);
    }
}
