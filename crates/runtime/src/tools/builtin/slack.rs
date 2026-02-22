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
use serde_json::{Value, json};
use snafu::ResultExt;
use std::borrow::Cow;
use tools::{SpiceModelTool, ToolCapability};
use tracing::Span;
use tracing_futures::Instrument;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::tools::utils::parameters;

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct SlackToolParams {
    /// The action to perform: "read" to read messages or "post" to send a message.
    action: String,

    /// The Slack channel to interact with.
    channel: String,

    /// The message text to send. Required when action is "post".
    message: Option<String>,

    /// If provided, post as a reply in this thread. Value is the `ts` from a previous post.
    thread_ts: Option<String>,
}

pub struct SlackTool {
    name: String,
    description: String,
    channels: Vec<String>,
    capability: ToolCapability,
}

impl SlackTool {
    #[must_use]
    pub fn new(
        name: Option<&str>,
        description: Option<&str>,
        channels: Vec<String>,
        capability: ToolCapability,
    ) -> Self {
        Self {
            name: name.unwrap_or("slack").to_string(),
            description: description
                .unwrap_or("Read or post messages in Slack channels")
                .to_string(),
            channels,
            capability,
        }
    }
}

#[async_trait]
impl SpiceModelTool for SlackTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<SlackToolParams>()
    }

    fn capability(&self) -> ToolCapability {
        self.capability
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span: Span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::slack", tool = self.name().to_string(), input = arg);

        let tool_use_result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let req: SlackToolParams = serde_json::from_str(arg)?;

            // Validate action
            if req.action != "read" && req.action != "post" {
                return Err(format!(
                    "Invalid action '{}'. Must be 'read' or 'post'",
                    req.action
                )
                .into());
            }

            // Validate channel is in allowed list
            if !self.channels.iter().any(|c| c == &req.channel) {
                return Err(format!(
                    "Channel '{}' is not in allowed channels: {:?}",
                    req.channel, self.channels
                )
                .into());
            }

            // Block post action in read-only mode
            if self.capability == ToolCapability::ReadOnly && req.action == "post" {
                return Err(
                    "Action 'post' is not permitted in read-only mode".to_string().into()
                );
            }

            // Validate message is provided for post action
            if req.action == "post" && req.message.is_none() {
                return Err(
                    "A 'message' is required when action is 'post'".to_string().into()
                );
            }

            // Stub implementation
            Ok(json!({
                "status": "stub",
                "message": "slack integration not yet implemented",
                "action": req.action,
                "channel": req.channel,
            }))
        }
        .instrument(span.clone())
        .await;

        match tool_use_result {
            Ok(value) => {
                let captured_output_json = serde_json::to_string(&value).boxed()?;
                tracing::info!(target: "task_history", parent: &span, captured_output = %captured_output_json);
                Ok(value)
            }
            Err(e) => {
                tracing::error!(target: "task_history", parent: &span, "{e}");
                Err(e)
            }
        }
    }
}
