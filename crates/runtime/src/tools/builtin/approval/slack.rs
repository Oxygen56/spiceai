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
use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::borrow::Cow;
use std::time::Duration;
use tokio::sync::oneshot;
use tools::SpiceModelTool;
use tracing::Span;
use tracing_futures::Instrument;

use crate::tools::utils::parameters;

use super::ApprovalTimeoutAction;
use super::store::{ApprovalResponse, ApprovalStore, PendingApproval, PendingApprovalInfo};

/// Parameters the agent passes when calling the Slack approval tool.
#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct ApprovalSlackToolParams {
    /// What needs approval and why.
    message: String,

    /// Optional additional context (e.g., diff summary, changed files).
    context: Option<String>,

    /// Slack channel to post the approval request to.
    channel: String,

    /// Thread timestamp — if provided, the approval request is posted as a
    /// reply in the existing thread. Get this from a previous slack tool call's `ts` field.
    thread_ts: Option<String>,
}

/// Slack approval tool — posts a message with approve/reject links to Slack,
/// then blocks until resolved via HTTP callback or timeout.
///
/// Supports threading: if `thread_ts` is provided, the approval request is
/// posted as a reply in the existing conversation thread.
pub struct ApprovalSlackTool {
    name: String,
    description: String,
    store: ApprovalStore,
    timeout: Duration,
    timeout_action: ApprovalTimeoutAction,
    base_url: Option<String>,
    slack_token: String,
    client: reqwest::Client,
}

impl ApprovalSlackTool {
    #[must_use]
    pub fn new(
        name: Option<&str>,
        description: Option<&str>,
        store: ApprovalStore,
        timeout: Duration,
        timeout_action: ApprovalTimeoutAction,
        base_url: Option<String>,
        slack_token: String,
    ) -> Self {
        Self {
            name: name.unwrap_or("approval_slack").to_string(),
            description: description
                .unwrap_or(
                    "Request human approval via Slack. Posts an approval request with \
                     approve/reject links. Supports threading via thread_ts parameter.",
                )
                .to_string(),
            store,
            timeout,
            timeout_action,
            base_url,
            slack_token,
            client: reqwest::Client::new(),
        }
    }

    /// Post the approval request to Slack via `chat.postMessage`.
    async fn post_to_slack(
        &self,
        params: &ApprovalSlackToolParams,
        approval_id: &str,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let mut text = format!("*Approval Required*\n\n{}", params.message);
        if let Some(ref context) = params.context {
            text.push_str(&format!("\n\n*Context:*\n```\n{context}\n```"));
        }

        // Add approve/reject links if base_url is configured
        if let Some(ref base_url) = self.base_url {
            let approve_url = format!("{base_url}/v1/agents/approvals/{approval_id}/approve");
            let reject_url = format!("{base_url}/v1/agents/approvals/{approval_id}/reject");
            text.push_str(&format!(
                "\n\n<{approve_url}|:white_check_mark: Approve>  |  <{reject_url}|:x: Reject>"
            ));
        }

        let mut body = json!({
            "channel": params.channel,
            "text": text,
            "unfurl_links": false,
        });

        // Post as thread reply if thread_ts is provided
        if let Some(ref thread_ts) = params.thread_ts {
            body["thread_ts"] = json!(thread_ts);
        }

        let response = self
            .client
            .post("https://slack.com/api/chat.postMessage")
            .header("Authorization", format!("Bearer {}", self.slack_token))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        let response_body: Value = response.json().await?;

        if !status.is_success() || response_body["ok"] == false {
            let error = response_body["error"]
                .as_str()
                .unwrap_or("unknown error");
            return Err(format!("Slack API error: {error}").into());
        }

        Ok(response_body)
    }
}

#[async_trait]
impl SpiceModelTool for ApprovalSlackTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<ApprovalSlackToolParams>()
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span: Span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::approval_slack", tool = self.name().to_string(), input = arg);

        let tool_use_result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let params: ApprovalSlackToolParams = serde_json::from_str(arg)?;
            let approval_id = uuid::Uuid::now_v7().to_string();
            let (tx, rx) = oneshot::channel();

            let info = PendingApprovalInfo {
                id: approval_id.clone(),
                agent_name: "agent".to_string(),
                message: params.message.clone(),
                context: params.context.clone(),
                created_at: Utc::now(),
            };

            self.store.store(PendingApproval { info, tx }).await;

            // Post to Slack
            match self.post_to_slack(&params, &approval_id).await {
                Ok(slack_response) => {
                    let ts = slack_response["ts"].as_str().unwrap_or("");
                    tracing::info!(
                        target: "task_history",
                        parent: &span,
                        approval_id = %approval_id,
                        slack_ts = %ts,
                        channel = %params.channel,
                        "Approval posted to Slack, waiting for response"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        target: "task_history",
                        parent: &span,
                        approval_id = %approval_id,
                        error = %e,
                        "Failed to post approval to Slack, still waiting for HTTP resolution"
                    );
                }
            }

            // Wait for response or timeout
            let response = tokio::select! {
                result = rx => {
                    result.map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
                        "Approval channel closed unexpectedly".into()
                    })?
                }
                _ = tokio::time::sleep(self.timeout) => {
                    self.store.remove(&approval_id).await;
                    let approved = matches!(self.timeout_action, ApprovalTimeoutAction::Approve);
                    ApprovalResponse {
                        approved,
                        comment: Some(format!("Timed out after {}s", self.timeout.as_secs())),
                        responded_at: Utc::now(),
                    }
                }
            };

            Ok(json!({
                "approved": response.approved,
                "comment": response.comment,
                "approval_id": approval_id,
            }))
        }
        .instrument(span.clone())
        .await;

        match tool_use_result {
            Ok(value) => {
                let captured_output_json = serde_json::to_string(&value)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_name_and_description() {
        let store = ApprovalStore::default();
        let tool = ApprovalSlackTool::new(
            None,
            None,
            store,
            Duration::from_secs(3600),
            ApprovalTimeoutAction::Reject,
            None,
            "xoxb-test-token".to_string(),
        );
        assert_eq!(tool.name(), "approval_slack");
        assert!(tool.description().is_some());
        assert!(tool.parameters().is_some());
    }

    #[test]
    fn test_custom_name() {
        let store = ApprovalStore::default();
        let tool = ApprovalSlackTool::new(
            Some("my_slack_approval"),
            Some("Custom approval tool"),
            store,
            Duration::from_secs(3600),
            ApprovalTimeoutAction::Reject,
            Some("https://example.com".to_string()),
            "xoxb-test-token".to_string(),
        );
        assert_eq!(tool.name(), "my_slack_approval");
        assert_eq!(
            tool.description().unwrap().to_string(),
            "Custom approval tool"
        );
    }

    #[tokio::test]
    async fn test_approval_slack_timeout() {
        let store = ApprovalStore::default();
        let tool = ApprovalSlackTool::new(
            None,
            None,
            store.clone(),
            Duration::from_millis(100),
            ApprovalTimeoutAction::Reject,
            None,
            "xoxb-test-token".to_string(), // Will fail to post, but timeout will resolve
        );

        let result = tool
            .call(r##"{"message": "Approve?", "channel": "#test"}"##)
            .await
            .unwrap();

        assert_eq!(result["approved"], false);
        assert!(result["comment"]
            .as_str()
            .unwrap()
            .contains("Timed out"));
    }
}
