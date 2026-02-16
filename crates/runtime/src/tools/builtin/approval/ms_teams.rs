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

/// Parameters the agent passes when calling the MS Teams approval tool.
#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct ApprovalMsTeamsToolParams {
    /// What needs approval and why.
    message: String,

    /// Optional additional context (e.g., diff summary, changed files).
    context: Option<String>,
}

/// MS Teams approval tool — posts an Adaptive Card with approve/reject action
/// buttons, then blocks until resolved via HTTP callback or timeout.
///
/// Note: Incoming webhooks don't support threading. Each approval is posted as
/// a new top-level card. Graph API support can be added later for threaded replies.
pub struct ApprovalMsTeamsTool {
    name: String,
    description: String,
    store: ApprovalStore,
    timeout: Duration,
    timeout_action: ApprovalTimeoutAction,
    base_url: Option<String>,
    webhook_url: String,
    client: reqwest::Client,
}

impl ApprovalMsTeamsTool {
    /// Create a new `ApprovalMsTeamsTool`.
    ///
    /// # Errors
    ///
    /// Returns an error if the webhook URL is not a valid Microsoft Teams domain.
    pub fn try_new(
        name: Option<&str>,
        description: Option<&str>,
        store: ApprovalStore,
        timeout: Duration,
        timeout_action: ApprovalTimeoutAction,
        base_url: Option<String>,
        webhook_url: String,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let parsed = url::Url::parse(&webhook_url)?;
        let host = parsed
            .host_str()
            .ok_or("Webhook URL does not contain a host")?;

        let allowed = host.ends_with(".webhook.office.com")
            || host.ends_with(".logic.azure.com")
            || host.ends_with(".office365.com");

        if !allowed {
            return Err(format!(
                "Webhook URL host '{host}' is not an allowed Microsoft Teams domain. \
                 Allowed domains: *.webhook.office.com, *.logic.azure.com, *.office365.com"
            )
            .into());
        }

        Ok(Self {
            name: name.unwrap_or("approval_ms_teams").to_string(),
            description: description
                .unwrap_or(
                    "Request human approval via Microsoft Teams. Posts an Adaptive Card \
                     with approve/reject action buttons.",
                )
                .to_string(),
            store,
            timeout,
            timeout_action,
            base_url,
            webhook_url,
            client: reqwest::Client::new(),
        })
    }

    /// Build and post an Adaptive Card with approve/reject action buttons.
    fn build_approval_card(
        &self,
        params: &ApprovalMsTeamsToolParams,
        approval_id: &str,
    ) -> Value {
        let mut body_elements: Vec<Value> = vec![
            json!({
                "type": "TextBlock",
                "text": "Approval Required",
                "weight": "Bolder",
                "size": "Medium"
            }),
            json!({
                "type": "TextBlock",
                "text": params.message,
                "wrap": true
            }),
        ];

        if let Some(ref context) = params.context {
            body_elements.push(json!({
                "type": "TextBlock",
                "text": "Context",
                "weight": "Bolder",
                "separator": true
            }));
            body_elements.push(json!({
                "type": "TextBlock",
                "text": context,
                "wrap": true,
                "fontType": "Monospace"
            }));
        }

        // Build actions if base_url is configured
        let mut actions: Vec<Value> = Vec::new();
        if let Some(ref base_url) = self.base_url {
            let approve_url =
                format!("{base_url}/v1/agents/approvals/{approval_id}/approve");
            let reject_url =
                format!("{base_url}/v1/agents/approvals/{approval_id}/reject");

            actions.push(json!({
                "type": "Action.OpenUrl",
                "title": "Approve",
                "url": approve_url,
                "style": "positive"
            }));
            actions.push(json!({
                "type": "Action.OpenUrl",
                "title": "Reject",
                "url": reject_url,
                "style": "destructive"
            }));
        }

        let mut card = json!({
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": body_elements,
        });

        if !actions.is_empty() {
            card["actions"] = json!(actions);
        }

        json!({
            "type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": null,
                "content": card
            }]
        })
    }

    /// Post the approval card to Teams via incoming webhook.
    async fn post_to_teams(
        &self,
        params: &ApprovalMsTeamsToolParams,
        approval_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let payload = self.build_approval_card(params, approval_id);

        let response = self
            .client
            .post(&self.webhook_url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "Microsoft Teams webhook returned HTTP {status}: {body}"
            )
            .into());
        }

        Ok(())
    }
}

#[async_trait]
impl SpiceModelTool for ApprovalMsTeamsTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<ApprovalMsTeamsToolParams>()
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span: Span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::approval_ms_teams", tool = self.name().to_string(), input = arg);

        let tool_use_result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let params: ApprovalMsTeamsToolParams = serde_json::from_str(arg)?;
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

            // Post to Teams
            match self.post_to_teams(&params, &approval_id).await {
                Ok(()) => {
                    tracing::info!(
                        target: "task_history",
                        parent: &span,
                        approval_id = %approval_id,
                        "Approval card posted to Teams, waiting for response"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        target: "task_history",
                        parent: &span,
                        approval_id = %approval_id,
                        error = %e,
                        "Failed to post approval to Teams, still waiting for HTTP resolution"
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

    fn make_tool(base_url: Option<String>) -> ApprovalMsTeamsTool {
        ApprovalMsTeamsTool::try_new(
            None,
            None,
            ApprovalStore::default(),
            Duration::from_secs(3600),
            ApprovalTimeoutAction::Reject,
            base_url,
            "https://myorg.webhook.office.com/webhookb2/guid".to_string(),
        )
        .unwrap()
    }

    #[test]
    fn test_reject_invalid_webhook_url() {
        let result = ApprovalMsTeamsTool::try_new(
            None,
            None,
            ApprovalStore::default(),
            Duration::from_secs(3600),
            ApprovalTimeoutAction::Reject,
            None,
            "https://evil.example.com/webhook".to_string(),
        );
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.to_string()
                .contains("not an allowed Microsoft Teams domain"),
            "Expected domain validation error, got: {err}"
        );
    }

    #[test]
    fn test_default_name_and_description() {
        let tool = make_tool(None);
        assert_eq!(tool.name(), "approval_ms_teams");
        assert!(tool.description().is_some());
        assert!(tool.parameters().is_some());
    }

    #[test]
    fn test_builds_approval_card_simple() {
        let tool = make_tool(Some("https://spice.example.com".to_string()));

        let params = ApprovalMsTeamsToolParams {
            message: "Release v1.2.3 ready to push".to_string(),
            context: None,
        };

        let card = tool.build_approval_card(&params, "test-approval-id");

        // Verify top-level structure
        assert_eq!(card["type"], "message");
        let attachments = card["attachments"].as_array().unwrap();
        assert_eq!(attachments.len(), 1);
        assert_eq!(
            attachments[0]["contentType"],
            "application/vnd.microsoft.card.adaptive"
        );

        let content = &attachments[0]["content"];
        assert_eq!(content["type"], "AdaptiveCard");
        assert_eq!(content["version"], "1.4");

        // Body: title + message = 2 elements
        let body = content["body"].as_array().unwrap();
        assert_eq!(body.len(), 2);
        assert_eq!(body[0]["text"], "Approval Required");
        assert_eq!(body[0]["weight"], "Bolder");
        assert_eq!(body[1]["text"], "Release v1.2.3 ready to push");

        // Actions: approve + reject
        let actions = content["actions"].as_array().unwrap();
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0]["title"], "Approve");
        assert_eq!(actions[0]["type"], "Action.OpenUrl");
        assert!(actions[0]["url"]
            .as_str()
            .unwrap()
            .contains("test-approval-id/approve"));
        assert_eq!(actions[1]["title"], "Reject");
        assert!(actions[1]["url"]
            .as_str()
            .unwrap()
            .contains("test-approval-id/reject"));
    }

    #[test]
    fn test_builds_approval_card_with_context() {
        let tool = make_tool(Some("https://spice.example.com".to_string()));

        let params = ApprovalMsTeamsToolParams {
            message: "Deploy to production".to_string(),
            context: Some("Changed files:\n- src/main.rs\n- Cargo.toml".to_string()),
        };

        let card = tool.build_approval_card(&params, "ctx-id");
        let content = &card["attachments"][0]["content"];
        let body = content["body"].as_array().unwrap();

        // Body: title + message + context title + context text = 4 elements
        assert_eq!(body.len(), 4);
        assert_eq!(body[2]["text"], "Context");
        assert_eq!(body[2]["weight"], "Bolder");
        assert!(body[3]["text"]
            .as_str()
            .unwrap()
            .contains("src/main.rs"));
        assert_eq!(body[3]["fontType"], "Monospace");
    }

    #[test]
    fn test_card_no_actions_without_base_url() {
        let tool = make_tool(None);

        let params = ApprovalMsTeamsToolParams {
            message: "No actions test".to_string(),
            context: None,
        };

        let card = tool.build_approval_card(&params, "no-url-id");
        let content = &card["attachments"][0]["content"];

        // No actions key when base_url is None
        assert!(content["actions"].is_null());
    }

    #[tokio::test]
    async fn test_approval_ms_teams_timeout() {
        let store = ApprovalStore::default();
        let tool = ApprovalMsTeamsTool::try_new(
            None,
            None,
            store.clone(),
            Duration::from_millis(100),
            ApprovalTimeoutAction::Reject,
            None,
            "https://myorg.webhook.office.com/webhookb2/guid".to_string(),
        )
        .unwrap();

        let result = tool
            .call(r#"{"message": "This will time out"}"#)
            .await
            .unwrap();

        assert_eq!(result["approved"], false);
        assert!(result["comment"]
            .as_str()
            .unwrap()
            .contains("Timed out"));
    }
}
