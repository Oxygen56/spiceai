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

pub mod ms_teams;
pub mod slack;
pub mod store;

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

pub use store::{ApprovalStore, PendingApprovalInfo};
use store::{ApprovalResponse, PendingApproval};

/// Timeout behavior when an approval expires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalTimeoutAction {
    Approve,
    Reject,
}

/// Parameters the agent passes when calling the base approval tool.
#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct ApprovalToolParams {
    /// What needs approval and why.
    message: String,

    /// Optional additional context (e.g., diff summary, changed files).
    context: Option<String>,
}

/// Base approval tool — blocks until resolved via HTTP API or timeout.
///
/// No external notification is sent. Approvals are resolved through:
/// - `POST /v1/agents/approvals/{id}` (programmatic)
/// - `GET /v1/agents/approvals/{id}/approve` (one-click)
/// - `GET /v1/agents/approvals/{id}/reject` (one-click)
pub struct ApprovalTool {
    name: String,
    description: String,
    store: ApprovalStore,
    timeout: Duration,
    timeout_action: ApprovalTimeoutAction,
    base_url: Option<String>,
}

impl ApprovalTool {
    #[must_use]
    pub fn new(
        name: Option<&str>,
        description: Option<&str>,
        store: ApprovalStore,
        timeout: Duration,
        timeout_action: ApprovalTimeoutAction,
        base_url: Option<String>,
    ) -> Self {
        Self {
            name: name.unwrap_or("approval").to_string(),
            description: description
                .unwrap_or("Request human approval before proceeding with an action")
                .to_string(),
            store,
            timeout,
            timeout_action,
            base_url,
        }
    }
}

#[async_trait]
impl SpiceModelTool for ApprovalTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<ApprovalToolParams>()
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span: Span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::approval", tool = self.name().to_string(), input = arg);

        let tool_use_result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let params: ApprovalToolParams = serde_json::from_str(arg)?;
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

            tracing::info!(
                target: "task_history",
                parent: &span,
                approval_id = %approval_id,
                "Approval requested, waiting for response"
            );

            // Log the callback URLs if base_url is configured
            if let Some(ref base_url) = self.base_url {
                tracing::info!(
                    target: "task_history",
                    parent: &span,
                    approve_url = %format!("{base_url}/v1/agents/approvals/{approval_id}/approve"),
                    reject_url = %format!("{base_url}/v1/agents/approvals/{approval_id}/reject"),
                    "Approval callback URLs"
                );
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

/// Parse a timeout duration from tool params. Supports formats like "1h", "24h", "30m", "7d".
pub(crate) fn parse_timeout(s: &str) -> Option<Duration> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    let (num_str, suffix) = if s.ends_with('d') {
        (&s[..s.len() - 1], "d")
    } else if s.ends_with('h') {
        (&s[..s.len() - 1], "h")
    } else if s.ends_with('m') {
        (&s[..s.len() - 1], "m")
    } else if s.ends_with('s') {
        (&s[..s.len() - 1], "s")
    } else {
        // Try parsing as seconds
        return s.parse::<u64>().ok().map(Duration::from_secs);
    };

    let num: u64 = num_str.parse().ok()?;
    match suffix {
        "d" => Some(Duration::from_secs(num * 86400)),
        "h" => Some(Duration::from_secs(num * 3600)),
        "m" => Some(Duration::from_secs(num * 60)),
        "s" => Some(Duration::from_secs(num)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_timeout() {
        assert_eq!(parse_timeout("1h"), Some(Duration::from_secs(3600)));
        assert_eq!(parse_timeout("24h"), Some(Duration::from_secs(86400)));
        assert_eq!(parse_timeout("30m"), Some(Duration::from_secs(1800)));
        assert_eq!(parse_timeout("7d"), Some(Duration::from_secs(604800)));
        assert_eq!(parse_timeout("60s"), Some(Duration::from_secs(60)));
        assert_eq!(parse_timeout("3600"), Some(Duration::from_secs(3600)));
        assert_eq!(parse_timeout(""), None);
        assert_eq!(parse_timeout("abc"), None);
    }

    #[test]
    fn test_default_name_and_description() {
        let store = ApprovalStore::default();
        let tool = ApprovalTool::new(
            None,
            None,
            store,
            Duration::from_secs(3600),
            ApprovalTimeoutAction::Reject,
            None,
        );
        assert_eq!(tool.name(), "approval");
        assert!(tool.description().is_some());
        assert!(tool.parameters().is_some());
    }

    #[tokio::test]
    async fn test_approval_tool_approved() {
        let store = ApprovalStore::default();
        let tool = ApprovalTool::new(
            None,
            None,
            store.clone(),
            Duration::from_secs(60),
            ApprovalTimeoutAction::Reject,
            None,
        );

        // Spawn tool call in background
        let handle = tokio::spawn(async move {
            tool.call(r#"{"message": "Deploy to production?"}"#)
                .await
                .unwrap()
        });

        // Wait briefly for the approval to be registered
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Resolve the approval
        let pending = store.list_pending().await;
        assert_eq!(pending.len(), 1);

        let approval_id = pending[0].id.clone();
        store
            .resolve(
                &approval_id,
                store::ApprovalResponse {
                    approved: true,
                    comment: Some("Go for it".to_string()),
                    responded_at: Utc::now(),
                },
            )
            .await
            .unwrap();

        let result = handle.await.unwrap();
        assert_eq!(result["approved"], true);
        assert_eq!(result["comment"], "Go for it");
        assert_eq!(result["approval_id"], approval_id);
    }

    #[tokio::test]
    async fn test_approval_tool_rejected() {
        let store = ApprovalStore::default();
        let tool = ApprovalTool::new(
            None,
            None,
            store.clone(),
            Duration::from_secs(60),
            ApprovalTimeoutAction::Reject,
            None,
        );

        let handle = tokio::spawn(async move {
            tool.call(r#"{"message": "Delete database?"}"#)
                .await
                .unwrap()
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let pending = store.list_pending().await;
        let approval_id = pending[0].id.clone();
        store
            .resolve(
                &approval_id,
                store::ApprovalResponse {
                    approved: false,
                    comment: Some("Too risky".to_string()),
                    responded_at: Utc::now(),
                },
            )
            .await
            .unwrap();

        let result = handle.await.unwrap();
        assert_eq!(result["approved"], false);
        assert_eq!(result["comment"], "Too risky");
    }

    #[tokio::test]
    async fn test_approval_tool_timeout_reject() {
        let store = ApprovalStore::default();
        let tool = ApprovalTool::new(
            None,
            None,
            store.clone(),
            Duration::from_millis(100), // Very short timeout
            ApprovalTimeoutAction::Reject,
            None,
        );

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

    #[tokio::test]
    async fn test_approval_tool_timeout_approve() {
        let store = ApprovalStore::default();
        let tool = ApprovalTool::new(
            None,
            None,
            store.clone(),
            Duration::from_millis(100),
            ApprovalTimeoutAction::Approve, // Auto-approve on timeout
            None,
        );

        let result = tool
            .call(r#"{"message": "This will auto-approve"}"#)
            .await
            .unwrap();

        assert_eq!(result["approved"], true);
        assert!(result["comment"]
            .as_str()
            .unwrap()
            .contains("Timed out"));
    }
}
