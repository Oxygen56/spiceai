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

use axum::{
    Extension, Json,
    extract::Path,
    http::StatusCode,
    response::{Html, IntoResponse, Response},
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::init::agent::WebhookRegistry;
use crate::tools::builtin::approval::store::{ApprovalResponse, ApprovalStore};
use crate::trigger::TriggerPayload;

/// Handle an incoming webhook for an agent pipeline.
///
/// Webhook paths are registered during agent loading. This handler looks up the
/// matching webhook trigger handler by path and dispatches the payload.
pub(crate) async fn webhook(
    Path(path): Path<String>,
    Extension(webhook_registry): Extension<WebhookRegistry>,
    Json(body): Json<Value>,
) -> Response {
    let lookup_path = if path.starts_with('/') {
        path.clone()
    } else {
        format!("/{path}")
    };

    let registry = webhook_registry.read().await;
    let handler = registry.get(&lookup_path).cloned();
    drop(registry);

    let Some(handler) = handler else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("No webhook handler registered for path '{lookup_path}'")
            })),
        )
            .into_response();
    };

    let payload = TriggerPayload {
        source: format!("webhook:{lookup_path}"),
        data: body,
        received_at: Utc::now(),
    };

    match handler.handle(payload).await {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "message": "Pipeline execution completed"
            })),
        )
            .into_response(),
        Err(e) => {
            tracing::error!("Webhook handler error for path '{lookup_path}': {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Pipeline execution failed: {e}")
                })),
            )
                .into_response()
        }
    }
}

/// List all registered webhook endpoints.
pub(crate) async fn list_webhooks(
    Extension(webhook_registry): Extension<WebhookRegistry>,
) -> Response {
    let registry = webhook_registry.read().await;
    let paths: Vec<&String> = registry.keys().collect();

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "webhooks": paths
        })),
    )
        .into_response()
}

// --- Approval endpoints ---

/// List all pending approval requests.
pub(crate) async fn list_approvals(
    Extension(store): Extension<ApprovalStore>,
) -> Response {
    let pending = store.list_pending().await;
    (
        StatusCode::OK,
        Json(serde_json::json!({ "approvals": pending })),
    )
        .into_response()
}

/// Request body for programmatic approval resolution.
#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct ResolveApprovalRequest {
    approved: bool,
    comment: Option<String>,
}

/// Resolve a pending approval via API (programmatic).
pub(crate) async fn resolve_approval(
    Path(approval_id): Path<String>,
    Extension(store): Extension<ApprovalStore>,
    Json(body): Json<ResolveApprovalRequest>,
) -> Response {
    let response = ApprovalResponse {
        approved: body.approved,
        comment: body.comment,
        responded_at: Utc::now(),
    };

    match store.resolve(&approval_id, response).await {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "resolved",
                "approval_id": approval_id,
                "approved": body.approved,
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": e.to_string()
            })),
        )
            .into_response(),
    }
}

/// One-click approve via GET (for Slack/Teams action buttons).
pub(crate) async fn approve_action(
    Path(approval_id): Path<String>,
    Extension(store): Extension<ApprovalStore>,
) -> Response {
    let response = ApprovalResponse {
        approved: true,
        comment: None,
        responded_at: Utc::now(),
    };

    match store.resolve(&approval_id, response).await {
        Ok(()) => Html(
            "<html><body><h1>Approved</h1>\
             <p>The approval has been recorded. You can close this tab.</p>\
             </body></html>"
                .to_string(),
        )
        .into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Html(
                "<html><body><h1>Not Found</h1>\
                 <p>This approval has already been resolved or does not exist.</p>\
                 </body></html>"
                    .to_string(),
            ),
        )
            .into_response(),
    }
}

/// One-click reject via GET (for Slack/Teams action buttons).
pub(crate) async fn reject_action(
    Path(approval_id): Path<String>,
    Extension(store): Extension<ApprovalStore>,
) -> Response {
    let response = ApprovalResponse {
        approved: false,
        comment: None,
        responded_at: Utc::now(),
    };

    match store.resolve(&approval_id, response).await {
        Ok(()) => Html(
            "<html><body><h1>Rejected</h1>\
             <p>The approval has been rejected. You can close this tab.</p>\
             </body></html>"
                .to_string(),
        )
        .into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Html(
                "<html><body><h1>Not Found</h1>\
                 <p>This approval has already been resolved or does not exist.</p>\
                 </body></html>"
                    .to_string(),
            ),
        )
            .into_response(),
    }
}
