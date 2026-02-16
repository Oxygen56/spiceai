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

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, oneshot};

/// The result of resolving an approval request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalResponse {
    pub approved: bool,
    pub comment: Option<String>,
    pub responded_at: DateTime<Utc>,
}

/// Metadata for a pending approval (safe to expose via API — no channel sender).
#[derive(Debug, Clone, Serialize)]
pub struct PendingApprovalInfo {
    pub id: String,
    pub agent_name: String,
    pub message: String,
    pub context: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Internal pending approval with the oneshot sender for resolution.
pub(crate) struct PendingApproval {
    pub info: PendingApprovalInfo,
    pub tx: oneshot::Sender<ApprovalResponse>,
}

/// Shared store for pending approvals. Thread-safe and cloneable.
///
/// Used by approval tools to register pending approvals (blocking via oneshot)
/// and by HTTP endpoints to resolve them.
#[derive(Clone, Default)]
pub struct ApprovalStore {
    pending: Arc<RwLock<HashMap<String, PendingApproval>>>,
}

impl ApprovalStore {
    /// Store a pending approval. The caller is responsible for creating the
    /// oneshot channel and keeping the receiver.
    pub async fn store(&self, approval: PendingApproval) {
        let id = approval.info.id.clone();
        let mut pending = self.pending.write().await;
        pending.insert(id, approval);
    }

    /// Resolve a pending approval by ID. Sends the response through the oneshot channel.
    pub async fn resolve(
        &self,
        id: &str,
        response: ApprovalResponse,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut pending = self.pending.write().await;
        let approval = pending
            .remove(id)
            .ok_or_else(|| format!("No pending approval found with id '{id}'"))?;
        approval
            .tx
            .send(response)
            .map_err(|_| "Failed to send approval response: receiver dropped")?;
        Ok(())
    }

    /// List all pending approvals (read-only view for the HTTP API).
    pub async fn list_pending(&self) -> Vec<PendingApprovalInfo> {
        let pending = self.pending.read().await;
        pending.values().map(|a| a.info.clone()).collect()
    }

    /// Remove a pending approval without resolving it (used for timeout cleanup).
    pub async fn remove(&self, id: &str) -> Option<PendingApproval> {
        let mut pending = self.pending.write().await;
        pending.remove(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_store_register_resolve() {
        let store = ApprovalStore::default();

        let (tx, rx) = oneshot::channel();
        let approval = PendingApproval {
            info: PendingApprovalInfo {
                id: "test-1".to_string(),
                agent_name: "test-agent".to_string(),
                message: "Please approve".to_string(),
                context: None,
                created_at: Utc::now(),
            },
            tx,
        };

        store.store(approval).await;

        // Should appear in pending list
        let pending = store.list_pending().await;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, "test-1");
        assert_eq!(pending[0].message, "Please approve");

        // Resolve it
        let response = ApprovalResponse {
            approved: true,
            comment: Some("LGTM".to_string()),
            responded_at: Utc::now(),
        };
        store.resolve("test-1", response).await.unwrap();

        // Should be gone from pending
        let pending = store.list_pending().await;
        assert!(pending.is_empty());

        // Receiver should get the response
        let result = rx.await.unwrap();
        assert!(result.approved);
        assert_eq!(result.comment, Some("LGTM".to_string()));
    }

    #[tokio::test]
    async fn test_store_resolve_not_found() {
        let store = ApprovalStore::default();

        let response = ApprovalResponse {
            approved: true,
            comment: None,
            responded_at: Utc::now(),
        };
        let result = store.resolve("nonexistent", response).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No pending approval found")
        );
    }

    #[tokio::test]
    async fn test_store_list_pending() {
        let store = ApprovalStore::default();

        for i in 0..3 {
            let (tx, _rx) = oneshot::channel();
            store
                .store(PendingApproval {
                    info: PendingApprovalInfo {
                        id: format!("approval-{i}"),
                        agent_name: "agent".to_string(),
                        message: format!("Approval {i}"),
                        context: None,
                        created_at: Utc::now(),
                    },
                    tx,
                })
                .await;
        }

        let pending = store.list_pending().await;
        assert_eq!(pending.len(), 3);
    }

    #[tokio::test]
    async fn test_store_remove() {
        let store = ApprovalStore::default();

        let (tx, _rx) = oneshot::channel();
        store
            .store(PendingApproval {
                info: PendingApprovalInfo {
                    id: "to-remove".to_string(),
                    agent_name: "agent".to_string(),
                    message: "Will be removed".to_string(),
                    context: None,
                    created_at: Utc::now(),
                },
                tx,
            })
            .await;

        assert_eq!(store.list_pending().await.len(), 1);

        let removed = store.remove("to-remove").await;
        assert!(removed.is_some());
        assert!(store.list_pending().await.is_empty());

        // Removing again returns None
        assert!(store.remove("to-remove").await.is_none());
    }
}
