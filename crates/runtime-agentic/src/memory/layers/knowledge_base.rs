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
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::memory::{MemoryEntry, MemoryFilter, MemoryLayer};

/// Permanent knowledge base layer (no expiry).
/// Stores long-term knowledge that persists indefinitely.
pub struct KnowledgeBaseLayer {
    entries: Arc<RwLock<Vec<MemoryEntry>>>,
}

impl KnowledgeBaseLayer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

impl Default for KnowledgeBaseLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryLayer for KnowledgeBaseLayer {
    fn name(&self) -> &str {
        "knowledge_base"
    }

    fn retention(&self) -> Option<Duration> {
        None
    }

    async fn store(
        &self,
        entries: Vec<MemoryEntry>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut store = self.entries.write().await;
        store.extend(entries);
        Ok(())
    }

    async fn query(
        &self,
        filter: MemoryFilter,
    ) -> Result<Vec<MemoryEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let store = self.entries.read().await;
        let mut results: Vec<MemoryEntry> = store
            .iter()
            .filter(|e| {
                if let Some(ref agent) = filter.agent {
                    if &e.agent != agent {
                        return false;
                    }
                }
                if let Some(ref layer) = filter.layer {
                    if &e.layer != layer {
                        return false;
                    }
                }
                if let Some(since) = filter.since {
                    if e.created_at < since {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        results.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Knowledge base entries never expire; always returns 0.
    async fn purge_expired(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        Ok(0)
    }

    async fn clear(
        &self,
        agent: &str,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let mut store = self.entries.write().await;
        let before = store.len();
        store.retain(|e| e.agent != agent);
        let removed = before.saturating_sub(store.len());
        Ok(u64::try_from(removed).unwrap_or(u64::MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;

    fn make_entry(agent: &str, content: &str) -> MemoryEntry {
        MemoryEntry {
            id: uuid::Uuid::now_v7().to_string(),
            agent: agent.to_string(),
            layer: "knowledge_base".to_string(),
            content: content.to_string(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            expires_at: None,
        }
    }

    #[tokio::test]
    async fn test_permanent_storage() {
        let layer = KnowledgeBaseLayer::new();
        layer
            .store(vec![make_entry("agent1", "fact 1")])
            .await
            .unwrap();

        let purged = layer.purge_expired().await.unwrap();
        assert_eq!(purged, 0);

        let results = layer
            .query(MemoryFilter {
                agent: Some("agent1".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "fact 1");
    }
}
