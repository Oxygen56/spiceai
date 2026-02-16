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
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::memory::{MemoryEntry, MemoryFilter, MemoryLayer};

/// Short-term session memory layer (default retention: 4 hours).
/// Stores raw conversation turns in memory.
pub struct SessionLayer {
    retention: Option<Duration>,
    entries: Arc<RwLock<Vec<MemoryEntry>>>,
}

impl SessionLayer {
    #[must_use]
    pub fn new(retention: Option<Duration>) -> Self {
        Self {
            retention,
            entries: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

#[async_trait]
impl MemoryLayer for SessionLayer {
    fn name(&self) -> &str {
        "session"
    }

    fn retention(&self) -> Option<Duration> {
        self.retention
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

    async fn purge_expired(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let now = Utc::now();
        let mut store = self.entries.write().await;
        let before = store.len();
        store.retain(|e| {
            e.expires_at
                .map_or(true, |expires_at| expires_at > now)
        });
        let purged = before.saturating_sub(store.len());
        Ok(u64::try_from(purged).unwrap_or(u64::MAX))
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
    use chrono::Duration as ChronoDuration;
    use std::collections::HashMap;

    fn make_entry(agent: &str, content: &str, expires_at: Option<chrono::DateTime<Utc>>) -> MemoryEntry {
        MemoryEntry {
            id: uuid::Uuid::now_v7().to_string(),
            agent: agent.to_string(),
            layer: "session".to_string(),
            content: content.to_string(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            expires_at,
        }
    }

    #[tokio::test]
    async fn test_store_and_query() {
        let layer = SessionLayer::new(Some(Duration::from_secs(4 * 3600)));
        let entries = vec![
            make_entry("agent1", "hello", None),
            make_entry("agent1", "world", None),
            make_entry("agent2", "other", None),
        ];
        layer.store(entries).await.unwrap();

        let filter = MemoryFilter {
            agent: Some("agent1".to_string()),
            ..Default::default()
        };
        let results = layer.query(filter).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_purge_expired() {
        let layer = SessionLayer::new(Some(Duration::from_secs(3600)));
        let expired_entry = MemoryEntry {
            expires_at: Some(Utc::now() - ChronoDuration::hours(1)),
            ..make_entry("agent1", "old", None)
        };
        let valid_entry = make_entry("agent1", "new", Some(Utc::now() + ChronoDuration::hours(1)));
        layer.store(vec![expired_entry, valid_entry]).await.unwrap();

        let purged = layer.purge_expired().await.unwrap();
        assert_eq!(purged, 1);

        let all = layer.query(MemoryFilter::default()).await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].content, "new");
    }
}
