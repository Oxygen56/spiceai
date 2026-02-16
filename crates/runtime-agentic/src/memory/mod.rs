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
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

pub mod compactors;
pub mod layers;

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub id: String,
    pub agent: String,
    pub layer: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default)]
pub struct MemoryFilter {
    pub agent: Option<String>,
    pub layer: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub limit: Option<usize>,
}

#[async_trait]
pub trait MemoryLayer: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn retention(&self) -> Option<Duration>;
    async fn store(
        &self,
        entries: Vec<MemoryEntry>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn query(
        &self,
        filter: MemoryFilter,
    ) -> Result<Vec<MemoryEntry>, Box<dyn std::error::Error + Send + Sync>>;
    async fn purge_expired(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>>;
    /// Remove all entries for a specific agent from this layer.
    async fn clear(
        &self,
        agent: &str,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>>;
}

#[async_trait]
pub trait MemoryCompactor: Send + Sync + 'static {
    /// Compact source entries into higher-level summary entries.
    async fn compact(
        &self,
        source_entries: Vec<MemoryEntry>,
    ) -> Result<Vec<MemoryEntry>, Box<dyn std::error::Error + Send + Sync>>;
}

/// Result of a compaction operation.
#[derive(Debug, Clone, Default)]
pub struct CompactionResult {
    pub entries_compacted: usize,
    pub summaries_produced: usize,
}

/// Manages the hierarchy of memory layers and compaction.
#[derive(Clone)]
pub struct MemoryManager {
    layers: Vec<Arc<dyn MemoryLayer>>,
    session_threshold: u32,
}

impl MemoryManager {
    #[must_use]
    pub fn new(layers: Vec<Arc<dyn MemoryLayer>>, session_threshold: u32) -> Self {
        Self {
            layers,
            session_threshold,
        }
    }

    /// Gather context from all memory layers for an agent.
    pub async fn gather_context(
        &self,
        agent: &str,
    ) -> Result<Vec<MemoryEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let mut all_entries = Vec::new();
        let filter = MemoryFilter {
            agent: Some(agent.to_string()),
            ..Default::default()
        };
        for layer in &self.layers {
            let entries = layer.query(filter.clone()).await?;
            all_entries.extend(entries);
        }
        // Sort by created_at descending (most recent first).
        all_entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(all_entries)
    }

    /// Store entries in the named layer.
    pub async fn store_in_layer(
        &self,
        layer_name: &str,
        entries: Vec<MemoryEntry>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for layer in &self.layers {
            if layer.name() == layer_name {
                return layer.store(entries).await;
            }
        }
        Err(format!("Memory layer not found: {layer_name}").into())
    }

    /// Purge expired entries from all layers.
    pub async fn purge_all_expired(
        &self,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let mut total = 0u64;
        for layer in &self.layers {
            total += layer.purge_expired().await?;
        }
        Ok(total)
    }

    #[must_use]
    pub fn session_threshold(&self) -> u32 {
        self.session_threshold
    }

    /// Query a specific layer for entries matching the given agent.
    pub async fn query_layer(
        &self,
        layer_name: &str,
        agent: &str,
    ) -> Result<Vec<MemoryEntry>, Box<dyn std::error::Error + Send + Sync>> {
        let filter = MemoryFilter {
            agent: Some(agent.to_string()),
            ..Default::default()
        };
        for layer in &self.layers {
            if layer.name() == layer_name {
                return layer.query(filter).await;
            }
        }
        Err(format!("Memory layer not found: {layer_name}").into())
    }

    /// Clear all entries for an agent from the named layer.
    pub async fn clear_layer(
        &self,
        layer_name: &str,
        agent: &str,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        for layer in &self.layers {
            if layer.name() == layer_name {
                return layer.clear(agent).await;
            }
        }
        Err(format!("Memory layer not found: {layer_name}").into())
    }

    /// Compact session memory entries into a summary.
    ///
    /// Queries the "session" layer for all entries for the given agent,
    /// runs them through `SessionSummaryCompactor`, stores the result
    /// in the "session_summary" layer, and clears the session layer.
    pub async fn compact_session(
        &self,
        agent: &str,
    ) -> Result<CompactionResult, Box<dyn std::error::Error + Send + Sync>> {
        let session_entries = self.query_layer("session", agent).await?;
        if session_entries.is_empty() {
            return Ok(CompactionResult::default());
        }

        let entries_count = session_entries.len();
        let compactor = compactors::session_summary::SessionSummaryCompactor::new();
        let summaries = compactor.compact(session_entries).await?;
        let summaries_count = summaries.len();

        self.store_in_layer("session_summary", summaries).await?;
        self.clear_layer("session", agent).await?;

        Ok(CompactionResult {
            entries_compacted: entries_count,
            summaries_produced: summaries_count,
        })
    }
}

/// Parse retention duration strings like "4h", "7d", "90d", "30m".
#[must_use]
pub fn parse_retention(retention: &str) -> Option<Duration> {
    let (num_str, unit) = retention.split_at(retention.len().checked_sub(1)?);
    let num: u64 = num_str.parse().ok()?;
    match unit {
        "h" => Some(Duration::from_secs(num * 3600)),
        "d" => Some(Duration::from_secs(num * 86400)),
        "m" => Some(Duration::from_secs(num * 60)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_retention_hours() {
        let d = parse_retention("4h").unwrap();
        assert_eq!(d, Duration::from_secs(4 * 3600));
    }

    #[test]
    fn test_parse_retention_days() {
        let d = parse_retention("7d").unwrap();
        assert_eq!(d, Duration::from_secs(7 * 86400));
    }

    #[test]
    fn test_parse_retention_minutes() {
        let d = parse_retention("30m").unwrap();
        assert_eq!(d, Duration::from_secs(30 * 60));
    }

    #[test]
    fn test_parse_retention_invalid() {
        assert!(parse_retention("").is_none());
        assert!(parse_retention("abc").is_none());
        assert!(parse_retention("10x").is_none());
    }
}
