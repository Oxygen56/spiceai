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
use std::collections::HashMap;

use crate::memory::{MemoryCompactor, MemoryEntry};

/// Compacts raw session turns into a summary entry.
///
/// This is a simple implementation that concatenates turn content.
/// The real LLM-powered compaction will be integrated during orchestration.
pub struct SessionSummaryCompactor;

impl SessionSummaryCompactor {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for SessionSummaryCompactor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryCompactor for SessionSummaryCompactor {
    async fn compact(
        &self,
        source_entries: Vec<MemoryEntry>,
    ) -> Result<Vec<MemoryEntry>, Box<dyn std::error::Error + Send + Sync>> {
        if source_entries.is_empty() {
            return Ok(Vec::new());
        }

        let agent = source_entries
            .first()
            .map(|e| e.agent.clone())
            .unwrap_or_default();

        let combined_content: String = source_entries
            .iter()
            .map(|e| e.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let summary = MemoryEntry {
            id: uuid::Uuid::now_v7().to_string(),
            agent,
            layer: "session_summary".to_string(),
            content: format!("Summary of session:\n{combined_content}"),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            expires_at: None,
        };

        Ok(vec![summary])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(agent: &str, content: &str) -> MemoryEntry {
        MemoryEntry {
            id: uuid::Uuid::now_v7().to_string(),
            agent: agent.to_string(),
            layer: "session".to_string(),
            content: content.to_string(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            expires_at: None,
        }
    }

    #[tokio::test]
    async fn test_compact_produces_summary() {
        let compactor = SessionSummaryCompactor::new();
        let entries = vec![
            make_entry("agent1", "User said hello"),
            make_entry("agent1", "Agent replied with greeting"),
        ];
        let result = compactor.compact(entries).await.unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].content.starts_with("Summary of session:"));
        assert!(result[0].content.contains("User said hello"));
        assert!(result[0].content.contains("Agent replied with greeting"));
        assert_eq!(result[0].layer, "session_summary");
        assert_eq!(result[0].agent, "agent1");
    }

    #[tokio::test]
    async fn test_compact_empty_entries() {
        let compactor = SessionSummaryCompactor::new();
        let result = compactor.compact(Vec::new()).await.unwrap();
        assert!(result.is_empty());
    }
}
