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

/// Extracts permanent knowledge entries from weekly rollups.
///
/// Promotes key facts and patterns from rollup summaries into the
/// knowledge base layer. Uses simple heuristic extraction; an LLM-powered
/// variant would use the configured compaction model to identify and
/// extract durable knowledge.
pub struct KnowledgeExtractor;

impl KnowledgeExtractor {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for KnowledgeExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryCompactor for KnowledgeExtractor {
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

        // Extract key information from rollups. A production implementation
        // would use an LLM to identify durable knowledge (facts, patterns,
        // preferences) that should persist permanently.
        let combined_content: String = source_entries
            .iter()
            .map(|e| e.content.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n");

        let knowledge = MemoryEntry {
            id: uuid::Uuid::now_v7().to_string(),
            agent,
            layer: "knowledge_base".to_string(),
            content: format!("Extracted knowledge:\n{combined_content}"),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            expires_at: None, // Knowledge base entries never expire
        };

        Ok(vec![knowledge])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(agent: &str, content: &str) -> MemoryEntry {
        MemoryEntry {
            id: uuid::Uuid::now_v7().to_string(),
            agent: agent.to_string(),
            layer: "weekly_rollup".to_string(),
            content: content.to_string(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            expires_at: None,
        }
    }

    #[tokio::test]
    async fn test_extract_knowledge() {
        let extractor = KnowledgeExtractor::new();
        let entries = vec![
            make_entry("agent1", "Week 1: User prefers concise reports"),
            make_entry("agent1", "Week 2: System has 3 production clusters"),
        ];
        let result = extractor.compact(entries).await.unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].content.starts_with("Extracted knowledge:"));
        assert!(result[0].content.contains("concise reports"));
        assert!(result[0].content.contains("3 production clusters"));
        assert_eq!(result[0].layer, "knowledge_base");
        assert!(result[0].expires_at.is_none());
    }

    #[tokio::test]
    async fn test_extract_empty() {
        let extractor = KnowledgeExtractor::new();
        let result = extractor.compact(Vec::new()).await.unwrap();
        assert!(result.is_empty());
    }
}
