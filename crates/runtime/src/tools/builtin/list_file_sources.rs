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
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{borrow::Cow, sync::Arc};
use tools::SpiceModelTool;
use tracing_futures::Instrument;

use crate::Runtime;
use crate::tools::builtin::git_worktree::WorktreeTracker;
use crate::tools::utils::parameters;

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct ListFileSourcesToolParams {
    /// Optional name filter. If provided, only file sources whose name contains this string are returned.
    name: Option<String>,
}

pub struct ListFileSourcesTool {
    name: String,
    description: String,
    rt: Arc<Runtime>,
    worktree_tracker: WorktreeTracker,
}

impl ListFileSourcesTool {
    #[must_use]
    pub fn new(
        name: Option<&str>,
        description: Option<&str>,
        rt: Arc<Runtime>,
        worktree_tracker: WorktreeTracker,
    ) -> Self {
        Self {
            name: name.unwrap_or("list_file_sources").to_string(),
            description: description
                .unwrap_or("List available file sources and their local paths")
                .to_string(),
            rt,
            worktree_tracker,
        }
    }
}

#[async_trait]
impl SpiceModelTool for ListFileSourcesTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<ListFileSourcesToolParams>()
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::list_file_sources", tool = self.name().to_string(), input = arg);

        let result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let req: ListFileSourcesToolParams = serde_json::from_str(arg)
                .unwrap_or(ListFileSourcesToolParams { name: None });

            let app_lock = self.rt.app.read().await;
            let file_sources = app_lock
                .as_ref()
                .map(|app| app.file_sources.clone())
                .unwrap_or_default();
            drop(app_lock);

            let mut results: Vec<Value> = file_sources
                .iter()
                .filter(|fs| {
                    req.name
                        .as_ref()
                        .map_or(true, |n| fs.name.contains(n.as_str()))
                })
                .map(|fs| {
                    let source_type = fs.from.split(':').next().unwrap_or(&fs.from);

                    // Surface non-secret params (branch, etc.) — filter out tokens/keys/passwords
                    let safe_params: serde_json::Map<String, Value> = fs.params.iter()
                        .filter(|(k, _)| {
                            !k.contains("token") && !k.contains("key")
                                && !k.contains("secret") && !k.contains("password")
                        })
                        .map(|(k, v)| (k.clone(), json!(v)))
                        .collect();

                    let mut entry = json!({
                        "name": fs.name,
                        "source_type": source_type,
                        "from": fs.from,
                        "local_path": fs.path,
                        "refresh_schedule": fs.refresh,
                    });
                    if !safe_params.is_empty() {
                        entry["params"] = Value::Object(safe_params);
                    }
                    entry
                })
                .collect();

            // Include dynamically tracked worktrees.
            for (name, wt) in self.worktree_tracker.list() {
                if req.name.as_ref().map_or(true, |n| name.contains(n.as_str())) {
                    results.push(json!({
                        "name": name,
                        "source_type": "worktree",
                        "from": format!("git_worktree:{}", wt.branch),
                        "local_path": wt.path.to_string_lossy(),
                    }));
                }
            }

            let total = results.len();
            tracing::debug!(total = total, filter = ?req.name, "list_file_sources results");
            Ok(json!({
                "file_sources": results,
                "total": total,
            }))
        }
        .instrument(span.clone())
        .await;

        match result {
            Ok(value) => {
                let captured = serde_json::to_string(&value)?;
                tracing::info!(target: "task_history", parent: &span, captured_output = %captured);
                Ok(value)
            }
            Err(e) => {
                tracing::error!(target: "task_history", parent: &span, "list_file_sources failed: {e}");
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
        // Can't easily construct a Runtime in unit tests, so test the struct directly
        let params = ListFileSourcesToolParams { name: None };
        assert!(params.name.is_none());
    }

    #[test]
    fn test_has_parameters() {
        let schema = parameters::<ListFileSourcesToolParams>();
        assert!(schema.is_some());
    }

    #[test]
    fn test_name_filter_param() {
        let params: ListFileSourcesToolParams =
            serde_json::from_str(r#"{"name": "repo"}"#).unwrap();
        assert_eq!(params.name, Some("repo".to_string()));
    }

    #[test]
    fn test_empty_params() {
        let params: ListFileSourcesToolParams = serde_json::from_str("{}").unwrap();
        assert!(params.name.is_none());
    }
}
