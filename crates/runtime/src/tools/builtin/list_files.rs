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
use serde_json::{Value, json};
use snafu::ResultExt;
use std::{borrow::Cow, fs, path::PathBuf};
use tools::{SpiceModelTool, ToolCapability};
use tracing::Span;
use tracing_futures::Instrument;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::tools::builtin::git_worktree::WorktreeTracker;
use crate::tools::utils::parameters;

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct ListFilesToolParams {
    /// Absolute path to the directory to list. Use list_file_sources to discover available paths.
    path: String,

    /// Whether to list files recursively. Defaults to false.
    recursive: Option<bool>,
}

pub struct ListFilesTool {
    name: String,
    description: String,
    base_paths: Vec<PathBuf>,
    worktree_tracker: Option<WorktreeTracker>,
}

impl ListFilesTool {
    #[must_use]
    pub fn new(
        name: Option<&str>,
        description: Option<&str>,
        base_paths: Vec<PathBuf>,
        worktree_tracker: Option<WorktreeTracker>,
    ) -> Self {
        Self {
            name: name.unwrap_or("list_files").to_string(),
            description: description
                .unwrap_or("List files and directories at the given path")
                .to_string(),
            base_paths,
            worktree_tracker,
        }
    }

    fn is_path_allowed(&self, path: &std::path::Path) -> bool {
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => return false,
        };

        if self.base_paths.iter().any(|base| {
            base.canonicalize()
                .map_or(false, |b| canonical.starts_with(&b))
        }) {
            return true;
        }

        if let Some(ref tracker) = self.worktree_tracker {
            for (_name, wt) in tracker.list() {
                if let Ok(canonical_wt) = wt.path.canonicalize() {
                    if canonical.starts_with(&canonical_wt) {
                        return true;
                    }
                }
            }
        }

        false
    }
}

fn collect_entries(
    dir: &std::path::Path,
    recursive: bool,
    entries: &mut Vec<Value>,
) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        let is_dir = metadata.is_dir();

        entries.push(json!({
            "name": entry.file_name().to_string_lossy(),
            "path": entry.path().to_string_lossy(),
            "size": metadata.len(),
            "is_directory": is_dir,
        }));

        if recursive && is_dir {
            collect_entries(&entry.path(), true, entries)?;
        }
    }
    Ok(())
}

#[async_trait]
impl SpiceModelTool for ListFilesTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<ListFilesToolParams>()
    }

    fn capability(&self) -> ToolCapability {
        ToolCapability::ReadOnly
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span: Span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::list_files", tool = self.name().to_string(), input = arg);

        let tool_use_result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let req: ListFilesToolParams = serde_json::from_str(arg)?;
            let dir_path = PathBuf::from(&req.path);

            if !self.is_path_allowed(&dir_path) {
                tracing::warn!(path = %req.path, allowed_dirs = ?self.base_paths, "list_files access denied: path is outside allowed directories");
                return Err(format!(
                    "Access denied: path '{}' is outside allowed directories",
                    req.path
                )
                .into());
            }

            let recursive = req.recursive.unwrap_or(false);
            let mut entries = Vec::new();
            collect_entries(&dir_path, recursive, &mut entries)?;

            Ok(json!({
                "path": req.path,
                "entries": entries,
                "total": entries.len(),
            }))
        }
        .instrument(span.clone())
        .await;

        match tool_use_result {
            Ok(value) => {
                let captured_output_json = serde_json::to_string(&value).boxed()?;
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
