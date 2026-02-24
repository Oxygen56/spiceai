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

/// Maximum file size that can be written (1 MB).
const MAX_FILE_SIZE: usize = 1_048_576;

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct WriteFileToolParams {
    /// Absolute path to the file to write.
    path: String,
    /// The content to write to the file.
    content: String,
}

pub struct WriteFileTool {
    name: String,
    description: String,
    base_paths: Vec<PathBuf>,
    worktree_tracker: Option<WorktreeTracker>,
}

impl WriteFileTool {
    #[must_use]
    pub fn new(
        name: Option<&str>,
        description: Option<&str>,
        base_paths: Vec<PathBuf>,
        worktree_tracker: Option<WorktreeTracker>,
    ) -> Self {
        Self {
            name: name.unwrap_or("write_file").to_string(),
            description: description
                .unwrap_or("Write content to a file at the given path")
                .to_string(),
            base_paths,
            worktree_tracker,
        }
    }

    /// Check if the target path is within allowed directories.
    ///
    /// For write operations, the file may not exist yet, so we walk up the path
    /// to find the nearest existing ancestor and canonicalize that.
    fn is_path_allowed(&self, path: &std::path::Path) -> bool {
        // Find the nearest existing ancestor to canonicalize
        let canonical = if path.exists() {
            match path.canonicalize() {
                Ok(p) => p,
                Err(_) => return false,
            }
        } else {
            // Walk up to the nearest existing ancestor
            let mut ancestor = path.parent();
            loop {
                match ancestor {
                    Some(a) if a.exists() => {
                        match a.canonicalize() {
                            Ok(canon_ancestor) => {
                                // Reconstruct the full path relative to the canonical ancestor
                                let remainder = path.strip_prefix(a).unwrap_or(path.as_ref());
                                break canon_ancestor.join(remainder);
                            }
                            Err(_) => return false,
                        }
                    }
                    Some(a) => ancestor = a.parent(),
                    None => return false,
                }
            }
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

#[async_trait]
impl SpiceModelTool for WriteFileTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<WriteFileToolParams>()
    }

    fn capability(&self) -> ToolCapability {
        ToolCapability::ReadWrite
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span: Span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::write_file", tool = self.name().to_string(), input = arg);

        let tool_use_result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let req: WriteFileToolParams = serde_json::from_str(arg)?;
            let file_path = PathBuf::from(&req.path);

            if !self.is_path_allowed(&file_path) {
                tracing::warn!(path = %req.path, allowed_dirs = ?self.base_paths, "write_file access denied: path is outside allowed directories");
                return Err(format!(
                    "Access denied: path '{}' is outside allowed directories",
                    req.path
                )
                .into());
            }

            if req.content.len() > MAX_FILE_SIZE {
                return Err(format!(
                    "Content too large: {} bytes (max {} bytes)",
                    req.content.len(),
                    MAX_FILE_SIZE
                )
                .into());
            }

            // Create parent directories if needed
            if let Some(parent) = file_path.parent() {
                if !parent.exists() {
                    fs::create_dir_all(parent)?;
                }
            }

            let bytes_written = req.content.len();
            fs::write(&file_path, &req.content)?;

            Ok(json!({
                "path": req.path,
                "bytes_written": bytes_written,
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_name_and_description() {
        let tool = WriteFileTool::new(None, None, vec![], None);
        assert_eq!(tool.name(), "write_file");
        assert!(tool.description().is_some());
        assert!(tool.parameters().is_some());
        assert_eq!(tool.capability(), ToolCapability::ReadWrite);
    }

    #[test]
    fn test_custom_name() {
        let tool = WriteFileTool::new(Some("my_writer"), Some("Custom writer"), vec![], None);
        assert_eq!(tool.name(), "my_writer");
        assert_eq!(tool.description().unwrap(), "Custom writer");
    }

    #[tokio::test]
    async fn test_write_new_file() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool::new(None, None, vec![dir.path().to_path_buf()], None);
        let file_path = dir.path().join("test.txt");

        let arg = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "content": "hello world"
        })
        .to_string();

        let result = tool.call(&arg).await.unwrap();
        assert_eq!(result["bytes_written"], 11);
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "hello world");
    }

    #[tokio::test]
    async fn test_overwrite_existing_file() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool::new(None, None, vec![dir.path().to_path_buf()], None);
        let file_path = dir.path().join("existing.txt");
        fs::write(&file_path, "old content").unwrap();

        let arg = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "content": "new content"
        })
        .to_string();

        let result = tool.call(&arg).await.unwrap();
        assert_eq!(result["bytes_written"], 11);
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "new content");
    }

    #[tokio::test]
    async fn test_reject_path_outside_base_paths() {
        let dir = TempDir::new().unwrap();
        let other_dir = TempDir::new().unwrap();
        let tool = WriteFileTool::new(None, None, vec![dir.path().to_path_buf()], None);
        let file_path = other_dir.path().join("forbidden.txt");

        let arg = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "content": "should fail"
        })
        .to_string();

        let result = tool.call(&arg).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Access denied")
        );
    }

    #[tokio::test]
    async fn test_reject_oversized_content() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool::new(None, None, vec![dir.path().to_path_buf()], None);
        let file_path = dir.path().join("big.txt");

        let big_content = "x".repeat(MAX_FILE_SIZE + 1);
        let arg = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "content": big_content
        })
        .to_string();

        let result = tool.call(&arg).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Content too large")
        );
    }

    #[tokio::test]
    async fn test_create_parent_directories() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool::new(None, None, vec![dir.path().to_path_buf()], None);
        let file_path = dir.path().join("sub").join("dir").join("file.txt");

        let arg = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "content": "nested"
        })
        .to_string();

        let result = tool.call(&arg).await.unwrap();
        assert_eq!(result["bytes_written"], 6);
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "nested");
    }

    #[tokio::test]
    async fn test_allow_worktree_path() {
        let dir = TempDir::new().unwrap();
        let wt_dir = TempDir::new().unwrap();
        let tracker = WorktreeTracker::default();
        tracker.register(
            "test-wt".to_string(),
            crate::tools::builtin::git_worktree::tracker::TrackedWorktree {
                path: wt_dir.path().to_path_buf(),
                branch: "test-branch".to_string(),
                repo_path: dir.path().to_path_buf(),
            },
        );

        // base_paths does NOT include wt_dir, but tracker does
        let tool = WriteFileTool::new(
            None,
            None,
            vec![dir.path().to_path_buf()],
            Some(tracker),
        );
        let file_path = wt_dir.path().join("worktree-file.txt");

        let arg = serde_json::json!({
            "path": file_path.to_str().unwrap(),
            "content": "in worktree"
        })
        .to_string();

        let result = tool.call(&arg).await.unwrap();
        assert_eq!(result["bytes_written"], 11);
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "in worktree");
    }
}
