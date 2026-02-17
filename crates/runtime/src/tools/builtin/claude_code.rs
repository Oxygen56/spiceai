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
use serde_json::{json, Value};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use tools::SpiceModelTool;
use tracing::Span;
use tracing_futures::Instrument;

use crate::tools::builtin::git_worktree::WorktreeTracker;
use crate::tools::utils::parameters;

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct ClaudeCodeToolParams {
    /// The prompt/instruction to send to Claude Code.
    prompt: String,

    /// Working directory for Claude Code to operate in.
    /// Use a worktree name (from git_worktree) or an absolute path from list_file_sources.
    working_directory: Option<String>,

    /// Tools Claude Code is allowed to use (e.g., ["Bash", "Read", "Edit"]).
    allowed_tools: Option<Vec<String>>,

    /// Maximum number of agentic turns Claude Code can take.
    max_turns: Option<u32>,

    /// Model to use (e.g., "sonnet", "opus").
    model: Option<String>,
}

#[derive(Debug)]
pub struct ClaudeCodeTool {
    name: String,
    description: String,
    claude_binary: String,
    default_model: Option<String>,
    default_max_turns: u32,
    default_allowed_tools: Vec<String>,
    allowed_working_dirs: Vec<PathBuf>,
    worktree_tracker: Option<WorktreeTracker>,
}

impl ClaudeCodeTool {
    /// Create a new `ClaudeCodeTool`.
    ///
    /// # Errors
    ///
    /// Returns an error if `allowed_working_dirs` is empty.
    pub fn try_new(
        name: Option<&str>,
        description: Option<&str>,
        claude_binary: Option<&str>,
        default_model: Option<String>,
        default_max_turns: Option<u32>,
        default_allowed_tools: Vec<String>,
        allowed_working_dirs: Vec<PathBuf>,
        worktree_tracker: Option<WorktreeTracker>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if allowed_working_dirs.is_empty() {
            return Err("allowed_working_dirs must not be empty".into());
        }

        Ok(Self {
            name: name.unwrap_or("claude_code").to_string(),
            description: description
                .unwrap_or(
                    "Invoke Claude Code CLI for complex coding tasks such as merge conflict resolution, \
                     code refactoring, and multi-file edits",
                )
                .to_string(),
            claude_binary: claude_binary.unwrap_or("claude").to_string(),
            default_model,
            default_max_turns: default_max_turns.unwrap_or(5),
            default_allowed_tools,
            allowed_working_dirs,
            worktree_tracker,
        })
    }

    /// Returns `true` if `path` is within one of the allowed working directories
    /// or within a tracked worktree.
    fn is_path_allowed(&self, path: &Path) -> bool {
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => return false,
        };

        // Check static allowed directories.
        if self.allowed_working_dirs.iter().any(|base| {
            base.canonicalize()
                .map_or(false, |b| canonical.starts_with(&b))
        }) {
            return true;
        }

        // Check tracked worktree paths.
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
impl SpiceModelTool for ClaudeCodeTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<ClaudeCodeToolParams>()
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span: Span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::claude_code", tool = self.name().to_string(), input = arg);

        let tool_use_result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let params: ClaudeCodeToolParams = serde_json::from_str(arg)?;

            // Determine and validate working directory.
            let working_dir = if let Some(ref dir) = params.working_directory {
                // Try to resolve as a worktree name first
                let resolved_from_tracker = self.worktree_tracker.as_ref()
                    .and_then(|tracker| tracker.get(dir))
                    .map(|wt| wt.path);

                if let Some(wt_path) = resolved_from_tracker {
                    tracing::debug!(worktree_name = %dir, path = %wt_path.display(), "Resolved working directory from worktree tracker");
                    wt_path
                } else {
                    let path = PathBuf::from(dir);
                    if !self.is_path_allowed(&path) {
                        tracing::warn!(working_directory = %dir, allowed_dirs = ?self.allowed_working_dirs, "claude_code access denied: working directory is outside allowed directories");
                        return Err(format!(
                            "Access denied: working directory '{}' is outside allowed directories",
                            dir
                        )
                        .into());
                    }
                    path
                }
            } else {
                self.allowed_working_dirs[0].clone()
            };

            let max_turns = params.max_turns.unwrap_or(self.default_max_turns);
            let model = params.model.as_deref().or(self.default_model.as_deref());
            let allowed_tools = params
                .allowed_tools
                .as_ref()
                .unwrap_or(&self.default_allowed_tools);

            // Build the Claude Code CLI command.
            let mut cmd = tokio::process::Command::new(&self.claude_binary);
            cmd.arg("-p")
                .arg(&params.prompt)
                .arg("--output-format")
                .arg("json")
                .arg("--max-turns")
                .arg(max_turns.to_string())
                // TODO: Make it proper
                .arg("--permission-mode")
                .arg("acceptEdits");

            // TODO: REMOVE
            println!("Claude code: command: {:#?}", cmd);

            for tool in allowed_tools {
                cmd.arg("--allowedTools").arg(tool);
            }

            if let Some(m) = model {
                cmd.arg("--model").arg(m);
            }

            cmd.current_dir(&working_dir);

            let output = cmd.output().await?;
            let exit_code = output.status.code().unwrap_or(-1);

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!(
                    "Claude Code exited with code {exit_code}: {stderr}"
                )
                .into());
            }

            let stdout = String::from_utf8_lossy(&output.stdout);

            // Attempt to parse JSON output and extract the result field.
            let result_text = if let Ok(parsed) = serde_json::from_str::<Value>(&stdout) {
                if let Some(result) = parsed.get("result").and_then(Value::as_str) {
                    result.to_string()
                } else {
                    stdout.to_string()
                }
            } else {
                stdout.to_string()
            };

            Ok(json!({
                "result": result_text,
                "model": model.unwrap_or("default"),
                "exit_code": exit_code,
            }))
        }
        .instrument(span.clone())
        .await;

        match tool_use_result {
            Ok(value) => {
                let captured_output_json = serde_json::to_string(&value)?;
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
    fn test_reject_disallowed_working_dir() {
        let allowed = TempDir::new().unwrap();
        let tool = ClaudeCodeTool::try_new(
            None,
            None,
            None,
            None,
            None,
            vec![],
            vec![allowed.path().to_path_buf()],
            None,
        )
        .unwrap();

        let disallowed = TempDir::new().unwrap();
        assert!(
            !tool.is_path_allowed(disallowed.path()),
            "Path outside allowed directories should be rejected"
        );
        assert!(
            tool.is_path_allowed(allowed.path()),
            "Path within allowed directories should be accepted"
        );
    }

    #[test]
    fn test_default_values() {
        let dir = TempDir::new().unwrap();
        let tool = ClaudeCodeTool::try_new(
            None,
            None,
            None,
            None,
            None,
            vec![],
            vec![dir.path().to_path_buf()],
            None,
        )
        .unwrap();

        assert_eq!(tool.name, "claude_code");
        assert_eq!(tool.claude_binary, "claude");
        assert_eq!(tool.default_max_turns, 5);
        assert!(tool.default_model.is_none());
        assert!(tool.default_allowed_tools.is_empty());
    }

    #[test]
    fn test_empty_allowed_dirs_rejected() {
        let result = ClaudeCodeTool::try_new(None, None, None, None, None, vec![], vec![], None);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("allowed_working_dirs must not be empty"));
    }

    #[test]
    fn test_params_schema_generation() {
        let dir = TempDir::new().unwrap();
        let tool = ClaudeCodeTool::try_new(
            None,
            None,
            None,
            None,
            None,
            vec![],
            vec![dir.path().to_path_buf()],
            None,
        )
        .unwrap();

        let params = tool.parameters();
        assert!(params.is_some(), "parameters() should return Some");
    }

    #[test]
    fn test_custom_values() {
        let dir = TempDir::new().unwrap();
        let tool = ClaudeCodeTool::try_new(
            Some("my_claude"),
            Some("Custom description"),
            Some("/usr/local/bin/claude"),
            Some("opus".to_string()),
            Some(10),
            vec!["Bash".to_string(), "Read".to_string()],
            vec![dir.path().to_path_buf()],
            None,
        )
        .unwrap();

        assert_eq!(tool.name, "my_claude");
        assert_eq!(tool.description, "Custom description");
        assert_eq!(tool.claude_binary, "/usr/local/bin/claude");
        assert_eq!(tool.default_model, Some("opus".to_string()));
        assert_eq!(tool.default_max_turns, 10);
        assert_eq!(tool.default_allowed_tools, vec!["Bash", "Read"]);
    }
}
