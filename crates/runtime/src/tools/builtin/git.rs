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
use tools::{SpiceModelTool, ToolCapability};
use tracing::Span;
use tracing_futures::Instrument;

use crate::tools::builtin::git_worktree::WorktreeTracker;
use crate::tools::utils::parameters;

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct GitToolParams {
    /// The git operation: "status", "log", "diff", "add", "branch", "checkout", "cherry-pick", "commit", "push", "fetch".
    operation: String,
    /// Working directory. Must be an absolute path within repo_path or a tracked worktree.
    /// Prefer using worktree_name instead for worktree operations.
    working_directory: Option<String>,
    /// For cherry-pick: list of commit SHAs to cherry-pick.
    commits: Option<Vec<String>>,
    /// For commit: the commit message.
    message: Option<String>,
    /// For branch/checkout: the branch name.
    branch: Option<String>,
    /// For push: the remote name (defaults to "origin").
    remote: Option<String>,
    /// For add: list of file paths to stage. If empty or omitted, stages all changes (".").
    files: Option<Vec<String>>,
    /// Additional CLI arguments.
    args: Option<Vec<String>>,
    /// Optional worktree name to resolve from WorktreeTracker.
    /// When provided, operations execute in the worktree instead of repo_path.
    worktree_name: Option<String>,
}

const READ_OPERATIONS: &[&str] = &["status", "log", "diff"];
const WRITE_OPERATIONS: &[&str] = &["add", "branch", "checkout", "cherry-pick", "commit", "push", "fetch"];

#[derive(Debug)]
pub struct GitTool {
    name: String,
    description: String,
    repo_path: PathBuf,
    capability: ToolCapability,
    allowed_operations: Vec<String>,
    worktree_tracker: Option<WorktreeTracker>,
    github_token: Option<String>,
}

impl GitTool {
    /// Create a new `GitTool`.
    ///
    /// # Errors
    ///
    /// Returns an error if the `repo_path` does not exist or is not a directory.
    pub fn try_new(
        name: Option<&str>,
        description: Option<&str>,
        repo_path: PathBuf,
        capability: ToolCapability,
        allowed_operations: Vec<String>,
        worktree_tracker: Option<WorktreeTracker>,
        github_token: Option<String>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if !repo_path.exists() {
            return Err(
                format!("repo_path '{}' does not exist", repo_path.display()).into(),
            );
        }
        if !repo_path.is_dir() {
            return Err(
                format!("repo_path '{}' is not a directory", repo_path.display()).into(),
            );
        }

        Ok(Self {
            name: name.unwrap_or("git").to_string(),
            description: description
                .unwrap_or("Perform git operations on a repository")
                .to_string(),
            repo_path,
            capability,
            allowed_operations,
            worktree_tracker,
            github_token,
        })
    }

    /// Validate the incoming request parameters before executing.
    fn validate_request(
        &self,
        req: &GitToolParams,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Validate operation is recognized
        if !READ_OPERATIONS.contains(&req.operation.as_str())
            && !WRITE_OPERATIONS.contains(&req.operation.as_str())
        {
            return Err(format!(
                "Unknown operation '{}'. Recognized operations: {}",
                req.operation,
                READ_OPERATIONS
                    .iter()
                    .chain(WRITE_OPERATIONS.iter())
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .into());
        }

        // Validate operation is in allowed list
        if !self
            .allowed_operations
            .iter()
            .any(|op| op == &req.operation)
        {
            return Err(format!(
                "Operation '{}' is not in allowed operations: {:?}",
                req.operation, self.allowed_operations
            )
            .into());
        }

        // Block write operations in read-only mode
        if self.capability == ToolCapability::ReadOnly
            && WRITE_OPERATIONS.contains(&req.operation.as_str())
        {
            return Err(format!(
                "Operation '{}' is not permitted in read-only mode",
                req.operation
            )
            .into());
        }

        // Reject --force in push args
        if req.operation == "push" {
            if let Some(ref args) = req.args {
                if args.iter().any(|a| a == "--force" || a == "-f") {
                    return Err(
                        "Force push is not allowed. Remove '--force' / '-f' from args."
                            .to_string()
                            .into(),
                    );
                }
            }
        }

        // Validate cherry-pick has commits (unless --abort/--continue/--skip)
        if req.operation == "cherry-pick" {
            let has_control_flag = req.args.as_ref().map_or(false, |args| {
                args.iter()
                    .any(|a| a == "--abort" || a == "--continue" || a == "--skip")
            });
            if !has_control_flag {
                match &req.commits {
                    None | Some(_) if req.commits.as_ref().map_or(true, Vec::is_empty) => {
                        return Err(
                            "Operation 'cherry-pick' requires a non-empty 'commits' list (or use --abort/--continue/--skip in args)"
                                .to_string()
                                .into(),
                        );
                    }
                    _ => {}
                }
            }
        }

        // Validate commit has message
        if req.operation == "commit" && req.message.as_deref().unwrap_or("").is_empty() {
            return Err(
                "Operation 'commit' requires a non-empty 'message'"
                    .to_string()
                    .into(),
            );
        }

        Ok(())
    }

    /// Determine the working directory for the command. Three resolution strategies:
    /// 1. worktree_name: Look up in WorktreeTracker and use worktree path
    /// 2. working_directory: Validate it's within repo_path and use it
    /// 3. Default: Use repo_path
    fn resolve_working_directory(
        &self,
        req: &GitToolParams,
    ) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
        // Priority 1: worktree_name lookup
        if let Some(wt_name) = &req.worktree_name {
            if let Some(tracker) = &self.worktree_tracker {
                if let Some(tracked_wt) = tracker.get(wt_name) {
                    tracing::debug!(
                        target: "task_history",
                        worktree_name = %wt_name,
                        worktree_path = %tracked_wt.path.display(),
                        "Resolved worktree from tracker"
                    );
                    return Ok(tracked_wt.path);
                }
                tracing::warn!(
                    worktree_name = %wt_name,
                    "Worktree not found in tracker, falling back to repo_path"
                );
            }
        }

        // Priority 2: working_directory parameter (must be within repo_path or a tracked worktree)
        if let Some(wd) = &req.working_directory {
            let wd_path = Path::new(wd);
            if !wd_path.exists() {
                return Err(format!("working_directory '{wd}' does not exist").into());
            }
            let canonical_wd = wd_path.canonicalize()?;
            let canonical_repo = self.repo_path.canonicalize()?;

            // Allow if within repo_path
            if canonical_wd.starts_with(&canonical_repo) {
                return Ok(canonical_wd);
            }

            // Allow if it matches a tracked worktree path
            if let Some(tracker) = &self.worktree_tracker {
                for (_name, wt) in tracker.list() {
                    if let Ok(canonical_wt) = wt.path.canonicalize() {
                        if canonical_wd.starts_with(&canonical_wt) {
                            return Ok(canonical_wd);
                        }
                    }
                }
            }

            return Err(format!(
                "working_directory '{}' is outside the configured repo_path '{}' and is not a tracked worktree",
                wd,
                self.repo_path.display()
            )
            .into());
        }

        // Priority 3: repo_path default
        Ok(self.repo_path.clone())
    }

    /// Build a `tokio::process::Command` for the given operation.
    fn build_command(
        &self,
        req: &GitToolParams,
        work_dir: &Path,
    ) -> Result<tokio::process::Command, Box<dyn std::error::Error + Send + Sync>> {
        let mut cmd = tokio::process::Command::new("git");
        cmd.current_dir(work_dir);

        // Disable interactive credential prompts — fail fast instead of hanging
        cmd.env("GIT_TERMINAL_PROMPT", "0");
        cmd.env("GIT_EDITOR", "true");       // prevent editor from opening
        cmd.env("GIT_PAGER", "cat");         // prevent pager from opening (e.g., git log)

        // Inject HTTPS credentials via inline credential helper
        if let Some(ref token) = self.github_token {
            let helper = format!(
                "!f() {{ echo \"username=x-access-token\"; echo \"password={token}\"; }}; f"
            );
            cmd.arg("-c").arg(format!("credential.helper={helper}"));
        }

        match req.operation.as_str() {
            "status" => {
                cmd.arg("status").arg("--porcelain");
                if let Some(ref args) = req.args {
                    cmd.args(args);
                }
            }
            "log" => {
                cmd.arg("log").arg("--oneline");
                if let Some(ref args) = req.args {
                    cmd.args(args);
                } else {
                    cmd.arg("-20");
                }
            }
            "diff" => {
                cmd.arg("diff");
                if let Some(ref args) = req.args {
                    cmd.args(args);
                }
            }
            "add" => {
                cmd.arg("add");
                if let Some(ref files) = req.files {
                    if !files.is_empty() {
                        cmd.args(files);
                    } else {
                        cmd.arg(".");
                    }
                } else {
                    cmd.arg(".");
                }
                if let Some(ref args) = req.args {
                    cmd.args(args);
                }
            }
            "branch" => {
                cmd.arg("branch");
                if let Some(ref branch) = req.branch {
                    cmd.arg(branch);
                } else {
                    cmd.arg("--list");
                }
                if let Some(ref args) = req.args {
                    cmd.args(args);
                }
            }
            "checkout" => {
                cmd.arg("checkout");
                // When args contain -b, don't pass branch as a positional arg
                // (it would be interpreted as a path, not a start point)
                let has_branch_flag = req.args.as_ref().map_or(false, |args| {
                    args.iter().any(|a| a == "-b" || a == "-B")
                });
                if !has_branch_flag {
                    if let Some(ref branch) = req.branch {
                        cmd.arg(branch);
                    }
                }
                if let Some(ref args) = req.args {
                    cmd.args(args);
                }
            }
            "cherry-pick" => {
                cmd.arg("cherry-pick");
                // commits is validated to be non-empty in validate_request
                if let Some(ref commits) = req.commits {
                    cmd.args(commits);
                }
                if let Some(ref args) = req.args {
                    cmd.args(args);
                }
            }
            "commit" => {
                cmd.arg("commit").arg("-m");
                // message is validated to be non-empty in validate_request
                if let Some(ref message) = req.message {
                    cmd.arg(message);
                }
                if let Some(ref args) = req.args {
                    cmd.args(args);
                }
            }
            "push" => {
                cmd.arg("push");
                let remote = req
                    .remote
                    .as_deref()
                    .unwrap_or("origin");
                cmd.arg(remote);
                if let Some(ref branch) = req.branch {
                    cmd.arg(branch);
                } else {
                    // Auto-detect current branch to avoid origin/HEAD resolution
                    // failures (e.g. "refs/remotes/origin/HEAD cannot be resolved
                    // to branch" in worktrees with broken symrefs).
                    if let Ok(repo) = git2::Repository::open(work_dir) {
                        if let Ok(head) = repo.head() {
                            if let Some(branch_name) = head.shorthand() {
                                cmd.arg(branch_name);
                            }
                        }
                    }
                }
                if let Some(ref args) = req.args {
                    cmd.args(args);
                }
            }
            "fetch" => {
                cmd.arg("fetch");
                let remote = req
                    .remote
                    .as_deref()
                    .unwrap_or("origin");
                cmd.arg(remote);
                if let Some(ref branch) = req.branch {
                    cmd.arg(branch);
                }
                if let Some(ref args) = req.args {
                    cmd.args(args);
                }
            }
            _ => {
                return Err(
                    format!("Unhandled operation: {}", req.operation).into()
                );
            }
        }

        Ok(cmd)
    }
}

#[async_trait]
impl SpiceModelTool for GitTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<GitToolParams>()
    }

    fn capability(&self) -> ToolCapability {
        self.capability
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span: Span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::git", tool = self.name().to_string(), input = arg);

        let tool_use_result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let req: GitToolParams = serde_json::from_str(arg)?;

            self.validate_request(&req)?;

            let work_dir = self.resolve_working_directory(&req)?;
            tracing::debug!(operation = %req.operation, work_dir = %work_dir.display(), "Executing git operation");

            let mut cmd = self.build_command(&req, &work_dir)?;

            let output = cmd.output().await?;

            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();

            if !output.status.success() {
                tracing::warn!(
                    operation = %req.operation,
                    exit_code = output.status.code().unwrap_or(-1),
                    stderr = %stderr,
                    "Git command exited with non-zero status");
            }

            Ok(json!({
                "exit_code": output.status.code().unwrap_or(-1),
                "stdout": stdout,
                "stderr": stderr,
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
                tracing::error!(target: "task_history", parent: &span, "Git tool failed: {e}");
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    /// Helper to create a `GitTool` pointing at a real directory (the system temp dir)
    /// so `try_new` succeeds, but we never actually run git commands in these tests.
    fn make_tool(ops: Vec<&str>, capability: ToolCapability) -> GitTool {
        let repo_path = env::temp_dir();
        GitTool::try_new(
            None,
            None,
            repo_path,
            capability,
            ops.into_iter().map(ToString::to_string).collect(),
            None,
            None,
        )
        .expect("temp_dir should exist")
    }

    #[test]
    fn test_reject_unknown_operation() {
        let tool = make_tool(
            vec!["status", "log", "diff", "commit", "push"],
            ToolCapability::ReadWrite,
        );
        let req = GitToolParams {
            operation: "rebase".to_string(),
            working_directory: None,
            commits: None,
            message: None,
            branch: None,
            remote: None,
            files: None,
            args: None,
            worktree_name: None,
        };
        let result = tool.validate_request(&req);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unknown operation 'rebase'"),
            "Expected unknown operation error"
        );
    }

    #[test]
    fn test_readonly_rejects_write_operation() {
        let tool = make_tool(
            vec!["status", "log", "diff", "commit", "push"],
            ToolCapability::ReadOnly,
        );
        let req = GitToolParams {
            operation: "commit".to_string(),
            working_directory: None,
            commits: None,
            message: Some("test commit".to_string()),
            branch: None,
            remote: None,
            files: None,
            args: None,
            worktree_name: None,
        };
        let result = tool.validate_request(&req);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not permitted in read-only mode"),
            "Expected read-only mode error"
        );
    }

    #[test]
    fn test_force_push_rejected() {
        let tool = make_tool(
            vec!["push"],
            ToolCapability::ReadWrite,
        );

        // --force flag
        let req = GitToolParams {
            operation: "push".to_string(),
            working_directory: None,
            commits: None,
            message: None,
            branch: Some("main".to_string()),
            remote: Some("origin".to_string()),
            files: None,
            args: Some(vec!["--force".to_string()]),
            worktree_name: None,
        };
        let result = tool.validate_request(&req);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Force push is not allowed"),
            "Expected force push rejection"
        );

        // -f shorthand
        let req_short = GitToolParams {
            operation: "push".to_string(),
            working_directory: None,
            commits: None,
            message: None,
            branch: Some("main".to_string()),
            remote: Some("origin".to_string()),
            files: None,
            args: Some(vec!["-f".to_string()]),
            worktree_name: None,
        };
        let result_short = tool.validate_request(&req_short);
        assert!(result_short.is_err());
        assert!(
            result_short
                .unwrap_err()
                .to_string()
                .contains("Force push is not allowed"),
            "Expected force push rejection for -f shorthand"
        );
    }

    #[test]
    fn test_operation_allowlist() {
        let tool = make_tool(
            vec!["status", "log"],
            ToolCapability::ReadOnly,
        );
        let req = GitToolParams {
            operation: "diff".to_string(),
            working_directory: None,
            commits: None,
            message: None,
            branch: None,
            remote: None,
            files: None,
            args: None,
            worktree_name: None,
        };
        let result = tool.validate_request(&req);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not in allowed operations"),
            "Expected allowlist error"
        );
    }

    #[test]
    fn test_cherry_pick_requires_commits() {
        let tool = make_tool(
            vec!["cherry-pick"],
            ToolCapability::ReadWrite,
        );

        // No commits field
        let req = GitToolParams {
            operation: "cherry-pick".to_string(),
            working_directory: None,
            commits: None,
            message: None,
            branch: None,
            remote: None,
            files: None,
            args: None,
            worktree_name: None,
        };
        let result = tool.validate_request(&req);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("non-empty 'commits' list"),
        );

        // Empty commits list
        let req_empty = GitToolParams {
            operation: "cherry-pick".to_string(),
            working_directory: None,
            commits: Some(vec![]),
            message: None,
            branch: None,
            remote: None,
            files: None,
            args: None,
            worktree_name: None,
        };
        let result_empty = tool.validate_request(&req_empty);
        assert!(result_empty.is_err());
        assert!(
            result_empty
                .unwrap_err()
                .to_string()
                .contains("non-empty 'commits' list"),
        );
    }

    #[test]
    fn test_commit_requires_message() {
        let tool = make_tool(
            vec!["commit"],
            ToolCapability::ReadWrite,
        );
        let req = GitToolParams {
            operation: "commit".to_string(),
            working_directory: None,
            commits: None,
            message: None,
            branch: None,
            remote: None,
            files: None,
            args: None,
            worktree_name: None,
        };
        let result = tool.validate_request(&req);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("non-empty 'message'"),
        );
    }

    #[test]
    fn test_valid_read_operation() {
        let tool = make_tool(
            vec!["status", "log", "diff"],
            ToolCapability::ReadOnly,
        );
        let req = GitToolParams {
            operation: "status".to_string(),
            working_directory: None,
            commits: None,
            message: None,
            branch: None,
            remote: None,
            files: None,
            args: None,
            worktree_name: None,
        };
        assert!(tool.validate_request(&req).is_ok());
    }

    #[test]
    fn test_valid_write_operation() {
        let tool = make_tool(
            vec!["commit", "push", "cherry-pick"],
            ToolCapability::ReadWrite,
        );

        let req = GitToolParams {
            operation: "commit".to_string(),
            working_directory: None,
            commits: None,
            message: Some("fix: resolve issue".to_string()),
            branch: None,
            remote: None,
            files: None,
            args: None,
            worktree_name: None,
        };
        assert!(tool.validate_request(&req).is_ok());

        let req_push = GitToolParams {
            operation: "push".to_string(),
            working_directory: None,
            commits: None,
            message: None,
            branch: Some("main".to_string()),
            remote: Some("origin".to_string()),
            files: None,
            args: None,
            worktree_name: None,
        };
        assert!(tool.validate_request(&req_push).is_ok());

        let req_cp = GitToolParams {
            operation: "cherry-pick".to_string(),
            working_directory: None,
            commits: Some(vec!["abc123".to_string(), "def456".to_string()]),
            message: None,
            branch: None,
            remote: None,
            files: None,
            args: None,
            worktree_name: None,
        };
        assert!(tool.validate_request(&req_cp).is_ok());
    }

    #[test]
    fn test_try_new_rejects_nonexistent_path() {
        let result = GitTool::try_new(
            None,
            None,
            PathBuf::from("/nonexistent/path/to/repo"),
            ToolCapability::ReadWrite,
            vec!["status".to_string()],
            None,
            None,
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("does not exist"),
        );
    }
}
