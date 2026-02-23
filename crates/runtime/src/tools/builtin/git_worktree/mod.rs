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
use snafu::ResultExt;
use std::borrow::Cow;
use std::path::PathBuf;
use tools::SpiceModelTool;
use tracing::Span;
use tracing_futures::Instrument;

use crate::tools::utils::parameters;

pub mod tracker;
pub use tracker::WorktreeTracker;

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct GitWorktreeToolParams {
    /// The operation to perform: "create", "list", "status", or "remove".
    operation: String,

    /// For create: the branch name (will be created if it doesn't exist).
    branch: Option<String>,

    /// For create: subdirectory name under the worktree root.
    path_suffix: Option<String>,

    /// For status/remove: the worktree name.
    worktree_name: Option<String>,
}

#[derive(Debug)]
pub struct GitWorktreeTool {
    name: String,
    description: String,
    repo_path: PathBuf,
    worktree_root: PathBuf,
    tracker: WorktreeTracker,
}

impl GitWorktreeTool {
    /// Create a new `GitWorktreeTool`.
    ///
    /// Validates that `repo_path` is a valid git repository and creates `worktree_root`
    /// if it does not already exist.
    pub fn try_new(
        name: Option<&str>,
        description: Option<&str>,
        repo_path: PathBuf,
        worktree_root: PathBuf,
        tracker: WorktreeTracker,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Create the worktree root directory if it doesn't exist.
        if !worktree_root.exists() {
            std::fs::create_dir_all(&worktree_root).map_err(|e| {
                format!(
                    "Failed to create worktree root directory '{}': {e}",
                    worktree_root.display()
                )
            })?;
        }

        Ok(Self {
            name: name.unwrap_or("git_worktree").to_string(),
            description: description
                .unwrap_or("Create and manage git worktrees for parallel branch work")
                .to_string(),
            repo_path,
            worktree_root,
            tracker,
        })
    }

    fn handle_create(
        &self,
        branch: &str,
        path_suffix: Option<&str>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let repo = git2::Repository::open(&self.repo_path)?;

        let worktree_name = path_suffix.unwrap_or(branch);
        let worktree_path = self.worktree_root.join(worktree_name);

        tracing::debug!(branch = %branch, worktree_name = %worktree_name, worktree_path = %worktree_path.display(), "Creating git worktree");

        if worktree_path.exists() {
            return Err(format!(
                "Worktree path '{}' already exists",
                worktree_path.display()
            )
            .into());
        }

        // Resolve the branch: local first, then remote tracking, then create from HEAD.
        let reference = match repo.find_branch(branch, git2::BranchType::Local) {
            Ok(b) => b.into_reference(),
            Err(_) => {
                // Try remote tracking branch (e.g., origin/release/1.1.x)
                let remote_ref = format!("refs/remotes/origin/{branch}");
                match repo.find_reference(&remote_ref) {
                    Ok(remote_reference) => {
                        let commit = remote_reference
                            .peel_to_commit()
                            .map_err(|e| format!("Failed to resolve remote ref to commit: {e}"))?;
                        tracing::info!(branch = %branch, commit = %commit.id(), "Creating local branch from remote tracking branch");
                        repo.branch(branch, &commit, false)?.into_reference()
                    }
                    Err(_) => {
                        tracing::info!(branch = %branch, "Branch not found locally or on remote, creating from HEAD");
                        let head_commit = repo
                            .head()?
                            .peel_to_commit()
                            .map_err(|e| format!("Failed to resolve HEAD to commit: {e}"))?;
                        repo.branch(branch, &head_commit, false)?.into_reference()
                    }
                }
            }
        };

        let mut opts = git2::WorktreeAddOptions::new();
        opts.reference(Some(&reference));

        repo.worktree(worktree_name, &worktree_path, Some(&opts))?;

        self.tracker.register(
            worktree_name.to_string(),
            tracker::TrackedWorktree {
                path: worktree_path.clone(),
                branch: branch.to_string(),
                repo_path: self.repo_path.clone(),
            },
        );

        Ok(json!({
            "status": "created",
            "worktree_name": worktree_name,
            "path": worktree_path.to_string_lossy(),
            "branch": branch,
        }))
    }

    fn handle_list(&self) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let repo = git2::Repository::open(&self.repo_path)?;
        let worktree_names = repo.worktrees()?;

        let mut entries = Vec::new();
        for i in 0..worktree_names.len() {
            let Some(wt_name) = worktree_names.get(i) else {
                continue;
            };

            let mut entry = json!({ "name": wt_name });

            if let Ok(wt) = repo.find_worktree(wt_name) {
                entry["path"] = json!(wt.path().to_string_lossy());
                entry["valid"] = json!(wt.validate().is_ok());
            }

            entries.push(entry);
        }

        Ok(json!({
            "worktrees": entries,
            "total": entries.len(),
        }))
    }

    fn handle_status(
        &self,
        worktree_name: &str,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let repo = git2::Repository::open(&self.repo_path)?;
        let wt = repo.find_worktree(worktree_name)?;
        let wt_path = wt.path().to_path_buf();

        let wt_repo = git2::Repository::open(&wt_path)?;
        let statuses = wt_repo.statuses(None)?;

        let mut changed_files = Vec::new();
        for entry in statuses.iter() {
            let status = entry.status();
            let path = entry.path().unwrap_or("<non-utf8>").to_string();

            let state = if status.is_index_new() || status.is_wt_new() {
                "new"
            } else if status.is_index_modified() || status.is_wt_modified() {
                "modified"
            } else if status.is_index_deleted() || status.is_wt_deleted() {
                "deleted"
            } else if status.is_index_renamed() || status.is_wt_renamed() {
                "renamed"
            } else {
                "other"
            };

            changed_files.push(json!({
                "path": path,
                "state": state,
            }));
        }

        // Get current branch name and HEAD commit
        let branch_name = wt_repo
            .head()
            .ok()
            .and_then(|h| h.shorthand().map(String::from));
        let head_commit = wt_repo
            .head()
            .ok()
            .and_then(|h| h.peel_to_commit().ok())
            .map(|c| c.id().to_string());

        Ok(json!({
            "worktree_name": worktree_name,
            "path": wt_path.to_string_lossy(),
            "branch": branch_name,
            "head_commit": head_commit,
            "changed_files": changed_files,
            "total_changes": changed_files.len(),
        }))
    }

    fn handle_remove(
        &self,
        worktree_name: &str,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let repo = git2::Repository::open(&self.repo_path)?;
        let wt = repo.find_worktree(worktree_name)?;
        let wt_path = wt.path().to_path_buf();

        tracing::debug!(worktree_name = %worktree_name, path = %wt_path.display(), "Removing worktree");

        wt.prune(Some(
            &mut git2::WorktreePruneOptions::new()
                .valid(true)
                .working_tree(true),
        ))?;

        // Remove the directory from the filesystem.
        if wt_path.exists() {
            std::fs::remove_dir_all(&wt_path)?;
        }

        // Deregister from the tracker if it was tracked.
        self.tracker.deregister(worktree_name);

        Ok(json!({
            "status": "removed",
            "worktree_name": worktree_name,
            "path": wt_path.to_string_lossy(),
        }))
    }
}

#[async_trait]
impl SpiceModelTool for GitWorktreeTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<GitWorktreeToolParams>()
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        // Validate that repo_path is actually a git repository.
        git2::Repository::open(&self.repo_path).map_err(|e| {
            format!(
                "Failed to open git repository at '{}': {e}",
                self.repo_path.display()
            )
        })?;

        let span: Span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::git_worktree", tool = self.name().to_string(), input = arg);

        let tool_use_result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let req: GitWorktreeToolParams = serde_json::from_str(arg)?;

            match req.operation.as_str() {
                "create" => {
                    let branch = req.branch.as_deref().ok_or(
                        "The 'branch' parameter is required for the 'create' operation",
                    )?;
                    self.handle_create(branch, req.path_suffix.as_deref())
                }
                "list" => self.handle_list(),
                "status" => {
                    let worktree_name = req.worktree_name.as_deref().ok_or(
                        "The 'worktree_name' parameter is required for the 'status' operation",
                    )?;
                    self.handle_status(worktree_name)
                }
                "remove" => {
                    let worktree_name = req.worktree_name.as_deref().ok_or(
                        "The 'worktree_name' parameter is required for the 'remove' operation",
                    )?;
                    self.handle_remove(worktree_name)
                }
                other => Err(format!(
                    "Unknown operation '{other}'. Valid operations are: create, list, status, remove"
                )
                .into()),
            }
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
                tracing::error!(target: "task_history", parent: &span, "git_worktree failed: {e}");
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn init_test_repo() -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let repo = git2::Repository::init(dir.path()).unwrap();

        // Create an initial commit so HEAD is valid.
        let sig = git2::Signature::now("Test", "test@example.com").unwrap();
        let tree_id = {
            let mut index = repo.index().unwrap();
            index.write_tree().unwrap()
        };
        let tree = repo.find_tree(tree_id).unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "Initial commit", &tree, &[])
            .unwrap();

        let path = dir.path().to_path_buf();
        (dir, path)
    }

    #[test]
    fn test_reject_invalid_repo_path() {
        let tracker = WorktreeTracker::default();
        let result = GitWorktreeTool::try_new(
            None,
            None,
            PathBuf::from("/tmp/definitely_not_a_git_repo_12345"),
            PathBuf::from("/tmp/worktrees"),
            tracker,
        );
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to open git repository"),
            "Unexpected error: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_create_and_list() {
        let (_dir, repo_path) = init_test_repo();
        let wt_root = TempDir::new().unwrap();
        let tracker = WorktreeTracker::default();

        let tool = GitWorktreeTool::try_new(
            None,
            None,
            repo_path,
            wt_root.path().to_path_buf(),
            tracker.clone(),
        )
        .unwrap();

        // Create a worktree.
        let create_result = tool
            .call(r#"{"operation": "create", "branch": "feature-test"}"#)
            .await
            .unwrap();

        assert_eq!(create_result["status"], "created");
        assert_eq!(create_result["branch"], "feature-test");
        assert_eq!(create_result["worktree_name"], "feature-test");
        assert!(!tracker.is_empty());

        // List worktrees.
        let list_result = tool
            .call(r#"{"operation": "list"}"#)
            .await
            .unwrap();

        let worktrees = list_result["worktrees"].as_array().unwrap();
        assert!(
            !worktrees.is_empty(),
            "Expected at least one worktree in list"
        );

        let names: Vec<&str> = worktrees
            .iter()
            .filter_map(|w| w["name"].as_str())
            .collect();
        assert!(
            names.contains(&"feature-test"),
            "Expected 'feature-test' in worktree list, got: {names:?}"
        );
    }

    #[test]
    fn test_tracker_register_deregister() {
        let tracker = WorktreeTracker::default();
        assert!(tracker.is_empty());

        tracker.register(
            "wt-1".to_string(),
            tracker::TrackedWorktree {
                path: PathBuf::from("/tmp/wt-1"),
                branch: "branch-1".to_string(),
                repo_path: PathBuf::from("/tmp/repo"),
            },
        );
        assert!(!tracker.is_empty());

        let removed = tracker.deregister("wt-1");
        assert!(removed.is_some());
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_tracker_drain() {
        let tracker = WorktreeTracker::default();
        tracker.register(
            "wt-a".to_string(),
            tracker::TrackedWorktree {
                path: PathBuf::from("/tmp/wt-a"),
                branch: "branch-a".to_string(),
                repo_path: PathBuf::from("/tmp/repo"),
            },
        );
        tracker.register(
            "wt-b".to_string(),
            tracker::TrackedWorktree {
                path: PathBuf::from("/tmp/wt-b"),
                branch: "branch-b".to_string(),
                repo_path: PathBuf::from("/tmp/repo"),
            },
        );

        let drained = tracker.drain();
        assert_eq!(drained.len(), 2);
        assert!(tracker.is_empty());
    }
}
