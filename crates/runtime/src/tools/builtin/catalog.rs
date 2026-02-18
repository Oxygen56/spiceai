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

use runtime_datafusion::allowlist::ResolvedTableAwareAllowlist;
use secrecy::{ExposeSecret, SecretString};
use snafu::{ResultExt, Snafu};
use spicepod::component::tool::Tool;
use std::{collections::HashMap, path::{Path, PathBuf}, sync::Arc, time::Duration};
use tools::ToolCapability;

use crate::{
    Runtime,
    datafusion::{SPICE_DEFAULT_CATALOG, SPICE_DEFAULT_SCHEMA},
    tools::{
        catalog::SpiceToolCatalog, factory::IndividualToolFactory, options::SpiceToolsOptions,
    },
};

use super::{
    SpiceModelTool,
    approval::{
        ApprovalTimeoutAction, ApprovalTool,
        ms_teams::ApprovalMsTeamsTool,
        slack::ApprovalSlackTool,
        store::ApprovalStore,
    },
    claude_code::ClaudeCodeTool,
    debug::DebugTool,
    fail::FailTool,
    get_readiness::GetReadinessTool,
    git::GitTool,
    github::GitHubTool,
    git_worktree::{GitWorktreeTool, WorktreeTracker},
    grep::GrepTool,
    http_tool::HttpTool,
    kubectl::KubectlTool,
    list_datasets::ListDatasetsTool,
    list_file_sources::ListFileSourcesTool,
    list_files::ListFilesTool,
    ms_teams::TeamsTool,
    read_file::ReadFileTool,
    sample::{SampleTableMethod, tool::SampleDataTool},
    search::SearchTool,
    slack::SlackTool,
    sql::SqlTool,
    table_schema::TableSchemaTool,
    web_search::WebSearchTool,
};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Unknown builtin tool: {id}"))]
    UnknownBuiltinTool { id: String },

    #[snafu(display("Failed to construct tool '{id}'. Error: {source}"))]
    FailedToConstructTool {
        id: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Clone)]
pub struct BuiltinToolCatalog {
    rt: Arc<Runtime>,
    /// An optional table allowlist. Overriden by any per-tool `table_allowlist` param.
    model_table_allowlist: Option<ResolvedTableAwareAllowlist>,
    /// Shared worktree tracker for session-scoped cleanup of git worktrees.
    worktree_tracker: WorktreeTracker,
    /// Shared approval store for human-in-the-loop approval tools.
    approval_store: ApprovalStore,
}

/// Check if a string contains glob metacharacters.
fn contains_glob_chars(s: &str) -> bool {
    s.contains('*') || s.contains('?') || s.contains('[')
}

/// Expand a list of base path patterns (which may contain globs) into concrete directory paths.
///
/// Patterns without glob metacharacters are passed through as-is.
/// Glob patterns are expanded using filesystem walking + `globset` matching.
fn expand_base_paths(patterns: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut result = Vec::new();
    for pattern in &patterns {
        let pattern_str = pattern.to_string_lossy();
        if !contains_glob_chars(&pattern_str) {
            result.push(pattern.clone());
            continue;
        }

        // Find the longest non-glob prefix to determine where to start walking
        let parts: Vec<&str> = pattern_str.split('/').collect();
        let mut root = PathBuf::new();
        for part in &parts {
            if contains_glob_chars(part) {
                break;
            }
            if root.as_os_str().is_empty() && part.is_empty() {
                // Preserve leading "/" for absolute paths
                root.push("/");
            } else {
                root.push(part);
            }
        }
        if root.as_os_str().is_empty() {
            root = PathBuf::from(".");
        }

        let matcher = match globset::Glob::new(&pattern_str) {
            Ok(g) => g.compile_matcher(),
            Err(e) => {
                tracing::warn!(
                    "Invalid glob pattern '{}': {e}. Treating as literal path.",
                    pattern_str
                );
                result.push(pattern.clone());
                continue;
            }
        };

        expand_recursive(&root, &matcher, &mut result);
    }

    if result.is_empty() && !patterns.is_empty() {
        tracing::warn!(
            "Glob expansion of base_paths produced no results. Patterns: {:?}",
            patterns
        );
    }

    result.sort();
    result.dedup();
    result
}

/// Recursively walk a directory, collecting paths that match the glob matcher.
fn expand_recursive(dir: &Path, matcher: &globset::GlobMatcher, results: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if matcher.is_match(&path) {
                results.push(path.clone());
            }
            expand_recursive(&path, matcher, results);
        }
    }
}

/// Parse `base_paths` from tool params and expand any glob patterns.
fn parse_and_expand_base_paths(params: &HashMap<String, SecretString>) -> Vec<PathBuf> {
    params
        .get("base_paths")
        .map(|v| {
            let raw: Vec<PathBuf> = v
                .expose_secret()
                .split(',')
                .map(|s| PathBuf::from(s.trim()))
                .collect();
            expand_base_paths(raw)
        })
        .unwrap_or_else(|| vec![PathBuf::from(".")])
}

impl BuiltinToolCatalog {
    pub(crate) fn new(rt: Arc<Runtime>) -> Self {
        Self {
            rt,
            model_table_allowlist: None,
            worktree_tracker: WorktreeTracker::default(),
            approval_store: ApprovalStore::default(),
        }
    }

    /// Create a new `BuiltinToolCatalog` with a table allowlist applied to all tools.
    #[must_use]
    pub fn with_table_allowlist(mut self, allowlist: ResolvedTableAwareAllowlist) -> Self {
        self.model_table_allowlist = Some(allowlist);
        self
    }

    /// Set the worktree tracker for session-scoped cleanup.
    #[must_use]
    pub fn with_worktree_tracker(mut self, tracker: WorktreeTracker) -> Self {
        self.worktree_tracker = tracker;
        self
    }

    /// Get a reference to the worktree tracker.
    #[must_use]
    pub fn worktree_tracker(&self) -> &WorktreeTracker {
        &self.worktree_tracker
    }

    /// Set the approval store for human-in-the-loop approval tools.
    #[must_use]
    pub fn with_approval_store(mut self, store: ApprovalStore) -> Self {
        self.approval_store = store;
        self
    }

    /// Get the approval store.
    #[must_use]
    pub fn approval_store(&self) -> &ApprovalStore {
        &self.approval_store
    }

    pub(crate) fn name() -> &'static str {
        "auto"
    }

    pub(crate) fn is_builtin_tool(name: &str) -> bool {
        [
            "websearch",
            "get_readiness",
            "search",
            "table_schema",
            "sql",
            "sample_distinct_columns",
            "random_sample",
            "top_n_sample",
            "list_datasets",
            "list_file_sources",
            "grep",
            "read_file",
            "list_files",
            "kubectl",
            "slack",
            "ms_teams",
            "git_worktree",
            "git",
            "claude_code",
            "debug",
            "fail",
            "github",
            "http_tool",
            "approval",
            "approval_slack",
            "approval_ms_teams",
        ]
        .contains(&name)
    }

    pub(crate) fn construct_builtin(
        &self,
        id: &str,
        name: Option<&str>,
        description: Option<&str>,
        params: &HashMap<String, SecretString>,
    ) -> Result<Arc<dyn SpiceModelTool>> {
        let name = name.unwrap_or(id);

        // Get default description if none is provided
        let description = match (id, description) {
            (_, Some(desc)) => desc, // Use provided description if available
            ("websearch", None) => "Search the web for information",
            ("get_readiness", None) => "Get the readiness status of the Spice.ai runtime",
            ("search", None) => "Search across available, searchable datasets in Spice.ai runtime",
            ("table_schema", None) => "Get the schema of the Spice.ai dataset",
            ("sql", None) => "Execute SQL queries (PostgreSQL dialect) using the Spice.ai runtime",
            ("sample_distinct_columns", None) => {
                "Sample distinct column values from a Spice.ai dataset"
            }
            ("random_sample", None) => "Get a random sample of rows from a Spice.ai dataset",
            ("top_n_sample", None) => {
                "Get top N samples from a Spice.ai dataset based on a specified ordering"
            }
            ("list_datasets", None) => "List available datasets",
            ("list_file_sources", None) => "List available file sources and their local paths. This includes github repositories",
            ("grep", None) => "Search for a pattern in files within configured directories",
            ("read_file", None) => "Read the contents of a file at the given path",
            ("list_files", None) => "List files and directories at the given path",
            ("kubectl", None) => "Execute kubectl operations against a Kubernetes cluster",
            ("slack", None) => "Read or post messages in Slack channels",
            ("ms_teams", None) => "Post messages and reports to Microsoft Teams channels",
            ("git_worktree", None) => "Create and manage git worktrees for parallel branch work",
            ("git", None) => "Perform git operations on a repository",
            ("claude_code", None) => {
                "Invoke Claude Code CLI for complex coding tasks such as merge conflict resolution"
            }
            ("debug", None) => "Print a debug message to the task history log",
            ("fail", None) => "Signal that this step cannot be completed due to invalid or missing information",
            ("github", None) => "Interact with GitHub repositories: milestones, pull requests, commits, and issues. IMPORTANT: Always use the 'fields' parameter to request only the specific fields you need (e.g. [\"number\", \"title\", \"state\"]). Request as few fields as possible to satisfy your task.",
            ("http_tool", None) => {
                "Make HTTP requests to external services and APIs"
            }
            ("approval", None) => {
                "Request human approval before proceeding with an action"
            }
            ("approval_slack", None) => {
                "Request human approval via Slack with approve/reject links"
            }
            ("approval_ms_teams", None) => {
                "Request human approval via Microsoft Teams with approve/reject buttons"
            }
            (_, None) => "",
        };

        // Use model-level table allowlist if set, otherwise parse from params
        let table_allowlist: Option<ResolvedTableAwareAllowlist> =
            if let Some(allowlist) = params.get("table_allowlist") {
                let tables = allowlist
                    .expose_secret()
                    .split(',')
                    .map(ToString::to_string)
                    .collect::<Vec<String>>();
                Some(
                    ResolvedTableAwareAllowlist::with_defaults(
                        SPICE_DEFAULT_CATALOG,
                        SPICE_DEFAULT_SCHEMA,
                    )
                    .with_table_patterns(tables)
                    .boxed()
                    .context(FailedToConstructToolSnafu { id })?,
                )
            } else {
                self.model_table_allowlist.clone()
            };

        match id {
            "websearch" => Ok(Arc::new(
                WebSearchTool::try_new(name, Some(description), params)
                    .context(FailedToConstructToolSnafu { id: id.to_string() })?,
            )),
            "get_readiness" => Ok(Arc::new(GetReadinessTool::new(
                Arc::clone(&self.rt),
                Some(name),
                Some(description),
            ))),
            "search" => Ok(Arc::new(
                SearchTool::new(Arc::clone(&self.rt), Some(name), Some(description))
                    .with_table_allowlist(table_allowlist),
            )),
            "table_schema" => Ok(Arc::new(
                TableSchemaTool::new(Arc::clone(&self.rt), Some(name), Some(description))
                    .with_table_allowlist(table_allowlist),
            )),
            "sql" => Ok(Arc::new(SqlTool::new(
                self.rt.datafusion(),
                Some(name),
                Some(description),
                table_allowlist,
            ))),
            "sample_distinct_columns" => Ok(Arc::new(
                SampleDataTool::new(self.rt.datafusion(), SampleTableMethod::DistinctColumns)
                    .with_overrides(Some(name), Some(description))
                    .with_table_allowlist(table_allowlist),
            )),
            "random_sample" => Ok(Arc::new(
                SampleDataTool::new(self.rt.datafusion(), SampleTableMethod::RandomSample)
                    .with_overrides(Some(name), Some(description))
                    .with_table_allowlist(table_allowlist),
            )),
            "top_n_sample" => Ok(Arc::new(
                SampleDataTool::new(self.rt.datafusion(), SampleTableMethod::TopNSample)
                    .with_overrides(Some(name), Some(description))
                    .with_table_allowlist(table_allowlist),
            )),
            "list_datasets" => Ok(Arc::new(ListDatasetsTool::new(
                Some(name),
                Some(description),
                table_allowlist,
                Arc::clone(&self.rt),
            ))),
            "list_file_sources" => Ok(Arc::new(ListFileSourcesTool::new(
                Some(name),
                Some(description),
                Arc::clone(&self.rt),
                self.worktree_tracker.clone(),
            ))),
            "grep" => {
                let base_paths = parse_and_expand_base_paths(params);
                Ok(Arc::new(GrepTool::new(
                    Some(name),
                    Some(description),
                    base_paths,
                    Some(self.worktree_tracker.clone()),
                )))
            }
            "read_file" => {
                let base_paths = parse_and_expand_base_paths(params);
                Ok(Arc::new(ReadFileTool::new(
                    Some(name),
                    Some(description),
                    base_paths,
                    Some(self.worktree_tracker.clone()),
                )))
            }
            "list_files" => {
                let base_paths = parse_and_expand_base_paths(params);
                Ok(Arc::new(ListFilesTool::new(
                    Some(name),
                    Some(description),
                    base_paths,
                    Some(self.worktree_tracker.clone()),
                )))
            }
            "kubectl" => {
                let namespace = params
                    .get("namespace")
                    .map(|v| v.expose_secret().to_string())
                    .unwrap_or_else(|| "default".to_string());
                let allowed_operations = params
                    .get("allowed_operations")
                    .map(|v| {
                        v.expose_secret()
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect()
                    })
                    .unwrap_or_else(|| {
                        vec![
                            "get".to_string(),
                            "list".to_string(),
                            "describe".to_string(),
                            "logs".to_string(),
                        ]
                    });
                let capability = params
                    .get("capability")
                    .and_then(|v| match v.expose_secret() {
                        "read_write" => Some(ToolCapability::ReadWrite),
                        _ => Some(ToolCapability::ReadOnly),
                    })
                    .unwrap_or(ToolCapability::ReadOnly);
                Ok(Arc::new(KubectlTool::new(
                    Some(name),
                    Some(description),
                    namespace,
                    allowed_operations,
                    capability,
                )))
            }
            "slack" => {
                let channels = params
                    .get("channels")
                    .map(|v| {
                        v.expose_secret()
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect()
                    })
                    .unwrap_or_default();
                let capability = params
                    .get("capability")
                    .and_then(|v| match v.expose_secret() {
                        "read_write" => Some(ToolCapability::ReadWrite),
                        _ => Some(ToolCapability::ReadOnly),
                    })
                    .unwrap_or(ToolCapability::ReadOnly);
                Ok(Arc::new(SlackTool::new(
                    Some(name),
                    Some(description),
                    channels,
                    capability,
                )))
            }
            "ms_teams" => {
                let webhook_url = params
                    .get("webhook_url")
                    .map(|v| v.expose_secret().to_string())
                    .ok_or_else(|| Error::FailedToConstructTool {
                        id: id.to_string(),
                        source: "Missing required 'webhook_url' parameter".into(),
                    })?;
                Ok(Arc::new(
                    TeamsTool::try_new(Some(name), Some(description), webhook_url)
                        .context(FailedToConstructToolSnafu { id: id.to_string() })?,
                ))
            }
            "git_worktree" => {
                let repo_path = params
                    .get("repo_path")
                    .map(|v| PathBuf::from(v.expose_secret().to_string()))
                    .ok_or_else(|| Error::FailedToConstructTool {
                        id: id.to_string(),
                        source: "Missing required 'repo_path' parameter".into(),
                    })?;
                let worktree_root = params
                    .get("worktree_root")
                    .map(|v| PathBuf::from(v.expose_secret().to_string()))
                    .unwrap_or_else(|| {
                        std::env::temp_dir().join("spice-agent-worktrees")
                    });
                Ok(Arc::new(
                    GitWorktreeTool::try_new(
                        Some(name),
                        Some(description),
                        repo_path,
                        worktree_root,
                        self.worktree_tracker.clone(),
                    )
                    .context(FailedToConstructToolSnafu { id: id.to_string() })?,
                ))
            }
            "git" => {
                let repo_path = params
                    .get("repo_path")
                    .map(|v| PathBuf::from(v.expose_secret().to_string()))
                    .ok_or_else(|| Error::FailedToConstructTool {
                        id: id.to_string(),
                        source: "Missing required 'repo_path' parameter".into(),
                    })?;
                let capability = params
                    .get("capability")
                    .and_then(|v| match v.expose_secret() {
                        "read_write" => Some(ToolCapability::ReadWrite),
                        _ => Some(ToolCapability::ReadOnly),
                    })
                    .unwrap_or(ToolCapability::ReadWrite);
                let allowed_operations = params
                    .get("allowed_operations")
                    .map(|v| {
                        v.expose_secret()
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect()
                    })
                    .unwrap_or_else(|| {
                        vec![
                            "status".to_string(),
                            "log".to_string(),
                            "diff".to_string(),
                            "add".to_string(),
                            "branch".to_string(),
                            "checkout".to_string(),
                            "cherry-pick".to_string(),
                            "commit".to_string(),
                            "push".to_string(),
                        ]
                    });
                let github_token = params
                    .get("github_token")
                    .map(|v| v.expose_secret().to_string())
                    .or_else(|| std::env::var("GITHUB_TOKEN").ok());
                Ok(Arc::new(
                    GitTool::try_new(
                        Some(name),
                        Some(description),
                        repo_path,
                        capability,
                        allowed_operations,
                        Some(self.worktree_tracker.clone()),
                        github_token,
                    )
                    .context(FailedToConstructToolSnafu { id: id.to_string() })?,
                ))
            }
            "claude_code" => {
                let claude_binary = params
                    .get("claude_binary")
                    .map(|v| v.expose_secret().to_string());
                let default_model = params
                    .get("default_model")
                    .map(|v| v.expose_secret().to_string());
                let default_max_turns = params
                    .get("default_max_turns")
                    .and_then(|v| v.expose_secret().parse::<u32>().ok());
                let default_allowed_tools = params
                    .get("default_allowed_tools")
                    .map(|v| {
                        v.expose_secret()
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect()
                    })
                    .unwrap_or_default();
                let allowed_working_dirs: Vec<PathBuf> = params
                    .get("allowed_working_dirs")
                    .map(|v| {
                        v.expose_secret()
                            .split(',')
                            .map(|s| PathBuf::from(s.trim()))
                            .collect()
                    })
                    .unwrap_or_else(|| vec![PathBuf::from(".")]);
                Ok(Arc::new(
                    ClaudeCodeTool::try_new(
                        Some(name),
                        Some(description),
                        claude_binary.as_deref(),
                        default_model,
                        default_max_turns,
                        default_allowed_tools,
                        allowed_working_dirs,
                        Some(self.worktree_tracker.clone()),
                    )
                    .context(FailedToConstructToolSnafu { id: id.to_string() })?,
                ))
            }
            "approval" => {
                let timeout = params
                    .get("timeout")
                    .and_then(|v| super::approval::parse_timeout(v.expose_secret()))
                    .unwrap_or(Duration::from_secs(3600));
                let timeout_action = match params
                    .get("timeout_action")
                    .map(|v| v.expose_secret())
                {
                    Some("approve") => ApprovalTimeoutAction::Approve,
                    _ => ApprovalTimeoutAction::Reject,
                };
                let base_url = params
                    .get("base_url")
                    .map(|v| v.expose_secret().to_string());
                Ok(Arc::new(ApprovalTool::new(
                    Some(name),
                    Some(description),
                    self.approval_store.clone(),
                    timeout,
                    timeout_action,
                    base_url,
                )))
            }
            "approval_slack" => {
                let timeout = params
                    .get("timeout")
                    .and_then(|v| super::approval::parse_timeout(v.expose_secret()))
                    .unwrap_or(Duration::from_secs(3600));
                let timeout_action = match params
                    .get("timeout_action")
                    .map(|v| v.expose_secret())
                {
                    Some("approve") => ApprovalTimeoutAction::Approve,
                    _ => ApprovalTimeoutAction::Reject,
                };
                let base_url = params
                    .get("base_url")
                    .map(|v| v.expose_secret().to_string());
                let slack_token = params
                    .get("slack_token")
                    .map(|v| v.expose_secret().to_string())
                    .ok_or_else(|| Error::FailedToConstructTool {
                        id: id.to_string(),
                        source: "Missing required 'slack_token' parameter".into(),
                    })?;
                Ok(Arc::new(ApprovalSlackTool::new(
                    Some(name),
                    Some(description),
                    self.approval_store.clone(),
                    timeout,
                    timeout_action,
                    base_url,
                    slack_token,
                )))
            }
            "approval_ms_teams" => {
                let timeout = params
                    .get("timeout")
                    .and_then(|v| super::approval::parse_timeout(v.expose_secret()))
                    .unwrap_or(Duration::from_secs(3600));
                let timeout_action = match params
                    .get("timeout_action")
                    .map(|v| v.expose_secret())
                {
                    Some("approve") => ApprovalTimeoutAction::Approve,
                    _ => ApprovalTimeoutAction::Reject,
                };
                let base_url = params
                    .get("base_url")
                    .map(|v| v.expose_secret().to_string());
                let webhook_url = params
                    .get("webhook_url")
                    .map(|v| v.expose_secret().to_string())
                    .ok_or_else(|| Error::FailedToConstructTool {
                        id: id.to_string(),
                        source: "Missing required 'webhook_url' parameter".into(),
                    })?;
                Ok(Arc::new(
                    ApprovalMsTeamsTool::try_new(
                        Some(name),
                        Some(description),
                        self.approval_store.clone(),
                        timeout,
                        timeout_action,
                        base_url,
                        webhook_url,
                    )
                    .context(FailedToConstructToolSnafu { id: id.to_string() })?,
                ))
            }
            "debug" => Ok(Arc::new(DebugTool::new(Some(name), Some(description)))),
            "fail" => Ok(Arc::new(FailTool::new(Some(name), Some(description)))),
            "github" => {
                let remote = params
                    .get("remote")
                    .map(|v| v.expose_secret().to_string())
                    .ok_or_else(|| Error::FailedToConstructTool {
                        id: id.to_string(),
                        source: "Missing required 'remote' parameter".into(),
                    })?;
                let token = params
                    .get("github_token")
                    .map(|v| v.expose_secret().to_string())
                    .or_else(|| std::env::var("GITHUB_TOKEN").ok())
                    .ok_or_else(|| Error::FailedToConstructTool {
                        id: id.to_string(),
                        source: "Missing 'github_token' parameter and GITHUB_TOKEN env var not set"
                            .into(),
                    })?;
                Ok(Arc::new(
                    GitHubTool::try_new(Some(name), Some(description), &remote, token)
                        .context(FailedToConstructToolSnafu { id: id.to_string() })?,
                ))
            }
            "http_tool" => Ok(Arc::new(
                HttpTool::try_new(Some(name), Some(description), params)
                    .context(FailedToConstructToolSnafu { id: id.to_string() })?,
            )),
            _ => Err(Error::UnknownBuiltinTool { id: id.to_string() }),
        }
    }
}

impl IndividualToolFactory for BuiltinToolCatalog {
    fn construct(
        &self,
        component: &Tool,
        params_with_secrets: HashMap<String, SecretString>,
    ) -> Result<Arc<dyn SpiceModelTool>, Box<dyn std::error::Error + Send + Sync>> {
        let id = component
            .from
            .split_once(':')
            .map_or(component.from.as_str(), |(_, id)| id);

        self.construct_builtin(
            id,
            Some(component.name.as_str()),
            component.description.as_deref(),
            &params_with_secrets,
        )
        .boxed()
    }
}

#[async_trait]
impl SpiceToolCatalog for BuiltinToolCatalog {
    async fn all(&self) -> Vec<Arc<dyn SpiceModelTool>> {
        let mut tools = vec![];
        for t in SpiceToolsOptions::Auto.tools_by_name() {
            match self.construct_builtin(t, None, None, &HashMap::new()) {
                Ok(tool) => tools.push(tool),
                Err(e) => tracing::warn!("Failed to construct builtin tool: '{}'. Error: {}", t, e),
            }
        }
        tools
    }

    async fn get(&self, name: &str) -> Option<Arc<dyn SpiceModelTool>> {
        self.construct_builtin(name, None, None, &HashMap::new())
            .ok()
    }

    fn name(&self) -> &str {
        Self::name()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
