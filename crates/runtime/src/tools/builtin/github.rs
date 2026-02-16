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
use tools::SpiceModelTool;
use tracing::Span;
use tracing_futures::Instrument;

use crate::tools::utils::parameters;

const GITHUB_API_BASE: &str = "https://api.github.com";

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct GitHubToolParams {
    /// The GitHub operation: "list_milestones", "get_milestone", "list_milestone_issues",
    /// "create_pull_request", "get_pull_request", "add_issue_to_milestone", "list_commits".
    operation: String,

    /// Milestone title or number (for milestone operations).
    milestone: Option<String>,

    /// PR title (for create_pull_request).
    title: Option<String>,

    /// PR body/description (for create_pull_request).
    body: Option<String>,

    /// Head branch (for create_pull_request).
    head: Option<String>,

    /// Base branch (for create_pull_request, list_commits).
    base: Option<String>,

    /// SHA or branch ref (for list_commits).
    sha: Option<String>,

    /// Issue or PR number.
    number: Option<u64>,

    /// Results per page (default 100).
    per_page: Option<u32>,

    /// Page number for pagination.
    page: Option<u32>,
}

const KNOWN_OPERATIONS: &[&str] = &[
    "list_milestones",
    "get_milestone",
    "list_milestone_issues",
    "create_pull_request",
    "get_pull_request",
    "add_issue_to_milestone",
    "list_commits",
];

#[derive(Debug)]
pub struct GitHubTool {
    name: String,
    description: String,
    owner: String,
    repo: String,
    token: String,
    client: reqwest::Client,
}

impl GitHubTool {
    /// Create a new `GitHubTool`.
    ///
    /// # Errors
    ///
    /// Returns an error if the remote URL cannot be parsed into an owner/repo pair.
    pub fn try_new(
        name: Option<&str>,
        description: Option<&str>,
        remote: &str,
        token: String,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let (owner, repo) = parse_github_remote(remote)?;

        Ok(Self {
            name: name.unwrap_or("github").to_string(),
            description: description
                .unwrap_or(
                    "Interact with GitHub repositories. Supports milestones, pull requests, commits, and issues.",
                )
                .to_string(),
            owner,
            repo,
            token,
            client: reqwest::Client::new(),
        })
    }

    // -----------------------------------------------------------------------
    // API helpers
    // -----------------------------------------------------------------------

    async fn api_get(
        &self,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{GITHUB_API_BASE}{path}");
        let response = self
            .client
            .get(&url)
            .query(query)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "spice-ai-agent")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send()
            .await?;

        let status = response.status();
        let body: Value = response.json().await?;

        if !status.is_success() {
            return Err(format!(
                "GitHub API GET {path} returned HTTP {status}: {}",
                serde_json::to_string(&body).unwrap_or_default()
            )
            .into());
        }

        Ok(body)
    }

    async fn api_post(
        &self,
        path: &str,
        body: &Value,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{GITHUB_API_BASE}{path}");
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "spice-ai-agent")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .json(body)
            .send()
            .await?;

        let status = response.status();
        let resp_body: Value = response.json().await?;

        if !status.is_success() {
            return Err(format!(
                "GitHub API POST {path} returned HTTP {status}: {}",
                serde_json::to_string(&resp_body).unwrap_or_default()
            )
            .into());
        }

        Ok(resp_body)
    }

    async fn api_patch(
        &self,
        path: &str,
        body: &Value,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{GITHUB_API_BASE}{path}");
        let response = self
            .client
            .patch(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "spice-ai-agent")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .json(body)
            .send()
            .await?;

        let status = response.status();
        let resp_body: Value = response.json().await?;

        if !status.is_success() {
            return Err(format!(
                "GitHub API PATCH {path} returned HTTP {status}: {}",
                serde_json::to_string(&resp_body).unwrap_or_default()
            )
            .into());
        }

        Ok(resp_body)
    }

    // -----------------------------------------------------------------------
    // Operation handlers
    // -----------------------------------------------------------------------

    async fn handle_list_milestones(
        &self,
        params: &GitHubToolParams,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let per_page = params.per_page.unwrap_or(100).to_string();
        let page = params.page.unwrap_or(1).to_string();
        let path = format!("/repos/{}/{}/milestones", self.owner, self.repo);

        self.api_get(
            &path,
            &[
                ("state", "open"),
                ("per_page", &per_page),
                ("page", &page),
            ],
        )
        .await
    }

    async fn handle_get_milestone(
        &self,
        params: &GitHubToolParams,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        // Determine the milestone number. If `number` is provided directly, use it.
        // If `milestone` is a numeric string, parse it.
        // Otherwise, treat it as a title and search for it.
        let milestone_number: u64 = if let Some(n) = params.number {
            n
        } else if let Some(ref ms) = params.milestone {
            if let Ok(n) = ms.parse::<u64>() {
                n
            } else {
                // Search by title: list all milestones and find by title
                self.find_milestone_number_by_title(ms).await?
            }
        } else {
            return Err(
                "get_milestone requires a 'milestone' (title or number) or 'number' parameter"
                    .to_string()
                    .into(),
            );
        };

        let path = format!(
            "/repos/{}/{}/milestones/{milestone_number}",
            self.owner, self.repo
        );
        self.api_get(&path, &[]).await
    }

    async fn handle_list_milestone_issues(
        &self,
        params: &GitHubToolParams,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        // Resolve milestone number from either `number`, numeric `milestone`, or title.
        let milestone_number: u64 = if let Some(n) = params.number {
            n
        } else if let Some(ref ms) = params.milestone {
            if let Ok(n) = ms.parse::<u64>() {
                n
            } else {
                self.find_milestone_number_by_title(ms).await?
            }
        } else {
            return Err(
                "list_milestone_issues requires a 'milestone' (title or number) or 'number' parameter"
                    .to_string()
                    .into(),
            );
        };

        let per_page = params.per_page.unwrap_or(100).to_string();
        let page = params.page.unwrap_or(1).to_string();
        let milestone_str = milestone_number.to_string();
        let path = format!("/repos/{}/{}/issues", self.owner, self.repo);

        let raw = self
            .api_get(
                &path,
                &[
                    ("milestone", &milestone_str),
                    ("state", "all"),
                    ("per_page", &per_page),
                    ("page", &page),
                ],
            )
            .await?;

        // Return only number, title, and merge commit SHA for each issue/PR.
        let items = raw
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|issue| {
                        let commit = issue
                            .get("pull_request")
                            .and_then(|pr| pr.get("merge_commit_sha"))
                            .and_then(|v| v.as_str());
                        json!({
                            "number": issue.get("number"),
                            "title": issue.get("title"),
                            "commit": commit,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(json!(items))
    }

    async fn handle_list_commits(
        &self,
        params: &GitHubToolParams,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let per_page = params.per_page.unwrap_or(100).to_string();
        let page = params.page.unwrap_or(1).to_string();
        let path = format!("/repos/{}/{}/commits", self.owner, self.repo);

        let mut query: Vec<(&str, &str)> = vec![("per_page", &per_page), ("page", &page)];

        if let Some(ref sha) = params.sha {
            query.push(("sha", sha.as_str()));
        }

        self.api_get(&path, &query).await
    }

    async fn handle_create_pull_request(
        &self,
        params: &GitHubToolParams,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let title = params
            .title
            .as_deref()
            .ok_or("create_pull_request requires a 'title' parameter")?;
        let head = params
            .head
            .as_deref()
            .ok_or("create_pull_request requires a 'head' parameter")?;
        let base = params
            .base
            .as_deref()
            .ok_or("create_pull_request requires a 'base' parameter")?;

        let mut body_json = json!({
            "title": title,
            "head": head,
            "base": base,
        });

        if let Some(ref body_text) = params.body {
            body_json["body"] = json!(body_text);
        }

        let path = format!("/repos/{}/{}/pulls", self.owner, self.repo);
        self.api_post(&path, &body_json).await
    }

    async fn handle_get_pull_request(
        &self,
        params: &GitHubToolParams,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let number = params
            .number
            .ok_or("get_pull_request requires a 'number' parameter")?;

        let path = format!("/repos/{}/{}/pulls/{number}", self.owner, self.repo);
        self.api_get(&path, &[]).await
    }

    async fn handle_add_issue_to_milestone(
        &self,
        params: &GitHubToolParams,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let issue_number = params
            .number
            .ok_or("add_issue_to_milestone requires a 'number' parameter")?;

        // Resolve milestone number from the `milestone` param.
        let milestone_number: u64 = if let Some(ref ms) = params.milestone {
            if let Ok(n) = ms.parse::<u64>() {
                n
            } else {
                self.find_milestone_number_by_title(ms).await?
            }
        } else {
            return Err(
                "add_issue_to_milestone requires a 'milestone' (title or number) parameter"
                    .to_string()
                    .into(),
            );
        };

        let path = format!(
            "/repos/{}/{}/issues/{issue_number}",
            self.owner, self.repo
        );
        self.api_patch(&path, &json!({ "milestone": milestone_number }))
            .await
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Search open milestones by title, returning the milestone number.
    async fn find_milestone_number_by_title(
        &self,
        title: &str,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let path = format!("/repos/{}/{}/milestones", self.owner, self.repo);
        let milestones = self
            .api_get(&path, &[("state", "closed"), ("per_page", "100")])
            .await?;

        let milestones = milestones
            .as_array()
            .ok_or("Expected milestones to be an array")?;

        for ms in milestones {
            if ms["title"].as_str() == Some(title) {
                if let Some(n) = ms["number"].as_u64() {
                    return Ok(n);
                }
            }
        }

        Err(format!("Milestone with title '{title}' not found").into())
    }
}

/// Parse a GitHub remote URL into `(owner, repo)`.
///
/// Supported formats:
/// - `https://github.com/owner/repo`
/// - `https://github.com/owner/repo.git`
/// - `git@github.com:owner/repo.git`
fn parse_github_remote(
    remote: &str,
) -> Result<(String, String), Box<dyn std::error::Error + Send + Sync>> {
    let path = if let Some(stripped) = remote.strip_prefix("git@github.com:") {
        stripped.to_string()
    } else if let Ok(url) = url::Url::parse(remote) {
        let host = url.host_str().unwrap_or("");
        if host != "github.com" {
            return Err(format!(
                "Expected a github.com URL, but got host '{host}' in '{remote}'"
            )
            .into());
        }
        url.path().trim_start_matches('/').to_string()
    } else {
        return Err(format!("Cannot parse remote URL: '{remote}'").into());
    };

    // Strip trailing `.git` if present
    let path = path.strip_suffix(".git").unwrap_or(&path);

    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() < 2 || parts[0].is_empty() || parts[1].is_empty() {
        return Err(format!(
            "Cannot extract owner/repo from remote URL: '{remote}'"
        )
        .into());
    }

    Ok((parts[0].to_string(), parts[1].to_string()))
}

#[async_trait]
impl SpiceModelTool for GitHubTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<GitHubToolParams>()
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span: Span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::github", tool = self.name().to_string(), input = arg);

        let tool_use_result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let params: GitHubToolParams = serde_json::from_str(arg)?;

            if !KNOWN_OPERATIONS.contains(&params.operation.as_str()) {
                return Err(format!(
                    "Unknown operation '{}'. Supported operations: {}",
                    params.operation,
                    KNOWN_OPERATIONS.join(", ")
                )
                .into());
            }

            match params.operation.as_str() {
                "list_milestones" => self.handle_list_milestones(&params).await,
                "get_milestone" => self.handle_get_milestone(&params).await,
                "list_milestone_issues" => self.handle_list_milestone_issues(&params).await,
                "list_commits" => self.handle_list_commits(&params).await,
                "create_pull_request" => self.handle_create_pull_request(&params).await,
                "get_pull_request" => self.handle_get_pull_request(&params).await,
                "add_issue_to_milestone" => self.handle_add_issue_to_milestone(&params).await,
                _ => unreachable!(),
            }
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

    #[test]
    fn test_parse_remote_url() {
        // Standard HTTPS URL
        let (owner, repo) =
            parse_github_remote("https://github.com/spiceai/spiceai").unwrap();
        assert_eq!(owner, "spiceai");
        assert_eq!(repo, "spiceai");

        // HTTPS URL with .git suffix
        let (owner, repo) =
            parse_github_remote("https://github.com/spiceai/spiceai.git").unwrap();
        assert_eq!(owner, "spiceai");
        assert_eq!(repo, "spiceai");

        // SSH URL
        let (owner, repo) =
            parse_github_remote("git@github.com:spiceai/spiceai.git").unwrap();
        assert_eq!(owner, "spiceai");
        assert_eq!(repo, "spiceai");

        // SSH URL without .git suffix
        let (owner, repo) =
            parse_github_remote("git@github.com:spiceai/spiceai").unwrap();
        assert_eq!(owner, "spiceai");
        assert_eq!(repo, "spiceai");

        // Different owner/repo
        let (owner, repo) =
            parse_github_remote("https://github.com/rust-lang/rust.git").unwrap();
        assert_eq!(owner, "rust-lang");
        assert_eq!(repo, "rust");
    }

    #[test]
    fn test_reject_invalid_url() {
        // Non-GitHub HTTPS URL
        let result = parse_github_remote("https://gitlab.com/owner/repo");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("github.com"),
            "Expected github.com validation error, got: {err}"
        );

        // Completely invalid URL
        let result = parse_github_remote("not-a-url");
        assert!(result.is_err());

        // Missing repo component
        let result = parse_github_remote("https://github.com/owner-only");
        assert!(result.is_err());
    }

    #[test]
    fn test_default_name_and_description() {
        let tool = GitHubTool::try_new(
            None,
            None,
            "https://github.com/spiceai/spiceai",
            "test-token".to_string(),
        )
        .unwrap();

        assert_eq!(tool.name(), "github");
        let desc = tool.description().unwrap();
        assert!(
            desc.contains("GitHub"),
            "Expected description to mention GitHub, got: {desc}"
        );
        assert!(
            desc.contains("milestones"),
            "Expected description to mention milestones, got: {desc}"
        );
        assert!(
            desc.contains("pull requests"),
            "Expected description to mention pull requests, got: {desc}"
        );
    }

    #[test]
    fn test_custom_name_and_description() {
        let tool = GitHubTool::try_new(
            Some("my_github"),
            Some("Custom description"),
            "https://github.com/spiceai/spiceai",
            "test-token".to_string(),
        )
        .unwrap();

        assert_eq!(tool.name(), "my_github");
        assert_eq!(tool.description().unwrap(), "Custom description");
    }

    #[tokio::test]
    async fn test_invalid_operation() {
        let tool = GitHubTool::try_new(
            None,
            None,
            "https://github.com/spiceai/spiceai",
            "test-token".to_string(),
        )
        .unwrap();

        let result = tool
            .call(r#"{"operation": "delete_repo"}"#)
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Unknown operation 'delete_repo'"),
            "Expected unknown operation error, got: {err}"
        );
    }

    #[test]
    fn test_has_parameters() {
        let tool = GitHubTool::try_new(
            None,
            None,
            "https://github.com/spiceai/spiceai",
            "test-token".to_string(),
        )
        .unwrap();

        assert!(tool.parameters().is_some());
    }

    #[test]
    fn test_owner_repo_extracted() {
        let tool = GitHubTool::try_new(
            None,
            None,
            "https://github.com/rust-lang/cargo.git",
            "tok".to_string(),
        )
        .unwrap();

        assert_eq!(tool.owner, "rust-lang");
        assert_eq!(tool.repo, "cargo");
    }
}
