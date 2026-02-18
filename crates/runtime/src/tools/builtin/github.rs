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
use std::collections::HashMap;
use tools::SpiceModelTool;
use tracing::Span;
use tracing_futures::Instrument;

use crate::tools::utils::parameters;

const GITHUB_API_BASE: &str = "https://api.github.com";
const GITHUB_GRAPHQL_URL: &str = "https://api.github.com/graphql";

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct GitHubToolParams {
    /// The GitHub operation: "list_milestones", "get_milestone", "list_milestone_issues",
    /// "list_milestone_pull_requests", "create_pull_request", "get_pull_request",
    /// "add_issue_to_milestone", "list_commits", "compare", "api_get", "api_post",
    /// "api_patch", "graphql".
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

    /// API path for api_get/api_post/api_patch. Paths not starting with "/"
    /// are auto-prefixed with "/repos/{owner}/{repo}/".
    path: Option<String>,

    /// JSON body for api_post/api_patch, or {"query": "...", "variables": {...}} for graphql.
    json_body: Option<Value>,

    /// Query parameters for api_get (e.g. {"state": "open"}).
    query_params: Option<HashMap<String, String>>,

    /// Optional list of top-level field names to include in the response.
    /// When provided, each object in the response is filtered to only these keys.
    /// Useful for reducing context size (e.g. ["number", "title", "state"]).
    fields: Option<Vec<String>>,
}

const KNOWN_OPERATIONS: &[&str] = &[
    "list_milestones",
    "get_milestone",
    "list_milestone_issues",
    "list_milestone_pull_requests",
    "create_pull_request",
    "get_pull_request",
    "add_issue_to_milestone",
    "list_commits",
    "compare",
    "api_get",
    "api_post",
    "api_patch",
    "graphql",
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
                    "Interact with GitHub repositories. Supports milestones, pull requests, commits, and issues. IMPORTANT: Always use the 'fields' parameter to request only the specific fields you need (e.g. [\"number\", \"title\", \"state\"]). Request as few fields as possible to satisfy your task.",
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
        tracing::debug!(method = "GET", path = %path, "GitHub API request");
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
            let err_body = serde_json::to_string(&body).unwrap_or_default();
            tracing::warn!(method = "GET", path = %path, %status, body = %err_body, "GitHub API error response");
            return Err(format!(
                "GitHub API GET {path} returned HTTP {status}: {err_body}",
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
        tracing::debug!(method = "POST", path = %path, "GitHub API request");
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
            let err_body = serde_json::to_string(&resp_body).unwrap_or_default();
            tracing::warn!(method = "POST", path = %path, %status, body = %err_body, "GitHub API error response");
            return Err(format!(
                "GitHub API POST {path} returned HTTP {status}: {err_body}",
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
        tracing::debug!(method = "PATCH", path = %path, "GitHub API request");
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
            let err_body = serde_json::to_string(&resp_body).unwrap_or_default();
            tracing::warn!(method = "PATCH", path = %path, %status, body = %err_body, "GitHub API error response");
            return Err(format!(
                "GitHub API PATCH {path} returned HTTP {status}: {err_body}",
            )
            .into());
        }

        Ok(resp_body)
    }

    async fn api_graphql(
        &self,
        query: &str,
        variables: Value,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        tracing::debug!(method = "POST", path = "/graphql", "GitHub GraphQL request");
        let body = json!({ "query": query, "variables": variables });
        let response = self
            .client
            .post(GITHUB_GRAPHQL_URL)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/json")
            .header("User-Agent", "spice-ai-agent")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        let resp_body: Value = response.json().await?;

        if !status.is_success() {
            return Err(format!(
                "GitHub GraphQL returned HTTP {status}: {}",
                serde_json::to_string(&resp_body).unwrap_or_default()
            )
            .into());
        }

        if let Some(errors) = resp_body.get("errors") {
            return Err(format!(
                "GitHub GraphQL errors: {}",
                serde_json::to_string(errors).unwrap_or_default()
            )
            .into());
        }

        resp_body
            .get("data")
            .cloned()
            .ok_or_else(|| "GitHub GraphQL response missing 'data' field".to_string().into())
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
        let milestone_number = self
            .resolve_milestone_number(params, "get_milestone")
            .await?;

        let path = format!(
            "/repos/{}/{}/milestones/{milestone_number}",
            self.owner, self.repo
        );
        self.api_get(&path, &[]).await
    }

    /// Resolve a milestone number from the params (by direct number or title lookup).
    async fn resolve_milestone_number(
        &self,
        params: &GitHubToolParams,
        operation: &str,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(n) = params.number {
            Ok(n)
        } else if let Some(ref ms) = params.milestone {
            if let Ok(n) = ms.parse::<u64>() {
                Ok(n)
            } else {
                self.find_milestone_number_by_title(ms).await
            }
        } else {
            Err(format!(
                "{operation} requires a 'milestone' (title or number) or 'number' parameter"
            )
            .into())
        }
    }

    /// Fetch all items (issues + PRs) for a milestone from the REST API.
    async fn fetch_milestone_items(
        &self,
        milestone_number: u64,
        per_page: u32,
    ) -> Result<Vec<Value>, Box<dyn std::error::Error + Send + Sync>> {
        let path = format!("/repos/{}/{}/issues", self.owner, self.repo);
        let per_page = per_page.to_string();
        let ms_str = milestone_number.to_string();
        let items = self
            .api_get(
                &path,
                &[
                    ("milestone", &ms_str),
                    ("state", "all"),
                    ("per_page", &per_page),
                ],
            )
            .await?;

        items
            .as_array()
            .cloned()
            .ok_or_else(|| "Expected milestone items to be an array".into())
    }

    async fn handle_list_milestone_issues(
        &self,
        params: &GitHubToolParams,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let milestone_number = self
            .resolve_milestone_number(params, "list_milestone_issues")
            .await?;

        let items = self
            .fetch_milestone_items(milestone_number, params.per_page.unwrap_or(100))
            .await?;

        let results: Vec<Value> = items
            .iter()
            .filter(|item| item.get("pull_request").is_none())
            .map(|item| {
                json!({
                    "number": item.get("number"),
                    "title": item.get("title"),
                    "state": item.get("state"),
                })
            })
            .collect();

        Ok(json!(results))
    }

    async fn handle_list_milestone_pull_requests(
        &self,
        params: &GitHubToolParams,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let milestone_number = self
            .resolve_milestone_number(params, "list_milestone_pull_requests")
            .await?;

        let items = self
            .fetch_milestone_items(milestone_number, params.per_page.unwrap_or(100))
            .await?;

        let mut results = Vec::new();
        for item in &items {
            if item.get("pull_request").is_none() {
                continue;
            }
            let number = item["number"].as_u64().unwrap_or(0);
            let pr_path = format!(
                "/repos/{}/{}/pulls/{number}",
                self.owner, self.repo
            );
            let pr = self.api_get(&pr_path, &[]).await?;
            results.push(json!({
                "number": number,
                "title": item.get("title"),
                "state": item.get("state"),
                "merged": pr.get("merged"),
                "merge_commit_sha": pr.get("merge_commit_sha"),
                "merged_at": pr.get("merged_at"),
            }));
        }

        Ok(json!(results))
    }

    async fn handle_compare(
        &self,
        params: &GitHubToolParams,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let base = params
            .base
            .as_deref()
            .ok_or("compare requires a 'base' parameter")?;
        let head = params.head.as_deref().unwrap_or("trunk");

        let path = format!(
            "/repos/{}/{}/compare/{base}...{head}",
            self.owner, self.repo
        );
        let raw = self.api_get(&path, &[]).await?;

        let commits = raw
            .get("commits")
            .and_then(|c| c.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|c| {
                        json!({
                            "sha": c.get("sha"),
                            "message": c.get("commit")
                                .and_then(|cm| cm.get("message"))
                                .and_then(|m| m.as_str())
                                .map(|m| m.lines().next().unwrap_or(m)),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(json!({
            "total_commits": commits.len(),
            "commits": commits,
        }))
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
    // Arbitrary API operations
    // -----------------------------------------------------------------------

    /// Resolve API path. Paths not starting with "/" are prefixed with "/repos/{owner}/{repo}/".
    fn resolve_api_path(
        &self,
        params: &GitHubToolParams,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let raw = params
            .path
            .as_deref()
            .ok_or("api_get/api_post/api_patch requires a 'path' parameter")?;
        if raw.starts_with('/') {
            Ok(raw.to_string())
        } else {
            Ok(format!("/repos/{}/{}/{raw}", self.owner, self.repo))
        }
    }

    async fn handle_arbitrary_get(
        &self,
        params: &GitHubToolParams,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let path = self.resolve_api_path(params)?;
        let query: Vec<(&str, &str)> = params
            .query_params
            .as_ref()
            .map(|qp| qp.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect())
            .unwrap_or_default();
        self.api_get(&path, &query).await
    }

    async fn handle_arbitrary_post(
        &self,
        params: &GitHubToolParams,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let path = self.resolve_api_path(params)?;
        let body = params.json_body.as_ref().cloned().unwrap_or(json!({}));
        self.api_post(&path, &body).await
    }

    async fn handle_arbitrary_patch(
        &self,
        params: &GitHubToolParams,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let path = self.resolve_api_path(params)?;
        let body = params.json_body.as_ref().cloned().unwrap_or(json!({}));
        self.api_patch(&path, &body).await
    }

    async fn handle_graphql(
        &self,
        params: &GitHubToolParams,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let json_body = params
            .json_body
            .as_ref()
            .ok_or("graphql requires a 'json_body' with 'query' field")?;
        let query = json_body
            .get("query")
            .and_then(|q| q.as_str())
            .ok_or("graphql requires json_body.query to be a string")?;
        let variables = json_body
            .get("variables")
            .cloned()
            .unwrap_or(json!({}));
        self.api_graphql(query, variables).await
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Filter a JSON value to only include the specified fields.
    /// - For objects: keep only matching keys
    /// - For arrays: filter each element
    /// - For other types: return as-is
    fn filter_fields(value: Value, fields: &[String]) -> Value {
        match value {
            Value::Object(map) => {
                let filtered: serde_json::Map<String, Value> = map
                    .into_iter()
                    .filter(|(k, _)| fields.iter().any(|f| f == k))
                    .collect();
                Value::Object(filtered)
            }
            Value::Array(arr) => Value::Array(
                arr.into_iter()
                    .map(|v| Self::filter_fields(v, fields))
                    .collect(),
            ),
            other => other,
        }
    }

    /// Search milestones by title, returning the milestone number.
    async fn find_milestone_number_by_title(
        &self,
        title: &str,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let path = format!("/repos/{}/{}/milestones", self.owner, self.repo);
        let milestones = self
            .api_get(&path, &[("state", "all"), ("per_page", "100")])
            .await?;

        let milestones = milestones
            .as_array()
            .ok_or("Expected milestones to be an array")?;

        for ms in milestones {
            if ms["title"].as_str() == Some(title) {
                if let Some(n) = ms["number"].as_u64() {
                    tracing::debug!(title = %title, number = n, "Resolved milestone by title");
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

            tracing::debug!(operation = %params.operation, "Dispatching GitHub operation");

            let mut result = match params.operation.as_str() {
                "list_milestones" => self.handle_list_milestones(&params).await,
                "get_milestone" => self.handle_get_milestone(&params).await,
                "list_milestone_issues" => self.handle_list_milestone_issues(&params).await,
                "list_milestone_pull_requests" => {
                    self.handle_list_milestone_pull_requests(&params).await
                }
                "list_commits" => self.handle_list_commits(&params).await,
                "compare" => self.handle_compare(&params).await,
                "create_pull_request" => self.handle_create_pull_request(&params).await,
                "get_pull_request" => self.handle_get_pull_request(&params).await,
                "add_issue_to_milestone" => self.handle_add_issue_to_milestone(&params).await,
                "api_get" => self.handle_arbitrary_get(&params).await,
                "api_post" => self.handle_arbitrary_post(&params).await,
                "api_patch" => self.handle_arbitrary_patch(&params).await,
                "graphql" => self.handle_graphql(&params).await,
                _ => unreachable!(),
            };

            // Filter response fields if requested.
            if let Ok(ref mut value) = result {
                if let Some(ref fields) = params.fields {
                    if !fields.is_empty() {
                        *value = Self::filter_fields(std::mem::take(value), fields);
                    }
                }
            }

            result
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
                tracing::error!(target: "task_history", parent: &span, "GitHub tool failed: {e}");
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
