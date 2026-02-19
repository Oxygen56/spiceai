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
    /// The REST API endpoint path.
    /// Relative paths (no leading "/") are auto-prefixed with /repos/{owner}/{repo}/.
    /// Absolute paths (leading "/") are used as-is.
    endpoint: String,

    /// HTTP method: "GET" (default), "POST", "PATCH", "PUT", "DELETE".
    #[serde(default)]
    method: Option<String>,

    /// Request parameters. For POST/PATCH/PUT these are sent as the JSON request body.
    /// For GET/DELETE these are sent as URL query parameters.
    #[serde(default)]
    params: Option<Value>,
}

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

        let default_description = format!(
            "Access the GitHub REST API for the repository {owner}/{repo}.\n\
            \n\
            Works like `gh api` — provide an endpoint, optional HTTP method, and optional parameters.\n\
            \n\
            Endpoint path resolution:\n\
            - Relative paths (no leading /) are prefixed with /repos/{owner}/{repo}/\n\
            \x20 Example: \"pulls\" becomes /repos/{owner}/{repo}/pulls\n\
            \x20 Example: \"milestones\" becomes /repos/{owner}/{repo}/milestones\n\
            - Absolute paths (leading /) are used as-is\n\
            \x20 Example: \"/user\" stays /user\n\
            \n\
            HTTP methods:\n\
            - GET (default): Fetch resources. `params` become query string parameters.\n\
            - POST: Create resources. `params` become the JSON request body.\n\
            - PATCH: Update resources. `params` become the JSON request body.\n\
            - PUT: Replace resources. `params` become the JSON request body.\n\
            - DELETE: Remove resources.\n\
            \n\
            Common patterns:\n\
            \n\
            List open pull requests:\n\
            \x20 {{\"endpoint\": \"pulls\", \"params\": {{\"state\": \"open\", \"per_page\": 10}}}}\n\
            \n\
            Create a pull request:\n\
            \x20 {{\"endpoint\": \"pulls\", \"method\": \"POST\", \"params\": {{\"title\": \"My PR\", \"head\": \"feature-branch\", \"base\": \"main\"}}}}\n\
            \n\
            Get a specific issue:\n\
            \x20 {{\"endpoint\": \"issues/42\"}}\n\
            \n\
            List milestones:\n\
            \x20 {{\"endpoint\": \"milestones\", \"params\": {{\"state\": \"open\"}}}}\n\
            \n\
            Add labels to an issue:\n\
            \x20 {{\"endpoint\": \"issues/42/labels\", \"method\": \"POST\", \"params\": {{\"labels\": [\"bug\", \"urgent\"]}}}}\n\
            \n\
            Compare two refs:\n\
            \x20 {{\"endpoint\": \"compare/main...feature-branch\"}}\n\
            \n\
            List commits on a branch:\n\
            \x20 {{\"endpoint\": \"commits\", \"params\": {{\"sha\": \"main\", \"per_page\": 10}}}}\n\
            \n\
            Search across GitHub (absolute path):\n\
            \x20 {{\"endpoint\": \"/search/issues\", \"params\": {{\"q\": \"repo:{owner}/{repo} is:pr is:merged\"}}}}\n\
            \n\
            API documentation: https://docs.github.com/en/rest",
        );

        Ok(Self {
            name: name.unwrap_or("github").to_string(),
            description: description.unwrap_or(&default_description).to_string(),
            owner,
            repo,
            token,
            client: reqwest::Client::new(),
        })
    }

    /// Resolve endpoint path. Relative paths are prefixed with `/repos/{owner}/{repo}/`.
    fn resolve_path(&self, endpoint: &str) -> String {
        if endpoint.starts_with('/') {
            endpoint.to_string()
        } else {
            format!("/repos/{}/{}/{endpoint}", self.owner, self.repo)
        }
    }

    /// Execute a GitHub API request.
    async fn api_request(
        &self,
        method: &str,
        path: &str,
        params: Option<&Value>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        tracing::debug!(method = %method, path = %path, "GitHub API request");
        let url = format!("{GITHUB_API_BASE}{path}");

        let mut request = match method {
            "POST" => self.client.post(&url),
            "PATCH" => self.client.patch(&url),
            "PUT" => self.client.put(&url),
            "DELETE" => self.client.delete(&url),
            _ => self.client.get(&url),
        };

        request = request
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "spice-ai-agent")
            .header("X-GitHub-Api-Version", "2022-11-28");

        // For body-bearing methods, send as JSON body; otherwise as query params.
        if let Some(p) = params {
            match method {
                "POST" | "PATCH" | "PUT" => {
                    request = request.json(p);
                }
                _ => {
                    if let Some(obj) = p.as_object() {
                        let query_params: Vec<(String, String)> = obj
                            .iter()
                            .map(|(k, v)| {
                                (
                                    k.clone(),
                                    match v {
                                        Value::String(s) => s.clone(),
                                        other => other.to_string(),
                                    },
                                )
                            })
                            .collect();
                        request = request.query(&query_params);
                    }
                }
            }
        }

        let response = request.send().await?;
        let status = response.status();
        let body_text = response.text().await?;

        if !status.is_success() {
            tracing::warn!(method = %method, path = %path, %status, body = %body_text, "GitHub API error response");
            return Err(format!(
                "GitHub API {method} {path} returned HTTP {status}: {body_text}",
            )
            .into());
        }

        // Handle empty responses (e.g. 204 No Content).
        if body_text.is_empty() {
            return Ok(json!({"status": status.as_u16()}));
        }

        serde_json::from_str(&body_text).map_err(|e| {
            format!("Failed to parse GitHub API response as JSON: {e}").into()
        })
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

            let path = self.resolve_path(&params.endpoint);
            let method = params.method.as_deref().unwrap_or("GET").to_uppercase();

            self.api_request(&method, &path, params.params.as_ref()).await
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
            desc.contains("GitHub REST API"),
            "Expected description to mention GitHub REST API, got: {desc}"
        );
        assert!(
            desc.contains("spiceai/spiceai"),
            "Expected description to mention repo, got: {desc}"
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

    #[test]
    fn test_resolve_path() {
        let tool = GitHubTool::try_new(
            None,
            None,
            "https://github.com/spiceai/spiceai",
            "tok".to_string(),
        )
        .unwrap();

        // Relative path gets prefixed
        assert_eq!(
            tool.resolve_path("pulls"),
            "/repos/spiceai/spiceai/pulls"
        );
        assert_eq!(
            tool.resolve_path("issues/42"),
            "/repos/spiceai/spiceai/issues/42"
        );

        // Absolute path stays as-is
        assert_eq!(tool.resolve_path("/user"), "/user");
        assert_eq!(
            tool.resolve_path("/search/issues"),
            "/search/issues"
        );
    }
}
