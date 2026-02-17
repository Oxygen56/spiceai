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

use std::collections::HashMap;

use secrecy::SecretString;
use spicepod::component::file_source::FileSource;

/// Parse a tool shorthand string into `(tool_id, is_write)`.
///
/// `"git:write"` → `("git", true)`
/// `"list_files"` → `("list_files", false)`
pub fn parse_tool_shorthand(s: &str) -> (&str, bool) {
    match s.strip_suffix(":write") {
        Some(id) => (id, true),
        None => (s, false),
    }
}

/// Derive a GitHub remote URL from a file_source `from` field.
///
/// `"github:/owner/repo"` → `Some("https://github.com/owner/repo")`
/// `"local:/path"` → `None`
pub fn derive_github_remote(from: &str) -> Option<String> {
    let path = from.strip_prefix("github:/")?;
    Some(format!("https://github.com/{path}"))
}

/// Derive tool construction params from a [`FileSource`] for the given `tool_id`.
///
/// The file_source's `path` is used as `base_paths` (for file-access tools) or
/// `repo_path` (for git tools), ensuring tools are scoped to the file_source's
/// local checkout directory.
pub fn derive_tool_params(
    file_source: &FileSource,
    tool_id: &str,
) -> HashMap<String, SecretString> {
    let mut params = HashMap::new();

    match tool_id {
        "list_files" | "read_file" | "grep" => {
            params.insert(
                "base_paths".to_string(),
                SecretString::from(file_source.path.clone()),
            );
        }
        "git" => {
            params.insert(
                "repo_path".to_string(),
                SecretString::from(file_source.path.clone()),
            );
            params.insert(
                "capability".to_string(),
                SecretString::from("read_write".to_string()),
            );
            if let Some(token) = file_source.params.get("github_token") {
                params.insert(
                    "github_token".to_string(),
                    SecretString::from(token.clone()),
                );
            }
        }
        "github" => {
            if let Some(remote) = derive_github_remote(&file_source.from) {
                params.insert("remote".to_string(), SecretString::from(remote));
            }
            if let Some(token) = file_source.params.get("github_token") {
                params.insert(
                    "github_token".to_string(),
                    SecretString::from(token.clone()),
                );
            }
        }
        "git_worktree" => {
            params.insert(
                "repo_path".to_string(),
                SecretString::from(file_source.path.clone()),
            );
            if let Some(worktree_root) = file_source.params.get("worktree_root") {
                params.insert(
                    "worktree_root".to_string(),
                    SecretString::from(worktree_root.clone()),
                );
            }
        }
        "claude_code" => {
            params.insert(
                "allowed_working_dirs".to_string(),
                SecretString::from(file_source.path.clone()),
            );
            for key in &[
                "claude_binary",
                "default_model",
                "default_max_turns",
                "default_allowed_tools",
            ] {
                if let Some(val) = file_source.params.get(*key) {
                    params.insert(key.to_string(), SecretString::from(val.clone()));
                }
            }
        }
        _ => {}
    }

    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;

    #[test]
    fn test_parse_tool_shorthand() {
        assert_eq!(parse_tool_shorthand("git:write"), ("git", true));
        assert_eq!(parse_tool_shorthand("github:write"), ("github", true));
        assert_eq!(
            parse_tool_shorthand("git_worktree:write"),
            ("git_worktree", true)
        );
        assert_eq!(parse_tool_shorthand("list_files"), ("list_files", false));
        assert_eq!(parse_tool_shorthand("read_file"), ("read_file", false));
        assert_eq!(parse_tool_shorthand("grep"), ("grep", false));
        assert_eq!(
            parse_tool_shorthand("claude_code:write"),
            ("claude_code", true)
        );
        assert_eq!(parse_tool_shorthand("claude_code"), ("claude_code", false));
    }

    #[test]
    fn test_derive_github_remote() {
        assert_eq!(
            derive_github_remote("github:/owner/repo"),
            Some("https://github.com/owner/repo".to_string())
        );
        assert_eq!(
            derive_github_remote("github:/krinart/demo_agent"),
            Some("https://github.com/krinart/demo_agent".to_string())
        );
        assert_eq!(derive_github_remote("local:/some/path"), None);
        assert_eq!(derive_github_remote("s3:/bucket/key"), None);
    }

    fn make_file_source(from: &str, path: &str, params: Vec<(&str, &str)>) -> FileSource {
        FileSource {
            from: from.to_string(),
            name: "test_source".to_string(),
            path: path.to_string(),
            params: params
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            env: HashMap::new(),
            refresh: None,
            tools: vec![],
            depends_on: vec![],
        }
    }

    #[test]
    fn test_derive_tool_params_file_tools() {
        let fs = make_file_source("github:/owner/repo", "/path/to/repo", vec![]);
        for tool_id in &["list_files", "read_file", "grep"] {
            let params = derive_tool_params(&fs, tool_id);
            assert_eq!(
                params.get("base_paths").unwrap().expose_secret(),
                "/path/to/repo",
                "base_paths should match fs.path for {tool_id}"
            );
            assert_eq!(params.len(), 1);
        }
    }

    #[test]
    fn test_derive_tool_params_git() {
        let fs = make_file_source(
            "github:/owner/repo",
            "/path/to/repo",
            vec![("github_token", "ghp_test123")],
        );
        let params = derive_tool_params(&fs, "git");
        assert_eq!(
            params.get("repo_path").unwrap().expose_secret(),
            "/path/to/repo"
        );
        assert_eq!(
            params.get("capability").unwrap().expose_secret(),
            "read_write"
        );
        assert_eq!(
            params.get("github_token").unwrap().expose_secret(),
            "ghp_test123"
        );
    }

    #[test]
    fn test_derive_tool_params_git_no_token() {
        let fs = make_file_source("github:/owner/repo", "/path/to/repo", vec![]);
        let params = derive_tool_params(&fs, "git");
        assert_eq!(
            params.get("repo_path").unwrap().expose_secret(),
            "/path/to/repo"
        );
        assert_eq!(
            params.get("capability").unwrap().expose_secret(),
            "read_write"
        );
        assert!(params.get("github_token").is_none());
    }

    #[test]
    fn test_derive_tool_params_github() {
        let fs = make_file_source(
            "github:/krinart/demo_agent",
            "/path/to/repo",
            vec![("github_token", "ghp_abc")],
        );
        let params = derive_tool_params(&fs, "github");
        assert_eq!(
            params.get("remote").unwrap().expose_secret(),
            "https://github.com/krinart/demo_agent"
        );
        assert_eq!(
            params.get("github_token").unwrap().expose_secret(),
            "ghp_abc"
        );
    }

    #[test]
    fn test_derive_tool_params_git_worktree() {
        let fs = make_file_source(
            "github:/owner/repo",
            "/path/to/repo",
            vec![("worktree_root", "/tmp/my-worktrees")],
        );
        let params = derive_tool_params(&fs, "git_worktree");
        assert_eq!(
            params.get("repo_path").unwrap().expose_secret(),
            "/path/to/repo"
        );
        assert_eq!(
            params.get("worktree_root").unwrap().expose_secret(),
            "/tmp/my-worktrees"
        );
    }

    #[test]
    fn test_derive_tool_params_git_worktree_no_root() {
        let fs = make_file_source("github:/owner/repo", "/path/to/repo", vec![]);
        let params = derive_tool_params(&fs, "git_worktree");
        assert_eq!(
            params.get("repo_path").unwrap().expose_secret(),
            "/path/to/repo"
        );
        assert!(
            params.get("worktree_root").is_none(),
            "worktree_root should fall through to catalog default"
        );
    }

    #[test]
    fn test_derive_tool_params_claude_code() {
        let fs = make_file_source(
            "github:/owner/repo",
            "/path/to/repo",
            vec![
                ("default_model", "claude-sonnet-4-5-20250929"),
                ("default_max_turns", "5"),
            ],
        );
        let params = derive_tool_params(&fs, "claude_code");
        assert_eq!(
            params.get("allowed_working_dirs").unwrap().expose_secret(),
            "/path/to/repo"
        );
        assert_eq!(
            params.get("default_model").unwrap().expose_secret(),
            "claude-sonnet-4-5-20250929"
        );
        assert_eq!(
            params.get("default_max_turns").unwrap().expose_secret(),
            "5"
        );
        assert!(params.get("claude_binary").is_none());
        assert!(params.get("default_allowed_tools").is_none());
    }

    #[test]
    fn test_derive_tool_params_claude_code_minimal() {
        let fs = make_file_source("github:/owner/repo", "/path/to/repo", vec![]);
        let params = derive_tool_params(&fs, "claude_code");
        assert_eq!(
            params.get("allowed_working_dirs").unwrap().expose_secret(),
            "/path/to/repo"
        );
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn test_derive_tool_params_unknown_tool() {
        let fs = make_file_source("github:/owner/repo", "/path/to/repo", vec![]);
        let params = derive_tool_params(&fs, "unknown_tool");
        assert!(params.is_empty());
    }
}
