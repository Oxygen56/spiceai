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
use globset::{Glob, GlobMatcher};
use regex::Regex;
use serde_json::{Value, json};
use snafu::ResultExt;
use std::{
    borrow::Cow,
    fs,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
};
use tools::{SpiceModelTool, ToolCapability};
use tracing::Span;
use tracing_futures::Instrument;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::tools::builtin::git_worktree::WorktreeTracker;
use crate::tools::utils::parameters;

/// Number of bytes sampled from the start of a file to detect binary content.
const BINARY_DETECTION_BYTES: usize = 8192;

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct GrepToolParams {
    /// The regex pattern to search for in file contents.
    pattern: String,

    /// Optional glob pattern to filter which files are searched (e.g. "**/*.rs", "src/**/*.py").
    path: Option<String>,

    /// Maximum number of matching lines to return. Defaults to 50.
    max_results: Option<usize>,

    /// Number of context lines to include before and after each match. Defaults to 0.
    context_lines: Option<usize>,
}

pub struct GrepTool {
    name: String,
    description: String,
    base_paths: Vec<PathBuf>,
    worktree_tracker: Option<WorktreeTracker>,
}

impl GrepTool {
    #[must_use]
    pub fn new(
        name: Option<&str>,
        description: Option<&str>,
        base_paths: Vec<PathBuf>,
        worktree_tracker: Option<WorktreeTracker>,
    ) -> Self {
        Self {
            name: name.unwrap_or("grep").to_string(),
            description: description
                .unwrap_or("Search for a regex pattern in files within configured directories")
                .to_string(),
            base_paths,
            worktree_tracker,
        }
    }

    fn is_path_allowed(&self, path: &Path) -> bool {
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

    /// Returns `base_paths` extended with tracked worktree paths (for `visit_dirs` symlink validation).
    fn all_allowed_paths(&self) -> Vec<PathBuf> {
        let mut paths = self.base_paths.clone();
        if let Some(ref tracker) = self.worktree_tracker {
            for (_name, wt) in tracker.list() {
                paths.push(wt.path.clone());
            }
        }
        paths
    }
}

/// Recursively collect files, resolving symlinks and verifying access.
fn visit_dirs(
    dir: &Path,
    base_paths: &[PathBuf],
    files: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        // Resolve symlinks to their real path
        let resolved = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => continue, // Skip broken symlinks
        };

        // Verify the resolved path is within allowed base paths
        let allowed = base_paths.iter().any(|base| {
            base.canonicalize()
                .map_or(false, |b| resolved.starts_with(&b))
        });
        if !allowed {
            continue;
        }

        if resolved.is_dir() {
            visit_dirs(&resolved, base_paths, files)?;
        } else {
            files.push(resolved);
        }
    }
    Ok(())
}

/// Returns `true` if the file appears to be binary (contains null bytes in the first chunk).
fn is_binary(path: &Path) -> bool {
    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return true, // Treat unreadable files as binary
    };
    let mut buf = vec![0u8; BINARY_DETECTION_BYTES];
    let n = match file.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return true,
    };
    buf[..n].contains(&0)
}

/// A single match with optional surrounding context lines.
struct MatchResult {
    file: String,
    line: usize,
    content: String,
    context_before: Vec<String>,
    context_after: Vec<String>,
}

/// Search a single file for matches, collecting context lines when requested.
fn search_file(
    file_path: &Path,
    re: &Regex,
    context_lines: usize,
    max_results: usize,
    current_count: usize,
) -> Vec<MatchResult> {
    let file = match fs::File::open(file_path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let reader = BufReader::new(file);
    let file_str = file_path.to_string_lossy().to_string();

    let lines: Vec<String> = reader
        .lines()
        .map(|l| l.unwrap_or_default())
        .collect();

    let mut results = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if current_count + results.len() >= max_results {
            break;
        }
        if !re.is_match(line) {
            continue;
        }

        let before_start = idx.saturating_sub(context_lines);
        let context_before: Vec<String> = lines[before_start..idx].to_vec();

        let after_end = (idx + 1 + context_lines).min(lines.len());
        let context_after: Vec<String> = lines[idx + 1..after_end].to_vec();

        results.push(MatchResult {
            file: file_str.clone(),
            line: idx + 1,
            content: line.clone(),
            context_before,
            context_after,
        });
    }
    results
}

#[async_trait]
impl SpiceModelTool for GrepTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<GrepToolParams>()
    }

    fn capability(&self) -> ToolCapability {
        ToolCapability::ReadOnly
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span: Span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::grep", tool = self.name().to_string(), input = arg);

        let tool_use_result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let req: GrepToolParams = serde_json::from_str(arg)?;
            let max_results = req.max_results.unwrap_or(50);
            let context_lines = req.context_lines.unwrap_or(0);

            let re = Regex::new(&req.pattern)?;

            // Build glob matcher from path pattern
            let glob_matcher: Option<GlobMatcher> = match req.path {
                Some(ref pattern) => {
                    let glob = Glob::new(pattern).map_err(|e| {
                        format!("Invalid glob pattern '{pattern}': {e}")
                    })?;
                    Some(glob.compile_matcher())
                }
                None => None,
            };

            let allowed_paths = self.all_allowed_paths();
            let mut all_files = Vec::new();
            for base in &allowed_paths {
                visit_dirs(base, &allowed_paths, &mut all_files).ok();
            }

            // Apply glob filter
            if let Some(ref matcher) = glob_matcher {
                all_files.retain(|p| {
                    // Match against relative paths from each base
                    self.base_paths.iter().any(|base| {
                        if let Ok(base_canonical) = base.canonicalize() {
                            if let Ok(rel) = p.strip_prefix(&base_canonical) {
                                return matcher.is_match(rel);
                            }
                        }
                        // Fallback: match against the file name
                        p.file_name()
                            .map_or(false, |name| matcher.is_match(Path::new(name)))
                    })
                });
            }

            // Skip binary files
            all_files.retain(|p| !is_binary(p));

            let mut matches = Vec::new();
            for file_path in &all_files {
                if matches.len() >= max_results {
                    break;
                }
                let file_matches =
                    search_file(file_path, &re, context_lines, max_results, matches.len());
                matches.extend(file_matches);
            }

            let result_values: Vec<Value> = matches
                .iter()
                .map(|m| {
                    let mut entry = json!({
                        "file": m.file,
                        "line": m.line,
                        "content": m.content,
                    });
                    if !m.context_before.is_empty() {
                        entry["context_before"] = json!(m.context_before);
                    }
                    if !m.context_after.is_empty() {
                        entry["context_after"] = json!(m.context_after);
                    }
                    entry
                })
                .collect();

            Ok(json!({
                "matches": result_values,
                "total": result_values.len(),
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
    use std::io::Write;
    use tempfile::TempDir;

    fn setup_test_dir() -> TempDir {
        let dir = TempDir::new().unwrap();

        // Create a text file
        let mut f = fs::File::create(dir.path().join("hello.rs")).unwrap();
        writeln!(f, "fn main() {{").unwrap();
        writeln!(f, "    println!(\"hello world\");").unwrap();
        writeln!(f, "    println!(\"goodbye world\");").unwrap();
        writeln!(f, "    let x = 42;").unwrap();
        writeln!(f, "}}").unwrap();

        // Create a nested file
        fs::create_dir_all(dir.path().join("src")).unwrap();
        let mut f2 = fs::File::create(dir.path().join("src/lib.rs")).unwrap();
        writeln!(f2, "pub fn greet() {{").unwrap();
        writeln!(f2, "    println!(\"hello from lib\");").unwrap();
        writeln!(f2, "}}").unwrap();

        // Create a non-Rust file
        let mut f3 = fs::File::create(dir.path().join("notes.txt")).unwrap();
        writeln!(f3, "hello from notes").unwrap();

        // Create a binary file
        let mut fb = fs::File::create(dir.path().join("image.bin")).unwrap();
        fb.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x00, 0x00]).unwrap();

        dir
    }

    #[tokio::test]
    async fn test_basic_search() {
        let dir = setup_test_dir();
        let tool = GrepTool::new(None, None, vec![dir.path().to_path_buf()], None);

        let result = tool
            .call(r#"{"pattern": "hello"}"#)
            .await
            .unwrap();

        let total = result["total"].as_u64().unwrap();
        assert!(total >= 3, "Expected at least 3 'hello' matches, got {total}");
    }

    #[tokio::test]
    async fn test_glob_filter() {
        let dir = setup_test_dir();
        let tool = GrepTool::new(None, None, vec![dir.path().to_path_buf()], None);

        let result = tool
            .call(r#"{"pattern": "hello", "path": "**/*.rs"}"#)
            .await
            .unwrap();

        // Should only match .rs files, not notes.txt
        let matches = result["matches"].as_array().unwrap();
        for m in matches {
            let file = m["file"].as_str().unwrap();
            assert!(file.ends_with(".rs"), "Expected .rs file, got {file}");
        }
        let total = result["total"].as_u64().unwrap();
        assert!(total >= 2, "Expected at least 2 'hello' matches in .rs files, got {total}");
    }

    #[tokio::test]
    async fn test_skips_binary_files() {
        let dir = setup_test_dir();
        let tool = GrepTool::new(None, None, vec![dir.path().to_path_buf()], None);

        // Search for the PNG magic bytes pattern — should find nothing since binary is skipped
        let result = tool
            .call(r#"{"pattern": "PNG"}"#)
            .await
            .unwrap();

        let matches = result["matches"].as_array().unwrap();
        for m in matches {
            let file = m["file"].as_str().unwrap();
            assert!(!file.ends_with(".bin"), "Binary file should have been skipped: {file}");
        }
    }

    #[tokio::test]
    async fn test_context_lines() {
        let dir = setup_test_dir();
        let tool = GrepTool::new(None, None, vec![dir.path().to_path_buf()], None);

        let result = tool
            .call(r#"{"pattern": "goodbye", "path": "**/*.rs", "context_lines": 1}"#)
            .await
            .unwrap();

        let matches = result["matches"].as_array().unwrap();
        assert_eq!(matches.len(), 1);

        let m = &matches[0];
        assert!(m.get("context_before").is_some(), "Expected context_before");
        assert!(m.get("context_after").is_some(), "Expected context_after");

        let before = m["context_before"].as_array().unwrap();
        assert_eq!(before.len(), 1);
        assert!(before[0].as_str().unwrap().contains("hello world"));

        let after = m["context_after"].as_array().unwrap();
        assert_eq!(after.len(), 1);
        assert!(after[0].as_str().unwrap().contains("let x = 42"));
    }

    #[tokio::test]
    async fn test_max_results() {
        let dir = setup_test_dir();
        let tool = GrepTool::new(None, None, vec![dir.path().to_path_buf()], None);

        let result = tool
            .call(r#"{"pattern": "hello", "max_results": 1}"#)
            .await
            .unwrap();

        let total = result["total"].as_u64().unwrap();
        assert_eq!(total, 1);
    }

    #[tokio::test]
    async fn test_symlink_outside_base_skipped() {
        let dir = setup_test_dir();
        let outside = TempDir::new().unwrap();
        let mut secret = fs::File::create(outside.path().join("secret.rs")).unwrap();
        writeln!(secret, "fn secret() {{ /* hello */ }}").unwrap();

        // Create symlink inside base dir pointing outside
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(
                outside.path().join("secret.rs"),
                dir.path().join("link_to_secret.rs"),
            )
            .unwrap();
        }

        let tool = GrepTool::new(None, None, vec![dir.path().to_path_buf()], None);
        let result = tool
            .call(r#"{"pattern": "secret"}"#)
            .await
            .unwrap();

        let matches = result["matches"].as_array().unwrap();
        for m in matches {
            let file = m["file"].as_str().unwrap();
            assert!(
                !file.contains("secret.rs"),
                "Symlink outside base dir should have been skipped: {file}"
            );
        }
    }
}
