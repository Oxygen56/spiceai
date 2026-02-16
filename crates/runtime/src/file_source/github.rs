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
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use git2::{
    Cred, FetchOptions, RemoteCallbacks, Repository, build::RepoBuilder,
};

use super::{FileSourceConnector, FileSourceConnectorFactory, RefreshStats};

/// A [`FileSourceConnector`] that syncs files from a GitHub repository using git
/// clone/pull operations.
#[derive(Debug)]
pub struct GitFileSource {
    repo_url: String,
    branch: String,
    token: Option<String>,
    target_path: PathBuf,
}

impl GitFileSource {
    fn build_fetch_options(&self) -> FetchOptions<'_> {
        let mut callbacks = RemoteCallbacks::new();
        if let Some(ref token) = self.token {
            let token = token.clone();
            callbacks.credentials(move |_url, _username, _allowed| {
                Cred::userpass_plaintext("x-access-token", &token)
            });
        }
        let mut fetch_opts = FetchOptions::new();
        fetch_opts.remote_callbacks(callbacks);
        fetch_opts
    }

    fn clone_repo(&self, target_path: &Path) -> Result<usize, git2::Error> {
        tracing::info!(repo = %self.repo_url, branch = %self.branch, "Cloning repository");

        let fetch_opts = self.build_fetch_options();
        let repo = RepoBuilder::new()
            .branch(&self.branch)
            .fetch_options(fetch_opts)
            .clone(&self.repo_url, target_path)?;

        let head = repo.head()?;
        let tree = head.peel_to_tree()?;
        Ok(tree.len())
    }

    fn pull_repo(&self, target_path: &Path) -> Result<usize, git2::Error> {
        tracing::info!(repo = %self.repo_url, branch = %self.branch, "Pulling latest changes");

        let repo = Repository::open(target_path)?;
        let mut remote = repo.find_remote("origin")?;

        let fetch_opts = self.build_fetch_options();
        remote.fetch(&[&self.branch], Some(&mut { fetch_opts }), None)?;

        let fetch_head = repo.find_reference("FETCH_HEAD")?;
        let fetch_commit = repo.reference_to_annotated_commit(&fetch_head)?;
        let (analysis, _) = repo.merge_analysis(&[&fetch_commit])?;

        if analysis.is_fast_forward() || analysis.is_normal() {
            let refname = format!("refs/heads/{}", self.branch);
            if let Ok(mut reference) = repo.find_reference(&refname) {
                reference.set_target(fetch_commit.id(), "fast-forward")?;
            }
            repo.set_head(&refname)?;
            repo.checkout_head(Some(git2::build::CheckoutBuilder::new().force()))?;
            tracing::info!("Repository updated via fast-forward");
        } else {
            tracing::debug!("Repository already up to date");
        }

        let head = repo.head()?;
        let tree = head.peel_to_tree()?;
        Ok(tree.len())
    }
}

#[async_trait]
impl FileSourceConnector for GitFileSource {
    async fn refresh(&self) -> Result<RefreshStats, Box<dyn std::error::Error + Send + Sync>> {
        let start = Instant::now();
        let target = self.target_path.clone();
        let is_existing_repo = target.join(".git").is_dir();

        let repo_url = self.repo_url.clone();
        let branch = self.branch.clone();
        let token = self.token.clone();
        let target_clone = target.clone();

        let source = GitFileSource {
            repo_url,
            branch,
            token,
            target_path: target_clone,
        };

        let files_synced = tokio::task::spawn_blocking(move || {
            if is_existing_repo {
                source.pull_repo(&target)
            } else {
                source.clone_repo(&target)
            }
        })
        .await??;

        Ok(RefreshStats {
            files_synced,
            bytes_transferred: 0,
            duration: start.elapsed(),
        })
    }
}

pub struct GitFileSourceFactory;

#[async_trait]
impl FileSourceConnectorFactory for GitFileSourceFactory {
    fn prefix(&self) -> &'static str {
        "github"
    }

    async fn create(
        &self,
        target_path: PathBuf,
        params: HashMap<String, String>,
    ) -> Result<Arc<dyn FileSourceConnector>, Box<dyn std::error::Error + Send + Sync>> {
        let repo = params
            .get("from")
            .ok_or("Missing 'from' parameter (expected github:owner/repo)")?
            .strip_prefix("github:")
            .ok_or("'from' must start with 'github:'")?;

        let repo_url = format!("https://github.com/{repo}.git");
        let branch = params
            .get("branch")
            .cloned()
            .unwrap_or_else(|| "main".to_string());
        let token = params.get("github_token").cloned();

        Ok(Arc::new(GitFileSource {
            repo_url,
            branch,
            token,
            target_path,
        }))
    }
}
