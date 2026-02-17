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
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use runtime_agentic::workflow::SessionResetHandler;

/// Tracks worktrees created during agent execution for session-scoped cleanup.
#[derive(Debug, Clone, Default)]
pub struct WorktreeTracker {
    inner: Arc<Mutex<HashMap<String, TrackedWorktree>>>,
}

#[derive(Debug, Clone)]
pub struct TrackedWorktree {
    pub path: PathBuf,
    pub branch: String,
    pub repo_path: PathBuf,
}

impl WorktreeTracker {
    pub fn register(&self, name: String, tracked: TrackedWorktree) {
        match self.inner.lock() {
            Ok(mut map) => {
                map.insert(name, tracked);
            }
            Err(e) => {
                tracing::error!("WorktreeTracker lock poisoned in register(): {e}");
            }
        }
    }

    pub fn deregister(&self, name: &str) -> Option<TrackedWorktree> {
        match self.inner.lock() {
            Ok(mut map) => map.remove(name),
            Err(e) => {
                tracing::error!("WorktreeTracker lock poisoned in deregister(): {e}");
                None
            }
        }
    }

    /// Get a worktree by name without removing it from the tracker.
    pub fn get(&self, name: &str) -> Option<TrackedWorktree> {
        match self.inner.lock() {
            Ok(map) => map.get(name).cloned(),
            Err(e) => {
                tracing::error!("WorktreeTracker lock poisoned in get(): {e}");
                None
            }
        }
    }

    /// List all tracked worktrees without removing them.
    pub fn list(&self) -> Vec<(String, TrackedWorktree)> {
        match self.inner.lock() {
            Ok(map) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            Err(e) => {
                tracing::error!("WorktreeTracker lock poisoned in list(): {e}");
                Vec::new()
            }
        }
    }

    pub fn drain(&self) -> HashMap<String, TrackedWorktree> {
        match self.inner.lock() {
            Ok(mut map) => std::mem::take(&mut *map),
            Err(e) => {
                tracing::error!("WorktreeTracker lock poisoned in drain(): {e}");
                HashMap::new()
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match self.inner.lock() {
            Ok(map) => map.is_empty(),
            Err(e) => {
                tracing::error!("WorktreeTracker lock poisoned in is_empty(): {e}");
                true
            }
        }
    }
}

impl SessionResetHandler for WorktreeTracker {
    fn on_session_reset(&self) {
        let entries = self.drain();
        for (name, tracked) in entries {
            if let Ok(repo) = git2::Repository::open(&tracked.repo_path) {
                if let Ok(wt) = repo.find_worktree(&name) {
                    let _ = wt.prune(Some(
                        &mut git2::WorktreePruneOptions::new()
                            .valid(true)
                            .working_tree(true),
                    ));
                }
            }
            let _ = std::fs::remove_dir_all(&tracked.path);
            tracing::info!(
                target: "task_history",
                worktree = %name,
                path = %tracked.path.display(),
                "Cleaned up worktree on session reset"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tracked(suffix: &str) -> TrackedWorktree {
        TrackedWorktree {
            path: PathBuf::from(format!("/tmp/wt-{suffix}")),
            branch: format!("branch-{suffix}"),
            repo_path: PathBuf::from("/tmp/repo"),
        }
    }

    #[test]
    fn test_tracker_register_deregister() {
        let tracker = WorktreeTracker::default();
        assert!(tracker.is_empty());

        tracker.register("wt-1".to_string(), make_tracked("1"));
        assert!(!tracker.is_empty());

        let removed = tracker.deregister("wt-1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().branch, "branch-1");
        assert!(tracker.is_empty());

        // Deregistering a non-existent entry returns None.
        let removed = tracker.deregister("wt-nonexistent");
        assert!(removed.is_none());
    }

    #[test]
    fn test_tracker_drain() {
        let tracker = WorktreeTracker::default();
        tracker.register("wt-a".to_string(), make_tracked("a"));
        tracker.register("wt-b".to_string(), make_tracked("b"));
        assert!(!tracker.is_empty());

        let drained = tracker.drain();
        assert_eq!(drained.len(), 2);
        assert!(drained.contains_key("wt-a"));
        assert!(drained.contains_key("wt-b"));
        assert!(tracker.is_empty());

        // Draining again yields an empty map.
        let drained_again = tracker.drain();
        assert!(drained_again.is_empty());
    }
}
