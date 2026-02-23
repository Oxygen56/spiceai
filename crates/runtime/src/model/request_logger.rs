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

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Logs each model API request to a numbered JSON file within a session directory.
///
/// Each chat session creates a timestamped folder under `.spice/data/requests/`.
/// Every model API call (including recursive tool-use calls) is saved as
/// `{NNN}_request.json`, forming a replayable checkpoint of the conversation.
#[derive(Clone)]
pub struct RequestLogger {
    session_dir: PathBuf,
    counter: Arc<AtomicUsize>,
}

impl RequestLogger {
    /// Creates a new session directory under `.spice/data/requests/{timestamp}/`.
    pub fn new() -> Self {
        let timestamp = chrono::Local::now().format("%Y-%m-%dT%H-%M-%S").to_string();
        let session_dir = PathBuf::from(crate::spice_data_base_path())
            .join("requests")
            .join(&timestamp);

        if let Err(e) = std::fs::create_dir_all(&session_dir) {
            tracing::warn!(error = %e, path = %session_dir.display(), "Failed to create request log session directory");
        } else {
            tracing::info!(path = %session_dir.display(), "Request logging session started");
        }

        Self {
            session_dir,
            counter: Arc::new(AtomicUsize::new(1)),
        }
    }

    /// Saves `req` as `{NNN}_request.json`. Fire-and-forget — warns on error.
    pub fn log_request(&self, req: &impl serde::Serialize) {
        let n = self.counter.fetch_add(1, Ordering::SeqCst);
        let filename = format!("{n:03}_request.json");
        let path = self.session_dir.join(&filename);

        match serde_json::to_string_pretty(req) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, json) {
                    tracing::warn!(error = %e, path = %path.display(), "Failed to write request log");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to serialize request for logging");
            }
        }
    }
}
