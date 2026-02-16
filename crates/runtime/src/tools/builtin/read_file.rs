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
use serde_json::{Value, json};
use snafu::ResultExt;
use std::{borrow::Cow, fs, path::PathBuf};
use tools::SpiceModelTool;
use tracing::Span;
use tracing_futures::Instrument;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::tools::utils::parameters;

/// Maximum file size that can be read (1 MB).
const MAX_FILE_SIZE: u64 = 1_048_576;

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct ReadFileToolParams {
    /// The path to the file to read.
    path: String,
}

pub struct ReadFileTool {
    name: String,
    description: String,
    base_paths: Vec<PathBuf>,
}

impl ReadFileTool {
    #[must_use]
    pub fn new(
        name: Option<&str>,
        description: Option<&str>,
        base_paths: Vec<PathBuf>,
    ) -> Self {
        Self {
            name: name.unwrap_or("read_file").to_string(),
            description: description
                .unwrap_or("Read the contents of a file at the given path")
                .to_string(),
            base_paths,
        }
    }

    fn is_path_allowed(&self, path: &std::path::Path) -> bool {
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => return false,
        };

        self.base_paths.iter().any(|base| {
            base.canonicalize()
                .map_or(false, |b| canonical.starts_with(&b))
        })
    }
}

#[async_trait]
impl SpiceModelTool for ReadFileTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<ReadFileToolParams>()
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span: Span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::read_file", tool = self.name().to_string(), input = arg);

        let tool_use_result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let req: ReadFileToolParams = serde_json::from_str(arg)?;
            let file_path = PathBuf::from(&req.path);

            if !self.is_path_allowed(&file_path) {
                return Err(format!(
                    "Access denied: path '{}' is outside allowed directories",
                    req.path
                )
                .into());
            }

            let metadata = fs::metadata(&file_path)?;
            if metadata.len() > MAX_FILE_SIZE {
                return Err(format!(
                    "File too large: {} bytes (max {} bytes)",
                    metadata.len(),
                    MAX_FILE_SIZE
                )
                .into());
            }

            let content = fs::read_to_string(&file_path)?;
            Ok(json!({
                "path": req.path,
                "content": content,
                "size": metadata.len(),
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
