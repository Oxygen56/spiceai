/*
Copyright 2024-2026 The Spice.ai OSS Authors

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

//! `spice replay` command - Replay a saved chat request JSON file.

use crate::context::RuntimeContext;
use crate::error::{ConnectionFailedSnafu, InvalidResponseSnafu, Result};
use clap::Args;
use futures::StreamExt;
use serde::Deserialize;
use snafu::ResultExt;
use std::io::{self, Write};

/// Arguments for the `replay` command.
#[derive(Args, Debug)]
pub struct ReplayArgs {
    /// Path to a saved request JSON file (from chat_log/)
    pub file: String,

    /// Remote Spice instance HTTP endpoint (e.g., `http://localhost:8090`)
    #[arg(long)]
    pub endpoint: Option<String>,

    /// Custom HTTP headers in format 'Key:Value' (can be specified multiple times)
    #[arg(long = "headers", value_name = "KEY:VALUE")]
    pub custom_headers: Vec<String>,
}

#[derive(Deserialize)]
struct ChatChunk {
    choices: Vec<ChunkChoice>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    delta: Delta,
}

#[derive(Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Deserialize)]
struct DeltaToolCall {
    #[serde(default)]
    function: Option<DeltaFunction>,
}

#[derive(Deserialize)]
struct DeltaFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct SseError {
    error: SseErrorBody,
}

#[derive(Deserialize)]
struct SseErrorBody {
    message: String,
}

/// Execute the `replay` command.
///
/// # Errors
///
/// Returns an error if the file cannot be read, parsed, or the API request fails.
pub async fn execute(ctx: &RuntimeContext, args: &ReplayArgs) -> Result<()> {
    let file_content = std::fs::read_to_string(&args.file).map_err(|e| {
        InvalidResponseSnafu {
            message: format!("Failed to read replay file '{}': {e}", args.file),
        }
        .build()
    })?;

    let mut body: serde_json::Value = serde_json::from_str(&file_content).map_err(|e| {
        InvalidResponseSnafu {
            message: format!("Failed to parse replay file as JSON: {e}"),
        }
        .build()
    })?;

    // Ensure streaming is enabled
    if let Some(obj) = body.as_object_mut() {
        obj.insert("stream".to_string(), serde_json::Value::Bool(true));
    }

    let base_endpoint = args
        .endpoint
        .as_deref()
        .unwrap_or_else(|| ctx.http_endpoint());
    let url = format!("{base_endpoint}/v1/chat/completions");

    eprintln!("Replaying {} -> {url}", args.file);

    let streaming_client = reqwest::Client::builder().build().unwrap_or_default();

    let mut request = streaming_client
        .post(&url)
        .header("Content-Type", "application/json")
        .header("Accept", "text/event-stream")
        .header("X-Spice-Single-Shot", "true")
        .json(&body);

    for (key, value) in ctx.get_headers() {
        request = request.header(&key, &value);
    }

    for header in &args.custom_headers {
        if let Some((key, value)) = header.split_once(':') {
            request = request.header(key.trim(), value.trim());
        }
    }

    let response = request
        .send()
        .await
        .context(ConnectionFailedSnafu { endpoint: &url })?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(InvalidResponseSnafu {
            message: format!("Replay request failed: {status} - {text}"),
        }
        .build());
    }

    let mut stream = response.bytes_stream();
    while let Some(chunk_result) = stream.next().await {
        let chunk = match chunk_result {
            Ok(chunk) => chunk,
            Err(e) => {
                eprintln!("\n\x1b[33mWarning:\x1b[0m Stream interrupted: {e}");
                break;
            }
        };

        let text = String::from_utf8_lossy(&chunk);
        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    continue;
                }
                if let Ok(chat_chunk) = serde_json::from_str::<ChatChunk>(data) {
                    for choice in &chat_chunk.choices {
                        if let Some(content) = &choice.delta.content {
                            print!("{content}");
                            let _ = io::stdout().flush();
                        }
                        if let Some(tool_calls) = &choice.delta.tool_calls {
                            for tc in tool_calls {
                                if let Some(ref func) = tc.function {
                                    if let Some(ref name) = func.name {
                                        print!("\n[tool_call: {name}");
                                    }
                                    if let Some(ref args) = func.arguments {
                                        print!("{args}");
                                    }
                                }
                            }
                            let _ = io::stdout().flush();
                        }
                    }
                } else if let Ok(sse_error) = serde_json::from_str::<SseError>(data) {
                    eprintln!("\n\x1b[31mError:\x1b[0m {}", sse_error.error.message);
                }
            }
        }
    }

    println!();
    Ok(())
}
