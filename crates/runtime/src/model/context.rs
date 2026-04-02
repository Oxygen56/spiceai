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

//! Context management for the tool-use loop.
//!
//! Provides token budget tracking, tool output truncation, and message pruning
//! to prevent context window overflow during multi-iteration tool calls.
//!
//! All logic is provider-agnostic — it operates on token counts from
//! `CompletionUsage` / `ResponseUsage` (returned by every provider) and the
//! universal message types (`ChatCompletionRequestMessage` / `InputItem`).

use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestToolMessageArgs,
};
use async_openai::types::responses::{
    EasyInputContent, EasyInputMessage, FunctionCallOutput, InputItem, Item, MessageType, Role,
};
use serde_json::Value;

/// Default maximum characters for a single tool output.
pub const DEFAULT_MAX_TOOL_OUTPUT_CHARS: usize = 50_000;

/// Default context utilization threshold to trigger pruning (75%).
pub const DEFAULT_PRUNE_THRESHOLD: f32 = 0.75;

/// Default number of recent tool call/response pairs to keep intact.
pub const DEFAULT_KEEP_RECENT_TOOL_PAIRS: usize = 3;

/// Default tokens reserved for model response generation.
pub const DEFAULT_RESPONSE_RESERVE: u32 = 16_384;

/// Configuration for context management within the tool-use loop.
///
/// Extracted from model parameters at construction time. All fields
/// have sensible defaults; budget-based pruning is only active when
/// `context_window` is `Some`.
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Total context window size in tokens. Required for budget-based pruning.
    pub context_window: Option<u32>,

    /// Tokens reserved for response generation (default: 16384).
    pub response_reserve: u32,

    /// Utilization threshold to trigger pruning (default: 0.75).
    pub prune_threshold: f32,

    /// Max chars for any single tool output (default: 50000).
    pub max_tool_output_chars: usize,

    /// Whether to inject budget status into messages (default: true).
    pub budget_awareness: bool,

    /// Number of recent tool response messages to always keep intact (default: 3).
    pub keep_recent_tool_pairs: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            context_window: None,
            response_reserve: DEFAULT_RESPONSE_RESERVE,
            prune_threshold: DEFAULT_PRUNE_THRESHOLD,
            max_tool_output_chars: DEFAULT_MAX_TOOL_OUTPUT_CHARS,
            budget_awareness: true,
            keep_recent_tool_pairs: DEFAULT_KEEP_RECENT_TOOL_PAIRS,
        }
    }
}

impl ContextConfig {
    /// Check whether context-budget-based pruning should be triggered.
    ///
    /// Returns `true` if a `context_window` is configured AND the current
    /// prompt utilization exceeds `prune_threshold`.
    pub fn should_prune(&self, prompt_tokens: u32) -> bool {
        let Some(context_window) = self.context_window else {
            return false;
        };
        let usable = context_window.saturating_sub(self.response_reserve);
        if usable == 0 {
            return true;
        }
        let utilization = prompt_tokens as f32 / usable as f32;
        utilization >= self.prune_threshold
    }

    /// Format a human-readable budget status string for injection into messages.
    ///
    /// Returns `None` when `context_window` is not configured.
    pub fn format_budget_status(&self, prompt_tokens: u32) -> Option<String> {
        let context_window = self.context_window?;
        let usable = context_window.saturating_sub(self.response_reserve);
        let remaining = usable.saturating_sub(prompt_tokens);
        let used_k = f64::from(prompt_tokens) / 1000.0;
        let total_k = f64::from(usable) / 1000.0;
        let remaining_k = f64::from(remaining) / 1000.0;
        Some(format!(
            "[Context: ~{used_k:.0}K/{total_k:.0}K tokens used, ~{remaining_k:.0}K remaining]"
        ))
    }
}

// ---------------------------------------------------------------------------
// Tool output truncation
// ---------------------------------------------------------------------------

/// Truncate a tool output `Value` to at most `max_chars` characters.
///
/// If the serialized JSON exceeds `max_chars` bytes, returns a `Value::String`
/// with the truncated content and a marker indicating truncation.
pub fn truncate_tool_output(output: Value, max_chars: usize) -> Value {
    let serialized = output.to_string();
    if serialized.len() <= max_chars {
        return output;
    }
    // Walk backwards from max_chars to find a valid UTF-8 boundary.
    let mut end = max_chars;
    while end > 0 && !serialized.is_char_boundary(end) {
        end -= 1;
    }
    Value::String(format!(
        "{}... [truncated: showing {} of {} bytes]",
        &serialized[..end],
        end,
        serialized.len()
    ))
}

// ---------------------------------------------------------------------------
// Chat API — message pruning & budget awareness
// ---------------------------------------------------------------------------

/// Prune older tool outputs in a Chat API message history.
///
/// Replaces the content of older `Tool` messages with a placeholder,
/// keeping the most recent `keep_recent_pairs` tool messages intact.
/// System messages, user messages, and assistant messages are never pruned.
pub fn prune_chat_messages(
    messages: &mut Vec<ChatCompletionRequestMessage>,
    keep_recent_pairs: usize,
) {
    let tool_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| {
            if matches!(m, ChatCompletionRequestMessage::Tool(_)) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    if tool_indices.len() <= keep_recent_pairs {
        return;
    }

    let prune_count = tool_indices.len() - keep_recent_pairs;
    for &idx in &tool_indices[..prune_count] {
        if let ChatCompletionRequestMessage::Tool(tool_msg) = &messages[idx] {
            let tool_call_id = tool_msg.tool_call_id.clone();
            if let Ok(pruned) = ChatCompletionRequestToolMessageArgs::default()
                .content("[Previous tool output cleared to save context]")
                .tool_call_id(tool_call_id)
                .build()
            {
                messages[idx] = pruned.into();
            }
        }
    }
}

/// Apply context management to a Chat API message list.
///
/// If pruning is triggered (based on `prompt_tokens` and the config threshold),
/// prunes older tool outputs and optionally injects a budget status message.
pub fn manage_chat_context(
    config: &ContextConfig,
    messages: &mut Vec<ChatCompletionRequestMessage>,
    prompt_tokens: u32,
) {
    if config.should_prune(prompt_tokens) {
        prune_chat_messages(messages, config.keep_recent_tool_pairs);
        tracing::info!(
            prompt_tokens = prompt_tokens,
            context_window = ?config.context_window,
            "Pruned older tool outputs to manage context budget"
        );
    }

    if config.budget_awareness {
        if let Some(status) = config.format_budget_status(prompt_tokens) {
            if let Ok(msg) = ChatCompletionRequestSystemMessageArgs::default()
                .content(status)
                .build()
            {
                messages.push(msg.into());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Responses API — item pruning & budget awareness
// ---------------------------------------------------------------------------

/// Prune older function-call outputs in a Responses API input item list.
///
/// Replaces the output of older `FunctionCallOutput` items with a placeholder,
/// keeping the most recent `keep_recent_pairs` output items intact.
pub fn prune_response_items(items: &mut [InputItem], keep_recent_pairs: usize) {
    let output_indices: Vec<usize> = items
        .iter()
        .enumerate()
        .filter_map(|(i, item)| {
            if matches!(item, InputItem::Item(Item::FunctionCallOutput(_))) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    if output_indices.len() <= keep_recent_pairs {
        return;
    }

    let prune_count = output_indices.len() - keep_recent_pairs;
    for &idx in &output_indices[..prune_count] {
        if let Some(InputItem::Item(Item::FunctionCallOutput(output))) = items.get_mut(idx) {
            output.output = FunctionCallOutput::Text(
                "[Previous tool output cleared to save context]".to_string(),
            );
        }
    }
}

/// Apply context management to a Responses API input item list.
///
/// If pruning is triggered, prunes older function-call outputs and optionally
/// injects a budget-awareness developer message.
pub fn manage_responses_context(
    config: &ContextConfig,
    items: &mut Vec<InputItem>,
    input_tokens: u32,
) {
    if config.should_prune(input_tokens) {
        prune_response_items(items, config.keep_recent_tool_pairs);
        tracing::info!(
            input_tokens = input_tokens,
            context_window = ?config.context_window,
            "Pruned older tool outputs to manage context budget (responses API)"
        );
    }

    if config.budget_awareness {
        if let Some(status) = config.format_budget_status(input_tokens) {
            items.push(InputItem::EasyMessage(EasyInputMessage {
                content: EasyInputContent::Text(status),
                role: Role::Developer,
                r#type: MessageType::Message,
            }));
        }
    }
}

// ---------------------------------------------------------------------------
// Param extraction helper
// ---------------------------------------------------------------------------

/// Extract a [`ContextConfig`] from model parameters.
///
/// Recognised parameter keys (with optional model-prefix):
/// - `context_window` — total context window size in tokens
/// - `context_response_reserve` — tokens reserved for response
/// - `context_prune_threshold` — utilization threshold for pruning
/// - `max_tool_output_chars` — max chars per tool output
/// - `context_budget_awareness` — `"true"` / `"false"`
/// - `context_keep_recent_pairs` — number of recent pairs to keep
pub fn extract_context_config<F>(get: F) -> ContextConfig
where
    F: Fn(&str) -> Option<String>,
{
    let mut config = ContextConfig::default();

    if let Some(v) = get("context_window") {
        if let Ok(w) = v.parse::<u32>() {
            config.context_window = Some(w);
        }
    }
    if let Some(v) = get("context_response_reserve") {
        if let Ok(r) = v.parse::<u32>() {
            config.response_reserve = r;
        }
    }
    if let Some(v) = get("context_prune_threshold") {
        if let Ok(t) = v.parse::<f32>() {
            config.prune_threshold = t;
        }
    }
    if let Some(v) = get("max_tool_output_chars") {
        if let Ok(m) = v.parse::<usize>() {
            config.max_tool_output_chars = m;
        }
    }
    if let Some(v) = get("context_budget_awareness") {
        config.budget_awareness = v == "true" || v == "1";
    }
    if let Some(v) = get("context_keep_recent_pairs") {
        if let Ok(p) = v.parse::<usize>() {
            config.keep_recent_tool_pairs = p;
        }
    }

    config
}

/// Return a default context window size (in tokens) for known model families.
/// Returns `None` for unrecognised models — the caller should leave pruning
/// disabled in that case.
pub fn default_context_window_for_model(model_id: &str) -> Option<u32> {
    let id = model_id.to_lowercase();
    if id.starts_with("gpt-4") || id.starts_with("gpt-5") || id.starts_with("o1") || id.starts_with("o3") || id.starts_with("o4") {
        return Some(128_000);
    }
    if id.starts_with("gpt-3") {
        return Some(16_384);
    }
    if id.starts_with("claude") {
        return Some(200_000);
    }
    if id.starts_with("gemini") {
        return Some(1_000_000);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::chat::{
        ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs,
    };
    use async_openai::types::responses::FunctionCallOutputItemParam;

    #[test]
    fn test_truncate_within_limit() {
        let output = Value::String("short".to_string());
        let result = truncate_tool_output(output.clone(), 100);
        assert_eq!(result, output);
    }

    #[test]
    fn test_truncate_exceeds_limit() {
        let long_text = "a".repeat(200);
        let output = Value::String(long_text);
        let result = truncate_tool_output(output, 50);
        let s = result.as_str().unwrap();
        assert!(s.contains("[truncated:"));
        assert!(s.contains("showing 50 of"));
    }

    #[test]
    fn test_truncate_json_object() {
        let output = serde_json::json!({"key": "a".repeat(60_000)});
        let result = truncate_tool_output(output, 100);
        let s = result.as_str().unwrap();
        assert!(s.contains("[truncated:"));
    }

    #[test]
    fn test_should_prune_below_threshold() {
        let config = ContextConfig {
            context_window: Some(200_000),
            response_reserve: 16_384,
            prune_threshold: 0.75,
            ..Default::default()
        };
        // usable = 200_000 - 16_384 = 183_616
        // threshold tokens = 183_616 * 0.75 ≈ 137_712
        assert!(!config.should_prune(100_000));
    }

    #[test]
    fn test_should_prune_above_threshold() {
        let config = ContextConfig {
            context_window: Some(200_000),
            response_reserve: 16_384,
            prune_threshold: 0.75,
            ..Default::default()
        };
        assert!(config.should_prune(140_000));
    }

    #[test]
    fn test_should_prune_no_window() {
        let config = ContextConfig::default();
        assert!(!config.should_prune(999_999));
    }

    #[test]
    fn test_format_budget_status() {
        let config = ContextConfig {
            context_window: Some(200_000),
            response_reserve: 16_384,
            ..Default::default()
        };
        let status = config.format_budget_status(120_000).unwrap();
        assert!(status.contains("120K"));
        assert!(status.contains("184K")); // 200000 - 16384 ≈ 184K
    }

    #[test]
    fn test_format_budget_status_no_window() {
        let config = ContextConfig::default();
        assert!(config.format_budget_status(120_000).is_none());
    }

    #[test]
    fn test_prune_chat_messages_basic() {
        let mut messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestUserMessageArgs::default()
                .content("hello")
                .build()
                .unwrap()
                .into(),
        ];

        // Add 5 tool messages
        for i in 0..5 {
            messages.push(
                ChatCompletionRequestToolMessageArgs::default()
                    .content(format!("tool output {i} with some data"))
                    .tool_call_id(format!("call_{i}"))
                    .build()
                    .unwrap()
                    .into(),
            );
        }

        assert_eq!(messages.len(), 6);
        prune_chat_messages(&mut messages, 2);
        assert_eq!(messages.len(), 6); // same count, content replaced

        // First message (user) untouched
        assert!(matches!(
            messages[0],
            ChatCompletionRequestMessage::User(_)
        ));

        // All tool messages still present
        for msg in &messages[1..] {
            assert!(matches!(msg, ChatCompletionRequestMessage::Tool(_)));
        }
    }

    #[test]
    fn test_prune_chat_messages_few_tools() {
        let mut messages: Vec<ChatCompletionRequestMessage> = vec![];
        for i in 0..2 {
            messages.push(
                ChatCompletionRequestToolMessageArgs::default()
                    .content(format!("output {i}"))
                    .tool_call_id(format!("call_{i}"))
                    .build()
                    .unwrap()
                    .into(),
            );
        }

        // With keep_recent_pairs=3, 2 tools should NOT be pruned
        prune_chat_messages(&mut messages, 3);
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_prune_response_items_basic() {
        let mut items: Vec<InputItem> = vec![];
        for i in 0..5 {
            items.push(InputItem::Item(Item::FunctionCallOutput(
                FunctionCallOutputItemParam {
                    call_id: format!("call_{i}"),
                    output: FunctionCallOutput::Text(format!("output {i}")),
                    id: Some(format!("id_{i}")),
                    status: None,
                },
            )));
        }

        prune_response_items(&mut items, 2);

        // First 3 should be pruned, last 2 kept
        for item in &items[..3] {
            if let InputItem::Item(Item::FunctionCallOutput(output)) = item {
                assert!(matches!(&output.output, FunctionCallOutput::Text(t) if t.contains("cleared")));
            }
        }
        for item in &items[3..] {
            if let InputItem::Item(Item::FunctionCallOutput(output)) = item {
                assert!(matches!(&output.output, FunctionCallOutput::Text(t) if t.starts_with("output")));
            }
        }
    }

    #[test]
    fn test_extract_context_config_defaults() {
        let config = extract_context_config(|_| None);
        assert!(config.context_window.is_none());
        assert_eq!(config.response_reserve, DEFAULT_RESPONSE_RESERVE);
        assert!((config.prune_threshold - DEFAULT_PRUNE_THRESHOLD).abs() < f32::EPSILON);
        assert_eq!(config.max_tool_output_chars, DEFAULT_MAX_TOOL_OUTPUT_CHARS);
        assert!(config.budget_awareness);
        assert_eq!(config.keep_recent_tool_pairs, DEFAULT_KEEP_RECENT_TOOL_PAIRS);
    }

    #[test]
    fn test_extract_context_config_custom() {
        let config = extract_context_config(|key| match key {
            "context_window" => Some("128000".to_string()),
            "context_prune_threshold" => Some("0.8".to_string()),
            "max_tool_output_chars" => Some("25000".to_string()),
            "context_budget_awareness" => Some("false".to_string()),
            _ => None,
        });
        assert_eq!(config.context_window, Some(128_000));
        assert!((config.prune_threshold - 0.8).abs() < f32::EPSILON);
        assert_eq!(config.max_tool_output_chars, 25_000);
        assert!(!config.budget_awareness);
    }
}
