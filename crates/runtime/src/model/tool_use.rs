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
#![allow(clippy::missing_errors_doc)]
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use std::pin::Pin;
use std::task::{Context, Poll};

use itertools::Itertools;
use llms::chat::nsql::SqlGeneration;
use llms::chat::{Chat, Result as ChatResult};
use llms::streaming_utils::{create_stream_choice, create_stream_response, generate_stream_id};

use async_openai::error::OpenAIError;
use async_openai::types::chat::{
    ChatChoiceStream, ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestToolMessageArgs, ChatCompletionRequestToolMessageContent,
    ChatCompletionResponseStream, ChatCompletionTool,
    ChatCompletionToolChoiceOption, ChatCompletionTools, CompletionTokensDetails, CompletionUsage,
    CreateChatCompletionRequest, CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse, FinishReason, FunctionCall, FunctionObject,
    PromptTokensDetails, Role, ToolChoiceOptions,
};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use pin_project::pin_project;
use serde_json::Value;

use tokio::sync::mpsc;
use tools::SpiceModelTool;
use tracing::{Instrument, Span};

use async_openai::types::chat::{
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessageContent,
};

use crate::Runtime;
use crate::model::{ModelContextExtension, SingleShotExtension};
use crate::model::context::{ContextConfig, manage_chat_context, truncate_tool_output};
use crate::tools::builtin::plan_mode::{
    EnterPlanModeTool, ExitPlanModeTool, ENTER_PLAN_MODE_TOOL_NAME, EXIT_PLAN_MODE_TOOL_NAME,
};
use llms::progress::Progress;
use runtime_request_context::{AsyncMarker, RequestContext};

/// Get the current git state (branch + last 3 commits) as a context string.
/// Returns empty string on any error (not a git repo, git not available, etc.).
pub(super) async fn get_git_state_context() -> String {
    let branch = tokio::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    let Some(branch) = branch else {
        return String::new();
    };

    let log = tokio::process::Command::new("git")
        .args(["log", "--oneline", "-3"])
        .output()
        .await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    let commits: String = log
        .lines()
        .map(|l| format!("  - {l}"))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "\n\n[Current Git State]\nBranch: {branch}\nLast 3 commits:\n{commits}"
    )
}

/// Append git state context to the last user message in the request.
fn append_git_state_to_request(
    req: &mut CreateChatCompletionRequest,
    git_state: &str,
) {
    if git_state.is_empty() {
        return;
    }

    // Find the last user message and append the git state
    for msg in req.messages.iter_mut().rev() {
        if let ChatCompletionRequestMessage::User(user_msg) = msg {
            match &mut user_msg.content {
                ChatCompletionRequestUserMessageContent::Text(text) => {
                    text.push_str(git_state);
                    return;
                }
                ChatCompletionRequestUserMessageContent::Array(_) => {
                    // For array content (multimodal), skip — don't modify
                    return;
                }
            }
        }
    }
}

pub struct ToolUsingChat {
    inner_chat: Arc<dyn Chat>,
    rt: Arc<Runtime>,
    tools: Vec<Arc<dyn SpiceModelTool>>,
    recursion_limit: Option<usize>,
    context_config: ContextConfig,
    /// Whether the tool-calling loop starts in plan mode.
    /// In plan mode, all tools are available but the model is instructed
    /// via the system prompt to only use them for read-only purposes.
    /// Plan mode is toggled by `enter_plan_mode` / `exit_plan_mode` tools.
    plan_mode: bool,
}

impl ToolUsingChat {
    #[must_use]
    pub fn new(
        inner_chat: Arc<dyn Chat>,
        rt: Arc<Runtime>,
        tools: Vec<Arc<dyn SpiceModelTool>>,
        recursion_limit: Option<usize>,
        context_config: ContextConfig,
    ) -> Self {
        let mut tools = tools;
        tools.push(Arc::new(EnterPlanModeTool::new(None, None)));
        tools.push(Arc::new(ExitPlanModeTool::new(None, None)));
        Self {
            inner_chat,
            rt,
            tools,
            recursion_limit,
            context_config,
            plan_mode: false,
        }
    }

    /// Enable plan mode for this `ToolUsingChat` instance.
    ///
    /// In plan mode, all tools remain available but the model is instructed
    /// to only use them for read-only operations. When `exit_plan_mode` is
    /// called, the approval tool (if configured) is auto-invoked. If approved
    /// (or no approval tool), plan mode is deactivated and the model proceeds.
    #[must_use]
    pub fn with_plan_mode(mut self, plan_mode: bool) -> Self {
        self.plan_mode = plan_mode;
        self
    }

    fn new_for_recursion(
        inner_chat: Arc<dyn Chat>,
        rt: Arc<Runtime>,
        tools: Vec<Arc<dyn SpiceModelTool>>,
        recursion_limit: Option<usize>,
        context_config: ContextConfig,
        plan_mode: bool,
    ) -> Self {
        Self {
            inner_chat,
            rt,
            tools,
            recursion_limit,
            context_config,
            plan_mode,
        }
    }

    #[must_use]
    pub fn runtime_tools(&self) -> Vec<ChatCompletionTool> {
        self.tools
            .iter()
            .map(|t| ChatCompletionTool {
                function: FunctionObject {
                    strict: t.strict(),
                    name: encode_tool_name(t.name().to_string().as_str()),
                    description: t.description().map(|d| d.to_string()),
                    parameters: t.parameters(),
                },
            })
            .collect_vec()
    }

    /// Create a new [`CreateChatCompletionRequest`] with the system prompt injected as the first message.
    async fn prepare_req(
        &self,
        mut req: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionRequest, OpenAIError> {
        if let Some(list_datasets) = self.tools.iter().find(|t| t.name() == "list_datasets") {
            let list_dataset_messages = self.create_list_dataset_messages(list_datasets).await?;
            req.messages =
                insert_initial_tools(req.messages, "list_datasets", &list_dataset_messages);
        }

        Ok(req)
    }

    /// Create the messagges expected from a model if it has called the `list_datasets` tool, and recieved a response.
    /// This is useful to prime the model as if it has already asked to list the available datasets.
    async fn create_list_dataset_messages(
        &self,
        list_datasets: &Arc<dyn SpiceModelTool>,
    ) -> Result<Vec<ChatCompletionRequestMessage>, OpenAIError> {
        let t_resp = list_datasets
            .call("")
            .await
            .map_err(|e| OpenAIError::InvalidArgument(e.to_string()))?;
        Ok(vec![
            ChatCompletionRequestAssistantMessageArgs::default()
                .tool_calls(vec![ChatCompletionMessageToolCalls::Function(
                    ChatCompletionMessageToolCall {
                        id: "initial_list_datasets".to_string(),
                        function: FunctionCall {
                            name: list_datasets.name().to_string(),
                            arguments: String::new(),
                        },
                    },
                )])
                .build()?
                .into(),
            ChatCompletionRequestToolMessageArgs::default()
                .content(t_resp.to_string())
                .tool_call_id("initial_list_datasets".to_string())
                .build()?
                .into(),
        ])
    }

    /// Check if a tool call is a spiced runtime tool.
    fn as_spiced_tool(&self, t: &ChatCompletionMessageToolCall) -> Option<Arc<dyn SpiceModelTool>> {
        self.tools
            .iter()
            .find(|tool| encode_tool_name(tool.name().as_ref()) == t.function.name)
            .cloned()
    }

    /// Call a spiced runtime tool.
    ///
    /// Return the result as a JSON value.
    async fn call_tool(&self, tool_call: &ChatCompletionMessageToolCall) -> Value {
        match self.as_spiced_tool(tool_call) {
            Some(t) => match t.call(&tool_call.function.arguments).await {
                Ok(v) => {
                    let v = truncate_tool_output(v, self.context_config.max_tool_output_chars);
                    tracing::debug!(
                        target: "task_history",
                        progress = Progress::log()
                            .id(Some(tool_call.id.clone()))
                            .title(format!("'{}' tool completed successfully", tool_call.function.name))
                            .json_content(v.clone())
                            .to_jsonl(),
                    );
                    v
                }
                Err(e) => {
                    tracing::debug!(
                        target: "task_history",
                        progress = Progress::error()
                            .id(Some(tool_call.id.clone()))
                            .title(format!("'{}' tool completed unsuccessfully", tool_call.function.name))
                            .content(e.to_string())
                            .to_jsonl(),
                    );
                    Value::String(format!(
                        "Failed to call the tool {}.\nAn error occurred: {e}",
                        t.name()
                    ))
                }
            },
            None => {
                // All calls to `call_tool` should have previously checked that `tool_call` has an associated tool.
                if cfg!(feature = "dev") {
                    panic!(
                        "Tool '{}' was provided to LLM, but now no longer exists. This should not be possible.",
                        tool_call.function.name
                    );
                } else {
                    tracing::warn!(
                        "Tool '{}' was provided to LLM, but now no longer exists. This should not be possible.",
                        tool_call.function.name
                    );
                    Value::Null
                }
            }
        }
    }

    /// For `requested_tools` requested from processing `original_messages` through a model, check
    /// if any are spiced runtime tools, and if so, run them locally and create new messages to be
    ///  reprocessed by the model.
    ///
    /// Returns
    /// - `None` if no spiced runtime tools were used. Note: external tools may still have been
    ///   requested.
    /// - `Some(messages)` if spiced runtime tools were used. The returned messages are ready to be
    ///   reprocessed by the model.
    async fn process_tool_calls_and_run_spice_tools(
        &self,
        original_messages: Vec<ChatCompletionRequestMessage>,
        requested_tools: Vec<ChatCompletionMessageToolCall>,
    ) -> Result<Option<Vec<ChatCompletionRequestMessage>>, OpenAIError> {
        let spiced_tools = requested_tools
            .iter()
            .filter(|&t| self.as_spiced_tool(t).is_some())
            .cloned()
            .collect_vec();

        tracing::trace!(
            "spiced_tools available: {:?}. Used {:?}",
            self.tools.iter().map(|t| t.name()).collect_vec(),
            spiced_tools
        );

        // Return early if no spiced runtime tools used.
        if spiced_tools.is_empty() {
            tracing::trace!("No spiced tools used by chat model, returning early");
            return Ok(None);
        }

        // Tell model the assistant has these tools
        let assistant_message: ChatCompletionRequestMessage =
            ChatCompletionRequestAssistantMessageArgs::default()
                .tool_calls(
                    spiced_tools
                        .iter()
                        .map(|t| ChatCompletionMessageToolCalls::Function(t.clone()))
                        .collect::<Vec<_>>(),
                ) // TODO - should this include non-spiced tools?
                .build()?
                .into();

        let mut tool_and_response_content = vec![];
        for t in spiced_tools.clone() {
            tracing::debug!(
                target: "task_history",
                progress = Progress::log()
                    .id(Some(t.id.clone()))
                    .title(format!("Calling '{}' tool", t.function.name))
                    .content(t.function.arguments.clone())
                    .to_jsonl(),
            );
            tracing::trace!(tool = %t.function.name, args = %t.function.arguments, "Calling tool");

            let content = self.call_tool(&t).await;
            let result_preview = match &content {
                Value::String(s) => s.chars().take(500).collect::<String>(),
                other => other.to_string().chars().take(500).collect::<String>(),
            };
            tracing::trace!(tool = %t.function.name, result_chars = result_preview.len(), result = %result_preview, "Tool returned");
            tool_and_response_content.push((t, content));
        }

        let total_result_chars: usize = tool_and_response_content
            .iter()
            .map(|(_, v)| match v {
                Value::String(s) => s.len(),
                other => other.to_string().len(),
            })
            .sum();
        tracing::trace!(
            tool_count = tool_and_response_content.len(),
            total_result_chars,
            "Sending tool results back to model"
        );

        // Tell model the assistant used these tools, and provided result.
        let tool_messages: Vec<ChatCompletionRequestMessage> = tool_and_response_content
            .iter()
            .map(|(tool_call, response_content)| {
                Ok(ChatCompletionRequestToolMessageArgs::default()
                    .content(match response_content {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .tool_call_id(tool_call.id.clone())
                    .build()?
                    .into())
            })
            .collect::<Result<_, OpenAIError>>()?;

        let mut messages = original_messages.clone();
        messages.push(assistant_message);
        messages.extend(tool_messages);

        if !messages.is_empty() {
            let used_tools = spiced_tools.len();
            if used_tools > 0 {
                let context = RequestContext::current(AsyncMarker::new().await);
                crate::model::add_tools_used(&context, used_tools);
            }
        }

        Ok(Some(messages))
    }

    async fn chat_request_inner(
        &self,
        req: CreateChatCompletionRequest,
        recursion_limit: Option<usize>,
        recent_tool_fingerprints: Vec<u64>,
        plan_mode: bool,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        // Don't use spice runtime tools if users has explicitly chosen to not use any tools.
        if req.tool_choice.as_ref().is_some_and(|c| {
            *c == ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::None)
        }) {
            tracing::trace!("User asked for no tools, calling inner chat model");
            return self.inner_chat.chat_request(req).await;
        }

        let mut current_req = req;
        let mut remaining = recursion_limit;
        let mut fingerprints = recent_tool_fingerprints;
        let mut accumulated_usage: Option<CompletionUsage> = None;
        let mut current_plan_mode = plan_mode;

        loop {
            if remaining.is_some_and(|f| f == 0) {
                tracing::warn!(
                    "Tool-use iteration limit reached. Will call model, but not process further tool calls."
                );
                let mut inner_req = self.add_runtime_tools(&current_req);
                let git_state = get_git_state_context().await;
                append_git_state_to_request(&mut inner_req, &git_state);
                let mut resp = self.inner_chat.chat_request(inner_req).await?;
                resp.usage = combine_usage(accumulated_usage, resp.usage);
                return Ok(resp);
            }

            // --- Request log ---
            let last_msg_preview: String = current_req.messages.last()
                .map(|m| match m {
                    ChatCompletionRequestMessage::Tool(t) => {
                        let content = serde_json::to_string(&t.content).unwrap_or_default();
                        format!("[tool:{}] {}", t.tool_call_id, content.chars().take(150).collect::<String>())
                    },
                    ChatCompletionRequestMessage::Assistant(a) => {
                        if let Some(ref tc) = a.tool_calls {
                            let names: Vec<String> = tc.iter().map(|t| match t {
                                ChatCompletionMessageToolCalls::Function(f) => f.function.name.clone(),
                                ChatCompletionMessageToolCalls::Custom(_) => "custom".into(),
                            }).collect();
                            format!("[assistant] tool_calls: {}", names.join(", "))
                        } else {
                            let content = a.content.as_ref().map(|c| serde_json::to_string(c).unwrap_or_default()).unwrap_or_default();
                            format!("[assistant] {}", content.chars().take(150).collect::<String>())
                        }
                    },
                    ChatCompletionRequestMessage::System(s) => match &s.content {
                        ChatCompletionRequestSystemMessageContent::Text(t) => format!("[system] {}", t.chars().take(150).collect::<String>()),
                        _ => "[system]".into(),
                    },
                    ChatCompletionRequestMessage::User(u) => match &u.content {
                        ChatCompletionRequestUserMessageContent::Text(t) => format!("[user] {}", t.chars().take(150).collect::<String>()),
                        _ => "[user]".into(),
                    },
                    _ => "[other]".into(),
                })
                .unwrap_or_default();
            let acc_tokens = accumulated_usage.as_ref().map_or(0, |u| u.total_tokens);
            tracing::debug!(
                last_message = %last_msg_preview,
                iterations_remaining = remaining,
                message_count = current_req.messages.len(),
                accumulated_tokens = acc_tokens,
                "Tool loop → request"
            );

            // Append spiced runtime tools to the request.
            let mut inner_req = self.add_runtime_tools(&current_req);

            // Inject current git state into the last user message
            let git_state = get_git_state_context().await;
            append_git_state_to_request(&mut inner_req, &git_state);

            let resp = match self.inner_chat.chat_request(inner_req.clone()).await {
                Ok(resp) => resp,
                Err(e) => {
                    tracing::warn!(error = %e, "Model API request failed during tool-use loop");
                    return Err(e);
                }
            };

            // Extract tool calls from response
            let tools_used = resp
                .choices
                .first()
                .and_then(|c| c.message.tool_calls.clone());

            let tool_calls: Vec<ChatCompletionMessageToolCall> = tools_used
                .unwrap_or_default()
                .iter()
                .filter_map(|tc| match tc {
                    ChatCompletionMessageToolCalls::Function(call) => Some(call.clone()),
                    ChatCompletionMessageToolCalls::Custom(_) => None,
                })
                .collect();

            // --- Response log ---
            let content_preview: String = resp.choices.first()
                .and_then(|c| c.message.content.as_ref())
                .map(|c| c.chars().take(200).collect::<String>())
                .unwrap_or_default();
            let tool_names: Vec<String> = tool_calls.iter()
                .map(|tc| decode_tool_name(&tc.function.name))
                .collect();
            let resp_usage = resp.usage.as_ref();
            let prompt_tok = resp_usage.map_or(0, |u| u.prompt_tokens);
            let context_window = self.context_config.context_window.unwrap_or(0);
            let context_budget = if context_window > 0 {
                format!("~{}K/{}K tokens used", prompt_tok / 1000, context_window / 1000)
            } else {
                format!("~{}K tokens used", prompt_tok / 1000)
            };
            tracing::info!(
                tools = ?tool_names,
                content_preview = %content_preview,
                iterations_remaining = remaining,
                context = %context_budget,
                completion_tokens = resp_usage.map_or(0, |u| u.completion_tokens),
                tool_calls = tool_calls.len(),
                "Tool loop ← response"
            );

            // Compute fingerprint for loop detection.
            let fingerprint = chat_tool_calls_fingerprint(&tool_calls);
            fingerprints.push(fingerprint);

            // Intercept plan mode tools before normal processing.
            let mut next_plan_mode = current_plan_mode;
            let mut plan_mode_messages: Vec<ChatCompletionRequestMessage> = Vec::new();
            let mut remaining_tool_calls: Vec<ChatCompletionMessageToolCall> = Vec::new();

            for tc in &tool_calls {
                let decoded_name = decode_tool_name(&tc.function.name);
                if decoded_name == ENTER_PLAN_MODE_TOOL_NAME {
                    next_plan_mode = true;
                    plan_mode_messages.push(
                        ChatCompletionRequestToolMessageArgs::default()
                            .content("Plan mode activated. You have access to all tools but MUST only use them for read-only purposes. Research and gather information, then call exit_plan_mode with your plan.")
                            .tool_call_id(tc.id.clone())
                            .build()?
                            .into(),
                    );
                } else if decoded_name == EXIT_PLAN_MODE_TOOL_NAME {
                    let plan_text = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                        .ok()
                        .and_then(|v| v.get("plan").and_then(|p| p.as_str()).map(String::from))
                        .unwrap_or_default();

                    if !plan_text.is_empty() {
                        tracing::debug!(target: "task_history", plan = %plan_text, "Agent plan submitted");
                    }

                    // Auto-call approval tool if one exists
                    let approval_tool = self.tools.iter().find(|t| {
                        let n = t.name();
                        n == "approval" || n == "approval_slack" || n == "approval_ms_teams"
                    });

                    let tool_response = if let Some(approval) = approval_tool {
                        let approval_arg = serde_json::json!({
                            "message": format!("Agent plan requires approval:\n\n{plan_text}"),
                            "context": plan_text,
                        }).to_string();

                        tracing::debug!(target: "task_history", "Plan submitted for approval, waiting...");

                        match approval.call(&approval_arg).await {
                            Ok(result) => {
                                let approved = result.get("approved")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false);
                                let comment = result.get("comment")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");

                                if approved {
                                    next_plan_mode = false;
                                    let suffix = if comment.is_empty() { String::new() } else { format!(": {comment}") };
                                    format!("Plan approved{suffix}. You may now proceed with execution.")
                                } else {
                                    next_plan_mode = true;
                                    format!("Plan rejected: {comment}. Revise your plan and call exit_plan_mode again.")
                                }
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "Approval tool failed, proceeding without approval");
                                next_plan_mode = false;
                                format!("Approval tool error ({e}). Proceeding with plan execution.")
                            }
                        }
                    } else {
                        next_plan_mode = false;
                        "Plan mode deactivated. All tools are now available. You may proceed with your plan.".to_string()
                    };

                    plan_mode_messages.push(
                        ChatCompletionRequestToolMessageArgs::default()
                            .content(tool_response)
                            .tool_call_id(tc.id.clone())
                            .build()?
                            .into(),
                    );
                } else {
                    remaining_tool_calls.push(tc.clone());
                }
            }

            // If plan mode tools were intercepted, we need to handle messages manually
            let has_plan_mode_tools = !plan_mode_messages.is_empty();

            let new_messages = match self
                .process_tool_calls_and_run_spice_tools(current_req.messages.clone(), remaining_tool_calls)
                .await?
            {
                Some(mut messages) => {
                    if has_plan_mode_tools {
                        messages = vec![
                            ChatCompletionRequestAssistantMessageArgs::default()
                                .tool_calls(
                                    tool_calls
                                        .iter()
                                        .map(|t| ChatCompletionMessageToolCalls::Function(t.clone()))
                                        .collect::<Vec<_>>(),
                                )
                                .build()?
                                .into(),
                        ];
                        messages.extend(plan_mode_messages);
                        for tc in &tool_calls {
                            let decoded = decode_tool_name(&tc.function.name);
                            if decoded != ENTER_PLAN_MODE_TOOL_NAME && decoded != EXIT_PLAN_MODE_TOOL_NAME {
                                let content = self.call_tool(tc).await;
                                messages.push(
                                    ChatCompletionRequestToolMessageArgs::default()
                                        .content(content.to_string())
                                        .tool_call_id(tc.id.clone())
                                        .build()?
                                        .into(),
                                );
                            }
                        }
                    }
                    Some(messages)
                }
                None if has_plan_mode_tools => {
                    let mut messages = vec![
                        ChatCompletionRequestAssistantMessageArgs::default()
                            .tool_calls(
                                tool_calls
                                    .iter()
                                    .map(|t| ChatCompletionMessageToolCalls::Function(t.clone()))
                                    .collect::<Vec<_>>(),
                            )
                            .build()?
                            .into(),
                    ];
                    messages.extend(plan_mode_messages);
                    Some(messages)
                }
                None => None,
            };

            match new_messages {
                Some(mut messages) => {
                    // Detect repeated identical tool calls (3 consecutive identical fingerprints).
                    if is_tool_loop_detected(&fingerprints) {
                        tracing::warn!("Tool-use loop detected: identical tool calls repeated 3 times");
                        messages.push(
                            ChatCompletionRequestSystemMessageArgs::default()
                                .content(LOOP_DETECTION_MESSAGE)
                                .build()?
                                .into(),
                        );
                    }

                    // Prepare next iteration (no recursion)
                    let usage_ref = resp.usage.clone();
                    accumulated_usage = combine_usage(accumulated_usage, resp.usage);
                    current_req = create_new_recursive_req(
                        &inner_req,
                        messages,
                        usage_ref.as_ref(),
                        &self.context_config,
                    );
                    remaining = remaining.map(|r| r - 1);
                    current_plan_mode = next_plan_mode;
                    // continue loop
                }
                None => {
                    // No tool calls to process — return final response.
                    let mut resp = resp;
                    resp.usage = combine_usage(accumulated_usage, resp.usage);
                    return Ok(resp);
                }
            }
        }
    }

    /// Add the spice runtime tools to a list of tools (may contain external tools too), and ensure no duplicates.
    fn add_runtime_tools(&self, req: &CreateChatCompletionRequest) -> CreateChatCompletionRequest {
        let mut runtime_tools = self.runtime_tools();
        if let Some(ref request_tools) = req.tools {
            runtime_tools.extend(request_tools.iter().filter_map(|t| match t {
                ChatCompletionTools::Function(f) => Some(f.clone()),
                ChatCompletionTools::Custom(_) => None,
            }));
        }
        // Ensure function names are unique. Tool-use recursion sometimes creates duplicates.
        runtime_tools.sort_by(|a, b| a.function.name.cmp(&b.function.name));
        runtime_tools.dedup_by(|a, b| a.function.name == b.function.name);
        let mut req = req.clone();
        req.tools = Some(
            runtime_tools
                .into_iter()
                .map(ChatCompletionTools::Function)
                .collect(),
        );
        req
    }

    async fn chat_stream_inner(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        // Don't use spice runtime tools if users has explicitly chosen to not use any tools.
        if req
            .tool_choice
            .as_ref()
            .is_some_and(|c| *c == ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::None))
        {
            return self.inner_chat.chat_stream(req).await;
        }

        if self.recursion_limit.is_some_and(|f| f == 0) {
            tracing::warn!(
                "Tool-use recursion limit reached. Will call model, but not process further tool calls."
            );
            let mut updated_req = self.add_runtime_tools(&req);
            let git_state = get_git_state_context().await;
            append_git_state_to_request(&mut updated_req, &git_state);
            return self.inner_chat.chat_stream(updated_req).await;
        }

        tracing::debug!(recursion_remaining = ?self.recursion_limit, "Calling model (streaming)");

        // Append spiced runtime tools to the request. Avoid clone if no runtime tools.
        let mut updated_req = self.add_runtime_tools(&req);
        let git_state = get_git_state_context().await;
        append_git_state_to_request(&mut updated_req, &git_state);
        let s = self.inner_chat.chat_stream(updated_req.clone()).await?;

        Ok(make_a_stream(
            Span::current(),
            RequestContext::current(AsyncMarker::new().await),
            Self::new_for_recursion(
                Arc::clone(&self.inner_chat),
                Arc::clone(&self.rt),
                self.tools.clone(),
                self.recursion_limit.map(|r| r - 1),
                self.context_config.clone(),
                self.plan_mode,
            ),
            req,
            s,
        ))
    }
}

#[async_trait]
impl Chat for ToolUsingChat {
    async fn run(&self, prompt: String) -> ChatResult<Option<String>> {
        self.inner_chat.run(prompt).await
    }

    async fn stream<'a>(
        &self,
        prompt: String,
    ) -> ChatResult<Pin<Box<dyn Stream<Item = ChatResult<Option<String>>> + Send>>> {
        self.inner_chat.stream(prompt).await
    }

    async fn chat_stream(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        let context = RequestContext::current(AsyncMarker::new().await);
        if context.extension::<ModelContextExtension>().is_none() {
            context.insert_extension(ModelContextExtension::new());
        }

        // Single-shot mode: skip tool execution, return raw model response.
        let recursion_limit = if context.extension::<SingleShotExtension>().is_some() {
            Some(0)
        } else {
            self.recursion_limit
        };

        let inner_req = self.prepare_req(req).await?;

        let session = Self::new_for_recursion(
            Arc::clone(&self.inner_chat),
            Arc::clone(&self.rt),
            self.tools.clone(),
            recursion_limit,
            self.context_config.clone(),
            self.plan_mode,
        );

        // wrap the completion stream to track the `ai_inferences_with_spice_count` when it is ready.
        let stream = session.chat_stream_inner(inner_req).await?;
        Ok(Box::pin(InferenceTrackingStream::new(stream, context)))
    }

    async fn chat_request(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        let context = RequestContext::current(AsyncMarker::new().await);
        if context.extension::<ModelContextExtension>().is_none() {
            context.insert_extension(ModelContextExtension::new());
        }

        // Single-shot mode: skip tool execution, return raw model response.
        let recursion_limit = if context.extension::<SingleShotExtension>().is_some() {
            Some(0)
        } else {
            self.recursion_limit
        };

        let inner_req = self.prepare_req(req).await?;

        let session = Self::new_for_recursion(
            Arc::clone(&self.inner_chat),
            Arc::clone(&self.rt),
            self.tools.clone(),
            recursion_limit,
            self.context_config.clone(),
            self.plan_mode,
        );

        let response = session
            .chat_request_inner(inner_req, recursion_limit, vec![], self.plan_mode)
            .await;

        // track ai_inferences_with_spice_count metric
        crate::model::track_ai_inferences_with_spice_count(&context);

        response
    }

    fn as_sql(&self) -> Option<&dyn SqlGeneration> {
        self.inner_chat.as_sql()
    }

    /// Override health endpoint to 1. avoid passing tools in request, 2. pre-calling `list_datasets` in [`ToolUsingChat::prepare_req`].
    async fn health(&self) -> ChatResult<()> {
        self.inner_chat.health().await
    }
}

/// Create a new [`CreateChatCompletionRequest`] with new messages.
///
/// Remove `tool_choice` if it is named (since it was just used), and set it to `Auto`.
/// Applies context management (pruning + budget awareness) when usage data is available.
fn create_new_recursive_req(
    req: &CreateChatCompletionRequest,
    mut new_msg: Vec<ChatCompletionRequestMessage>,
    marginal_usage: Option<&CompletionUsage>,
    context_config: &ContextConfig,
) -> CreateChatCompletionRequest {
    let mut new_req = req.clone();

    // Context management: prune old tool outputs and inject budget status if needed.
    if let Some(usage) = marginal_usage {
        tracing::trace!(
            prompt_tokens = usage.prompt_tokens,
            context_window = ?context_config.context_window,
            message_count = new_msg.len(),
            should_prune = context_config.should_prune(usage.prompt_tokens),
            "Creating recursive request with context management"
        );
        manage_chat_context(context_config, &mut new_msg, usage.prompt_tokens);
    }

    // Append ephemeral guidance to the last tool result message so the model
    // sees it right before generating its next response. Strip any previous
    // copy first so it never accumulates across iterations.
    strip_ephemeral_guidance(&mut new_msg);
    append_ephemeral_guidance(&mut new_msg);

    new_req.messages = new_msg;

    // Remove tool_choice if it is named (since it was just used), and set it to `Auto`.
    // This also includes when a tool_choice is not set. It could be set as a default (in spicepod.yaml via openai_tool_choice), but will appear as None here. We want to set it to Auto here to ensure named tool is used once and does not cause infinite tool use.
    if matches!(
        new_req.tool_choice,
        Some(ChatCompletionToolChoiceOption::Function(_)) | None
    ) {
        // Auto is default when tools exist.
        tracing::trace!("Not recursively using named tool_choice in subsequent calls.");
        new_req.tool_choice = Some(ChatCompletionToolChoiceOption::Mode(
            ToolChoiceOptions::Auto,
        ));
    }

    // Adjust input `max_completion_tokens` if usage is known to ensure we don't exceed the limit.
    if let Some(max_completion_tokens) = new_req.max_completion_tokens
        && let Some(usage) = marginal_usage
    {
        new_req.max_completion_tokens =
            Some(max_completion_tokens.saturating_sub(usage.completion_tokens));
    }

    new_req
}

pub fn combine_usage(
    u1: Option<CompletionUsage>,
    u2: Option<CompletionUsage>,
) -> Option<CompletionUsage> {
    match (u1, u2) {
        (Some(u1), Some(u2)) => Some(CompletionUsage {
            prompt_tokens: u1.prompt_tokens + u2.prompt_tokens,
            completion_tokens: u1.completion_tokens + u2.completion_tokens,
            total_tokens: u1.total_tokens + u2.total_tokens,
            prompt_tokens_details: combine_token_details(
                u1.prompt_tokens_details,
                u2.prompt_tokens_details,
            ),
            completion_tokens_details: combine_completion_details(
                u1.completion_tokens_details,
                u2.completion_tokens_details,
            ),
        }),
        (Some(u1), None) => Some(u1),
        (None, Some(u2)) => Some(u2),
        (None, None) => None,
    }
}
fn combine_token_details(
    a: Option<PromptTokensDetails>,
    b: Option<PromptTokensDetails>,
) -> Option<PromptTokensDetails> {
    match (a, b) {
        (Some(a), Some(b)) => Some(PromptTokensDetails {
            audio_tokens: combine_opt_u32(a.audio_tokens, b.audio_tokens),
            cached_tokens: combine_opt_u32(a.cached_tokens, b.cached_tokens),
        }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn combine_completion_details(
    a: Option<CompletionTokensDetails>,
    b: Option<CompletionTokensDetails>,
) -> Option<CompletionTokensDetails> {
    match (a, b) {
        (Some(a), Some(b)) => Some(CompletionTokensDetails {
            accepted_prediction_tokens: combine_opt_u32(
                a.accepted_prediction_tokens,
                b.accepted_prediction_tokens,
            ),
            audio_tokens: combine_opt_u32(a.audio_tokens, b.audio_tokens),
            reasoning_tokens: combine_opt_u32(a.reasoning_tokens, b.reasoning_tokens),
            rejected_prediction_tokens: combine_opt_u32(
                a.rejected_prediction_tokens,
                b.rejected_prediction_tokens,
            ),
        }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}
pub fn combine_opt_u32(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a + b),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

// Ensure that `tool_messages` have been added to `messages` after all initial developer/system messages and after initial user messages (i.e. not including user messages after assistant messages).
fn insert_initial_tools(
    messages: Vec<ChatCompletionRequestMessage>,
    tool_name: &str,
    tool_messages: &[ChatCompletionRequestMessage],
) -> Vec<ChatCompletionRequestMessage> {
    // Do not add `tool_messages` if already in `messages`.
    if messages.iter().any(|m| {
        let ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
            tool_calls: Some(tools),
            ..
        }) = m
        else {
            return false;
        };
        tools.iter().any(|t| match t {
            ChatCompletionMessageToolCalls::Function(call) => call.function.name == tool_name,
            ChatCompletionMessageToolCalls::Custom(_) => false,
        })
    }) {
        return messages;
    }

    // Find index to insert at
    let idx = messages
        .iter()
        .enumerate()
        .find_map(|(i, m)| {
            if matches!(
                m,
                ChatCompletionRequestMessage::Assistant(_)
                    | ChatCompletionRequestMessage::Tool(_)
                    | ChatCompletionRequestMessage::Function(_)
            ) {
                return Some(i);
            }
            None
        })
        .unwrap_or(messages.len());

    let Some((a, b)) = messages.split_at_checked(idx) else {
        return messages;
    };

    [a, tool_messages, b].concat()
}

struct CustomStream {
    receiver: mpsc::Receiver<Result<CreateChatCompletionStreamResponse, OpenAIError>>,
}

impl Stream for CustomStream {
    type Item = Result<CreateChatCompletionStreamResponse, OpenAIError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

fn make_a_stream(
    span: Span,
    request_context: Arc<RequestContext>,
    model: ToolUsingChat,
    req: CreateChatCompletionRequest,
    mut s: ChatCompletionResponseStream,
) -> ChatCompletionResponseStream {
    let (sender, receiver) = mpsc::channel(100);
    let sender_clone = sender;

    tokio::spawn(
        request_context
            .scope(async move {
                let tool_call_states: Arc<
                    Mutex<HashMap<(i32, i32), ChatCompletionMessageToolCall>>,
                > = Arc::new(Mutex::new(HashMap::new()));

                let mut chat_output = String::new();
                let stream_id = generate_stream_id(&req.model);

                while let Some(result) = s.next().await {
                    let response = match result {
                        Ok(response) => response,
                        Err(e) => {
                            tracing::error!("Error in chat completion stream: {e}");
                            let error_msg = format!("An error occurred: {e}");
                            if let Ok(error_resp) = create_stream_response(
                                &stream_id,
                                &req.model,
                                vec![create_stream_choice(0, Some(error_msg), Some(Role::Assistant), Some(FinishReason::Stop))],
                                None,
                            ) {
                                let _ = sender_clone.send(Ok(error_resp)).await;
                            }
                            return;
                        }
                    };
                    let mut finished_choices: Vec<ChatChoiceStream> = vec![];
                    for chat_choice1 in &response.choices {
                        let chat_choice = chat_choice1.clone();

                        // Appending the tool call chunks
                        // TODO: only concatenate, spiced tools
                        if let Some(ref tool_calls) = chat_choice.delta.tool_calls {
                            for tool_call_chunk in tool_calls {
                                let key: (i32, i32) = if let (Ok(index), Ok(tool_call_index)) = (chat_choice.index.try_into(), tool_call_chunk.index.try_into()) { (index, tool_call_index) } else {
                                    tracing::error!(
                                        "chat_choice.index value {} or tool_call_chunk.index value {} is too large to fit in an i32",
                                        chat_choice.index,
                                        tool_call_chunk.index
                                    );
                                    return;
                                };

                                let states = Arc::clone(&tool_call_states);
                                let tool_call_data = tool_call_chunk.clone();

                                let mut states_lock = match states.lock() {
                                    Ok(lock) => lock,
                                    Err(e) => {
                                        tracing::error!("Failed to lock tool_call_states: {}", e);
                                        return;
                                    }
                                };

                                let state = states_lock.entry(key).or_insert_with(|| {
                                    ChatCompletionMessageToolCall {
                                        id: tool_call_data.id.clone().unwrap_or_default(),
                                        function: FunctionCall {
                                            name: tool_call_data
                                                .function
                                                .as_ref()
                                                .and_then(|f| f.name.clone())
                                                .unwrap_or_default(),
                                            arguments: String::new(),
                                        },
                                    }
                                });

                                if let Some(arguments) = tool_call_chunk
                                    .function
                                    .as_ref()
                                    .and_then(|f| f.arguments.as_ref())
                                {
                                    state.function.arguments.push_str(arguments);
                                }
                            }
                        }
                        if chat_choice.delta.content.is_some() {
                            finished_choices.push(chat_choice.clone());
                        }

                        // If a tool has finished (i.e. we have all chunks), process them.
                        if let Some(finish_reason) = &chat_choice.finish_reason {
                            if matches!(finish_reason, FinishReason::ToolCalls) {
                                let tool_call_states_clone = Arc::clone(&tool_call_states);

                                let tool_calls_to_process = {
                                    match tool_call_states_clone.lock() {
                                        Ok(states_lock) => states_lock
                                            .values()
                                            .cloned()
                                            .collect(),
                                        Err(e) => {
                                            tracing::error!(
                                                "Failed to lock tool_call_states: {}",
                                                e
                                            );
                                            return;
                                        }
                                    }
                                };

                                let new_messages = match model
                                    .process_tool_calls_and_run_spice_tools(
                                        req.messages.clone(),
                                        tool_calls_to_process,
                                    )
                                    .await
                                {
                                    Ok(Some(messages)) => messages,
                                    Ok(None) => {
                                        // No spice tools within returned tools, so return as message in stream.
                                        finished_choices.push(chat_choice);
                                        continue;
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            error = %e,
                                            message_count = req.messages.len(),
                                            model = %req.model,
                                            "Error processing tool calls in streaming loop"
                                        );
                                        let error_msg = format!("An error occurred while processing tool calls: {e}");
                                        if let Ok(error_resp) = create_stream_response(
                                            &stream_id,
                                            &req.model,
                                            vec![create_stream_choice(0, Some(error_msg), Some(Role::Assistant), Some(FinishReason::Stop))],
                                            None,
                                        ) {
                                            let _ = sender_clone.send(Ok(error_resp)).await;
                                        }
                                        return;
                                    }
                                };

                                match model
                                    .chat_stream_inner(create_new_recursive_req(
                                        &req,
                                        new_messages,
                                        response.usage.as_ref(),
                                        &model.context_config,
                                    ))
                                    .await
                                {
                                    Ok(mut s) => {
                                        while let Some(resp) = s.next().await {
                                            // TODO check if this works for choices > 1.
                                            if let Err(e) = sender_clone.send(resp).await {
                                                if !sender_clone.is_closed() {
                                                    tracing::error!("Error sending error: {}", e);
                                                }
                                                return;
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            error = %e,
                                            message_count = req.messages.len(),
                                            model = %req.model,
                                            "Error from recursive chat_stream"
                                        );
                                        let error_msg = format!("An error occurred: {e}");
                                        if let Ok(error_resp) = create_stream_response(
                                            &stream_id,
                                            &req.model,
                                            vec![create_stream_choice(0, Some(error_msg), Some(Role::Assistant), Some(FinishReason::Stop))],
                                            None,
                                        ) {
                                            let _ = sender_clone.send(Ok(error_resp)).await;
                                        }
                                        return;
                                    }
                                }
                            } else if matches!(finish_reason, FinishReason::Stop)
                                || matches!(finish_reason, FinishReason::Length)
                            {
                                // If complete, return to stream original.
                                finished_choices.push(chat_choice.clone());
                            }
                        }
                    }

                    if let Some(choice) = finished_choices.first() {
                        if let Some(intermediate_chat_output) = &choice.delta.content {
                            chat_output.push_str(intermediate_chat_output);
                        }

                        let mut resp2 = response.clone();
                        resp2.choices = finished_choices;
                        if let Err(e) = sender_clone.send(Ok(resp2)).await
                            && !sender_clone.is_closed() {
                                tracing::error!("Error sending error: {}", e);
                            }
                    }

                    // When there are no [`ChatChoiceStream`]s, but the model has usage, send the response (with no choices).
                    if response.choices.is_empty() && response.usage.is_some()
                        && let Err(e) = sender_clone.send(Ok(response)).await
                            && !sender_clone.is_closed() {
                                tracing::error!("Error sending error: {}", e);
                            }
                }

                tracing::debug!(target: "task_history", captured_output = %chat_output);
            })
            .instrument(span),
    );
    Box::pin(CustomStream { receiver }) as ChatCompletionResponseStream
}

const LOOP_DETECTION_MESSAGE: &str = "You have called the same tool(s) with identical arguments multiple times and received the same results. The state has not changed. Please take a different action, proceed to the next step of your task, or report what is blocking you.";

/// Ephemeral guidance appended to the last tool result each iteration.
/// Stripped from previous messages before being re-appended so it never accumulates.
const TOOL_USE_GUIDANCE_SUFFIX: &str = "\n\nIMPORTANT: Before making your next tool call, review your recent actions above. Make sure each call makes forward progress. Do not call the same tool or query the same data source again — even with different parameters, limits, or formatting. If your last few steps look repetitive or circular, stop immediately and produce your response with the information you already have.";

/// Strip the ephemeral guidance suffix from any tool result message that contains it.
fn strip_ephemeral_guidance(messages: &mut [ChatCompletionRequestMessage]) {
    for msg in messages.iter_mut() {
        if let ChatCompletionRequestMessage::Tool(tool_msg) = msg {
            if let ChatCompletionRequestToolMessageContent::Text(text) = &mut tool_msg.content {
                if let Some(idx) = text.find(TOOL_USE_GUIDANCE_SUFFIX) {
                    text.truncate(idx);
                }
            }
        }
    }
}

/// Append the ephemeral guidance to the last tool result message.
fn append_ephemeral_guidance(messages: &mut [ChatCompletionRequestMessage]) {
    for msg in messages.iter_mut().rev() {
        if let ChatCompletionRequestMessage::Tool(tool_msg) = msg {
            if let ChatCompletionRequestToolMessageContent::Text(text) = &mut tool_msg.content {
                text.push_str(TOOL_USE_GUIDANCE_SUFFIX);
                return;
            }
        }
    }
}

/// Compute a fingerprint (hash) of the tool calls for loop detection.
fn chat_tool_calls_fingerprint(calls: &[ChatCompletionMessageToolCall]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut keys: Vec<(&str, &str)> = calls
        .iter()
        .map(|c| (c.function.name.as_str(), c.function.arguments.as_str()))
        .collect();
    keys.sort();
    keys.hash(&mut hasher);
    hasher.finish()
}

/// Check if the last 3 fingerprints are identical (indicating a tool-use loop).
fn is_tool_loop_detected(fingerprints: &[u64]) -> bool {
    let len = fingerprints.len();
    len >= 3
        && fingerprints[len - 1] == fingerprints[len - 2]
        && fingerprints[len - 2] == fingerprints[len - 3]
}

// OpenAI tools must satisfy '^[a-zA-Z0-9_-]+$'. Commonly external tools may have '/' in their name.
pub fn encode_tool_name(name: &str) -> String {
    if name.contains('/') {
        name.replace('_', "__").replace('/', "_")
    } else {
        name.to_string()
    }
}

/// Decode an encoded tool name back to its original form.
fn decode_tool_name(encoded: &str) -> String {
    // Reverse the encoding: single _ → /, __ → _
    // We need to handle __ first to avoid double-replacing.
    if encoded.contains('_') {
        // Use a two-pass approach with a sentinel
        encoded
            .replace("__", "\x00")
            .replace('_', "/")
            .replace('\x00', "_")
    } else {
        encoded.to_string()
    }
}

#[pin_project]
struct InferenceTrackingStream<S> {
    #[pin]
    stream: S,
    context: Arc<RequestContext>,
}

impl<S: Stream> InferenceTrackingStream<S> {
    pub fn new(stream: S, context: Arc<RequestContext>) -> Self {
        InferenceTrackingStream { stream, context }
    }
}

impl<S: Stream> Stream for InferenceTrackingStream<S> {
    type Item = S::Item;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        let stream = &mut this.stream;
        let context = this.context;

        match stream.as_mut().poll_next(cx) {
            Poll::Ready(None) => {
                let context = Arc::clone(context);
                crate::model::track_ai_inferences_with_spice_count(&context);
                Poll::Ready(None)
            }
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::chat::{
        ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
        ChatCompletionRequestUserMessageArgs, FunctionCall,
    };

    fn create_system_message(content: &str) -> ChatCompletionRequestMessage {
        ChatCompletionRequestSystemMessageArgs::default()
            .content(content)
            .build()
            .expect("couldn't create system message")
            .into()
    }

    fn create_user_message(content: &str) -> ChatCompletionRequestMessage {
        ChatCompletionRequestUserMessageArgs::default()
            .content(content)
            .build()
            .expect("couldn't create user message")
            .into()
    }

    fn create_assistant_message_with_tool_calls(
        tool_calls: Vec<ChatCompletionMessageToolCall>,
    ) -> ChatCompletionRequestMessage {
        ChatCompletionRequestAssistantMessageArgs::default()
            .tool_calls(
                tool_calls
                    .into_iter()
                    .map(ChatCompletionMessageToolCalls::Function)
                    .collect::<Vec<_>>(),
            )
            .build()
            .expect("couldn't create assistant message w. tools")
            .into()
    }

    fn create_tool_message(tool_call_id: &str, content: &str) -> ChatCompletionRequestMessage {
        ChatCompletionRequestToolMessageArgs::default()
            .tool_call_id(tool_call_id)
            .content(content)
            .build()
            .expect("couldn't create tool message")
            .into()
    }

    fn create_list_datasets_tool_call() -> ChatCompletionMessageToolCall {
        ChatCompletionMessageToolCall {
            id: "test_id".to_string(),
            function: FunctionCall {
                name: "list_datasets".to_string(),
                arguments: "{}".to_string(),
            },
        }
    }

    #[test]
    fn test_insert_initial_tools_empty_messages() {
        let messages = vec![];
        let tool_messages = vec![
            create_assistant_message_with_tool_calls(vec![create_list_datasets_tool_call()]),
            create_tool_message("test_id", "dataset1, dataset2"),
        ];

        let result = insert_initial_tools(messages, "list_datasets", &tool_messages);

        insta::assert_json_snapshot!(result, {
            "[].Assistant.tool_calls[].id" => "[tool_call_id]",
            "[].Tool.tool_call_id" => "[tool_call_id]"
        });
    }

    #[test]
    fn test_insert_initial_tools_with_system_and_user_messages() {
        let messages = vec![
            create_system_message("You are a helpful assistant"),
            create_user_message("Hello"),
        ];
        let tool_messages = vec![
            create_assistant_message_with_tool_calls(vec![create_list_datasets_tool_call()]),
            create_tool_message("test_id", "dataset1, dataset2"),
        ];

        let result = insert_initial_tools(messages, "list_datasets", &tool_messages);

        insta::assert_json_snapshot!(result, {
            "[].Assistant.tool_calls[].id" => "[tool_call_id]",
            "[].Tool.tool_call_id" => "[tool_call_id]"
        });
    }

    #[test]
    fn test_insert_initial_tools_with_existing_assistant_message() {
        let existing_tool_call = ChatCompletionMessageToolCall {
            id: "existing_id".to_string(),
            function: FunctionCall {
                name: "other_tool".to_string(),
                arguments: "{}".to_string(),
            },
        };

        let messages = vec![
            create_system_message("You are a helpful assistant"),
            create_user_message("Hello"),
            create_assistant_message_with_tool_calls(vec![existing_tool_call]),
        ];
        let tool_messages = vec![
            create_assistant_message_with_tool_calls(vec![create_list_datasets_tool_call()]),
            create_tool_message("test_id", "dataset1, dataset2"),
        ];

        let result = insert_initial_tools(messages, "list_datasets", &tool_messages);

        insta::assert_json_snapshot!(result, {
            "[].Assistant.tool_calls[].id" => "[tool_call_id]",
            "[].Tool.tool_call_id" => "[tool_call_id]"
        });
    }

    #[test]
    fn test_insert_initial_tools_skips_if_tool_already_exists() {
        let existing_list_datasets_call = create_list_datasets_tool_call();
        let messages = vec![
            create_system_message("You are a helpful assistant"),
            create_user_message("Hello"),
            create_assistant_message_with_tool_calls(vec![existing_list_datasets_call]),
        ];
        let tool_messages = vec![
            create_assistant_message_with_tool_calls(vec![create_list_datasets_tool_call()]),
            create_tool_message("test_id", "dataset1, dataset2"),
        ];

        let result = insert_initial_tools(messages, "list_datasets", &tool_messages);

        insta::assert_json_snapshot!(result, {
            "[].Assistant.tool_calls[].id" => "[tool_call_id]"
        });
    }

    #[test]
    fn test_insert_initial_tools_with_different_tool_name() {
        let existing_tool_call = ChatCompletionMessageToolCall {
            id: "other_id".to_string(),
            function: FunctionCall {
                name: "other_tool".to_string(),
                arguments: "{}".to_string(),
            },
        };

        let messages = vec![
            create_system_message("You are a helpful assistant"),
            create_assistant_message_with_tool_calls(vec![existing_tool_call]),
        ];
        let tool_messages = vec![
            create_assistant_message_with_tool_calls(vec![create_list_datasets_tool_call()]),
            create_tool_message("test_id", "dataset1, dataset2"),
        ];

        let result = insert_initial_tools(messages, "list_datasets", &tool_messages);

        insta::assert_json_snapshot!(result, {
            "[].Assistant.tool_calls[].id" => "[tool_call_id]",
            "[].Tool.tool_call_id" => "[tool_call_id]"
        });
    }

    #[test]
    fn test_insert_initial_tools_insertion_point_end_of_messages() {
        let messages = vec![
            create_system_message("You are a helpful assistant"),
            create_user_message("What datasets are available?"),
            create_user_message("And what about tables?"),
        ];
        let tool_messages = vec![
            create_assistant_message_with_tool_calls(vec![create_list_datasets_tool_call()]),
            create_tool_message("test_id", "dataset1, dataset2"),
        ];

        let result = insert_initial_tools(messages, "list_datasets", &tool_messages);

        insta::assert_json_snapshot!(result, {
            "[].Assistant.tool_calls[].id" => "[tool_call_id]",
            "[].Tool.tool_call_id" => "[tool_call_id]"
        });
    }
}
