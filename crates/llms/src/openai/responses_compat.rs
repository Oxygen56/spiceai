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

//! Conversion layer between Chat Completions API types and Responses API types.
//!
//! Responses-only models (e.g. `codex`, `gpt-5.2-pro`) reject `/v1/chat/completions`.
//! This module converts `CreateChatCompletionRequest` → `CreateResponse` and
//! `Response` → `CreateChatCompletionResponse` so that the rest of the runtime
//! can continue to use the `Chat` trait uniformly.

#![allow(deprecated)] // function_call / system_fingerprint fields are deprecated but we must set them.

use async_openai::error::OpenAIError;
use async_openai::types::chat::{
    ChatChoice, ChatChoiceStream, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestToolMessageContentPart,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta, ChatCompletionToolChoiceOption, CompletionUsage,
    CreateChatCompletionRequest, CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse, FinishReason, FunctionCall, FunctionCallStream,
    FunctionType, Role, ToolChoiceOptions as ChatToolChoiceOptions,
};
use async_openai::types::responses::{
    CreateResponse, EasyInputContent, EasyInputMessage, FunctionCallOutput,
    FunctionCallOutputItemParam, FunctionTool, FunctionToolCall, InputItem, InputParam, Item,
    MessageType, OutputItem, OutputMessageContent, Reasoning, Response, Status, Tool,
    ToolChoiceFunction, ToolChoiceOptions as ResponseToolChoiceOptions, ToolChoiceParam,
};

use async_openai::types::chat::ChatCompletionTools;

/// Convert a `CreateChatCompletionRequest` into a `CreateResponse` for the Responses API.
pub(crate) fn chat_request_to_create_response(
    req: &CreateChatCompletionRequest,
) -> Result<CreateResponse, OpenAIError> {
    let instructions = extract_instructions(&req.messages);
    let input_items = convert_messages(&req.messages)?;
    let tools = convert_tools(&req.tools);
    let tool_choice = convert_tool_choice(&req.tool_choice);

    let max_output_tokens = req
        .max_completion_tokens
        .map(|t| t as u32)
        .or(req.max_tokens.map(|t| t as u32));

    let reasoning = req.reasoning_effort.as_ref().map(|effort| Reasoning {
        effort: Some(effort.clone()),
        ..Default::default()
    });

    Ok(CreateResponse {
        input: InputParam::Items(input_items),
        model: Some(req.model.clone()),
        instructions,
        tools,
        tool_choice,
        max_output_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        parallel_tool_calls: req.parallel_tool_calls,
        reasoning,
        ..Default::default()
    })
}

/// Convert a `Response` from the Responses API back to a `CreateChatCompletionResponse`.
#[expect(clippy::cast_possible_truncation)]
pub(crate) fn response_to_chat_completion(
    resp: Response,
) -> Result<CreateChatCompletionResponse, OpenAIError> {
    let (content, tool_calls, refusal, finish_reason) = extract_response_output(&resp)?;

    Ok(CreateChatCompletionResponse {
        id: resp.id,
        model: resp.model,
        created: resp.created_at as u32,
        object: "chat.completion".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            logprobs: None,
            finish_reason,
            message: ChatCompletionResponseMessage {
                role: Role::Assistant,
                content,
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                refusal,
                annotations: None,
                function_call: None,
                audio: None,
            },
        }],
        usage: resp.usage.map(|u| CompletionUsage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.total_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }),
        service_tier: None,
        system_fingerprint: None,
    })
}

/// Convert a `Response` from the Responses API into two stream chunks: one with content/tool_calls
/// and one with the finish reason. This matches the normal OpenAI streaming behavior where content
/// and finish_reason arrive in separate chunks, which the `make_a_stream` tool-use wrapper expects.
#[expect(clippy::cast_possible_truncation)]
pub(crate) fn response_to_stream_chunks(
    resp: Response,
) -> Result<
    (
        CreateChatCompletionStreamResponse,
        CreateChatCompletionStreamResponse,
    ),
    OpenAIError,
> {
    let (content, tool_calls, _refusal, finish_reason) = extract_response_output(&resp)?;

    let tool_call_chunks: Option<Vec<ChatCompletionMessageToolCallChunk>> = if tool_calls.is_empty()
    {
        None
    } else {
        Some(
            tool_calls
                .into_iter()
                .enumerate()
                .map(|(i, tc)| match tc {
                    ChatCompletionMessageToolCalls::Function(f) => {
                        ChatCompletionMessageToolCallChunk {
                            index: i as u32,
                            id: Some(f.id),
                            r#type: Some(FunctionType::Function),
                            function: Some(FunctionCallStream {
                                name: Some(f.function.name),
                                arguments: Some(f.function.arguments),
                            }),
                        }
                    }
                    ChatCompletionMessageToolCalls::Custom(_) => {
                        ChatCompletionMessageToolCallChunk {
                            index: i as u32,
                            id: None,
                            r#type: None,
                            function: None,
                        }
                    }
                })
                .collect(),
        )
    };

    let usage = resp.usage.map(|u| CompletionUsage {
        prompt_tokens: u.input_tokens,
        completion_tokens: u.output_tokens,
        total_tokens: u.total_tokens,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });

    // Chunk 1: content and/or tool calls, no finish_reason.
    let content_chunk = CreateChatCompletionStreamResponse {
        id: resp.id.clone(),
        model: resp.model.clone(),
        created: resp.created_at as u32,
        object: "chat.completion.chunk".to_string(),
        choices: vec![ChatChoiceStream {
            index: 0,
            logprobs: None,
            delta: ChatCompletionStreamResponseDelta {
                content,
                tool_calls: tool_call_chunks,
                role: Some(Role::Assistant),
                refusal: None,
                function_call: None,
            },
            finish_reason: None,
        }],
        usage: None,
        service_tier: None,
        system_fingerprint: None,
    };

    // Chunk 2: finish_reason and usage, no content.
    let finish_chunk = CreateChatCompletionStreamResponse {
        id: resp.id,
        model: resp.model,
        created: resp.created_at as u32,
        object: "chat.completion.chunk".to_string(),
        choices: vec![ChatChoiceStream {
            index: 0,
            logprobs: None,
            delta: ChatCompletionStreamResponseDelta {
                content: None,
                tool_calls: None,
                role: None,
                refusal: None,
                function_call: None,
            },
            finish_reason,
        }],
        usage,
        service_tier: None,
        system_fingerprint: None,
    };

    Ok((content_chunk, finish_chunk))
}

/// Extract text content, tool calls, refusal, and finish reason from a Responses API output.
fn extract_response_output(
    resp: &Response,
) -> Result<
    (
        Option<String>,
        Vec<ChatCompletionMessageToolCalls>,
        Option<String>,
        Option<FinishReason>,
    ),
    OpenAIError,
> {
    let mut content = String::new();
    let mut tool_calls: Vec<ChatCompletionMessageToolCalls> = Vec::new();
    let mut refusal: Option<String> = None;

    for item in &resp.output {
        match item {
            OutputItem::Message(msg) => {
                for c in &msg.content {
                    match c {
                        OutputMessageContent::OutputText(t) => content.push_str(&t.text),
                        OutputMessageContent::Refusal(r) => {
                            refusal = Some(r.refusal.clone());
                        }
                    }
                }
            }
            OutputItem::FunctionCall(fc) => {
                tool_calls.push(ChatCompletionMessageToolCalls::Function(
                    ChatCompletionMessageToolCall {
                        id: fc.call_id.clone(),
                        function: FunctionCall {
                            name: fc.name.clone(),
                            arguments: fc.arguments.clone(),
                        },
                    },
                ));
            }
            _ => {} // Ignore reasoning, web search, etc.
        }
    }

    let finish_reason = if !tool_calls.is_empty() {
        Some(FinishReason::ToolCalls)
    } else {
        match resp.status {
            Status::Completed => Some(FinishReason::Stop),
            Status::Incomplete => Some(FinishReason::Length),
            _ => None,
        }
    };

    let content = if content.is_empty() {
        None
    } else {
        Some(content)
    };

    Ok((content, tool_calls, refusal, finish_reason))
}

/// Extract system/developer messages into a single `instructions` string.
fn extract_instructions(messages: &[ChatCompletionRequestMessage]) -> Option<String> {
    let system_texts: Vec<String> = messages
        .iter()
        .filter_map(|m| match m {
            ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content,
                ..
            }) => match content {
                ChatCompletionRequestSystemMessageContent::Text(text) => Some(text.clone()),
                ChatCompletionRequestSystemMessageContent::Array(parts) => {
                    let texts: Vec<&str> = parts
                        .iter()
                        .map(|p| match p {
                            ChatCompletionRequestSystemMessageContentPart::Text(
                                ChatCompletionRequestMessageContentPartText { text },
                            ) => text.as_str(),
                        })
                        .collect();
                    Some(texts.join("\n"))
                }
            },
            ChatCompletionRequestMessage::Developer(dev) => {
                match &dev.content {
                    async_openai::types::chat::ChatCompletionRequestDeveloperMessageContent::Text(text) => Some(text.clone()),
                    async_openai::types::chat::ChatCompletionRequestDeveloperMessageContent::Array(parts) => {
                        let texts: Vec<String> = parts
                            .iter()
                            .map(|p| match p {
                                async_openai::types::chat::ChatCompletionRequestDeveloperMessageContentPart::Text(
                                    ChatCompletionRequestMessageContentPartText { text },
                                ) => text.clone(),
                            })
                            .collect();
                        Some(texts.join("\n"))
                    }
                }
            }
            _ => None,
        })
        .collect();

    if system_texts.is_empty() {
        None
    } else {
        Some(system_texts.join("\n"))
    }
}

/// Convert chat completion messages to Responses API input items.
fn convert_messages(
    messages: &[ChatCompletionRequestMessage],
) -> Result<Vec<InputItem>, OpenAIError> {
    let mut items = Vec::new();

    for msg in messages {
        match msg {
            // System/Developer messages are handled via `instructions`
            ChatCompletionRequestMessage::System(_)
            | ChatCompletionRequestMessage::Developer(_) => continue,

            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content,
                ..
            }) => {
                let text = match content {
                    ChatCompletionRequestUserMessageContent::Text(t) => t.clone(),
                    ChatCompletionRequestUserMessageContent::Array(parts) => {
                        let texts: Vec<String> = parts
                            .iter()
                            .filter_map(|p| match p {
                                ChatCompletionRequestUserMessageContentPart::Text(
                                    ChatCompletionRequestMessageContentPartText { text },
                                ) => Some(text.clone()),
                                _ => None, // Skip image/audio/file parts
                            })
                            .collect();
                        texts.join("\n")
                    }
                };
                items.push(InputItem::EasyMessage(EasyInputMessage {
                    r#type: MessageType::default(),
                    role: async_openai::types::responses::Role::User,
                    content: EasyInputContent::Text(text),
                }));
            }

            ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                content,
                tool_calls,
                ..
            }) => {
                // Add text content as an assistant message
                let text = match content {
                    Some(ChatCompletionRequestAssistantMessageContent::Text(t)) => Some(t.clone()),
                    Some(ChatCompletionRequestAssistantMessageContent::Array(parts)) => {
                        let texts: Vec<String> = parts
                            .iter()
                            .filter_map(|p| match p {
                                async_openai::types::chat::ChatCompletionRequestAssistantMessageContentPart::Text(
                                    ChatCompletionRequestMessageContentPartText { text },
                                ) => Some(text.clone()),
                                _ => None,
                            })
                            .collect();
                        if texts.is_empty() {
                            None
                        } else {
                            Some(texts.join(""))
                        }
                    }
                    None => None,
                };

                if let Some(text) = text {
                    if !text.is_empty() {
                        items.push(InputItem::EasyMessage(EasyInputMessage {
                            r#type: MessageType::default(),
                            role: async_openai::types::responses::Role::Assistant,
                            content: EasyInputContent::Text(text),
                        }));
                    }
                }

                // Add tool calls as FunctionCall items
                if let Some(calls) = tool_calls {
                    for call in calls {
                        match call {
                            ChatCompletionMessageToolCalls::Function(tc) => {
                                items.push(InputItem::Item(Item::FunctionCall(FunctionToolCall {
                                    call_id: tc.id.clone(),
                                    name: tc.function.name.clone(),
                                    arguments: tc.function.arguments.clone(),
                                    id: None,
                                    status: None,
                                })));
                            }
                            ChatCompletionMessageToolCalls::Custom(_) => {
                                // Custom tool calls not supported in this conversion
                            }
                        }
                    }
                }
            }

            ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
                content,
                tool_call_id,
            }) => {
                let text = match content {
                    ChatCompletionRequestToolMessageContent::Text(t) => t.clone(),
                    ChatCompletionRequestToolMessageContent::Array(parts) => {
                        let texts: Vec<String> = parts
                            .iter()
                            .map(|p| match p {
                                ChatCompletionRequestToolMessageContentPart::Text(
                                    ChatCompletionRequestMessageContentPartText { text },
                                ) => text.clone(),
                            })
                            .collect();
                        texts.join("\n")
                    }
                };
                items.push(InputItem::Item(Item::FunctionCallOutput(
                    FunctionCallOutputItemParam {
                        call_id: tool_call_id.clone(),
                        output: FunctionCallOutput::Text(text),
                        id: None,
                        status: None,
                    },
                )));
            }

            ChatCompletionRequestMessage::Function(_) => {
                // Deprecated function messages are not supported
            }
        }
    }

    Ok(items)
}

/// Convert chat completion tools to Responses API tools.
fn convert_tools(tools: &Option<Vec<ChatCompletionTools>>) -> Option<Vec<Tool>> {
    tools.as_ref().map(|tools| {
        tools
            .iter()
            .filter_map(|t| match t {
                ChatCompletionTools::Function(ct) => Some(Tool::Function(FunctionTool {
                    name: ct.function.name.clone(),
                    parameters: ct.function.parameters.clone(),
                    strict: ct.function.strict,
                    description: ct.function.description.clone(),
                })),
                ChatCompletionTools::Custom(_) => None,
            })
            .collect()
    })
}

/// Convert chat completion tool choice to Responses API tool choice.
fn convert_tool_choice(
    choice: &Option<ChatCompletionToolChoiceOption>,
) -> Option<ToolChoiceParam> {
    match choice {
        Some(ChatCompletionToolChoiceOption::Mode(ChatToolChoiceOptions::None)) => {
            Some(ToolChoiceParam::Mode(ResponseToolChoiceOptions::None))
        }
        Some(ChatCompletionToolChoiceOption::Mode(ChatToolChoiceOptions::Auto)) => {
            Some(ToolChoiceParam::Mode(ResponseToolChoiceOptions::Auto))
        }
        Some(ChatCompletionToolChoiceOption::Mode(ChatToolChoiceOptions::Required)) => {
            Some(ToolChoiceParam::Mode(ResponseToolChoiceOptions::Required))
        }
        Some(ChatCompletionToolChoiceOption::Function(named)) => {
            Some(ToolChoiceParam::Function(ToolChoiceFunction {
                name: named.function.name.clone(),
            }))
        }
        _ => None,
    }
}
