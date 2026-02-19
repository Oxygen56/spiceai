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

use crate::chat::Chat;
use crate::chat::nsql::structured_output::StructuredOutputSqlGeneration;
use crate::chat::nsql::{SqlGeneration, json::JsonSchemaSqlGeneration};
use crate::streaming_utils::{create_stream_choice, create_stream_response, generate_stream_id};
use async_openai::config::Config;
use async_openai::error::OpenAIError;
use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionResponseStream,
    CreateChatCompletionRequest, CreateChatCompletionResponse, FinishReason, Role,
};
use async_trait::async_trait;
use futures::TryStreamExt;
use tracing_futures::Instrument;

use super::Openai;
use super::responses_compat::{
    chat_request_to_create_response, response_to_chat_completion, response_to_stream_chunks,
};

#[async_trait]
impl<C: Config + Send + Sync + Clone> Chat for Openai<C> {
    fn as_sql(&self) -> Option<&dyn SqlGeneration> {
        // Only use structured output schema for OpenAI, not openai compatible.
        if self.supports_structured_output() {
            Some(&StructuredOutputSqlGeneration {})
        } else {
            Some(&JsonSchemaSqlGeneration {})
        }
    }

    async fn chat_stream(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        // Responses-only models: convert to Responses API call and wrap as stream.
        if self.is_responses_only() {
            let responses_req = chat_request_to_create_response(&req);
            let mut responses_req = match responses_req {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("Failed to convert chat request to Responses API format: {e}");
                    return Ok(Self::error_stream(
                        &req.model,
                        &format!("Failed to prepare request: {e}"),
                    ));
                }
            };
            responses_req.model = Some(self.model.clone());

            let permit = self
                .rate_controller
                .acquire()
                .await
                .map_err(|e| OpenAIError::InvalidArgument(e.to_string()))?;

            let resp = match self.client.responses().create(responses_req).await {
                Ok(r) => r,
                Err(e) => {
                    drop(permit);
                    tracing::error!("Responses API call failed for model '{}': {e}", self.model);
                    return Ok(Self::error_stream(
                        &req.model,
                        &format!("Model API call failed: {e}"),
                    ));
                }
            };

            drop(permit);

            let outer_model = req.model.clone();
            let (mut content_chunk, mut finish_chunk) = response_to_stream_chunks(resp)?;
            content_chunk.model.clone_from(&outer_model);
            finish_chunk.model = outer_model;
            return Ok(Box::pin(futures::stream::iter(vec![
                Ok(content_chunk),
                Ok(finish_chunk),
            ])));
        }

        let outer_model = req.model.clone();
        let mut inner_req = req.clone();
        inner_req.model.clone_from(&self.model);

        let permit = self
            .rate_controller
            .acquire()
            .await
            .map_err(|e| OpenAIError::InvalidArgument(e.to_string()))?;

        let stream = self.client.chat().create_stream(inner_req).await?;

        drop(permit); // drop the permit after acquiring the stream, instead of after receiving the response
        // semaphore permits aren't `Copy`, so we can't move it into the closure in `.map_ok`

        Ok(Box::pin(stream.map_ok(move |mut s| {
            s.model.clone_from(&outer_model);
            s
        })))
    }

    // Custom healthcheck for OpenAI because Azure dosn't support `max_completion_tokens`.
    #[expect(deprecated)]
    async fn health(&self) -> Result<(), crate::chat::Error> {
        let span = tracing::span!(target: "task_history", tracing::Level::INFO, "health", input = "health");

        // Responses-only models (codex, gpt-5.2-pro) don't support /v1/chat/completions.
        // Use model retrieval as a health check instead (same pattern as xAI).
        if self.is_responses_only() {
            return match self.client.models().retrieve(&self.model).await {
                Ok(_) => Ok(()),
                Err(e) => {
                    tracing::error!(target: "task_history", parent: &span, "{e}");
                    Err(crate::chat::Error::HealthCheckError {
                        source: Box::new(e),
                    })
                }
            };
        }

        let mut req = CreateChatCompletionRequest {
            messages: vec![ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    name: None,
                    content: ChatCompletionRequestUserMessageContent::Text(
                        "Respond with 'ok'".to_string(),
                    ),
                },
            )],
            ..Default::default()
        };

        if self.supports_reasoning_effort() {
            req.reasoning_effort = Some(async_openai::types::chat::ReasoningEffort::Low);
        }

        if self.supports_max_completion_tokens() {
            req.max_completion_tokens = Some(300);
        } else {
            req.max_tokens = Some(300);
        }

        let result = self.chat_request(req).instrument(span.clone()).await;
        tracing::debug!("{} model health check response: {:?}", self.model, result);
        if let Err(e) = result {
            tracing::error!(target: "task_history", parent: &span, "{e}");
            return Err(crate::chat::Error::HealthCheckError {
                source: Box::new(e),
            });
        }
        Ok(())
    }

    async fn chat_request(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        let outer_model = req.model.clone();

        // Responses-only models: convert chat request → Responses API request → convert back.
        if self.is_responses_only() {
            let mut responses_req = chat_request_to_create_response(&req)?;
            responses_req.model = Some(self.model.clone());

            let permit = self
                .rate_controller
                .acquire()
                .await
                .map_err(|e| OpenAIError::InvalidArgument(e.to_string()))?;

            let resp = self.client.responses().create(responses_req).await?;

            drop(permit);

            let mut chat_resp = response_to_chat_completion(resp)?;
            chat_resp.model = outer_model;
            return Ok(chat_resp);
        }

        let mut inner_req = req.clone();
        inner_req.model.clone_from(&self.model);

        let permit = self
            .rate_controller
            .acquire()
            .await
            .map_err(|e| OpenAIError::InvalidArgument(e.to_string()))?;

        let mut resp = self.client.chat().create(inner_req).await?;

        drop(permit);

        resp.model = outer_model;
        Ok(resp)
    }
}

impl<C: Config + Clone> Openai<C> {
    /// Create a stream that yields a single error message as assistant content.
    /// Used to gracefully report errors back through the chat interface instead of
    /// breaking the stream with an `Err`.
    fn error_stream(model: &str, error_msg: &str) -> ChatCompletionResponseStream {
        let stream_id = generate_stream_id(model);
        let choice =
            create_stream_choice(0, Some(error_msg.to_string()), Some(Role::Assistant), None);
        let finish = create_stream_choice(0, None, None, Some(FinishReason::Stop));

        let mut chunks = Vec::with_capacity(2);
        if let Ok(content_resp) =
            create_stream_response(&stream_id, model, vec![choice], None)
        {
            chunks.push(Ok(content_resp));
        }
        if let Ok(finish_resp) =
            create_stream_response(&stream_id, model, vec![finish], None)
        {
            chunks.push(Ok(finish_resp));
        }

        Box::pin(futures::stream::iter(chunks))
    }
}
