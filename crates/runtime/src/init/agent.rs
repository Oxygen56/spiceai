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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_openai::types::chat::{
    ChatChoice, ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionTools,
    CreateChatCompletionRequestArgs, FunctionObject,
};
use async_trait::async_trait;
use futures::TryStreamExt;
use tools::SpiceModelTool;
use tracing_futures::Instrument;

use crate::tools::builtin::fail::FailTool;
use crate::tools::builtin::plan_mode::{
    EnterPlanModeTool, ExitPlanModeTool, ENTER_PLAN_MODE_TOOL_NAME, EXIT_PLAN_MODE_TOOL_NAME,
};
use crate::memory::layers::{
    knowledge_base::KnowledgeBaseLayer, session::SessionLayer,
    session_summary::SessionSummaryLayer, weekly_rollup::WeeklyRollupLayer,
};
use crate::memory::{MemoryLayer, MemoryManager, parse_retention};
use crate::model::LLMChatCompletionsModelStore;
use crate::pipeline::executor;
use crate::pipeline::resolve::resolve_workflow;
use crate::pipeline::vote::ModelCaller;
use crate::pipeline::ResolvedWorkflow;
use crate::session::in_memory::InMemorySessionStore;
use crate::session::SessionStore;
use crate::trigger::schedule::ScheduleTriggerFactory;
use crate::trigger::webhook::WebhookTriggerFactory;
use crate::trigger::{Trigger, TriggerHandler, TriggerPayload, TriggerRegistry};
use crate::datafusion::DataFusion;
use crate::tools::file_source_tools;
use crate::Runtime;
use app::App;
use spicepod::component::agent::Agent;
use spicepod::component::session::SessionConfig;
use tokio::sync::RwLock;

/// Registry of webhook trigger handlers keyed by path.
/// Shared between agent initialization and the HTTP webhook handler.
pub type WebhookRegistry = Arc<RwLock<HashMap<String, Arc<dyn TriggerHandler>>>>;

/// Holds the runtime state for a loaded agent.
pub struct LoadedAgent {
    pub name: String,
    pub config: Agent,
    pub pipelines: Vec<ResolvedWorkflow>,
    pub triggers: Vec<Box<dyn Trigger>>,
    pub session_store: Arc<dyn SessionStore>,
    pub session_config: SessionConfig,
    pub memory_manager: Option<MemoryManager>,
}

/// A `TriggerHandler` that executes a pipeline when invoked.
struct PipelineTriggerHandler {
    pipeline: ResolvedWorkflow,
    session_store: Arc<dyn SessionStore>,
    session_config: SessionConfig,
    memory_manager: Option<MemoryManager>,
    model_caller: Arc<dyn ModelCaller>,
    df: Arc<DataFusion>,
}

#[async_trait]
impl TriggerHandler for PipelineTriggerHandler {
    async fn handle(
        &self,
        payload: TriggerPayload,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let result = executor::execute_workflow(
            &self.pipeline,
            &payload,
            self.session_store.as_ref(),
            &self.session_config,
            self.memory_manager.as_ref(),
            self.model_caller.as_ref(),
            None,
        )
        .await?;

        tracing::info!(
            pipeline = %result.workflow_name,
            session_id = %result.session_id,
            steps = result.steps_executed,
            output_length = result.final_output.len(),
            output_preview = %result.final_output.chars().take(300).collect::<String>(),
            "Pipeline execution completed"
        );

        // Persist execution result to agentic.default.agent_results table.
        let id = uuid::Uuid::now_v7().to_string();
        let agent_name = sql_escape(&self.pipeline.agent_name);
        let pipeline_name = sql_escape(&result.workflow_name);
        let trigger_source = sql_escape(&payload.source);
        let output = sql_escape(&result.final_output);
        let insert_sql = format!(
            "INSERT INTO agentic.default.agent_results VALUES \
             ('{id}', '{agent_name}', '{pipeline_name}', '{trigger_source}', '{output}', NOW())"
        );
        match self.df.query_builder(&insert_sql).build().run().await {
            Ok(query_result) => {
                // Drain the stream to ensure the write is committed.
                if let Err(e) = query_result.data.try_collect::<Vec<_>>().await {
                    tracing::warn!(
                        pipeline = %result.workflow_name,
                        error = %e,
                        "Failed to commit agent result write"
                    );
                } else {
                    tracing::info!(
                        id = %id,
                        pipeline = %result.workflow_name,
                        "Agent result stored in agentic.default.agent_results"
                    );
                }
            }
            Err(e) => tracing::warn!(
                pipeline = %result.workflow_name,
                error = %e,
                "Failed to persist agent result to agentic.default.agent_results"
            ),
        }

        Ok(())
    }
}

/// Maximum number of retries when the model fails to use required tools.
const REQUIRED_TOOL_RETRIES: usize = 3;

/// Name of the auto-injected fail tool.
const FAIL_TOOL_NAME: &str = "fail";

/// Outcome of a single iteration in the tool-calling loop.
enum IterationOutcome {
    /// Model returned content without calling any tools — iteration is done.
    Done(String, HashSet<String>),
    /// Model called tools and results were fed back — continue looping.
    /// The `Option<bool>` carries an optional plan mode state change.
    Continue(HashSet<String>, Option<bool>),
}

/// A `ModelCaller` that looks up models from the runtime's `completion_llms` store.
/// Handles step-specific tool execution with a tool-calling loop and required tool enforcement.
struct RuntimeModelCaller {
    llms: Arc<RwLock<LLMChatCompletionsModelStore>>,
}

impl RuntimeModelCaller {
    /// Convert `SpiceModelTool` instances to `ChatCompletionTool` schemas for the request.
    fn tools_to_schemas(tools: &[Arc<dyn SpiceModelTool>]) -> Vec<ChatCompletionTools> {
        tools
            .iter()
            .map(|t| {
                ChatCompletionTools::Function(ChatCompletionTool {
                    function: FunctionObject {
                        strict: t.strict(),
                        name: crate::model::tool_use::encode_tool_name(
                            t.name().to_string().as_str(),
                        ),
                        description: t.description().map(|d| d.to_string()),
                        parameters: t.parameters(),
                    },
                })
            })
            .collect()
    }

    /// Find a step tool by its encoded name.
    fn find_tool<'a>(
        tools: &'a [Arc<dyn SpiceModelTool>],
        encoded_name: &str,
    ) -> Option<&'a Arc<dyn SpiceModelTool>> {
        tools.iter().find(|t| {
            crate::model::tool_use::encode_tool_name(t.name().as_ref()) == encoded_name
        })
    }

    /// Extract tool calls from a model response, returning only step-level tool calls.
    fn extract_step_tool_calls(
        response: &async_openai::types::chat::CreateChatCompletionResponse,
        all_step_tools: &[Arc<dyn SpiceModelTool>],
    ) -> Vec<ChatCompletionMessageToolCall> {
        response
            .choices
            .first()
            .and_then(|c| c.message.tool_calls.as_ref())
            .map(|calls| {
                calls
                    .iter()
                    .filter_map(|tc| match tc {
                        ChatCompletionMessageToolCalls::Function(call)
                            if Self::find_tool(all_step_tools, &call.function.name).is_some() =>
                        {
                            Some(call.clone())
                        }
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Run a single model call with tool-calling loop. Returns (content, set of tools called).
    ///
    /// If the model calls the `fail` tool, returns an error immediately.
    /// If `initial_plan_mode` is true, only read-only tools are presented to the model
    /// until `exit_plan_mode` is called.
    async fn call_with_tool_loop(
        model: &Arc<dyn llms::chat::Chat>,
        step_name: &str,
        model_name: &str,
        messages: Vec<ChatCompletionRequestMessage>,
        all_step_tools: &[Arc<dyn SpiceModelTool>],
        max_iterations: usize,
        initial_plan_mode: bool,
    ) -> Result<(String, HashSet<String>), Box<dyn std::error::Error + Send + Sync>> {
        let mut current_messages = messages;
        let mut tools_called = HashSet::new();
        // Plan mode state is tracked for logging and plan mode tool toggling.
        // Tool availability is NOT filtered by plan mode — it's a behavioral
        // constraint via system prompt, not a technical sandbox.
        let mut plan_mode = initial_plan_mode;
        tracing::info!(
            target: "task_history",
            plan_mode = plan_mode,
            "Starting tool loop"
        );

        tracing::info!(
            step = %step_name,
            model = %model_name,
            max_iterations = max_iterations,
            tool_count = all_step_tools.len(),
            tools = ?all_step_tools.iter().map(|t| t.name().to_string()).collect::<Vec<_>>(),
            "Starting tool loop for step"
        );

        for iteration in 0..max_iterations {
            tracing::debug!(
                target: "task_history",
                iteration = iteration,
                "Starting tool iteration"
            );

            let iter_span = tracing::info_span!(
                target: "task_history",
                "tool_iteration",
                step = %step_name,
                iteration = iteration,
                model = %model_name,
            );

            let iter_result: Result<IterationOutcome, Box<dyn std::error::Error + Send + Sync>> = async {
                // All tools are always available — plan mode is a behavioral constraint
                // via the system prompt, not a technical tool filter.
                let active_tools = all_step_tools.to_vec();
                let active_schemas = Self::tools_to_schemas(&active_tools);

                let mut req_builder = CreateChatCompletionRequestArgs::default();
                req_builder.model(model_name).messages(current_messages.clone());
                if !active_schemas.is_empty() {
                    req_builder.tools(active_schemas);
                }
                let req = req_builder.build()?;

                tracing::info!(
                    step = %step_name,
                    model = %model_name,
                    iteration = iteration,
                    message_count = current_messages.len(),
                    "Sending request to model"
                );

                let response = model.chat_request(req).await?;

                // Check for step-level tool calls (against all tools, not just active)
                let step_tool_calls = Self::extract_step_tool_calls(&response, all_step_tools);

                if step_tool_calls.is_empty() {
                    // No step tools called — return the text content
                    let content = response
                        .choices
                        .first()
                        .and_then(|ChatChoice { message, .. }| message.content.clone())
                        .unwrap_or_default();
                    tracing::info!(
                        step = %step_name,
                        iteration = iteration,
                        content_length = content.len(),
                        "Model returned final response (no tool calls)"
                    );
                    return Ok(IterationOutcome::Done(content, HashSet::new()));
                }

                let tool_names: Vec<String> = step_tool_calls
                    .iter()
                    .filter_map(|tc| Self::find_tool(all_step_tools, &tc.function.name))
                    .map(|t| t.name().to_string())
                    .collect();
                tracing::info!(
                    step = %step_name,
                    iteration = iteration,
                    count = step_tool_calls.len(),
                    tools = ?tool_names,
                    "Model requested tool calls"
                );

                // Execute the step tools and build messages for the next round
                let assistant_message: ChatCompletionRequestMessage =
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .tool_calls(
                            step_tool_calls
                                .iter()
                                .map(|t| ChatCompletionMessageToolCalls::Function(t.clone()))
                                .collect::<Vec<_>>(),
                        )
                        .build()?
                        .into();
                current_messages.push(assistant_message);

                let mut iter_tools = HashSet::new();
                let mut new_plan_mode: Option<bool> = None;

                for tool_call in &step_tool_calls {
                    if let Some(tool) = Self::find_tool(all_step_tools, &tool_call.function.name) {
                        // Check for `fail` tool — short-circuit immediately
                        if tool.name() == FAIL_TOOL_NAME {
                            let reason = serde_json::from_str::<serde_json::Value>(
                                &tool_call.function.arguments,
                            )
                            .ok()
                            .and_then(|v| v.get("reason").and_then(|r| r.as_str()).map(String::from))
                            .unwrap_or_else(|| tool_call.function.arguments.clone());
                            tracing::error!(
                                target: "task_history",
                                tool = FAIL_TOOL_NAME,
                                reason = %reason,
                                "Step failed via fail tool"
                            );
                            return Err(format!("Step failed: {reason}").into());
                        }

                        // Check for plan mode tools — toggle plan mode state
                        if tool.name() == ENTER_PLAN_MODE_TOOL_NAME {
                            let _ = tool.call(&tool_call.function.arguments).await;
                            new_plan_mode = Some(true);
                            let tool_message: ChatCompletionRequestMessage =
                                ChatCompletionRequestToolMessageArgs::default()
                                    .content("Plan mode activated. You have access to all tools but MUST only use them for read-only purposes. Research and gather information, then call exit_plan_mode with your plan.")
                                    .tool_call_id(tool_call.id.clone())
                                    .build()?
                                    .into();
                            current_messages.push(tool_message);
                            continue;
                        }
                        if tool.name() == EXIT_PLAN_MODE_TOOL_NAME {
                            let _ = tool.call(&tool_call.function.arguments).await;
                            let plan_text = serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
                                .ok()
                                .and_then(|v| v.get("plan").and_then(|p| p.as_str()).map(String::from))
                                .unwrap_or_default();

                            if !plan_text.is_empty() {
                                println!("\n📋 Agent Plan:\n{plan_text}\n");
                            }

                            // Auto-call approval tool if one exists among step tools
                            let approval_tool = all_step_tools.iter().find(|t| {
                                let n = t.name();
                                n == "approval" || n == "approval_slack" || n == "approval_ms_teams"
                            });

                            let tool_response = if let Some(approval) = approval_tool {
                                let approval_arg = serde_json::json!({
                                    "message": format!("Agent plan requires approval:\n\n{plan_text}"),
                                    "context": plan_text,
                                }).to_string();

                                tracing::info!(
                                    target: "task_history",
                                    "Plan submitted for approval, waiting..."
                                );

                                match approval.call(&approval_arg).await {
                                    Ok(result) => {
                                        let approved = result.get("approved")
                                            .and_then(|v| v.as_bool())
                                            .unwrap_or(false);
                                        let comment = result.get("comment")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");

                                        if approved {
                                            new_plan_mode = Some(false);
                                            let suffix = if comment.is_empty() { String::new() } else { format!(": {comment}") };
                                            format!("Plan approved{suffix}. You may now proceed with execution.")
                                        } else {
                                            // Rejected — stay in plan mode
                                            new_plan_mode = Some(true);
                                            format!("Plan rejected: {comment}. Revise your plan and call exit_plan_mode again.")
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!(error = %e, "Approval tool failed, proceeding without approval");
                                        new_plan_mode = Some(false);
                                        format!("Approval tool error ({e}). Proceeding with plan execution.")
                                    }
                                }
                            } else {
                                // No approval tool — proceed immediately
                                new_plan_mode = Some(false);
                                "Plan mode deactivated. All tools are now available. You may proceed with your plan.".to_string()
                            };

                            let tool_message: ChatCompletionRequestMessage =
                                ChatCompletionRequestToolMessageArgs::default()
                                    .content(tool_response)
                                    .tool_call_id(tool_call.id.clone())
                                    .build()?
                                    .into();
                            current_messages.push(tool_message);
                            continue;
                        }
                    }

                    let tool_result =
                        if let Some(tool) = Self::find_tool(all_step_tools, &tool_call.function.name) {
                            iter_tools.insert(tool.name().to_string());
                            tracing::info!(
                                target: "task_history",
                                tool = %tool.name(),
                                args = %tool_call.function.arguments,
                                "Executing tool"
                            );
                            let tool_start = std::time::Instant::now();
                            let result = match tool.call(&tool_call.function.arguments).await {
                                Ok(v) => {
                                    let r = v.to_string();
                                    tracing::info!(
                                        tool = %tool.name(),
                                        result_length = r.len(),
                                        duration_ms = tool_start.elapsed().as_millis() as u64,
                                        result_preview = %r.chars().take(200).collect::<String>(),
                                        "Tool returned result"
                                    );
                                    r
                                },
                                Err(e) => {
                                    tracing::warn!(
                                        tool = %tool.name(),
                                        error = %e,
                                        duration_ms = tool_start.elapsed().as_millis() as u64,
                                        "Tool execution failed"
                                    );
                                    format!("Tool error: {e}")
                                },
                            };
                            result
                        } else {
                            tracing::warn!(
                                tool = %tool_call.function.name,
                                "Unknown tool called by model"
                            );
                            "Unknown tool".to_string()
                        };

                    let tool_message: ChatCompletionRequestMessage =
                        ChatCompletionRequestToolMessageArgs::default()
                            .content(tool_result)
                            .tool_call_id(tool_call.id.clone())
                            .build()?
                            .into();
                    current_messages.push(tool_message);
                }

                Ok(IterationOutcome::Continue(iter_tools, new_plan_mode))
            }
            .instrument(iter_span)
            .await;

            match iter_result? {
                IterationOutcome::Done(content, _) => return Ok((content, tools_called)),
                IterationOutcome::Continue(iter_tools, plan_mode_change) => {
                    if let Some(new_mode) = plan_mode_change {
                        plan_mode = new_mode;
                        tracing::info!(
                            target: "task_history",
                            plan_mode = plan_mode,
                            "Plan mode state changed"
                        );
                    }
                    let count = iter_tools.len();
                    tools_called.extend(iter_tools);
                    tracing::debug!(
                        target: "task_history",
                        tools_called = count,
                        "Iteration completed, continuing loop"
                    );
                }
            }
        }

        // Hit the loop limit — return whatever content we have
        Err(format!(
            "Step tool calling loop exceeded maximum iterations ({max_iterations})"
        )
        .into())
    }
}

#[async_trait]
impl ModelCaller for RuntimeModelCaller {
    async fn call_model(
        &self,
        step_name: &str,
        model_name: &str,
        system_prompt: &str,
        user_message: &str,
        required_tools: &[Arc<dyn SpiceModelTool>],
        optional_tools: &[Arc<dyn SpiceModelTool>],
        max_iterations: usize,
        plan_mode: bool,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let llms = self.llms.read().await;
        let model = llms
            .get(model_name)
            .ok_or_else(|| format!("Model '{model_name}' not found in loaded LLMs"))?;
        let model = Arc::clone(model);
        drop(llms);

        // Auto-inject the fail tool + merge all step tools
        let fail_tool: Arc<dyn SpiceModelTool> = Arc::new(FailTool::new(None, None));
        let mut all_step_tools: Vec<Arc<dyn SpiceModelTool>> = vec![fail_tool];

        // Auto-inject plan mode tools when plan mode is configured
        if plan_mode {
            all_step_tools.push(Arc::new(EnterPlanModeTool::new(None, None)));
            all_step_tools.push(Arc::new(ExitPlanModeTool::new(None, None)));
        }

        all_step_tools.extend(required_tools.iter().cloned());
        all_step_tools.extend(optional_tools.iter().cloned());

        // Build the user message, injecting required tool instructions if needed
        let effective_user_message = if required_tools.is_empty() {
            user_message.to_string()
        } else {
            let tool_names: Vec<String> =
                required_tools.iter().map(|t| t.name().to_string()).collect();
            format!(
                "{user_message}\n\nYou MUST use the following tools in this step: [{}]. Call each at least once before responding.",
                tool_names.join(", ")
            )
        };

        // Build initial messages
        let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();
        if !system_prompt.is_empty() {
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt)
                    .build()?
                    .into(),
            );
        }
        messages.push(
            ChatCompletionRequestUserMessageArgs::default()
                .content(effective_user_message.as_str())
                .build()?
                .into(),
        );

        // Required tool tracking — persists across retries
        let required_names: HashSet<String> = required_tools
            .iter()
            .map(|t| t.name().to_string())
            .collect();
        let mut all_tools_used = HashSet::new();

        for attempt in 0..=REQUIRED_TOOL_RETRIES {
            let attempt_span = tracing::info_span!(
                target: "task_history",
                "model_call_attempt",
                step = %step_name,
                model = %model_name,
                attempt = attempt,
            );

            let attempt_result: Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> = async {
                let mut attempt_messages = messages.clone();

                // On retry, add a nudge about missing required tools
                if attempt > 0 {
                    let still_missing: Vec<String> = required_names
                        .iter()
                        .filter(|n| !all_tools_used.contains(*n))
                        .cloned()
                        .collect();
                    tracing::warn!(
                        target: "task_history",
                        attempt = attempt,
                        missing = ?still_missing,
                        "Retrying step because required tools were not used"
                    );
                    attempt_messages.push(
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(format!(
                                "You did not use the required tools: [{}]. You MUST call them before responding.",
                                still_missing.join(", ")
                            ))
                            .build()?
                            .into(),
                    );
                }

                let (content, tools_called) = Self::call_with_tool_loop(
                    &model,
                    step_name,
                    model_name,
                    attempt_messages,
                    &all_step_tools,
                    max_iterations,
                    plan_mode,
                )
                .await?;

                // Accumulate tools used across retries
                all_tools_used.extend(tools_called);

                // Check if all required tools were used (across all attempts)
                if required_names.is_empty() || required_names.is_subset(&all_tools_used) {
                    return Ok(Some(content));
                }

                let missing: Vec<&str> = required_names
                    .iter()
                    .filter(|n| !all_tools_used.contains(*n))
                    .map(String::as_str)
                    .collect();
                if !missing.is_empty() {
                    tracing::warn!(
                        target: "task_history",
                        missing_tools = ?missing,
                        attempt = attempt,
                        "Required tools not used by model"
                    );
                }

                Ok(None)
            }
            .instrument(attempt_span)
            .await;

            match attempt_result? {
                Some(content) => return Ok(content),
                None => continue,
            }
        }

        Err(format!(
            "Model failed to use required tools [{}] after {} retries",
            required_names
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
            REQUIRED_TOOL_RETRIES
        )
        .into())
    }
}

impl Runtime {
    pub(crate) async fn load_agents(self: Arc<Self>) {
        let app_lock = self.app.read().await;

        if let Some(app) = app_lock.as_ref() {
            if app.agents.is_empty() {
                return;
            }

            ensure_agent_results_table(&self.df).await;

            for agent in &app.agents {
                tracing::info!("Loading agent [{}]...", agent.name);
                match Self::load_agent(Arc::clone(&self), app, agent).await {
                    Ok(loaded) => {
                        tracing::info!(
                            "Agent [{}] loaded with {} pipeline(s), {} trigger(s)",
                            loaded.name,
                            loaded.pipelines.len(),
                            loaded.triggers.len(),
                        );
                        let mut agents = self.agents.write().await;
                        agents.insert(loaded.name.clone(), loaded);
                    }
                    Err(e) => {
                        tracing::error!("Failed to load agent [{}]: {e}", agent.name);
                    }
                }
            }
        }
    }

    async fn load_agent(
        rt: Arc<Runtime>,
        app: &App,
        agent: &Agent,
    ) -> Result<LoadedAgent, Box<dyn std::error::Error + Send + Sync>> {
        // Build read tools map from runtime's loaded tools
        let tools_lock = rt.tools.read().await;
        let mut available_read_tools: HashMap<String, Arc<dyn tools::SpiceModelTool>> =
            HashMap::new();
        for (name, tooling) in tools_lock.iter() {
            if let Some(tool) = tooling.as_individual() {
                available_read_tools.insert(name.clone(), Arc::clone(tool));
            }
        }
        drop(tools_lock);

        // Build write tools map from runtime's loaded write tools
        let write_tools_lock = rt.write_tools.read().await;
        let mut available_write_tools: HashMap<String, Arc<dyn tools::SpiceModelTool>> =
            HashMap::new();
        for (name, tooling) in write_tools_lock.iter() {
            if let Some(tool) = tooling.as_individual() {
                available_write_tools.insert(name.clone(), Arc::clone(tool));
            }
        }
        drop(write_tools_lock);

        // Add file_source tool names to agent's read_tools for pipeline resolution.
        // The tools themselves are already in rt.tools/rt.write_tools (registered during load_tools).
        let mut agent = agent.clone();
        for fs_name in &agent.file_sources.clone() {
            let Some(file_source) = app.file_sources.iter().find(|fs| fs.name == *fs_name) else {
                tracing::warn!(
                    "Agent '{}' references unknown file_source '{fs_name}', skipping",
                    agent.name
                );
                continue;
            };

            for tool_shorthand in &file_source.tools {
                let (tool_id, _is_write) =
                    file_source_tools::parse_tool_shorthand(tool_shorthand);
                if !agent.read_tools.contains(&tool_id.to_string()) {
                    agent.read_tools.push(tool_id.to_string());
                }
            }
        }

        // Resolve pipelines
        let mut resolved_pipelines = Vec::new();
        for pipeline_config in &agent.pipelines {
            let resolved = resolve_workflow(
                pipeline_config,
                &agent,
                &available_read_tools,
                &available_write_tools,
            )?;
            resolved_pipelines.push(resolved);
        }

        // Build trigger registry
        let mut trigger_registry = TriggerRegistry::new();
        trigger_registry.register(Arc::new(WebhookTriggerFactory));
        trigger_registry.register(Arc::new(ScheduleTriggerFactory));

        // Build session store
        let session_store: Arc<dyn SessionStore> = Arc::new(InMemorySessionStore::new());

        // Build session config
        let session_config = agent.session.clone().unwrap_or(SessionConfig {
            scope: Some("per_task".to_string()),
            reset: Some("never".to_string()),
            params: HashMap::new(),
        });

        // Build memory manager
        let memory_manager = build_memory_manager(&agent);

        // Build model caller
        let model_caller: Arc<dyn ModelCaller> = Arc::new(RuntimeModelCaller {
            llms: rt.completion_llms(),
        });

        // Create and start triggers for each pipeline
        let mut triggers: Vec<Box<dyn Trigger>> = Vec::new();
        let webhook_registry = rt.webhook_registry.clone();

        for pipeline in &resolved_pipelines {
            // Find the matching pipeline config to get the trigger config
            let pipeline_config = agent
                .pipelines
                .iter()
                .find(|p| p.name == pipeline.name);

            let Some(pipeline_config) = pipeline_config else {
                tracing::warn!(
                    "No pipeline config found for resolved pipeline '{}', skipping trigger",
                    pipeline.name
                );
                continue;
            };

            let trigger = match trigger_registry.create(&pipeline_config.trigger) {
                Ok(trigger) => trigger,
                Err(e) => {
                    tracing::error!(
                        "Failed to create trigger for pipeline '{}': {e}",
                        pipeline.name
                    );
                    continue;
                }
            };

            let handler: Arc<dyn TriggerHandler> = Arc::new(PipelineTriggerHandler {
                pipeline: pipeline.clone(),
                session_store: Arc::clone(&session_store),
                session_config: session_config.clone(),
                memory_manager: memory_manager.clone(),
                model_caller: Arc::clone(&model_caller),
                df: rt.datafusion(),
            });

            // For webhook triggers, register the handler in the webhook registry
            if pipeline_config.trigger.r#type == "webhook" {
                let path = pipeline_config
                    .trigger
                    .params
                    .get("path")
                    .cloned()
                    .unwrap_or_else(|| "/webhook".to_string());

                let mut registry = webhook_registry.write().await;
                registry.insert(path.clone(), Arc::clone(&handler));
                tracing::info!(
                    "Registered webhook handler for agent '{}' pipeline '{}' at path '{path}'",
                    agent.name,
                    pipeline.name,
                );
            }

            if let Err(e) = trigger.start(handler).await {
                tracing::error!(
                    "Failed to start trigger for pipeline '{}': {e}",
                    pipeline.name
                );
                continue;
            }

            triggers.push(trigger);
        }

        Ok(LoadedAgent {
            name: agent.name.clone(),
            config: agent.clone(),
            pipelines: resolved_pipelines,
            triggers,
            session_store,
            session_config,
            memory_manager,
        })
    }
}

fn build_memory_manager(agent: &Agent) -> Option<MemoryManager> {
    let memory_config = agent.memory.as_ref()?;
    let mut layers: Vec<Arc<dyn MemoryLayer>> = Vec::new();

    for layer_config in &memory_config.layers {
        let retention = layer_config
            .retention
            .as_deref()
            .and_then(parse_retention);

        let layer: Arc<dyn MemoryLayer> = match layer_config.r#type.as_str() {
            "session" => Arc::new(SessionLayer::new(retention)),
            "session_summary" => Arc::new(SessionSummaryLayer::new(retention)),
            "weekly_rollup" => Arc::new(WeeklyRollupLayer::new(retention)),
            "knowledge_base" => Arc::new(KnowledgeBaseLayer::new()),
            other => {
                tracing::warn!(
                    "Unknown memory layer type '{}' for agent '{}', skipping",
                    other,
                    agent.name
                );
                continue;
            }
        };
        layers.push(layer);
    }

    let session_threshold = memory_config
        .compaction
        .as_ref()
        .and_then(|c| c.session_threshold)
        .unwrap_or(50);

    Some(MemoryManager::new(layers, session_threshold))
}

/// Escape single quotes in a string for safe SQL insertion.
fn sql_escape(s: &str) -> String {
    s.replace('\'', "''")
}

const AGENT_RESULTS_DDL: &str = "\
CREATE TABLE IF NOT EXISTS agentic.default.agent_results (\
  \"id\" TEXT NOT NULL, \
  \"agent_name\" TEXT, \
  \"pipeline_name\" TEXT, \
  \"trigger_source\" TEXT, \
  \"output\" TEXT, \
  \"created_at\" TIMESTAMP, \
  PRIMARY KEY (\"id\")\
)";

/// Ensure the `agentic.default.agent_results` table exists.
/// Silently skips if the `agentic` catalog is not configured.
pub async fn ensure_agent_results_table(df: &Arc<DataFusion>) {
    match df.query_builder(AGENT_RESULTS_DDL).build().run().await {
        Ok(query_result) => {
            if let Err(e) = query_result.data.try_collect::<Vec<_>>().await {
                tracing::warn!("Failed to commit agent_results table creation: {e}");
            } else {
                tracing::info!("Agent results table ready");
            }
        }
        Err(e) => tracing::warn!("Could not create agent_results table (is the 'agentic' catalog configured?): {e}"),
    }
}
