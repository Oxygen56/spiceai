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

use std::time::Instant;

use chrono::Utc;
use tracing_futures::Instrument;

use crate::memory::{MemoryEntry, MemoryManager};
use crate::session::{self, Session, SessionStore, Turn};
use crate::trigger::TriggerPayload;
use spicepod::component::session::SessionConfig;

use super::vote::{self, ModelCaller};
use super::{ResolvedStep, ResolvedWorkflow, SessionResetHandler};

/// Result of executing a workflow.
#[derive(Debug, Clone)]
pub struct WorkflowExecutionResult {
    pub workflow_name: String,
    pub session_id: String,
    pub steps_executed: usize,
    pub final_output: String,
}

/// Execute a resolved workflow in response to a trigger payload.
///
/// Execution flow:
/// 1. Resolve session via `SessionStore` (scope key from config)
/// 2. Check reset policy — close old session if expired, create new
/// 3. Gather context from `MemoryManager`
/// 4. For each step sequentially:
///    a. Build messages with system prompt + step prompt + memory + session + prev output
///    b. If vote step: fan out to proposer models, judge selects
///    c. If standard step: invoke model
///    d. Step output becomes context for next step
/// 5. Store interaction as Turn
/// 6. If threshold exceeded, trigger compaction notification
pub async fn execute_workflow(
    workflow: &ResolvedWorkflow,
    payload: &TriggerPayload,
    session_store: &dyn SessionStore,
    session_config: &SessionConfig,
    memory_manager: Option<&MemoryManager>,
    model_caller: &dyn ModelCaller,
    on_session_reset: Option<&dyn SessionResetHandler>,
) -> Result<WorkflowExecutionResult, Box<dyn std::error::Error + Send + Sync>> {
    let workflow_span = tracing::info_span!(
        target: "task_history",
        "workflow_execution",
        workflow = %workflow.name,
        agent = %workflow.agent_name,
        default_model = %workflow.default_model,
        step_count = workflow.steps.len(),
    );

    execute_workflow_inner(
        workflow,
        payload,
        session_store,
        session_config,
        memory_manager,
        model_caller,
        on_session_reset,
    )
    .instrument(workflow_span)
    .await
}

async fn execute_workflow_inner(
    workflow: &ResolvedWorkflow,
    payload: &TriggerPayload,
    session_store: &dyn SessionStore,
    session_config: &SessionConfig,
    memory_manager: Option<&MemoryManager>,
    model_caller: &dyn ModelCaller,
    on_session_reset: Option<&dyn SessionResetHandler>,
) -> Result<WorkflowExecutionResult, Box<dyn std::error::Error + Send + Sync>> {
    let workflow_start = Instant::now();

    // 1. Resolve session
    let scope_key = session::resolve_scope_key(
        session_config,
        &workflow.name,
        &workflow.agent_name,
        &payload.source,
    );
    let mut current_session = session_store
        .get_or_create(&workflow.agent_name, &scope_key)
        .await?;

    // 2. Check reset policy
    if session::should_reset(&current_session, session_config) {
        tracing::info!(
            target: "task_history",
            session_id = %current_session.id,
            workflow = %workflow.name,
            "Session reset policy triggered, closing old session"
        );
        if let Some(handler) = on_session_reset {
            handler.on_session_reset();
        }
        session_store.close(&current_session.id).await?;
        current_session = session_store
            .create(&workflow.agent_name, &scope_key)
            .await?;
    }

    tracing::info!(
        target: "task_history",
        session_id = %current_session.id,
        session_turns = current_session.turns.len(),
        "Workflow using session"
    );

    // 3. Gather memory context
    let memory_context = if let Some(mm) = memory_manager {
        let entries = mm.gather_context(&workflow.agent_name).await?;
        format_memory_context(&entries)
    } else {
        String::new()
    };

    // 4. Build session context from prior turns
    let session_context = format_session_context(&current_session);

    // 5. Execute steps sequentially
    let mut step_output = serde_json::to_string(&payload.data).unwrap_or_default();
    let mut steps_executed = 0;

    tracing::info!(
        workflow = %workflow.name,
        agent = %workflow.agent_name,
        model = %workflow.default_model,
        step_count = workflow.steps.len(),
        trigger_source = %payload.source,
        payload_size = step_output.len(),
        "Starting workflow execution"
    );

    for (step_index, resolved_step) in workflow.steps.iter().enumerate() {
        let step_start = Instant::now();

        match resolved_step {
            ResolvedStep::Standard(step) => {
                let model_name = step
                    .model
                    .as_deref()
                    .unwrap_or(&workflow.default_model);

                let step_span = tracing::info_span!(
                    target: "task_history",
                    "workflow_step",
                    step = %step.name,
                    model = %model_name,
                    workflow = %workflow.name,
                    step_index = step_index,
                    step_type = "standard",
                );

                let required_tool_names: Vec<String> = step.required_write_tools.iter().map(|t| t.name().to_string()).collect();
                let optional_tool_names: Vec<String> = step.optional_write_tools.iter().map(|t| t.name().to_string()).collect();
                tracing::info!(
                    step = %step.name,
                    model = %model_name,
                    step_index = step_index,
                    required_tools = ?required_tool_names,
                    optional_tools = ?optional_tool_names,
                    read_tools = step.read_tools.len(),
                    max_iterations = step.max_iterations,
                    input_length = step_output.len(),
                    "Starting step execution"
                );

                let step_result: Result<String, Box<dyn std::error::Error + Send + Sync>> = async {
                    let user_message = build_step_message(
                        &step.prompt,
                        &step_output,
                        &memory_context,
                        &session_context,
                    );

                    // Combine read tools + agent read tools + optional write tools as optional
                    let mut optional_tools = step.read_tools.clone();
                    optional_tools.extend(step.optional_write_tools.iter().cloned());
                    optional_tools.extend(workflow.agent_read_tools.iter().cloned());

                    model_caller
                        .call_model(
                            &step.name,
                            model_name,
                            &workflow.system_prompt,
                            &user_message,
                            &step.required_write_tools,
                            &optional_tools,
                            step.max_iterations,
                            workflow.plan_mode,
                        )
                        .await
                }
                .instrument(step_span.clone())
                .await;

                let duration_ms = step_start.elapsed().as_millis() as u64;

                match step_result {
                    Ok(output) => {
                        let captured: String = output.chars().take(500).collect();
                        tracing::info!(
                            target: "task_history",
                            parent: &step_span,
                            captured_output = %captured,
                            duration_ms = duration_ms,
                            "Standard step completed"
                        );
                        step_output = output;
                    }
                    Err(e) => {
                        tracing::error!(
                            target: "task_history",
                            parent: &step_span,
                            duration_ms = duration_ms,
                            "{e}"
                        );
                        return Err(e);
                    }
                }
            }
            ResolvedStep::Vote(vote_step) => {
                let step_span = tracing::info_span!(
                    target: "task_history",
                    "workflow_step",
                    workflow = %workflow.name,
                    step = %vote_step.name,
                    step_index = step_index,
                    step_type = "vote",
                    proposer_count = vote_step.proposer_models.len(),
                    judge_model = %vote_step.judge_model,
                );

                let step_result = async {
                    vote::execute_vote_step(
                        &vote_step.name,
                        &vote_step.prompt,
                        &step_output,
                        &vote_step.proposer_models,
                        &vote_step.judge_model,
                        vote_step.judge_prompt.as_deref(),
                        model_caller,
                    )
                    .await
                }
                .instrument(step_span.clone())
                .await;

                let duration_ms = step_start.elapsed().as_millis() as u64;

                match step_result {
                    Ok(vote_result) => {
                        let captured: String =
                            vote_result.selected.content.chars().take(500).collect();
                        tracing::info!(
                            target: "task_history",
                            parent: &step_span,
                            captured_output = %captured,
                            selected_model = %vote_result.selected.model,
                            proposals_count = vote_result.proposals.len(),
                            duration_ms = duration_ms,
                            "Vote step completed"
                        );
                        step_output = vote_result.selected.content;
                    }
                    Err(e) => {
                        tracing::error!(
                            target: "task_history",
                            parent: &step_span,
                            duration_ms = duration_ms,
                            "{e}"
                        );
                        return Err(e);
                    }
                }
            }
        }
        steps_executed += 1;
    }

    // 6. Store the entire interaction as a Turn
    let turn = Turn {
        index: current_session.turns.len() as u32,
        role: "workflow".to_string(),
        content: step_output.clone(),
        created_at: Utc::now(),
    };
    session_store.add_turn(&current_session.id, turn).await?;

    // 7. Store in session memory layer if available
    if let Some(mm) = memory_manager {
        let entry = MemoryEntry {
            id: uuid::Uuid::now_v7().to_string(),
            agent: workflow.agent_name.clone(),
            layer: "session".to_string(),
            content: step_output.clone(),
            metadata: std::collections::HashMap::new(),
            created_at: Utc::now(),
            expires_at: None,
        };
        if let Err(e) = mm.store_in_layer("session", vec![entry]).await {
            tracing::warn!(
                target: "task_history",
                workflow = %workflow.name,
                "Failed to store workflow output in session memory: {e}"
            );
        }
    }

    // 8. Check compaction threshold and compact if needed
    if let Some(mm) = memory_manager {
        let updated_session = session_store.get(&current_session.id).await?;
        if let Some(s) = updated_session {
            if s.turns.len() as u32 >= mm.session_threshold() {
                match mm.compact_session(&workflow.agent_name).await {
                    Ok(result) => {
                        tracing::info!(
                            target: "task_history",
                            session_id = %s.id,
                            entries_compacted = result.entries_compacted,
                            summaries_produced = result.summaries_produced,
                            "Session memory compacted"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "task_history",
                            session_id = %s.id,
                            "Session memory compaction failed: {e}"
                        );
                    }
                }
            }
        }
    }

    let total_duration_ms = workflow_start.elapsed().as_millis() as u64;
    tracing::info!(
        target: "task_history",
        workflow = %workflow.name,
        session_id = %current_session.id,
        steps_executed = steps_executed,
        total_duration_ms = total_duration_ms,
        "Workflow execution completed"
    );

    Ok(WorkflowExecutionResult {
        workflow_name: workflow.name.clone(),
        session_id: current_session.id,
        steps_executed,
        final_output: step_output,
    })
}

fn build_step_message(
    step_prompt: &str,
    previous_output: &str,
    memory_context: &str,
    session_context: &str,
) -> String {
    let mut parts = Vec::new();

    if !memory_context.is_empty() {
        parts.push(format!("## Memory Context\n{memory_context}"));
    }
    if !session_context.is_empty() {
        parts.push(format!("## Session History\n{session_context}"));
    }
    if !previous_output.is_empty() {
        parts.push(format!("## Previous Step Output\n{previous_output}"));
    }
    parts.push(format!("## Current Task\n{step_prompt}"));

    parts.join("\n\n")
}

fn format_memory_context(entries: &[MemoryEntry]) -> String {
    if entries.is_empty() {
        return String::new();
    }
    entries
        .iter()
        .map(|e| format!("[{}] {}", e.layer, e.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_session_context(session: &Session) -> String {
    if session.turns.is_empty() {
        return String::new();
    }
    session
        .turns
        .iter()
        .map(|t| format!("[{}] {}: {}", t.created_at.format("%H:%M"), t.role, t.content))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::in_memory::InMemorySessionStore;
    use crate::workflow::StandardStep;
    use async_trait::async_trait;

    struct EchoModelCaller;

    #[async_trait]
    impl ModelCaller for EchoModelCaller {
        async fn call_model(
            &self,
            _step_name: &str,
            _model_name: &str,
            _system_prompt: &str,
            user_message: &str,
            _required_tools: &[std::sync::Arc<dyn tools::SpiceModelTool>],
            _optional_tools: &[std::sync::Arc<dyn tools::SpiceModelTool>],
            _max_iterations: usize,
            _plan_mode: bool,
        ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
            Ok(format!(
                "Response to: {}",
                user_message.chars().take(50).collect::<String>()
            ))
        }
    }

    #[tokio::test]
    async fn test_execute_workflow_basic() {
        let workflow = ResolvedWorkflow {
            name: "test-workflow".to_string(),
            agent_name: "test-agent".to_string(),
            system_prompt: "You are a helpful assistant.".to_string(),
            default_model: "gpt-4".to_string(),
            steps: vec![ResolvedStep::Standard(StandardStep {
                name: "step1".to_string(),
                prompt: "Analyze the input.".to_string(),
                model: None,
                read_tools: vec![],
                required_write_tools: vec![],
                optional_write_tools: vec![],
                max_iterations: 10,
            })],
            agent_read_tools: vec![],
            datasets: vec![],
            file_sources: vec![],
            plan_mode: false,
        };

        let session_store = InMemorySessionStore::new();
        let session_config = SessionConfig {
            scope: Some("per_task".to_string()),
            reset: Some("never".to_string()),
            params: std::collections::HashMap::new(),
        };

        let payload = TriggerPayload {
            source: "test".to_string(),
            data: serde_json::json!({"alert": "high cpu"}),
            received_at: Utc::now(),
        };

        let result = execute_workflow(
            &workflow,
            &payload,
            &session_store,
            &session_config,
            None,
            &EchoModelCaller,
            None,
        )
        .await
        .expect("should succeed");

        assert_eq!(result.workflow_name, "test-workflow");
        assert_eq!(result.steps_executed, 1);
        assert!(!result.final_output.is_empty());
        assert!(!result.session_id.is_empty());
    }

    #[tokio::test]
    async fn test_execute_workflow_multi_step() {
        let workflow = ResolvedWorkflow {
            name: "multi".to_string(),
            agent_name: "agent".to_string(),
            system_prompt: "System.".to_string(),
            default_model: "gpt-4".to_string(),
            steps: vec![
                ResolvedStep::Standard(StandardStep {
                    name: "step1".to_string(),
                    prompt: "Gather data.".to_string(),
                    model: None,
                    read_tools: vec![],
                    required_write_tools: vec![],
                    optional_write_tools: vec![],
                    max_iterations: 10,
                }),
                ResolvedStep::Standard(StandardStep {
                    name: "step2".to_string(),
                    prompt: "Summarize.".to_string(),
                    model: Some("claude-3".to_string()),
                    read_tools: vec![],
                    required_write_tools: vec![],
                    optional_write_tools: vec![],
                    max_iterations: 10,
                }),
            ],
            agent_read_tools: vec![],
            datasets: vec![],
            file_sources: vec![],
            plan_mode: false,
        };

        let session_store = InMemorySessionStore::new();
        let session_config = SessionConfig {
            scope: None,
            reset: None,
            params: std::collections::HashMap::new(),
        };

        let payload = TriggerPayload {
            source: "webhook".to_string(),
            data: serde_json::json!(null),
            received_at: Utc::now(),
        };

        let result = execute_workflow(
            &workflow,
            &payload,
            &session_store,
            &session_config,
            None,
            &EchoModelCaller,
            None,
        )
        .await
        .expect("should succeed");

        assert_eq!(result.steps_executed, 2);
    }
}
