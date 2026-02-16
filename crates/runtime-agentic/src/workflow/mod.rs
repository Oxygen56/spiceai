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

use std::sync::Arc;
use tools::SpiceModelTool;

pub mod executor;
pub mod resolve;
pub mod vote;

/// Hook called when a session reset is triggered.
///
/// Implementations can use this to clean up session-scoped resources
/// (e.g., git worktrees) before the old session is closed.
pub trait SessionResetHandler: Send + Sync {
    fn on_session_reset(&self);
}

/// A fully resolved workflow ready for execution.
#[derive(Clone)]
pub struct ResolvedWorkflow {
    pub name: String,
    pub agent_name: String,
    pub system_prompt: String,
    pub default_model: String,
    pub steps: Vec<ResolvedStep>,
    /// Read tools available to all steps (from agent.read_tools).
    pub agent_read_tools: Vec<Arc<dyn SpiceModelTool>>,
    pub datasets: Vec<String>,
    pub file_sources: Vec<String>,
}

impl std::fmt::Debug for ResolvedWorkflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedWorkflow")
            .field("name", &self.name)
            .field("agent_name", &self.agent_name)
            .field("default_model", &self.default_model)
            .field("steps", &self.steps.len())
            .field("agent_read_tools", &self.agent_read_tools.len())
            .finish()
    }
}

#[derive(Clone)]
pub enum ResolvedStep {
    Standard(StandardStep),
    Vote(ResolvedVoteStep),
}

#[derive(Clone)]
pub struct StandardStep {
    pub name: String,
    pub prompt: String,
    pub model: Option<String>,
    /// Read tools specific to this step (in addition to agent read tools).
    pub read_tools: Vec<Arc<dyn SpiceModelTool>>,
    /// Required write tools for this step.
    pub required_write_tools: Vec<Arc<dyn SpiceModelTool>>,
    /// Optional write tools for this step.
    pub optional_write_tools: Vec<Arc<dyn SpiceModelTool>>,
    /// Maximum number of tool-calling round-trips for this step.
    pub max_iterations: usize,
}

#[derive(Clone)]
pub struct ResolvedVoteStep {
    pub name: String,
    pub prompt: String,
    pub proposer_models: Vec<String>,
    pub judge_model: String,
    pub judge_prompt: Option<String>,
}
