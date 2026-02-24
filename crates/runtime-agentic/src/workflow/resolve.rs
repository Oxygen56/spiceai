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

use spicepod::component::agent::Agent;
use spicepod::component::pipeline::{PipelineConfig, StepConfig, StepTools};
use std::collections::HashMap;
use std::sync::Arc;
use tools::SpiceModelTool;

use super::{ResolvedStep, ResolvedVoteStep, ResolvedWorkflow, StandardStep};

/// Resolve a pipeline config into a fully resolved workflow with per-step tool scoping.
pub fn resolve_workflow(
    pipeline_config: &PipelineConfig,
    agent: &Agent,
    available_read_tools: &HashMap<String, Arc<dyn SpiceModelTool>>,
    available_write_tools: &HashMap<String, Arc<dyn SpiceModelTool>>,
) -> Result<ResolvedWorkflow, Box<dyn std::error::Error + Send + Sync>> {
    let mut missing_tools = Vec::new();
    let agent_read_tools: Vec<Arc<dyn SpiceModelTool>> = agent
        .read_tools
        .iter()
        .filter_map(|name| match available_read_tools.get(name) {
            Some(tool) => Some(tool.clone()),
            None => {
                missing_tools.push(name.clone());
                None
            }
        })
        .collect();

    if !missing_tools.is_empty() {
        return Err(format!(
            "Agent '{}' references unavailable read tools: [{}]",
            agent.name,
            missing_tools.join(", ")
        )
        .into());
    }

    let steps = pipeline_config
        .steps
        .iter()
        .map(|step| resolve_step(step, available_read_tools, available_write_tools))
        .collect::<Result<Vec<_>, _>>()?;

    let plan_mode = agent
        .planning
        .as_ref()
        .is_some_and(|p| p.start_in_plan_mode);

    let system_prompt = if plan_mode {
        format!(
            "{}\n\n## Planning Mode\n\n\
             You start in planning mode. In this mode:\n\
             - You have access to ALL tools, but you MUST only use them for \
             read-only purposes (queries, searches, file reads, git log, git status, etc.)\n\
             - Do NOT make any changes, write files, create commits, or perform \
             any operations with side effects\n\
             - Research and gather information, then call `exit_plan_mode` with \
             your plan describing what actions you will take\n\
             - After your plan is approved, you may proceed with write operations\n\n\
             Tools available:\n\
             - `enter_plan_mode(reason)` \u{2014} Re-enter planning mode to research further\n\
             - `exit_plan_mode(plan)` \u{2014} Submit your plan for approval and unlock all tools",
            agent.prompt
        )
    } else {
        agent.prompt.clone()
    };

    Ok(ResolvedWorkflow {
        name: pipeline_config.name.clone(),
        agent_name: agent.name.clone(),
        system_prompt,
        default_model: agent.model.clone(),
        steps,
        agent_read_tools,
        datasets: agent.datasets.clone(),
        file_sources: agent.file_sources.clone(),
        plan_mode,
    })
}

fn resolve_step(
    step: &StepConfig,
    available_read_tools: &HashMap<String, Arc<dyn SpiceModelTool>>,
    available_write_tools: &HashMap<String, Arc<dyn SpiceModelTool>>,
) -> Result<ResolvedStep, Box<dyn std::error::Error + Send + Sync>> {
    if step.r#type.as_deref() == Some("vote") {
        return resolve_vote_step(step);
    }

    let (read_tools, required_write, optional_write) = match &step.tools {
        Some(StepTools::Simple(names)) => {
            let mut missing = Vec::new();
            let tools = names
                .iter()
                .filter_map(|n| match available_read_tools.get(n) {
                    Some(tool) => Some(tool.clone()),
                    None => {
                        missing.push(n.clone());
                        None
                    }
                })
                .collect();
            if !missing.is_empty() {
                return Err(format!(
                    "Step '{}' references unavailable read tools: [{}]",
                    step.name,
                    missing.join(", ")
                )
                .into());
            }
            (tools, vec![], vec![])
        }
        Some(StepTools::Structured { required, optional }) => {
            let mut missing = Vec::new();
            let req = required
                .iter()
                .filter_map(|n| match available_write_tools.get(n) {
                    Some(tool) => Some(tool.clone()),
                    None => {
                        missing.push(n.clone());
                        None
                    }
                })
                .collect();
            if !missing.is_empty() {
                return Err(format!(
                    "Step '{}' references unavailable required write tools: [{}]",
                    step.name,
                    missing.join(", ")
                )
                .into());
            }

            let mut missing_opt = Vec::new();
            let opt = optional
                .iter()
                .filter_map(|n| match available_write_tools.get(n) {
                    Some(tool) => Some(tool.clone()),
                    None => {
                        missing_opt.push(n.clone());
                        None
                    }
                })
                .collect();
            if !missing_opt.is_empty() {
                return Err(format!(
                    "Step '{}' references unavailable optional write tools: [{}]",
                    step.name,
                    missing_opt.join(", ")
                )
                .into());
            }

            (vec![], req, opt)
        }
        None => (vec![], vec![], vec![]),
    };

    Ok(ResolvedStep::Standard(StandardStep {
        name: step.name.clone(),
        prompt: step.prompt.clone().unwrap_or_default(),
        model: step.model.clone(),
        read_tools,
        required_write_tools: required_write,
        optional_write_tools: optional_write,
        max_iterations: step.max_iterations.unwrap_or(10),
    }))
}

fn resolve_vote_step(
    step: &StepConfig,
) -> Result<ResolvedStep, Box<dyn std::error::Error + Send + Sync>> {
    let models = step
        .models
        .clone()
        .ok_or("Vote step requires 'models' field")?;
    let judge = step
        .judge
        .as_ref()
        .ok_or("Vote step requires 'judge' field")?;

    Ok(ResolvedStep::Vote(ResolvedVoteStep {
        name: step.name.clone(),
        prompt: step.prompt.clone().unwrap_or_default(),
        proposer_models: models,
        judge_model: judge.model.clone(),
        judge_prompt: judge.prompt.clone(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::Value;
    use spicepod::component::pipeline::{JudgeConfig, TriggerConfig};
    use std::borrow::Cow;

    #[derive(Clone)]
    struct MockTool {
        tool_name: String,
    }

    #[async_trait]
    impl SpiceModelTool for MockTool {
        fn name(&self) -> Cow<'_, str> {
            Cow::Borrowed(&self.tool_name)
        }

        fn description(&self) -> Option<Cow<'_, str>> {
            None
        }

        fn parameters(&self) -> Option<Value> {
            None
        }

        async fn call(
            &self,
            _arg: &str,
        ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
            Ok(Value::Null)
        }
    }

    fn make_agent(read_tools: Vec<&str>) -> Agent {
        Agent {
            name: "test-agent".to_string(),
            model: "gpt-4".to_string(),
            prompt: "You are a helpful assistant.".to_string(),
            datasets: vec!["ds1".to_string()],
            file_sources: vec!["fs1".to_string()],
            read_tools: read_tools.into_iter().map(String::from).collect(),
            session: None,
            memory: None,
            planning: None,
            pipelines: vec![],
            depends_on: vec![],
        }
    }

    fn make_trigger() -> TriggerConfig {
        TriggerConfig {
            r#type: "webhook".to_string(),
            params: HashMap::new(),
        }
    }

    fn make_read_tools(names: &[&str]) -> HashMap<String, Arc<dyn SpiceModelTool>> {
        names
            .iter()
            .map(|n| {
                (
                    n.to_string(),
                    Arc::new(MockTool {
                        tool_name: n.to_string(),
                    }) as Arc<dyn SpiceModelTool>,
                )
            })
            .collect()
    }

    fn make_write_tools(names: &[&str]) -> HashMap<String, Arc<dyn SpiceModelTool>> {
        make_read_tools(names)
    }

    #[test]
    fn test_resolve_workflow_with_simple_tools() {
        let agent = make_agent(vec!["search", "read_file"]);
        let read_tools = make_read_tools(&["search", "read_file", "list_files"]);
        let write_tools = make_write_tools(&[]);

        let pipeline = PipelineConfig {
            name: "analysis".to_string(),
            trigger: make_trigger(),
            steps: vec![StepConfig {
                name: "gather".to_string(),
                r#type: None,
                prompt: Some("Gather information".to_string()),
                model: None,
                tools: Some(StepTools::Simple(vec![
                    "search".to_string(),
                    "list_files".to_string(),
                ])),
                models: None,
                judge: None,
                max_iterations: None,
            }],
        };

        let resolved =
            resolve_workflow(&pipeline, &agent, &read_tools, &write_tools).unwrap();

        assert_eq!(resolved.name, "analysis");
        assert_eq!(resolved.agent_name, "test-agent");
        assert_eq!(resolved.default_model, "gpt-4");
        assert_eq!(resolved.system_prompt, "You are a helpful assistant.");
        assert_eq!(resolved.datasets, vec!["ds1"]);
        assert_eq!(resolved.file_sources, vec!["fs1"]);
        assert_eq!(resolved.agent_read_tools.len(), 2);

        assert_eq!(resolved.steps.len(), 1);
        match &resolved.steps[0] {
            ResolvedStep::Standard(step) => {
                assert_eq!(step.name, "gather");
                assert_eq!(step.prompt, "Gather information");
                assert_eq!(step.read_tools.len(), 2);
                assert!(step.required_write_tools.is_empty());
                assert!(step.optional_write_tools.is_empty());
            }
            ResolvedStep::Vote(_) => panic!("Expected standard step"),
        }
    }

    #[test]
    fn test_resolve_workflow_with_vote_step() {
        let agent = make_agent(vec![]);
        let read_tools = make_read_tools(&[]);
        let write_tools = make_write_tools(&[]);

        let pipeline = PipelineConfig {
            name: "consensus".to_string(),
            trigger: make_trigger(),
            steps: vec![StepConfig {
                name: "vote-step".to_string(),
                r#type: Some("vote".to_string()),
                prompt: Some("What is the best approach?".to_string()),
                model: None,
                tools: None,
                models: Some(vec![
                    "gpt-4".to_string(),
                    "claude-3".to_string(),
                    "gemini".to_string(),
                ]),
                judge: Some(JudgeConfig {
                    model: "gpt-4-turbo".to_string(),
                    prompt: Some("Pick the most thorough answer.".to_string()),
                }),
                max_iterations: None,
            }],
        };

        let resolved =
            resolve_workflow(&pipeline, &agent, &read_tools, &write_tools).unwrap();

        assert_eq!(resolved.steps.len(), 1);
        match &resolved.steps[0] {
            ResolvedStep::Vote(step) => {
                assert_eq!(step.name, "vote-step");
                assert_eq!(step.prompt, "What is the best approach?");
                assert_eq!(
                    step.proposer_models,
                    vec!["gpt-4", "claude-3", "gemini"]
                );
                assert_eq!(step.judge_model, "gpt-4-turbo");
                assert_eq!(
                    step.judge_prompt,
                    Some("Pick the most thorough answer.".to_string())
                );
            }
            ResolvedStep::Standard(_) => panic!("Expected vote step"),
        }
    }

    #[test]
    fn test_resolve_vote_step_missing_models() {
        let agent = make_agent(vec![]);
        let read_tools = make_read_tools(&[]);
        let write_tools = make_write_tools(&[]);

        let pipeline = PipelineConfig {
            name: "bad-vote".to_string(),
            trigger: make_trigger(),
            steps: vec![StepConfig {
                name: "vote-step".to_string(),
                r#type: Some("vote".to_string()),
                prompt: Some("Question?".to_string()),
                model: None,
                tools: None,
                models: None,
                judge: Some(JudgeConfig {
                    model: "judge-model".to_string(),
                    prompt: None,
                }),
                max_iterations: None,
            }],
        };

        let result = resolve_workflow(&pipeline, &agent, &read_tools, &write_tools);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Vote step requires 'models' field"));
    }
}
