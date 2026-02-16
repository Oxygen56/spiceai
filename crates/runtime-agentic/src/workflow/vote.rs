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

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::future::join_all;
use serde::{Deserialize, Serialize};
use tools::SpiceModelTool;

/// Result of executing a vote step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteExecution {
    pub proposals: Vec<Proposal>,
    pub selected: Proposal,
    pub judge_reasoning: String,
}

/// A single proposal from one of the proposer models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub model: String,
    pub content: String,
    pub created_at: DateTime<Utc>,
}

/// Abstraction for invoking an LLM model by name.
#[async_trait]
pub trait ModelCaller: Send + Sync {
    async fn call_model(
        &self,
        step_name: &str,
        model_name: &str,
        system_prompt: &str,
        user_message: &str,
        required_tools: &[Arc<dyn SpiceModelTool>],
        optional_tools: &[Arc<dyn SpiceModelTool>],
        max_iterations: usize,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
}

const DEFAULT_JUDGE_INSTRUCTION: &str =
    "Select the best proposal based on accuracy, completeness, and clarity.";

/// Execute a vote step: fan out to multiple proposer models in parallel, then
/// have a judge model select the best proposal.
pub async fn execute_vote_step(
    step_name: &str,
    prompt: &str,
    context: &str,
    proposer_models: &[String],
    judge_model: &str,
    judge_prompt: Option<&str>,
    caller: &dyn ModelCaller,
) -> Result<VoteExecution, Box<dyn std::error::Error + Send + Sync>> {
    // 1. Fan out: all proposer models receive same context + prompt in parallel.
    let user_message = if context.is_empty() {
        prompt.to_string()
    } else {
        format!("{context}\n\n{prompt}")
    };

    let futures: Vec<_> = proposer_models
        .iter()
        .map(|model| {
            let msg = user_message.clone();
            let model_name = model.clone();
            async move {
                let result = caller
                    .call_model(step_name, &model_name, "", &msg, &[], &[], 1)
                    .await;
                (model_name, result)
            }
        })
        .collect();

    let results = join_all(futures).await;

    // 2. Collect proposals, skipping failures with a warning.
    let mut proposals: Vec<Proposal> = Vec::new();
    for (model_name, result) in results {
        match result {
            Ok(content) => {
                proposals.push(Proposal {
                    model: model_name,
                    content,
                    created_at: Utc::now(),
                });
            }
            Err(e) => {
                tracing::warn!(
                    step = step_name,
                    model = %model_name,
                    error = %e,
                    "Proposer model failed in vote step, skipping"
                );
            }
        }
    }

    if proposals.is_empty() {
        return Err(format!(
            "Vote step '{step_name}': all proposer models failed, cannot proceed"
        )
        .into());
    }

    // If there is only one proposal, skip judging and return it directly.
    if proposals.len() == 1 {
        let selected = proposals[0].clone();
        return Ok(VoteExecution {
            proposals: proposals.clone(),
            selected,
            judge_reasoning: "Only one proposal available; selected by default.".to_string(),
        });
    }

    // 3. Build the judge prompt with all proposals numbered.
    let judge_instruction = judge_prompt.unwrap_or(DEFAULT_JUDGE_INSTRUCTION);
    let judge_system = build_judge_system_prompt(judge_instruction, &proposals);

    let judge_user_message = format!(
        "The original task was:\n{prompt}\n\nPlease evaluate the proposals above and select the best one."
    );

    // 4. Judge model selects the best proposal.
    let judge_response = match caller
        .call_model(step_name, judge_model, &judge_system, &judge_user_message, &[], &[], 1)
        .await
    {
        Ok(response) => response,
        Err(e) => {
            tracing::warn!(
                step = step_name,
                judge = judge_model,
                error = %e,
                "Judge model failed, falling back to first proposal"
            );
            let selected = proposals[0].clone();
            return Ok(VoteExecution {
                proposals: proposals.clone(),
                selected,
                judge_reasoning: format!("Judge model failed ({e}); fell back to first proposal."),
            });
        }
    };

    // 5. Parse judge output to identify the selected proposal.
    let selected_index = parse_selected_proposal(&judge_response, proposals.len());
    let selected = proposals[selected_index].clone();

    Ok(VoteExecution {
        proposals,
        selected,
        judge_reasoning: judge_response,
    })
}

/// Build the system prompt for the judge model, enumerating all proposals.
fn build_judge_system_prompt(judge_instruction: &str, proposals: &[Proposal]) -> String {
    let mut prompt = String::from("You are evaluating proposals from multiple models.\n\n");
    prompt.push_str(judge_instruction);
    prompt.push_str("\n\nProposals:\n");

    for (i, proposal) in proposals.iter().enumerate() {
        prompt.push_str(&format!(
            "\n## Proposal {} (from {})\n{}\n",
            i + 1,
            proposal.model,
            proposal.content,
        ));
    }

    prompt.push_str(
        "\nSelect the best proposal by responding with \"Selected: N\" where N is the proposal number, followed by your reasoning.",
    );

    prompt
}

/// Parse the judge's response to find which proposal was selected.
/// Looks for "Selected: N" pattern. Falls back to proposal 0 if unparseable.
fn parse_selected_proposal(response: &str, num_proposals: usize) -> usize {
    // Look for "Selected: N" pattern (case-insensitive).
    let lower = response.to_lowercase();
    if let Some(pos) = lower.find("selected:") {
        let after = &response[pos + "selected:".len()..];
        let trimmed = after.trim_start();
        // Parse the number at the start of the remaining text.
        let num_str: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = num_str.parse::<usize>() {
            if n >= 1 && n <= num_proposals {
                return n - 1; // Convert to 0-indexed.
            }
        }
    }

    tracing::warn!(
        "Could not parse judge selection from response, defaulting to first proposal"
    );
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_selected_proposal_basic() {
        assert_eq!(parse_selected_proposal("Selected: 2\nBecause it...", 3), 1);
        assert_eq!(parse_selected_proposal("Selected: 1", 3), 0);
        assert_eq!(parse_selected_proposal("Selected: 3", 3), 2);
    }

    #[test]
    fn test_parse_selected_proposal_case_insensitive() {
        assert_eq!(parse_selected_proposal("selected: 2", 3), 1);
        assert_eq!(parse_selected_proposal("SELECTED: 2", 3), 1);
    }

    #[test]
    fn test_parse_selected_proposal_out_of_range() {
        assert_eq!(parse_selected_proposal("Selected: 5", 3), 0);
        assert_eq!(parse_selected_proposal("Selected: 0", 3), 0);
    }

    #[test]
    fn test_parse_selected_proposal_no_match() {
        assert_eq!(parse_selected_proposal("I think proposal 2 is best", 3), 0);
        assert_eq!(parse_selected_proposal("", 3), 0);
    }

    #[test]
    fn test_parse_selected_embedded_in_text() {
        let response = "After careful consideration, Selected: 2\n\nProposal 2 is more complete.";
        assert_eq!(parse_selected_proposal(response, 3), 1);
    }

    #[test]
    fn test_build_judge_system_prompt() {
        let proposals = vec![
            Proposal {
                model: "gpt-4".to_string(),
                content: "Answer A".to_string(),
                created_at: Utc::now(),
            },
            Proposal {
                model: "claude-3".to_string(),
                content: "Answer B".to_string(),
                created_at: Utc::now(),
            },
        ];

        let prompt = build_judge_system_prompt("Pick the best one.", &proposals);
        assert!(prompt.contains("Proposal 1 (from gpt-4)"));
        assert!(prompt.contains("Proposal 2 (from claude-3)"));
        assert!(prompt.contains("Answer A"));
        assert!(prompt.contains("Answer B"));
        assert!(prompt.contains("Pick the best one."));
        assert!(prompt.contains("Selected: N"));
    }

    struct MockCaller {
        responses: std::collections::HashMap<String, Result<String, String>>,
    }

    #[async_trait]
    impl ModelCaller for MockCaller {
        async fn call_model(
            &self,
            _step_name: &str,
            model_name: &str,
            _system_prompt: &str,
            _user_message: &str,
            _required_tools: &[Arc<dyn SpiceModelTool>],
            _optional_tools: &[Arc<dyn SpiceModelTool>],
            _max_iterations: usize,
        ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
            match self.responses.get(model_name) {
                Some(Ok(content)) => Ok(content.clone()),
                Some(Err(e)) => Err(e.clone().into()),
                None => Err(format!("No mock response for model '{model_name}'").into()),
            }
        }
    }

    #[tokio::test]
    async fn test_execute_vote_step_success() {
        let mut responses = std::collections::HashMap::new();
        responses.insert("model-a".to_string(), Ok("Proposal from A".to_string()));
        responses.insert("model-b".to_string(), Ok("Proposal from B".to_string()));
        responses.insert(
            "judge".to_string(),
            Ok("Selected: 2\nProposal B is better.".to_string()),
        );
        let caller = MockCaller { responses };

        let result = execute_vote_step(
            "test-step",
            "What is 2+2?",
            "",
            &["model-a".to_string(), "model-b".to_string()],
            "judge",
            None,
            &caller,
        )
        .await
        .expect("should succeed");

        assert_eq!(result.proposals.len(), 2);
        assert_eq!(result.selected.model, "model-b");
        assert_eq!(result.selected.content, "Proposal from B");
    }

    #[tokio::test]
    async fn test_execute_vote_step_all_proposers_fail() {
        let mut responses = std::collections::HashMap::new();
        responses.insert("model-a".to_string(), Err("fail".to_string()));
        responses.insert("model-b".to_string(), Err("fail".to_string()));
        let caller = MockCaller { responses };

        let result = execute_vote_step(
            "test-step",
            "What is 2+2?",
            "",
            &["model-a".to_string(), "model-b".to_string()],
            "judge",
            None,
            &caller,
        )
        .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("all proposer models failed"));
    }
}
