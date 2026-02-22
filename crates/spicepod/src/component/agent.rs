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

use super::memory::MemoryConfig;
use super::pipeline::PipelineConfig;
use super::session::SessionConfig;
use super::{Nameable, WithDependsOn};
#[cfg(feature = "schemars")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Agent {
    pub name: String,
    pub model: String,
    pub prompt: String,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub datasets: Vec<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub file_sources: Vec<String>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub read_tools: Vec<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<SessionConfig>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryConfig>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planning: Option<PlanningModeConfig>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pipelines: Vec<PipelineConfig>,

    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(rename = "dependsOn", default)]
    pub depends_on: Vec<String>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct PlanningModeConfig {
    /// Whether the agent starts in plan mode. Default: true.
    #[serde(default = "default_true")]
    pub start_in_plan_mode: bool,
}

impl Nameable for Agent {
    fn name(&self) -> &str {
        &self.name
    }
}

impl WithDependsOn<Agent> for Agent {
    fn depends_on(&self, depends_on: &[String]) -> Agent {
        Agent {
            name: self.name.clone(),
            model: self.model.clone(),
            prompt: self.prompt.clone(),
            datasets: self.datasets.clone(),
            file_sources: self.file_sources.clone(),
            read_tools: self.read_tools.clone(),
            session: self.session.clone(),
            memory: self.memory.clone(),
            planning: self.planning.clone(),
            pipelines: self.pipelines.clone(),
            depends_on: depends_on.to_vec(),
        }
    }
}
