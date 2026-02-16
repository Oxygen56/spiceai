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

use std::collections::HashMap;

#[cfg(feature = "schemars")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
pub struct PipelineConfig {
    pub name: String,
    pub trigger: TriggerConfig,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<StepConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
pub struct TriggerConfig {
    #[serde(rename = "type")]
    pub r#type: String,

    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub params: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
pub struct StepConfig {
    pub name: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(rename = "type")]
    pub r#type: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<StepTools>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge: Option<JudgeConfig>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_iterations: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(untagged)]
pub enum StepTools {
    Simple(Vec<String>),
    Structured {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        required: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        optional: Vec<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
pub struct JudgeConfig {
    pub model: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}
