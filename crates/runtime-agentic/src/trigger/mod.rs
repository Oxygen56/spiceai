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

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub mod schedule;
pub mod webhook;

#[derive(Debug, Clone)]
pub struct TriggerPayload {
    pub source: String,
    pub data: Value,
    pub received_at: DateTime<Utc>,
}

#[async_trait]
pub trait Trigger: Send + Sync + 'static {
    async fn start(
        &self,
        handler: Arc<dyn TriggerHandler>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    fn name(&self) -> &str;
}

#[async_trait]
pub trait TriggerHandler: Send + Sync + 'static {
    async fn handle(
        &self,
        payload: TriggerPayload,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

pub trait TriggerFactory: Send + Sync {
    fn create(
        &self,
        config: &spicepod::component::pipeline::TriggerConfig,
    ) -> Result<Box<dyn Trigger>, Box<dyn std::error::Error + Send + Sync>>;

    fn trigger_type(&self) -> &'static str;
}

/// Registry of trigger factories keyed by trigger type.
pub struct TriggerRegistry {
    factories: HashMap<String, Arc<dyn TriggerFactory>>,
}

impl TriggerRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    pub fn register(&mut self, factory: Arc<dyn TriggerFactory>) {
        self.factories
            .insert(factory.trigger_type().to_string(), factory);
    }

    /// Creates a trigger from the given config by looking up the appropriate factory.
    ///
    /// # Errors
    ///
    /// Returns an error if the trigger type is unknown or if the factory fails to create the trigger.
    pub fn create(
        &self,
        config: &spicepod::component::pipeline::TriggerConfig,
    ) -> Result<Box<dyn Trigger>, Box<dyn std::error::Error + Send + Sync>> {
        let factory = self
            .factories
            .get(&config.r#type)
            .ok_or_else(|| format!("Unknown trigger type: {}", config.r#type))?;
        factory.create(config)
    }
}

impl Default for TriggerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
