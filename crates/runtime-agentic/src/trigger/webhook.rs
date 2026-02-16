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
use tokio::sync::RwLock;

use super::{Trigger, TriggerFactory, TriggerHandler};

pub struct WebhookTrigger {
    name: String,
    path: String,
    handler: RwLock<Option<Arc<dyn TriggerHandler>>>,
}

impl WebhookTrigger {
    #[must_use]
    pub fn new(name: String, path: String) -> Self {
        Self {
            name,
            path,
            handler: RwLock::new(None),
        }
    }

    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the stored handler, if any.
    pub async fn handler(&self) -> Option<Arc<dyn TriggerHandler>> {
        self.handler.read().await.clone()
    }
}

#[async_trait]
impl Trigger for WebhookTrigger {
    async fn start(
        &self,
        handler: Arc<dyn TriggerHandler>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut guard = self.handler.write().await;
        *guard = Some(handler);
        tracing::info!("Webhook trigger '{}' started on path '{}'", self.name, self.path);
        Ok(())
    }

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut guard = self.handler.write().await;
        *guard = None;
        tracing::info!("Webhook trigger '{}' stopped", self.name);
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

pub struct WebhookTriggerFactory;

impl TriggerFactory for WebhookTriggerFactory {
    fn create(
        &self,
        config: &spicepod::component::pipeline::TriggerConfig,
    ) -> Result<Box<dyn Trigger>, Box<dyn std::error::Error + Send + Sync>> {
        let path = config
            .params
            .get("path")
            .cloned()
            .unwrap_or_else(|| "/webhook".to_string());

        let name = config
            .params
            .get("name")
            .cloned()
            .unwrap_or_else(|| format!("webhook:{path}"));

        Ok(Box::new(WebhookTrigger::new(name, path)))
    }

    fn trigger_type(&self) -> &'static str {
        "webhook"
    }
}
