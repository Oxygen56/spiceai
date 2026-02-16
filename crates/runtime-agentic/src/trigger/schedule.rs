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
use std::time::Duration;

use async_trait::async_trait;
use chrono::{Local, Utc};
use croner::Cron;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::{Trigger, TriggerFactory, TriggerHandler, TriggerPayload};

pub struct ScheduleTrigger {
    name: String,
    cron_expr: String,
    cancel_token: CancellationToken,
}

impl ScheduleTrigger {
    #[must_use]
    pub fn new(name: String, cron_expr: String) -> Self {
        Self {
            name,
            cron_expr,
            cancel_token: CancellationToken::new(),
        }
    }
}

#[async_trait]
impl Trigger for ScheduleTrigger {
    async fn start(
        &self,
        handler: Arc<dyn TriggerHandler>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let cron: Cron = self.cron_expr.parse().map_err(|e| {
            format!(
                "Failed to parse cron expression '{}': {}",
                self.cron_expr, e
            )
        })?;

        let cancel_token = self.cancel_token.clone();
        let name = self.name.clone();

        tokio::spawn(async move {
            tracing::info!("Schedule trigger '{name}' started");
            loop {
                let now = Local::now();
                let next = match cron.find_next_occurrence(&now, false) {
                    Ok(next) => next,
                    Err(e) => {
                        tracing::error!(
                            "Schedule trigger '{name}': failed to determine next run time: {e}"
                        );
                        return;
                    }
                };

                let duration_till = next.signed_duration_since(now);
                let interval = duration_till.to_std().unwrap_or(Duration::from_secs(1));

                tokio::select! {
                    () = cancel_token.cancelled() => {
                        tracing::info!("Schedule trigger '{name}' cancelled");
                        return;
                    }
                    () = tokio::time::sleep(interval) => {
                        let payload = TriggerPayload {
                            source: name.clone(),
                            data: Value::Null,
                            received_at: Utc::now(),
                        };

                        if let Err(e) = handler.handle(payload).await {
                            tracing::error!("Schedule trigger '{name}': handler error: {e}");
                        }
                    }
                }
            }
        });

        Ok(())
    }

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.cancel_token.cancel();
        tracing::info!("Schedule trigger '{}' stopped", self.name);
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

pub struct ScheduleTriggerFactory;

impl TriggerFactory for ScheduleTriggerFactory {
    fn create(
        &self,
        config: &spicepod::component::pipeline::TriggerConfig,
    ) -> Result<Box<dyn Trigger>, Box<dyn std::error::Error + Send + Sync>> {
        let schedule = config
            .params
            .get("schedule")
            .ok_or("Schedule trigger requires a 'schedule' parameter with a cron expression")?
            .clone();

        let name = config
            .params
            .get("name")
            .cloned()
            .unwrap_or_else(|| format!("schedule:{schedule}"));

        Ok(Box::new(ScheduleTrigger::new(name, schedule)))
    }

    fn trigger_type(&self) -> &'static str {
        "schedule"
    }
}
