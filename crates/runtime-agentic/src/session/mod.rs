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
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use spicepod::component::session::SessionConfig;

pub mod in_memory;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub agent: String,
    pub scope_key: String,
    pub turns: Vec<Turn>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub status: SessionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionStatus {
    Active,
    Closed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub index: u32,
    pub role: String,
    pub content: String,
    pub created_at: DateTime<Utc>,
}

#[async_trait]
pub trait SessionStore: Send + Sync + 'static {
    async fn create(
        &self,
        agent: &str,
        scope_key: &str,
    ) -> Result<Session, Box<dyn std::error::Error + Send + Sync>>;

    async fn get_or_create(
        &self,
        agent: &str,
        scope_key: &str,
    ) -> Result<Session, Box<dyn std::error::Error + Send + Sync>>;

    async fn get(
        &self,
        session_id: &str,
    ) -> Result<Option<Session>, Box<dyn std::error::Error + Send + Sync>>;

    async fn add_turn(
        &self,
        session_id: &str,
        turn: Turn,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    async fn close(
        &self,
        session_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    async fn list_active(
        &self,
        agent: &str,
    ) -> Result<Vec<Session>, Box<dyn std::error::Error + Send + Sync>>;
}

/// Resolve the scope key from config, pipeline name, agent name, and trigger source.
pub fn resolve_scope_key(
    config: &SessionConfig,
    pipeline_name: &str,
    agent_name: &str,
    trigger_source: &str,
) -> String {
    match config.scope.as_deref() {
        Some("per_task") | None => format!("pipeline:{pipeline_name}"),
        Some("per_agent") => format!("agent:{agent_name}"),
        Some("per_trigger") => format!("trigger:{trigger_source}"),
        Some(other) => format!("custom:{other}"),
    }
}

/// Check if a session should be reset based on the reset policy (lazy evaluation).
pub fn should_reset(session: &Session, config: &SessionConfig) -> bool {
    match config.reset.as_deref() {
        Some("idle") => {
            let idle_minutes: i64 = config
                .params
                .get("idle_minutes")
                .and_then(|v| v.parse().ok())
                .unwrap_or(240);
            let threshold = session.updated_at + Duration::minutes(idle_minutes);
            Utc::now() > threshold
        }
        Some("daily") => {
            let at_hour: u32 = config
                .params
                .get("at_hour")
                .and_then(|v| v.parse().ok())
                .unwrap_or(4);
            let now = Utc::now();
            let today_reset = now.date_naive().and_hms_opt(at_hour, 0, 0);
            if let Some(reset_time) = today_reset {
                let reset_dt = reset_time.and_utc();
                session.updated_at < reset_dt && now >= reset_dt
            } else {
                false
            }
        }
        Some("never") | None => false,
        _ => false,
    }
}
