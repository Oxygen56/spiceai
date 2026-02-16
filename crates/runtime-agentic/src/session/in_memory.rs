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

use super::{Session, SessionStatus, SessionStore, Turn};
use async_trait::async_trait;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

pub struct InMemorySessionStore {
    sessions: Arc<RwLock<HashMap<String, Session>>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemorySessionStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn create(
        &self,
        agent: &str,
        scope_key: &str,
    ) -> Result<Session, Box<dyn std::error::Error + Send + Sync>> {
        let now = Utc::now();
        let session = Session {
            id: Uuid::now_v7().to_string(),
            agent: agent.to_string(),
            scope_key: scope_key.to_string(),
            turns: Vec::new(),
            created_at: now,
            updated_at: now,
            status: SessionStatus::Active,
        };
        let mut sessions = self.sessions.write().await;
        sessions.insert(session.id.clone(), session.clone());
        Ok(session)
    }

    async fn get_or_create(
        &self,
        agent: &str,
        scope_key: &str,
    ) -> Result<Session, Box<dyn std::error::Error + Send + Sync>> {
        let composite_key = format!("{agent}:{scope_key}");

        {
            let sessions = self.sessions.read().await;
            for session in sessions.values() {
                let key = format!("{}:{}", session.agent, session.scope_key);
                if key == composite_key && session.status == SessionStatus::Active {
                    return Ok(session.clone());
                }
            }
        }

        self.create(agent, scope_key).await
    }

    async fn get(
        &self,
        session_id: &str,
    ) -> Result<Option<Session>, Box<dyn std::error::Error + Send + Sync>> {
        let sessions = self.sessions.read().await;
        Ok(sessions.get(session_id).cloned())
    }

    async fn add_turn(
        &self,
        session_id: &str,
        turn: Turn,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut sessions = self.sessions.write().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| format!("session not found: {session_id}"))?;
        session.turns.push(turn);
        session.updated_at = Utc::now();
        Ok(())
    }

    async fn close(
        &self,
        session_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut sessions = self.sessions.write().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| format!("session not found: {session_id}"))?;
        session.status = SessionStatus::Closed;
        session.updated_at = Utc::now();
        Ok(())
    }

    async fn list_active(
        &self,
        agent: &str,
    ) -> Result<Vec<Session>, Box<dyn std::error::Error + Send + Sync>> {
        let sessions = self.sessions.read().await;
        Ok(sessions
            .values()
            .filter(|s| s.agent == agent && s.status == SessionStatus::Active)
            .cloned()
            .collect())
    }
}
