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
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

pub mod github;
pub mod s3;

#[derive(Debug)]
pub struct RefreshStats {
    pub files_synced: usize,
    pub bytes_transferred: u64,
    pub duration: Duration,
}

#[async_trait]
pub trait FileSourceConnector: Send + Sync + Debug + 'static {
    /// Sync remote files into the configured local directory.
    async fn refresh(&self) -> Result<RefreshStats, Box<dyn std::error::Error + Send + Sync>>;
}

#[async_trait]
pub trait FileSourceConnectorFactory: Send + Sync {
    fn prefix(&self) -> &'static str;

    async fn create(
        &self,
        target_path: PathBuf,
        params: HashMap<String, String>,
    ) -> Result<Arc<dyn FileSourceConnector>, Box<dyn std::error::Error + Send + Sync>>;
}
