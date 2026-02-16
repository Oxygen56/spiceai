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
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use chrono::Local;
use croner::Cron;
use tokio_util::sync::CancellationToken;

use crate::file_source::github::GitFileSourceFactory;
use crate::file_source::s3::S3FileSourceFactory;
use crate::file_source::{FileSourceConnector, FileSourceConnectorFactory};
use crate::Runtime;

/// Registry of file source connector factories keyed by prefix.
fn build_factory_registry() -> HashMap<String, Box<dyn FileSourceConnectorFactory>> {
    let mut registry: HashMap<String, Box<dyn FileSourceConnectorFactory>> = HashMap::new();
    let factories: Vec<Box<dyn FileSourceConnectorFactory>> = vec![
        Box::new(GitFileSourceFactory),
        Box::new(S3FileSourceFactory),
    ];
    for factory in factories {
        registry.insert(factory.prefix().to_string(), factory);
    }
    registry
}

impl Runtime {
    pub(crate) async fn load_file_sources(self: Arc<Self>) {
        let app_lock = self.app.read().await;
        let Some(app) = app_lock.as_ref() else {
            return;
        };

        if app.file_sources.is_empty() {
            return;
        }

        let file_sources = app.file_sources.clone();
        drop(app_lock);

        let factory_registry = build_factory_registry();

        for file_source in &file_sources {
            let prefix = file_source
                .from
                .split(':')
                .next()
                .unwrap_or(&file_source.from);

            let Some(factory) = factory_registry.get(prefix) else {
                tracing::error!(
                    "Unknown file source type '{}' for file source '{}'. \
                     Supported types: github, s3, gdrive",
                    file_source.from,
                    file_source.name,
                );
                continue;
            };

            let target_path = PathBuf::from(&file_source.path);

            // Create target directory if it doesn't exist
            if let Err(e) = tokio::fs::create_dir_all(&target_path).await {
                tracing::error!(
                    "Failed to create target directory '{}' for file source '{}': {e}",
                    target_path.display(),
                    file_source.name,
                );
                continue;
            }

            // Build params including the 'from' field for the factory
            let mut params = file_source.params.clone();
            params.insert("from".to_string(), file_source.from.clone());

            let connector = match factory.create(target_path.clone(), params).await {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!(
                        "Failed to create file source connector for '{}': {e}",
                        file_source.name,
                    );
                    continue;
                }
            };

            // Run initial refresh
            tracing::info!(
                "Loading file source [{}] from '{}' into '{}'...",
                file_source.name,
                file_source.from,
                target_path.display(),
            );

            match connector.refresh().await {
                Ok(stats) => {
                    tracing::info!(
                        "File source [{}] loaded: {} files synced, {} bytes in {:?}",
                        file_source.name,
                        stats.files_synced,
                        stats.bytes_transferred,
                        stats.duration,
                    );
                }
                Err(e) => {
                    tracing::error!(
                        "Failed initial refresh for file source '{}': {e}",
                        file_source.name,
                    );
                    // Continue to set up scheduled refresh even if initial fails
                }
            }

            // Set up periodic refresh if configured
            if let Some(ref cron_expr) = file_source.refresh {
                let name = file_source.name.clone();
                schedule_file_source_refresh(name, connector, cron_expr.clone());
            }
        }
    }
}

/// Spawn a background task that periodically refreshes a file source on a cron schedule.
fn schedule_file_source_refresh(
    name: String,
    connector: Arc<dyn FileSourceConnector>,
    cron_expr: String,
) {
    let cancel_token = CancellationToken::new();

    tokio::spawn(async move {
        let cron: Cron = match cron_expr.parse() {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(
                    "Invalid cron expression '{cron_expr}' for file source '{name}': {e}"
                );
                return;
            }
        };

        tracing::info!(
            "File source '{name}' refresh scheduled with cron '{cron_expr}'"
        );

        loop {
            let now = Local::now();
            let next = match cron.find_next_occurrence(&now, false) {
                Ok(next) => next,
                Err(e) => {
                    tracing::error!(
                        "File source '{name}': failed to determine next refresh time: {e}"
                    );
                    return;
                }
            };

            let duration_till = next.signed_duration_since(now);
            let interval = duration_till.to_std().unwrap_or(Duration::from_secs(60));

            tokio::select! {
                () = cancel_token.cancelled() => {
                    tracing::info!("File source '{name}' refresh schedule cancelled");
                    return;
                }
                () = tokio::time::sleep(interval) => {
                    tracing::info!("Refreshing file source '{name}'...");
                    match connector.refresh().await {
                        Ok(stats) => {
                            tracing::info!(
                                "File source '{name}' refreshed: {} files synced, {} bytes in {:?}",
                                stats.files_synced,
                                stats.bytes_transferred,
                                stats.duration,
                            );
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to refresh file source '{name}': {e}"
                            );
                        }
                    }
                }
            }
        }
    });
}
