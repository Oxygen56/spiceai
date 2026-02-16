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
use futures::TryStreamExt;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use tokio::io::AsyncWriteExt;

use super::{FileSourceConnector, FileSourceConnectorFactory, RefreshStats};

/// A [`FileSourceConnector`] for syncing files from Amazon S3 using `object_store`.
///
/// Downloads objects under an optional S3 prefix into a local directory,
/// preserving the relative key structure. On subsequent refreshes, only files
/// whose S3 `last_modified` is newer than the local file's modification time
/// are downloaded (incremental refresh).
#[derive(Debug)]
pub struct S3FileSource {
    store: Arc<dyn ObjectStore>,
    prefix: Option<ObjectPath>,
    target_path: PathBuf,
}

/// Convert a [`SystemTime`] to a [`DateTime<Utc>`].
fn system_time_to_utc(st: SystemTime) -> DateTime<Utc> {
    DateTime::<Utc>::from(st)
}

#[async_trait]
impl FileSourceConnector for S3FileSource {
    async fn refresh(&self) -> Result<RefreshStats, Box<dyn std::error::Error + Send + Sync>> {
        let start = Instant::now();
        tokio::fs::create_dir_all(&self.target_path).await?;

        let mut files_synced: usize = 0;
        let mut bytes_transferred: u64 = 0;

        let mut list_stream = self.store.list(self.prefix.as_ref());

        while let Some(meta) = list_stream.try_next().await? {
            let key = meta.location.as_ref();

            // Determine relative path: strip the prefix if set.
            let relative = if let Some(ref pfx) = self.prefix {
                key.strip_prefix(pfx.as_ref())
                    .unwrap_or(key)
                    .trim_start_matches('/')
            } else {
                key
            };

            if relative.is_empty() || relative.ends_with('/') {
                continue;
            }

            let local_path = self.target_path.join(relative);

            // Incremental: skip if local file exists and is at least as recent as S3 object
            if let Ok(local_meta) = tokio::fs::metadata(&local_path).await {
                if let Ok(local_mtime) = local_meta.modified() {
                    let local_dt = system_time_to_utc(local_mtime);
                    if local_dt >= meta.last_modified {
                        tracing::trace!(
                            key,
                            "Skipping (local file up to date)"
                        );
                        continue;
                    }
                }
            }

            if let Some(parent) = local_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            let get_result = self.store.get(&meta.location).await?;
            let bytes = get_result.bytes().await?;

            let mut file = tokio::fs::File::create(&local_path).await?;
            file.write_all(&bytes).await?;
            file.flush().await?;

            bytes_transferred += bytes.len() as u64;
            files_synced += 1;
        }

        tracing::info!(
            files_synced,
            bytes_transferred,
            "S3 file source sync completed"
        );

        Ok(RefreshStats {
            files_synced,
            bytes_transferred,
            duration: start.elapsed(),
        })
    }
}

pub struct S3FileSourceFactory;

#[async_trait]
impl FileSourceConnectorFactory for S3FileSourceFactory {
    fn prefix(&self) -> &'static str {
        "s3"
    }

    async fn create(
        &self,
        target_path: PathBuf,
        params: HashMap<String, String>,
    ) -> Result<Arc<dyn FileSourceConnector>, Box<dyn std::error::Error + Send + Sync>> {
        let bucket = params
            .get("from")
            .ok_or("Missing 'from' parameter (expected s3:<bucket-name>)")?
            .strip_prefix("s3:")
            .ok_or("'from' must start with 's3:'")?
            .to_string();

        let mut builder = AmazonS3Builder::new().with_bucket_name(&bucket);

        if let Some(region) = params.get("s3_region") {
            builder = builder.with_region(region);
        }
        if let Some(access_key) = params.get("s3_access_key_id") {
            builder = builder.with_access_key_id(access_key);
        }
        if let Some(secret_key) = params.get("s3_secret_access_key") {
            builder = builder.with_secret_access_key(secret_key);
        }
        if let Some(endpoint) = params.get("s3_endpoint") {
            builder = builder.with_endpoint(endpoint);
        }

        let store: Arc<dyn ObjectStore> = Arc::new(builder.build()?);

        let prefix = params
            .get("s3_path")
            .map(|p| ObjectPath::from(p.trim_start_matches('/')));

        Ok(Arc::new(S3FileSource {
            store,
            prefix,
            target_path,
        }))
    }
}
