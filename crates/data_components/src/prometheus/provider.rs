/*
Copyright 2024-2026 The Spice.ai OSS Authors

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

use std::any::Any;
use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchReceiverStream;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};

use super::client::PrometheusClient;
use super::convert;
use super::Result;

/// The type of Prometheus query to execute.
#[derive(Debug, Clone)]
pub enum QueryType {
    /// Instant query: evaluates the expression at a single point in time.
    Instant,
    /// Range query: evaluates the expression over a time range.
    Range {
        /// Step interval (e.g., "15s", "1m", "5m").
        step: String,
        /// How far back to query (e.g., "1h", "24h", "7d").
        lookback: String,
    },
}

/// Builder for constructing a `PrometheusTableProvider`.
///
/// Runs an initial probe query to infer the Arrow schema from the Prometheus response.
pub struct PrometheusTableProviderBuilder {
    client: Arc<PrometheusClient>,
}

impl PrometheusTableProviderBuilder {
    #[must_use]
    pub fn new(client: Arc<PrometheusClient>) -> Self {
        Self { client }
    }

    /// Builds a `PrometheusTableProvider` by running a probe query to infer the schema.
    ///
    /// # Errors
    ///
    /// Returns an error if the probe query fails.
    pub async fn build(
        self,
        query_expr: String,
        query_type: QueryType,
    ) -> Result<PrometheusTableProvider> {
        let now = current_timestamp_secs();

        let probe_response = match &query_type {
            QueryType::Instant => self.client.query(&query_expr, Some(now)).await?,
            QueryType::Range { step, lookback } => {
                let lookback_secs = parse_duration_to_secs(lookback);
                // For schema inference, query a small window
                let probe_start = now - lookback_secs.min(300.0);
                self.client
                    .query_range(&query_expr, probe_start, now, step)
                    .await?
            }
        };

        let schema = convert::infer_schema(&probe_response);

        Ok(PrometheusTableProvider {
            schema,
            client: self.client,
            query_expr,
            query_type,
        })
    }
}

/// A DataFusion `TableProvider` that queries Prometheus for metrics data.
pub struct PrometheusTableProvider {
    schema: SchemaRef,
    client: Arc<PrometheusClient>,
    query_expr: String,
    query_type: QueryType,
}

impl std::fmt::Debug for PrometheusTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrometheusTableProvider")
            .field("query_expr", &self.query_expr)
            .field("query_type", &self.query_type)
            .field("schema", &self.schema)
            .finish()
    }
}

#[async_trait]
impl TableProvider for PrometheusTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // Prometheus has its own query language (PromQL); SQL filters are applied post-fetch
        Ok(vec![
            TableProviderFilterPushDown::Unsupported;
            filters.len()
        ])
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(PrometheusExec::new(
            Arc::clone(&self.client),
            self.query_expr.clone(),
            self.query_type.clone(),
            Arc::clone(&self.schema),
        )))
    }
}

use datafusion::error::Result as DFResult;

/// An `ExecutionPlan` that executes a Prometheus query and returns results as Arrow batches.
pub struct PrometheusExec {
    client: Arc<PrometheusClient>,
    query_expr: String,
    query_type: QueryType,
    schema: SchemaRef,
    properties: PlanProperties,
}

impl PrometheusExec {
    #[must_use]
    pub fn new(
        client: Arc<PrometheusClient>,
        query_expr: String,
        query_type: QueryType,
        schema: SchemaRef,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            client,
            query_expr,
            query_type,
            schema,
            properties,
        }
    }
}

impl std::fmt::Debug for PrometheusExec {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "PrometheusExec [query={}, type={:?}]",
            self.query_expr, self.query_type
        )
    }
}

impl DisplayAs for PrometheusExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "PrometheusExec [query={}]",
            self.query_expr
        )
    }
}

impl ExecutionPlan for PrometheusExec {
    fn name(&self) -> &'static str {
        "PrometheusExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let client = Arc::clone(&self.client);
        let query_expr = self.query_expr.clone();
        let query_type = self.query_type.clone();
        let schema = Arc::clone(&self.schema);

        let mut builder = RecordBatchReceiverStream::builder(Arc::clone(&schema), 2);
        let tx = builder.tx();

        builder.spawn(async move {
            let now = current_timestamp_secs();

            let response = match &query_type {
                QueryType::Instant => client
                    .query(&query_expr, Some(now))
                    .await
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?,
                QueryType::Range { step, lookback } => {
                    let lookback_secs = parse_duration_to_secs(lookback);
                    let start = now - lookback_secs;
                    client
                        .query_range(&query_expr, start, now, step)
                        .await
                        .map_err(|e| DataFusionError::Execution(e.to_string()))?
                }
            };

            let batches = convert::response_to_record_batches(&response, &schema)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;

            for batch in batches {
                tx.send(Ok(batch)).await.map_err(|_| {
                    DataFusionError::Execution("Failed to send record batch".to_string())
                })?;
            }

            Ok(())
        });

        Ok(builder.build())
    }
}

fn current_timestamp_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Parses a duration string like "15s", "5m", "1h", "7d" into seconds.
///
/// Supported suffixes: `s` (seconds), `m` (minutes), `h` (hours), `d` (days).
/// Falls back to treating the input as raw seconds if no suffix matches.
pub fn parse_duration_to_secs(duration: &str) -> f64 {
    let duration = duration.trim();

    if duration.is_empty() {
        return 3600.0; // default 1h
    }

    let (num_str, multiplier) = if let Some(num) = duration.strip_suffix('s') {
        (num, 1.0)
    } else if let Some(num) = duration.strip_suffix('m') {
        (num, 60.0)
    } else if let Some(num) = duration.strip_suffix('h') {
        (num, 3600.0)
    } else if let Some(num) = duration.strip_suffix('d') {
        (num, 86400.0)
    } else {
        (duration, 1.0) // assume seconds
    };

    num_str.parse::<f64>().unwrap_or(3600.0) * multiplier
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_duration_seconds() {
        assert!((parse_duration_to_secs("15s") - 15.0).abs() < f64::EPSILON);
        assert!((parse_duration_to_secs("30s") - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_duration_minutes() {
        assert!((parse_duration_to_secs("5m") - 300.0).abs() < f64::EPSILON);
        assert!((parse_duration_to_secs("1m") - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_duration_hours() {
        assert!((parse_duration_to_secs("1h") - 3600.0).abs() < f64::EPSILON);
        assert!((parse_duration_to_secs("24h") - 86400.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_duration_days() {
        assert!((parse_duration_to_secs("7d") - 604_800.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_duration_raw_number() {
        assert!((parse_duration_to_secs("3600") - 3600.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_duration_empty() {
        assert!((parse_duration_to_secs("") - 3600.0).abs() < f64::EPSILON);
    }
}
