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

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::array::{Float64Builder, RecordBatch, StringBuilder, TimestampMillisecondBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use snafu::ResultExt;

use super::client::{MetricResult, PrometheusResponse};
use super::{ArrowInternalSnafu, Result};

/// Infers an Arrow schema from a Prometheus response.
///
/// The schema always includes:
/// - `timestamp` (`TimestampMillisecond` with UTC timezone)
/// - `value` (`Float64`)
/// - `metric_name` (`Utf8`, from the `__name__` label)
///
/// Plus one `Utf8` column per unique label key (sorted alphabetically), excluding `__name__`.
#[must_use]
pub fn infer_schema(response: &PrometheusResponse) -> SchemaRef {
    let label_names = collect_label_names(&response.results);

    let mut fields = vec![
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            false,
        ),
        Field::new("value", DataType::Float64, false),
        Field::new("metric_name", DataType::Utf8, true),
    ];

    for label in &label_names {
        fields.push(Field::new(label.as_str(), DataType::Utf8, true));
    }

    Arc::new(Schema::new(fields))
}

/// Converts a Prometheus response into Arrow `RecordBatch`es using the given schema.
///
/// # Errors
///
/// Returns an error if Arrow array building fails.
pub fn response_to_record_batches(
    response: &PrometheusResponse,
    schema: &SchemaRef,
) -> Result<Vec<RecordBatch>> {
    if response.results.is_empty() {
        return Ok(vec![]);
    }

    let label_columns: Vec<String> = schema
        .fields()
        .iter()
        .skip(3) // skip timestamp, value, metric_name
        .map(|f| f.name().clone())
        .collect();

    let total_rows = estimate_row_count(&response.results);

    let mut ts_builder =
        TimestampMillisecondBuilder::with_capacity(total_rows).with_timezone("UTC");
    let mut val_builder = Float64Builder::with_capacity(total_rows);
    let mut name_builder = StringBuilder::with_capacity(total_rows, total_rows * 16);

    let mut label_builders: Vec<StringBuilder> = label_columns
        .iter()
        .map(|_| StringBuilder::with_capacity(total_rows, total_rows * 16))
        .collect();

    for result in &response.results {
        let metric_name = result.metric.get("__name__").map(String::as_str);

        let data_points = collect_data_points(result);

        for (ts_secs, value_str) in &data_points {
            // Convert seconds to milliseconds
            #[allow(clippy::cast_possible_truncation)]
            let ts_ms = (*ts_secs * 1000.0) as i64;
            ts_builder.append_value(ts_ms);

            let value: f64 = value_str.parse().unwrap_or(f64::NAN);
            val_builder.append_value(value);

            match metric_name {
                Some(name) => name_builder.append_value(name),
                None => name_builder.append_null(),
            }

            for (i, label_name) in label_columns.iter().enumerate() {
                match result.metric.get(label_name) {
                    Some(label_value) => label_builders[i].append_value(label_value),
                    None => label_builders[i].append_null(),
                }
            }
        }
    }

    let mut columns: Vec<Arc<dyn arrow::array::Array>> = vec![
        Arc::new(ts_builder.finish()),
        Arc::new(val_builder.finish()),
        Arc::new(name_builder.finish()),
    ];

    for builder in &mut label_builders {
        columns.push(Arc::new(builder.finish()));
    }

    let batch =
        RecordBatch::try_new(Arc::clone(schema), columns).context(ArrowInternalSnafu)?;

    Ok(vec![batch])
}

/// Collects all unique label names across results, excluding `__name__`.
/// Returns a sorted set for deterministic column ordering.
fn collect_label_names(results: &[MetricResult]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for result in results {
        for key in result.metric.keys() {
            if key != "__name__" {
                names.insert(key.clone());
            }
        }
    }
    names
}

/// Collects all (timestamp, value) data points from a metric result.
fn collect_data_points(result: &MetricResult) -> Vec<(f64, String)> {
    if !result.values.is_empty() {
        result.values.clone()
    } else if let Some((ts, val)) = &result.value {
        vec![(*ts, val.clone())]
    } else {
        vec![]
    }
}

fn estimate_row_count(results: &[MetricResult]) -> usize {
    results
        .iter()
        .map(|r| {
            if !r.values.is_empty() {
                r.values.len()
            } else if r.value.is_some() {
                1
            } else {
                0
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use arrow::array::{Array, AsArray, Float64Array, StringArray};
    use arrow::datatypes::TimestampMillisecondType;

    fn make_matrix_response() -> PrometheusResponse {
        PrometheusResponse {
            result_type: "matrix".to_string(),
            results: vec![
                MetricResult {
                    metric: HashMap::from([
                        ("__name__".to_string(), "up".to_string()),
                        ("job".to_string(), "prometheus".to_string()),
                        ("instance".to_string(), "localhost:9090".to_string()),
                    ]),
                    values: vec![
                        (1_708_000_000.0, "1".to_string()),
                        (1_708_000_015.0, "1".to_string()),
                    ],
                    value: None,
                },
                MetricResult {
                    metric: HashMap::from([
                        ("__name__".to_string(), "up".to_string()),
                        ("job".to_string(), "node".to_string()),
                        ("instance".to_string(), "localhost:9100".to_string()),
                    ]),
                    values: vec![(1_708_000_000.0, "0".to_string())],
                    value: None,
                },
            ],
        }
    }

    #[test]
    fn test_infer_schema_matrix() {
        let response = make_matrix_response();
        let schema = infer_schema(&response);

        assert_eq!(schema.fields().len(), 5);
        assert_eq!(schema.field(0).name(), "timestamp");
        assert_eq!(schema.field(1).name(), "value");
        assert_eq!(schema.field(2).name(), "metric_name");
        // Labels sorted alphabetically
        assert_eq!(schema.field(3).name(), "instance");
        assert_eq!(schema.field(4).name(), "job");
    }

    #[test]
    fn test_convert_matrix_response() {
        let response = make_matrix_response();
        let schema = infer_schema(&response);
        let batches =
            response_to_record_batches(&response, &schema).expect("should convert");

        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 3); // 2 from first result + 1 from second

        // Check timestamps
        let ts_col = batch.column(0).as_primitive::<TimestampMillisecondType>();
        assert_eq!(ts_col.value(0), 1_708_000_000_000);
        assert_eq!(ts_col.value(1), 1_708_000_015_000);
        assert_eq!(ts_col.value(2), 1_708_000_000_000);

        // Check values
        let val_col = batch.column(1).as_any().downcast_ref::<Float64Array>().expect("float64");
        assert!((val_col.value(0) - 1.0).abs() < f64::EPSILON);
        assert!((val_col.value(1) - 1.0).abs() < f64::EPSILON);
        assert!((val_col.value(2) - 0.0).abs() < f64::EPSILON);

        // Check metric_name
        let name_col = batch.column(2).as_any().downcast_ref::<StringArray>().expect("string");
        assert_eq!(name_col.value(0), "up");

        // Check instance label
        let instance_col = batch.column(3).as_any().downcast_ref::<StringArray>().expect("string");
        assert_eq!(instance_col.value(0), "localhost:9090");
        assert_eq!(instance_col.value(2), "localhost:9100");

        // Check job label
        let job_col = batch.column(4).as_any().downcast_ref::<StringArray>().expect("string");
        assert_eq!(job_col.value(0), "prometheus");
        assert_eq!(job_col.value(2), "node");
    }

    #[test]
    fn test_convert_vector_response() {
        let response = PrometheusResponse {
            result_type: "vector".to_string(),
            results: vec![MetricResult {
                metric: HashMap::from([
                    ("__name__".to_string(), "up".to_string()),
                    ("job".to_string(), "prometheus".to_string()),
                ]),
                values: vec![],
                value: Some((1_708_000_000.0, "1".to_string())),
            }],
        };

        let schema = infer_schema(&response);
        let batches =
            response_to_record_batches(&response, &schema).expect("should convert");

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
    }

    #[test]
    fn test_convert_empty_response() {
        let response = PrometheusResponse {
            result_type: "vector".to_string(),
            results: vec![],
        };

        let schema = infer_schema(&response);
        let batches =
            response_to_record_batches(&response, &schema).expect("should convert");

        assert!(batches.is_empty());
    }

    #[test]
    fn test_sparse_labels() {
        let response = PrometheusResponse {
            result_type: "vector".to_string(),
            results: vec![
                MetricResult {
                    metric: HashMap::from([
                        ("__name__".to_string(), "metric_a".to_string()),
                        ("job".to_string(), "job1".to_string()),
                    ]),
                    values: vec![],
                    value: Some((1_708_000_000.0, "1".to_string())),
                },
                MetricResult {
                    metric: HashMap::from([
                        ("__name__".to_string(), "metric_b".to_string()),
                        ("instance".to_string(), "host:9090".to_string()),
                    ]),
                    values: vec![],
                    value: Some((1_708_000_001.0, "2".to_string())),
                },
            ],
        };

        let schema = infer_schema(&response);
        // Should have: timestamp, value, metric_name, instance, job
        assert_eq!(schema.fields().len(), 5);

        let batches =
            response_to_record_batches(&response, &schema).expect("should convert");

        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);

        // First row: instance should be null, job should be "job1"
        let instance_col = batch.column(3).as_any().downcast_ref::<StringArray>().expect("string");
        assert!(instance_col.is_null(0)); // first result has no instance
        assert_eq!(instance_col.value(1), "host:9090");

        let job_col = batch.column(4).as_any().downcast_ref::<StringArray>().expect("string");
        assert_eq!(job_col.value(0), "job1");
        assert!(job_col.is_null(1)); // second result has no job
    }

    #[test]
    fn test_nan_value_handling() {
        let response = PrometheusResponse {
            result_type: "vector".to_string(),
            results: vec![MetricResult {
                metric: HashMap::from([("__name__".to_string(), "test".to_string())]),
                values: vec![],
                value: Some((1_708_000_000.0, "NaN".to_string())),
            }],
        };

        let schema = infer_schema(&response);
        let batches =
            response_to_record_batches(&response, &schema).expect("should convert");

        let batch = &batches[0];
        let val_col = batch.column(1).as_any().downcast_ref::<Float64Array>().expect("float64");
        assert!(val_col.value(0).is_nan());
    }
}
