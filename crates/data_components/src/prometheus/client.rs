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

use std::collections::HashMap;
use std::sync::Arc;

use reqwest::RequestBuilder;
use serde::Deserialize;
use snafu::ResultExt;
use token_provider::TokenProvider;
use url::Url;

use super::{Error, ReqwestInternalSnafu, Result};

pub enum Auth {
    Basic(String, Option<String>),
    Bearer(Arc<dyn TokenProvider>),
}

impl std::fmt::Debug for Auth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Auth::Basic(user, _) => write!(f, "Basic({user}, ***)"),
            Auth::Bearer(_) => write!(f, "Bearer(***)"),
        }
    }
}

/// Raw Prometheus API response envelope.
#[derive(Debug, Deserialize)]
pub struct PrometheusApiResponse {
    pub status: String,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(rename = "errorType")]
    #[serde(default)]
    pub error_type: Option<String>,
    pub data: Option<PrometheusData>,
}

/// The `data` field of a successful Prometheus response.
#[derive(Debug, Deserialize)]
pub struct PrometheusData {
    #[serde(rename = "resultType")]
    pub result_type: String,
    pub result: Vec<serde_json::Value>,
}

/// A single metric result for vector/matrix result types.
#[derive(Debug, Clone)]
pub struct MetricResult {
    pub metric: HashMap<String, String>,
    /// For matrix results: list of (timestamp_seconds, value_string) pairs.
    pub values: Vec<(f64, String)>,
    /// For vector results: a single (timestamp_seconds, value_string).
    pub value: Option<(f64, String)>,
}

/// Parsed Prometheus query response with typed results.
#[derive(Debug)]
pub struct PrometheusResponse {
    pub result_type: String,
    pub results: Vec<MetricResult>,
}

#[derive(Debug)]
pub struct PrometheusClient {
    client: reqwest::Client,
    base_url: Url,
    auth: Option<Auth>,
}

impl PrometheusClient {
    /// Creates a new Prometheus client.
    ///
    /// # Errors
    ///
    /// Returns an error if the base URL is invalid.
    pub fn new(client: reqwest::Client, base_url: Url, auth: Option<Auth>) -> Result<Self> {
        if base_url.scheme() != "http" && base_url.scheme() != "https" {
            return Err(Error::InvalidConfiguration {
                message: format!(
                    "Prometheus endpoint must use http or https scheme, got '{}'",
                    base_url.scheme()
                ),
            });
        }

        Ok(Self {
            client,
            base_url,
            auth,
        })
    }

    /// Executes an instant query against Prometheus.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails or the response cannot be parsed.
    pub async fn query(
        &self,
        expr: &str,
        time: Option<f64>,
    ) -> Result<PrometheusResponse> {
        let url = self
            .base_url
            .join("/api/v1/query")
            .map_err(|e| Error::InvalidConfiguration {
                message: format!("Failed to construct query URL: {e}"),
            })?;

        let mut request = self.client.get(url).query(&[("query", expr)]);

        if let Some(t) = time {
            request = request.query(&[("time", &t.to_string())]);
        }

        request = apply_auth(request, self.auth.as_ref());

        self.execute_request(request).await
    }

    /// Executes a range query against Prometheus.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails or the response cannot be parsed.
    pub async fn query_range(
        &self,
        expr: &str,
        start: f64,
        end: f64,
        step: &str,
    ) -> Result<PrometheusResponse> {
        let url = self
            .base_url
            .join("/api/v1/query_range")
            .map_err(|e| Error::InvalidConfiguration {
                message: format!("Failed to construct query_range URL: {e}"),
            })?;

        let request = self
            .client
            .get(url)
            .query(&[
                ("query", expr),
                ("start", &start.to_string()),
                ("end", &end.to_string()),
                ("step", step),
            ]);

        let request = apply_auth(request, self.auth.as_ref());

        self.execute_request(request).await
    }

    async fn execute_request(&self, request: RequestBuilder) -> Result<PrometheusResponse> {
        let response = request.send().await.context(ReqwestInternalSnafu)?;
        let status = response.status();

        let response_text = response.text().await.context(ReqwestInternalSnafu)?;

        let api_response: PrometheusApiResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                let preview = response_text.chars().take(1000).collect::<String>();
                tracing::error!(
                    "Failed to decode Prometheus response as JSON.\nHTTP Status: {}\nJSON Parse Error: {}\nResponse preview:\n{}",
                    status,
                    e,
                    preview
                );
                Error::JsonDecodeError {
                    status,
                    error: e.to_string(),
                    response_preview: preview,
                }
            })?;

        if status.is_client_error() || status.is_server_error() {
            let message = api_response
                .error
                .unwrap_or_else(|| format!("HTTP {status}"));
            return Err(Error::InvalidReqwestStatus { status, message });
        }

        if api_response.status == "error" {
            return Err(Error::QueryError {
                message: api_response
                    .error
                    .unwrap_or_else(|| "Unknown Prometheus error".to_string()),
            });
        }

        let data = api_response.data.ok_or(Error::EmptyResult {})?;

        let results = parse_metric_results(&data.result_type, &data.result)?;

        Ok(PrometheusResponse {
            result_type: data.result_type,
            results,
        })
    }

    /// Returns the configured base URL.
    #[must_use]
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }
}

fn apply_auth(request: RequestBuilder, auth: Option<&Auth>) -> RequestBuilder {
    match auth {
        Some(Auth::Basic(user, pass)) => request.basic_auth(user, pass.clone()),
        Some(Auth::Bearer(token_provider)) => {
            request.bearer_auth(token_provider.get_token())
        }
        None => request,
    }
}

fn parse_metric_results(
    result_type: &str,
    raw_results: &[serde_json::Value],
) -> Result<Vec<MetricResult>> {
    let mut results = Vec::with_capacity(raw_results.len());

    for raw in raw_results {
        let metric: HashMap<String, String> = raw
            .get("metric")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();

        let values = if let Some(vals) = raw.get("values").and_then(|v| v.as_array()) {
            vals.iter()
                .filter_map(|v| {
                    let arr = v.as_array()?;
                    let ts = arr.first()?.as_f64()?;
                    let val = arr.get(1)?.as_str()?.to_string();
                    Some((ts, val))
                })
                .collect()
        } else {
            vec![]
        };

        let value = raw.get("value").and_then(|v| {
            let arr = v.as_array()?;
            let ts = arr.first()?.as_f64()?;
            let val = arr.get(1)?.as_str()?.to_string();
            Some((ts, val))
        });

        if result_type != "matrix" && result_type != "vector" {
            tracing::debug!(
                "Unsupported Prometheus result type '{result_type}', treating as vector"
            );
        }

        results.push(MetricResult {
            metric,
            values,
            value,
        });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_url_construction() {
        let base = Url::parse("http://localhost:9090").expect("valid url");
        let query_url = base.join("/api/v1/query").expect("valid join");
        assert_eq!(query_url.as_str(), "http://localhost:9090/api/v1/query");
    }

    #[test]
    fn test_query_range_url_construction() {
        let base = Url::parse("http://localhost:9090").expect("valid url");
        let url = base.join("/api/v1/query_range").expect("valid join");
        assert_eq!(
            url.as_str(),
            "http://localhost:9090/api/v1/query_range"
        );
    }

    #[test]
    fn test_parse_matrix_results() {
        let raw: Vec<serde_json::Value> = serde_json::from_str(
            r#"[
                {
                    "metric": {"__name__": "up", "job": "prometheus", "instance": "localhost:9090"},
                    "values": [[1708000000, "1"], [1708000015, "1"]]
                },
                {
                    "metric": {"__name__": "up", "job": "node", "instance": "localhost:9100"},
                    "values": [[1708000000, "0"]]
                }
            ]"#,
        )
        .expect("valid json");

        let results = parse_metric_results("matrix", &raw).expect("should parse");
        assert_eq!(results.len(), 2);

        assert_eq!(results[0].metric.get("__name__"), Some(&"up".to_string()));
        assert_eq!(results[0].metric.get("job"), Some(&"prometheus".to_string()));
        assert_eq!(results[0].values.len(), 2);
        assert!((results[0].values[0].0 - 1_708_000_000.0).abs() < f64::EPSILON);
        assert_eq!(results[0].values[0].1, "1");

        assert_eq!(results[1].values.len(), 1);
    }

    #[test]
    fn test_parse_vector_results() {
        let raw: Vec<serde_json::Value> = serde_json::from_str(
            r#"[
                {
                    "metric": {"__name__": "up", "job": "prometheus"},
                    "value": [1708000000, "1"]
                }
            ]"#,
        )
        .expect("valid json");

        let results = parse_metric_results("vector", &raw).expect("should parse");
        assert_eq!(results.len(), 1);
        assert!(results[0].value.is_some());
        let (ts, val) = results[0].value.as_ref().expect("has value");
        assert!((ts - 1_708_000_000.0).abs() < f64::EPSILON);
        assert_eq!(val, "1");
    }

    #[test]
    fn test_parse_empty_results() {
        let raw: Vec<serde_json::Value> = vec![];
        let results = parse_metric_results("vector", &raw).expect("should parse");
        assert!(results.is_empty());
    }

    #[test]
    fn test_invalid_scheme() {
        let base = Url::parse("ftp://localhost:9090").expect("valid url");
        let client = reqwest::Client::new();
        let result = PrometheusClient::new(client, base, None);
        assert!(result.is_err());
    }
}
