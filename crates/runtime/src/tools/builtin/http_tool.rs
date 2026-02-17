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

use std::{borrow::Cow, collections::HashMap, time::Duration};

use async_trait::async_trait;
use reqwest::{Client, Method, header::{HeaderMap, HeaderName, HeaderValue}};
use schemars::JsonSchema;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tools::SpiceModelTool;
use tracing_futures::Instrument;

use crate::tools::utils::parameters;

/// Parameters the LLM provides when calling the HTTP tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HttpToolParams {
    /// The JSON body to send in the request. For POST/PUT/PATCH this becomes the
    /// request body. For GET/DELETE these are sent as query parameters.
    #[serde(default)]
    pub body: Option<Value>,

    /// Optional path to append to the base URL (e.g. "/summarize").
    #[serde(default)]
    pub path: Option<String>,
}

pub struct HttpTool {
    name: String,
    description: String,
    client: Client,
    url: String,
    method: Method,
    headers: HeaderMap,
    timeout: Duration,
}

impl HttpTool {
    pub fn try_new(
        name: Option<&str>,
        description: Option<&str>,
        params: &HashMap<String, SecretString>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let url = params
            .get("url")
            .map(|v| v.expose_secret().to_string())
            .ok_or("Missing required 'url' parameter")?;

        let method = params
            .get("method")
            .map(|v| v.expose_secret().to_uppercase())
            .and_then(|m| m.parse::<Method>().ok())
            .unwrap_or(Method::POST);

        let headers_str = params
            .get("headers")
            .map(|v| v.expose_secret().to_string())
            .unwrap_or_default();
        let headers = parse_headers(&headers_str);

        let timeout = params
            .get("timeout")
            .and_then(|v| super::approval::parse_timeout(v.expose_secret()))
            .unwrap_or(Duration::from_secs(30));

        let client = Client::builder().timeout(timeout).build()?;

        Ok(Self {
            name: name.unwrap_or("http_tool").to_string(),
            description: description
                .unwrap_or("Make HTTP requests to external services and APIs")
                .to_string(),
            client,
            url,
            method,
            headers,
            timeout,
        })
    }
}

#[async_trait]
impl SpiceModelTool for HttpTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<HttpToolParams>()
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span = tracing::span!(
            target: "task_history",
            tracing::Level::INFO,
            "tool_use::http_tool",
            tool = self.name.as_str(),
            input = arg,
        );

        let result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let params: HttpToolParams = serde_json::from_str(arg)?;

            let url = match &params.path {
                Some(path) => format!("{}{path}", self.url.trim_end_matches('/')),
                None => self.url.clone(),
            };

            let mut request = self.client.request(self.method.clone(), &url);
            request = request.headers(self.headers.clone());

            // For body-bearing methods, send as JSON body; otherwise as query params.
            if let Some(body) = &params.body {
                match self.method {
                    Method::POST | Method::PUT | Method::PATCH => {
                        request = request.json(body);
                    }
                    _ => {
                        if let Some(obj) = body.as_object() {
                            let query_params: Vec<(String, String)> = obj
                                .iter()
                                .map(|(k, v)| {
                                    (
                                        k.clone(),
                                        match v {
                                            Value::String(s) => s.clone(),
                                            other => other.to_string(),
                                        },
                                    )
                                })
                                .collect();
                            request = request.query(&query_params);
                        }
                    }
                }
            }

            let response = request.send().await?;
            let status = response.status();
            let response_text = response.text().await?;

            if !status.is_success() {
                return Ok(json!({
                    "error": response_text,
                    "status": status.as_u16(),
                }));
            }

            // Try to parse as JSON; fall back to wrapping as text.
            match serde_json::from_str::<Value>(&response_text) {
                Ok(json_value) => Ok(json_value),
                Err(_) => Ok(json!({ "result": response_text })),
            }
        }
        .instrument(span.clone())
        .await;

        match &result {
            Ok(value) => {
                if let Ok(output_json) = serde_json::to_string(value) {
                    tracing::info!(
                        target: "task_history",
                        parent: &span,
                        captured_output = %output_json,
                    );
                }
            }
            Err(e) => {
                tracing::error!(target: "task_history", parent: &span, "{e}");
            }
        }

        result
    }
}

/// Parse a comma-separated `key:value` string into a [`HeaderMap`].
///
/// Example: `"Authorization:Bearer tok,Content-Type:application/json"`
fn parse_headers(raw: &str) -> HeaderMap {
    let mut map = HeaderMap::new();
    if raw.is_empty() {
        return map;
    }
    for pair in raw.split(',') {
        let pair = pair.trim();
        if let Some((key, value)) = pair.split_once(':') {
            let key = key.trim();
            let value = value.trim();
            if let (Ok(name), Ok(val)) = (
                key.parse::<HeaderName>(),
                HeaderValue::from_str(value),
            ) {
                map.insert(name, val);
            } else {
                tracing::warn!("Invalid HTTP header: {pair}");
            }
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_headers_basic() {
        let headers = parse_headers("Content-Type:application/json,Accept:text/plain");
        assert_eq!(
            headers.get("content-type").unwrap().to_str().unwrap(),
            "application/json"
        );
        assert_eq!(
            headers.get("accept").unwrap().to_str().unwrap(),
            "text/plain"
        );
    }

    #[test]
    fn test_parse_headers_with_bearer_token() {
        let headers = parse_headers("Authorization:Bearer my_secret_token");
        assert_eq!(
            headers.get("authorization").unwrap().to_str().unwrap(),
            "Bearer my_secret_token"
        );
    }

    #[test]
    fn test_parse_headers_empty() {
        let headers = parse_headers("");
        assert!(headers.is_empty());
    }

    #[test]
    fn test_http_tool_params_schema() {
        let tool = HttpTool::try_new(
            None,
            None,
            &HashMap::from([("url".to_string(), SecretString::from("http://example.com".to_string()))]),
        )
        .unwrap();
        let schema = tool.parameters();
        assert!(schema.is_some());
        let schema = schema.unwrap();
        // Should have "body" and "path" properties
        let props = schema
            .get("properties")
            .expect("should have properties");
        assert!(props.get("body").is_some());
        assert!(props.get("path").is_some());
    }

    #[test]
    fn test_default_method_is_post() {
        let tool = HttpTool::try_new(
            None,
            None,
            &HashMap::from([("url".to_string(), SecretString::from("http://example.com".to_string()))]),
        )
        .unwrap();
        assert_eq!(tool.method, Method::POST);
    }

    #[test]
    fn test_custom_method() {
        let tool = HttpTool::try_new(
            None,
            None,
            &HashMap::from([
                ("url".to_string(), SecretString::from("http://example.com".to_string())),
                ("method".to_string(), SecretString::from("GET".to_string())),
            ]),
        )
        .unwrap();
        assert_eq!(tool.method, Method::GET);
    }

    #[test]
    fn test_missing_url_errors() {
        let result = HttpTool::try_new(None, None, &HashMap::new());
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("Missing required 'url' parameter"));
    }

    #[test]
    fn test_default_name_and_description() {
        let tool = HttpTool::try_new(
            None,
            None,
            &HashMap::from([("url".to_string(), SecretString::from("http://example.com".to_string()))]),
        )
        .unwrap();
        assert_eq!(tool.name(), "http_tool");
        assert!(tool.description().is_some());
    }

    #[test]
    fn test_custom_name_and_description() {
        let tool = HttpTool::try_new(
            Some("my_api"),
            Some("Calls my API"),
            &HashMap::from([("url".to_string(), SecretString::from("http://example.com".to_string()))]),
        )
        .unwrap();
        assert_eq!(tool.name(), "my_api");
        assert_eq!(tool.description().unwrap(), "Calls my API");
    }

    #[test]
    fn test_timeout_parsing() {
        let tool = HttpTool::try_new(
            None,
            None,
            &HashMap::from([
                ("url".to_string(), SecretString::from("http://example.com".to_string())),
                ("timeout".to_string(), SecretString::from("60s".to_string())),
            ]),
        )
        .unwrap();
        assert_eq!(tool.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_default_timeout_30s() {
        let tool = HttpTool::try_new(
            None,
            None,
            &HashMap::from([("url".to_string(), SecretString::from("http://example.com".to_string()))]),
        )
        .unwrap();
        assert_eq!(tool.timeout, Duration::from_secs(30));
    }
}
