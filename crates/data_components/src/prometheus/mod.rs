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

use arrow::error::ArrowError;
use reqwest::StatusCode;
use snafu::Snafu;

pub mod client;
pub mod convert;
pub mod provider;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Failed to send Prometheus HTTP request: {source}"))]
    ReqwestInternal { source: reqwest::Error },

    #[snafu(display("HTTP {status}: {message}"))]
    InvalidReqwestStatus {
        status: reqwest::StatusCode,
        message: String,
    },

    #[snafu(display(
        "Cannot setup the Prometheus data connector with an invalid configuration. {message}"
    ))]
    InvalidConfiguration { message: String },

    #[snafu(display("Failed to process Prometheus response: {source}"))]
    ArrowInternal { source: ArrowError },

    #[snafu(display(
        "The Prometheus API returned an invalid response (HTTP {status}). Technical details: {error}"
    ))]
    JsonDecodeError {
        status: reqwest::StatusCode,
        error: String,
        response_preview: String,
    },

    #[snafu(display("Prometheus query returned an error: {message}"))]
    QueryError { message: String },

    #[snafu(display("Prometheus query returned no data"))]
    EmptyResult {},
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Determines if a Prometheus error is retriable (transient).
///
/// Retriable errors include:
/// - All HTTP 5xx server errors
/// - HTTP 408 Request Timeout
/// - Connection/timeout errors from reqwest
/// - JSON decode errors from server errors (truncated responses)
#[must_use]
pub fn is_retriable_error(error: &Error) -> bool {
    match error {
        Error::InvalidReqwestStatus { status, .. } => {
            status.is_server_error() || *status == StatusCode::REQUEST_TIMEOUT
        }
        Error::JsonDecodeError { status, .. } => status.is_server_error(),
        Error::ReqwestInternal { source } => {
            source.is_timeout()
                || source.is_connect()
                || source.is_body()
                || source.is_decode()
                || source
                    .status()
                    .is_some_and(|s| s.is_server_error() || s == StatusCode::REQUEST_TIMEOUT)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_errors_retriable() {
        let server_error_codes = [
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::GATEWAY_TIMEOUT,
        ];

        for status in server_error_codes {
            let error = Error::InvalidReqwestStatus {
                status,
                message: format!("Server error: {status}"),
            };
            assert!(
                is_retriable_error(&error),
                "InvalidReqwestStatus with status {status} should be retriable"
            );
        }

        let timeout_error = Error::InvalidReqwestStatus {
            status: StatusCode::REQUEST_TIMEOUT,
            message: "Request Timeout".to_string(),
        };
        assert!(
            is_retriable_error(&timeout_error),
            "408 Request Timeout should be retriable"
        );
    }

    #[test]
    fn test_client_errors_not_retriable() {
        let client_error_codes = [
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::NOT_FOUND,
        ];

        for status in client_error_codes {
            let error = Error::InvalidReqwestStatus {
                status,
                message: format!("Client error: {status}"),
            };
            assert!(
                !is_retriable_error(&error),
                "InvalidReqwestStatus with client status {status} should NOT be retriable"
            );
        }
    }

    #[test]
    fn test_json_decode_server_error_retriable() {
        let error = Error::JsonDecodeError {
            status: StatusCode::BAD_GATEWAY,
            error: "unexpected EOF".to_string(),
            response_preview: "<html>".to_string(),
        };
        assert!(is_retriable_error(&error));

        let error = Error::JsonDecodeError {
            status: StatusCode::BAD_REQUEST,
            error: "unexpected EOF".to_string(),
            response_preview: "bad request".to_string(),
        };
        assert!(!is_retriable_error(&error));
    }

    #[test]
    fn test_non_retriable_errors() {
        let errors = vec![
            Error::InvalidConfiguration {
                message: "bad config".to_string(),
            },
            Error::QueryError {
                message: "bad query".to_string(),
            },
            Error::EmptyResult {},
        ];

        for error in &errors {
            assert!(
                !is_retriable_error(error),
                "Error should NOT be retriable: {error:?}"
            );
        }
    }
}
