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

use async_trait::async_trait;
use data_components::prometheus::{
    client::{Auth, PrometheusClient},
    provider::{PrometheusTableProviderBuilder, QueryType},
};
use datafusion::datasource::TableProvider;
use runtime::component::dataset::Dataset;
use runtime::dataconnector::{
    ConnectorComponent, ConnectorParams, DataConnector, DataConnectorError, DataConnectorFactory,
    DataConnectorResult, NewDataConnectorResult, default_spice_client,
};
use runtime::parameters::{ParameterSpec, Parameters};
use snafu::prelude::*;
use std::{any::Any, future::Future, pin::Pin, sync::Arc};
use token_provider::{StaticTokenProvider, TokenProvider};
use url::Url;

#[derive(Debug)]
pub struct Prometheus {
    params: Parameters,
    client: Arc<PrometheusClient>,
}

#[derive(Default, Debug, Copy, Clone)]
pub struct PrometheusFactory {}

impl PrometheusFactory {
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    #[must_use]
    pub fn new_arc() -> Arc<dyn DataConnectorFactory> {
        Arc::new(Self {}) as Arc<dyn DataConnectorFactory>
    }
}

const PARAMETERS: &[ParameterSpec] = &[
    // Connector parameters
    ParameterSpec::component("endpoint")
        .description("The URL of the Prometheus server (e.g., http://localhost:9090).")
        .required(),
    ParameterSpec::component("auth_token")
        .description("The bearer token to use for authentication with Prometheus.")
        .secret(),
    ParameterSpec::component("auth_user")
        .description("The username to use for HTTP Basic Auth.")
        .secret(),
    ParameterSpec::component("auth_pass")
        .description("The password to use for HTTP Basic Auth.")
        .secret(),
    // Runtime parameters
    ParameterSpec::runtime("step")
        .description("Step interval for range queries (e.g., '15s', '1m', '5m'). Default: '15s'."),
    ParameterSpec::runtime("lookback")
        .description("How far back to query for range queries (e.g., '1h', '24h', '7d'). Default: '1h'."),
    ParameterSpec::runtime("query_type")
        .description("The query type: 'instant' or 'range'. Default: 'range'."),
];

impl DataConnectorFactory for PrometheusFactory {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn create(
        &self,
        params: ConnectorParams,
    ) -> Pin<Box<dyn Future<Output = NewDataConnectorResult> + Send>> {
        Box::pin(async move {
            let endpoint_str = params.parameters.get("endpoint").expose().ok_or_else(|p| {
                DataConnectorError::InvalidConfigurationNoSource {
                    dataconnector: "prometheus".to_string(),
                    message: format!(
                        "A required parameter was missing: `{}`.\nSpecify the Prometheus server URL.",
                        p.0
                    ),
                    connector_component: params.component.clone(),
                }
            })?;

            let endpoint = Url::parse(endpoint_str).boxed().map_err(|source| {
                DataConnectorError::InvalidConfiguration {
                    dataconnector: "prometheus".to_string(),
                    message: "The specified 'endpoint' URL is not valid.".to_string(),
                    connector_component: params.component.clone(),
                    source,
                }
            })?;

            let token = params
                .parameters
                .get("auth_token")
                .ok()
                .map(|token| {
                    Arc::new(StaticTokenProvider::new(token.clone())) as Arc<dyn TokenProvider>
                });

            let user = params
                .parameters
                .get("auth_user")
                .expose()
                .ok()
                .map(str::to_string);
            let pass = params
                .parameters
                .get("auth_pass")
                .expose()
                .ok()
                .map(str::to_string);

            let auth = match (token, user, pass) {
                (None, Some(user), pass) => Some(Auth::Basic(user, pass)),
                (Some(token), _, _) => Some(Auth::Bearer(token)),
                _ => None,
            };

            let http_client = default_spice_client("application/json")
                .boxed()
                .map_err(|source| DataConnectorError::InternalWithSource {
                    dataconnector: "prometheus".to_string(),
                    connector_component: params.component.clone(),
                    source,
                })?;

            let client = PrometheusClient::new(http_client, endpoint, auth)
                .boxed()
                .map_err(|source| DataConnectorError::InternalWithSource {
                    dataconnector: "prometheus".to_string(),
                    connector_component: params.component.clone(),
                    source,
                })?;

            let prometheus = Prometheus {
                params: params.parameters,
                client: Arc::new(client),
            };

            Ok(Arc::new(prometheus) as Arc<dyn DataConnector>)
        })
    }

    fn prefix(&self) -> &'static str {
        "prometheus"
    }

    fn parameters(&self) -> &'static [ParameterSpec] {
        PARAMETERS
    }
}

#[async_trait]
impl DataConnector for Prometheus {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn read_provider(
        &self,
        dataset: &Dataset,
    ) -> DataConnectorResult<Arc<dyn TableProvider>> {
        let query_expr = dataset.path().to_string();

        if query_expr.is_empty() {
            return Err(DataConnectorError::InvalidConfigurationNoSource {
                dataconnector: "prometheus".to_string(),
                message: "A PromQL query expression is required. Specify it in the 'from' field, e.g., 'from: prometheus:up{job=\"node\"}'.".to_string(),
                connector_component: ConnectorComponent::from(dataset),
            });
        }

        let query_type_str = self
            .params
            .get("query_type")
            .expose()
            .ok()
            .unwrap_or("range");

        let query_type = match query_type_str {
            "instant" => QueryType::Instant,
            "range" | _ => {
                let step = self
                    .params
                    .get("step")
                    .expose()
                    .ok()
                    .unwrap_or("15s")
                    .to_string();
                let lookback = self
                    .params
                    .get("lookback")
                    .expose()
                    .ok()
                    .unwrap_or("1h")
                    .to_string();
                QueryType::Range { step, lookback }
            }
        };

        let provider = PrometheusTableProviderBuilder::new(Arc::clone(&self.client))
            .build(query_expr, query_type)
            .await
            .map_err(|e| DataConnectorError::InternalWithSource {
                dataconnector: "prometheus".to_string(),
                connector_component: ConnectorComponent::from(dataset),
                source: e.into(),
            })?;

        Ok(Arc::new(provider))
    }
}

/// The name used to identify this connector in configuration.
pub const CONNECTOR_NAME: &str = "prometheus";

/// Returns a new instance of the `Prometheus` connector factory.
#[must_use]
pub fn factory() -> Arc<dyn DataConnectorFactory> {
    PrometheusFactory::new_arc()
}
