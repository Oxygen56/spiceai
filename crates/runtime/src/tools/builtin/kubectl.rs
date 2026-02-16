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
use serde_json::{Value, json};
use std::borrow::Cow;
use tools::{SpiceModelTool, ToolCapability};
use tracing::Span;
use tracing_futures::Instrument;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::tools::utils::parameters;

/// Operations that mutate cluster state and should be blocked in read-only mode.
const WRITE_OPERATIONS: &[&str] = &[
    "apply", "delete", "scale", "rollout", "patch", "edit", "create",
];

/// All recognized read-only operations.
const READ_OPERATIONS: &[&str] = &["get", "list", "describe", "logs"];

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct KubectlToolParams {
    /// The operation to perform: "get", "list", "describe", or "logs".
    command: String,

    /// The Kubernetes resource type (e.g., "pods", "deployments", "services", "nodes", "events",
    /// "configmaps", "namespaces", "replicasets", "statefulsets", "daemonsets", "jobs",
    /// "cronjobs", "ingresses", "pvcs").
    resource: String,

    /// Resource name. Required for "get", "describe", and "logs".
    name: Option<String>,

    /// Label selector for filtering resources when listing (e.g., "app=nginx,tier=frontend").
    label_selector: Option<String>,

    /// Container name for "logs" in multi-container pods. If not specified, uses the default container.
    container: Option<String>,

    /// Number of tail lines for "logs". Defaults to 100.
    tail_lines: Option<i64>,
}

pub struct KubectlTool {
    name: String,
    description: String,
    namespace: String,
    allowed_operations: Vec<String>,
    capability: ToolCapability,
    #[cfg(feature = "kubernetes")]
    client: kube::Client,
}

impl KubectlTool {
    /// Create a new `KubectlTool` without a live Kubernetes client.
    /// Used when the `kubernetes` feature is disabled.
    #[cfg(not(feature = "kubernetes"))]
    #[must_use]
    pub fn new(
        name: Option<&str>,
        description: Option<&str>,
        namespace: String,
        allowed_operations: Vec<String>,
        capability: ToolCapability,
    ) -> Self {
        Self {
            name: name.unwrap_or("kubectl").to_string(),
            description: description
                .unwrap_or("Execute kubectl operations against a Kubernetes cluster")
                .to_string(),
            namespace,
            allowed_operations,
            capability,
        }
    }

    /// Create a new `KubectlTool` with a live Kubernetes client.
    #[cfg(feature = "kubernetes")]
    pub async fn try_new(
        name: Option<&str>,
        description: Option<&str>,
        namespace: String,
        allowed_operations: Vec<String>,
        capability: ToolCapability,
        kubeconfig_path: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let client = if let Some(path) = kubeconfig_path {
            let kubeconfig = kube::config::Kubeconfig::read_from(path)?;
            let config =
                kube::Config::from_custom_kubeconfig(kubeconfig, &Default::default()).await?;
            kube::Client::try_from(config)?
        } else {
            kube::Client::try_default().await?
        };

        Ok(Self {
            name: name.unwrap_or("kubectl").to_string(),
            description: description
                .unwrap_or("Execute kubectl operations against a Kubernetes cluster")
                .to_string(),
            namespace,
            allowed_operations,
            capability,
            client,
        })
    }

    fn validate_request(
        &self,
        req: &KubectlToolParams,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Validate command is in allowed operations
        if !self.allowed_operations.iter().any(|op| op == &req.command) {
            return Err(format!(
                "Operation '{}' is not in allowed operations: {:?}",
                req.command, self.allowed_operations
            )
            .into());
        }

        // Block write operations in read-only mode
        if self.capability == ToolCapability::ReadOnly
            && WRITE_OPERATIONS.contains(&req.command.as_str())
        {
            return Err(format!(
                "Operation '{}' is not permitted in read-only mode",
                req.command
            )
            .into());
        }

        // Validate command is recognized
        if !READ_OPERATIONS.contains(&req.command.as_str())
            && !WRITE_OPERATIONS.contains(&req.command.as_str())
        {
            return Err(format!("Unknown command '{}'. Recognized commands: get, list, describe, logs", req.command).into());
        }

        // Validate name is provided for commands that require it
        if matches!(req.command.as_str(), "get" | "describe" | "logs") && req.name.is_none() {
            return Err(format!(
                "Operation '{}' requires a resource name",
                req.command
            )
            .into());
        }

        // Validate logs only works on pods
        if req.command == "logs" && req.resource != "pods" {
            return Err("'logs' command is only supported for pods".into());
        }

        Ok(())
    }
}

/// Lookup table mapping resource type names to their API group/version/kind/plural.
/// This avoids expensive API discovery calls.
#[cfg(feature = "kubernetes")]
fn resource_api_info(resource: &str) -> Option<(&str, &str, &str, &str)> {
    // Returns (group, version, kind, plural)
    match resource {
        "pods" | "pod" | "po" => Some(("", "v1", "Pod", "pods")),
        "services" | "service" | "svc" => Some(("", "v1", "Service", "services")),
        "configmaps" | "configmap" | "cm" => Some(("", "v1", "ConfigMap", "configmaps")),
        "secrets" | "secret" => Some(("", "v1", "Secret", "secrets")),
        "namespaces" | "namespace" | "ns" => Some(("", "v1", "Namespace", "namespaces")),
        "nodes" | "node" | "no" => Some(("", "v1", "Node", "nodes")),
        "events" | "event" | "ev" => Some(("", "v1", "Event", "events")),
        "persistentvolumeclaims" | "pvc" => {
            Some(("", "v1", "PersistentVolumeClaim", "persistentvolumeclaims"))
        }
        "deployments" | "deployment" | "deploy" => {
            Some(("apps", "v1", "Deployment", "deployments"))
        }
        "replicasets" | "replicaset" | "rs" => Some(("apps", "v1", "ReplicaSet", "replicasets")),
        "statefulsets" | "statefulset" | "sts" => {
            Some(("apps", "v1", "StatefulSet", "statefulsets"))
        }
        "daemonsets" | "daemonset" | "ds" => Some(("apps", "v1", "DaemonSet", "daemonsets")),
        "jobs" | "job" => Some(("batch", "v1", "Job", "jobs")),
        "cronjobs" | "cronjob" | "cj" => Some(("batch", "v1", "CronJob", "cronjobs")),
        "ingresses" | "ingress" | "ing" => {
            Some(("networking.k8s.io", "v1", "Ingress", "ingresses"))
        }
        _ => None,
    }
}

#[cfg(feature = "kubernetes")]
fn build_api_resource(
    group: &str,
    version: &str,
    kind: &str,
    plural: &str,
) -> kube::api::ApiResource {
    kube::api::ApiResource {
        group: group.to_string(),
        version: version.to_string(),
        kind: kind.to_string(),
        plural: plural.to_string(),
        api_version: if group.is_empty() {
            version.to_string()
        } else {
            format!("{group}/{version}")
        },
    }
}

#[cfg(feature = "kubernetes")]
impl KubectlTool {
    async fn handle_get(
        &self,
        resource: &str,
        name: &str,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let (group, version, kind, plural) = resource_api_info(resource)
            .ok_or_else(|| format!("Unknown resource type: {resource}"))?;
        let ar = build_api_resource(group, version, kind, plural);

        let api: kube::Api<kube::api::DynamicObject> =
            kube::Api::namespaced_with(self.client.clone(), &self.namespace, &ar);
        let obj = api.get(name).await?;

        Ok(serde_json::to_value(&obj)?)
    }

    async fn handle_list(
        &self,
        resource: &str,
        label_selector: Option<&str>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let (group, version, kind, plural) = resource_api_info(resource)
            .ok_or_else(|| format!("Unknown resource type: {resource}"))?;
        let ar = build_api_resource(group, version, kind, plural);

        let api: kube::Api<kube::api::DynamicObject> =
            kube::Api::namespaced_with(self.client.clone(), &self.namespace, &ar);

        let mut lp = kube::api::ListParams::default();
        if let Some(selector) = label_selector {
            lp = lp.labels(selector);
        }

        let list = api.list(&lp).await?;
        let count = list.items.len();
        let items: Vec<Value> = list
            .items
            .into_iter()
            .map(|obj| {
                json!({
                    "name": obj.metadata.name,
                    "namespace": obj.metadata.namespace,
                    "labels": obj.metadata.labels,
                    "creation_timestamp": obj.metadata.creation_timestamp.map(|t| t.0.to_rfc3339()),
                })
            })
            .collect();

        Ok(json!({
            "resource": resource,
            "count": count,
            "items": items,
        }))
    }

    async fn handle_describe(
        &self,
        resource: &str,
        name: &str,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let (group, version, kind, plural) = resource_api_info(resource)
            .ok_or_else(|| format!("Unknown resource type: {resource}"))?;
        let ar = build_api_resource(group, version, kind, plural);

        let api: kube::Api<kube::api::DynamicObject> =
            kube::Api::namespaced_with(self.client.clone(), &self.namespace, &ar);
        let obj = api.get(name).await?;

        // Also fetch related events
        let events_ar = build_api_resource("", "v1", "Event", "events");
        let events_api: kube::Api<kube::api::DynamicObject> =
            kube::Api::namespaced_with(self.client.clone(), &self.namespace, &events_ar);
        let field_selector = format!("involvedObject.name={name},involvedObject.kind={kind}");
        let events_lp = kube::api::ListParams::default().fields(&field_selector);
        let events = events_api.list(&events_lp).await.ok();

        let event_entries: Vec<Value> = events
            .map(|list| {
                list.items
                    .into_iter()
                    .map(|e| {
                        json!({
                            "reason": e.data.get("reason"),
                            "message": e.data.get("message"),
                            "type": e.data.get("type"),
                            "last_timestamp": e.data.get("lastTimestamp"),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(json!({
            "resource": serde_json::to_value(&obj)?,
            "events": event_entries,
        }))
    }

    async fn handle_logs(
        &self,
        name: &str,
        container: Option<&str>,
        tail_lines: Option<i64>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let pods: kube::Api<k8s_openapi::api::core::v1::Pod> =
            kube::Api::namespaced(self.client.clone(), &self.namespace);

        let mut log_params = kube::api::LogParams {
            tail_lines: Some(tail_lines.unwrap_or(100)),
            ..Default::default()
        };
        if let Some(c) = container {
            log_params.container = Some(c.to_string());
        }

        let logs = pods.logs(name, &log_params).await?;

        Ok(json!({
            "pod": name,
            "container": container,
            "namespace": self.namespace,
            "lines": logs.lines().count(),
            "logs": logs,
        }))
    }
}

#[async_trait]
impl SpiceModelTool for KubectlTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<KubectlToolParams>()
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span: Span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::kubectl", tool = self.name().to_string(), input = arg);

        let tool_use_result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let req: KubectlToolParams = serde_json::from_str(arg)?;

            self.validate_request(&req)?;

            #[cfg(feature = "kubernetes")]
            {
                match req.command.as_str() {
                    "get" => {
                        self.handle_get(&req.resource, req.name.as_deref().unwrap_or_default())
                            .await
                    }
                    "list" => {
                        self.handle_list(&req.resource, req.label_selector.as_deref())
                            .await
                    }
                    "describe" => {
                        self.handle_describe(
                            &req.resource,
                            req.name.as_deref().unwrap_or_default(),
                        )
                        .await
                    }
                    "logs" => {
                        self.handle_logs(
                            req.name.as_deref().unwrap_or_default(),
                            req.container.as_deref(),
                            req.tail_lines,
                        )
                        .await
                    }
                    _ => Err(format!("Unhandled command: {}", req.command).into()),
                }
            }

            #[cfg(not(feature = "kubernetes"))]
            {
                Ok(json!({
                    "status": "unavailable",
                    "message": "Kubernetes support not enabled. Compile with --features kubernetes.",
                    "command": req.command,
                    "resource": req.resource,
                    "name": req.name,
                    "namespace": self.namespace,
                }))
            }
        }
        .instrument(span.clone())
        .await;

        match tool_use_result {
            Ok(value) => {
                let captured_output_json = serde_json::to_string(&value)?;
                tracing::info!(target: "task_history", parent: &span, captured_output = %captured_output_json);
                Ok(value)
            }
            Err(e) => {
                tracing::error!(target: "task_history", parent: &span, "{e}");
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool(ops: Vec<&str>) -> KubectlTool {
        #[cfg(not(feature = "kubernetes"))]
        {
            KubectlTool::new(
                None,
                None,
                "default".to_string(),
                ops.into_iter().map(ToString::to_string).collect(),
                ToolCapability::ReadOnly,
            )
        }
        #[cfg(feature = "kubernetes")]
        {
            // In test mode with kubernetes feature, we can't create a real client easily.
            // This test path validates parameter logic only.
            panic!("Use integration tests for kubernetes-enabled kubectl tests");
        }
    }

    #[test]
    fn test_validate_allowed_operations() {
        let tool = make_tool(vec!["get", "list"]);
        let req = KubectlToolParams {
            command: "describe".to_string(),
            resource: "pods".to_string(),
            name: Some("my-pod".to_string()),
            label_selector: None,
            container: None,
            tail_lines: None,
        };
        let result = tool.validate_request(&req);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not in allowed operations"));
    }

    #[test]
    fn test_validate_readonly_blocks_writes() {
        let tool = make_tool(vec!["get", "delete"]);
        let req = KubectlToolParams {
            command: "delete".to_string(),
            resource: "pods".to_string(),
            name: Some("my-pod".to_string()),
            label_selector: None,
            container: None,
            tail_lines: None,
        };
        let result = tool.validate_request(&req);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not permitted in read-only mode"));
    }

    #[test]
    fn test_validate_name_required() {
        let tool = make_tool(vec!["get", "describe", "logs"]);
        let req = KubectlToolParams {
            command: "get".to_string(),
            resource: "pods".to_string(),
            name: None,
            label_selector: None,
            container: None,
            tail_lines: None,
        };
        let result = tool.validate_request(&req);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("requires a resource name"));
    }

    #[test]
    fn test_validate_logs_only_pods() {
        let tool = make_tool(vec!["logs"]);
        let req = KubectlToolParams {
            command: "logs".to_string(),
            resource: "deployments".to_string(),
            name: Some("my-deploy".to_string()),
            label_selector: None,
            container: None,
            tail_lines: None,
        };
        let result = tool.validate_request(&req);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("only supported for pods"));
    }

    #[test]
    fn test_validate_success() {
        let tool = make_tool(vec!["get", "list", "describe", "logs"]);

        // list doesn't require name
        let req = KubectlToolParams {
            command: "list".to_string(),
            resource: "pods".to_string(),
            name: None,
            label_selector: Some("app=nginx".to_string()),
            container: None,
            tail_lines: None,
        };
        assert!(tool.validate_request(&req).is_ok());

        // get with name
        let req = KubectlToolParams {
            command: "get".to_string(),
            resource: "deployments".to_string(),
            name: Some("my-deploy".to_string()),
            label_selector: None,
            container: None,
            tail_lines: None,
        };
        assert!(tool.validate_request(&req).is_ok());

        // logs with pod
        let req = KubectlToolParams {
            command: "logs".to_string(),
            resource: "pods".to_string(),
            name: Some("my-pod".to_string()),
            label_selector: None,
            container: Some("nginx".to_string()),
            tail_lines: Some(50),
        };
        assert!(tool.validate_request(&req).is_ok());
    }

    #[cfg(feature = "kubernetes")]
    #[test]
    fn test_resource_api_info_known() {
        assert!(resource_api_info("pods").is_some());
        assert!(resource_api_info("deployments").is_some());
        assert!(resource_api_info("services").is_some());
        assert!(resource_api_info("configmaps").is_some());
        assert!(resource_api_info("nodes").is_some());
        assert!(resource_api_info("events").is_some());
        assert!(resource_api_info("jobs").is_some());
        assert!(resource_api_info("cronjobs").is_some());
        assert!(resource_api_info("ingresses").is_some());
        assert!(resource_api_info("pvcs").is_some());
        // Short aliases
        assert!(resource_api_info("po").is_some());
        assert!(resource_api_info("svc").is_some());
        assert!(resource_api_info("deploy").is_some());
        assert!(resource_api_info("ds").is_some());
        assert!(resource_api_info("sts").is_some());
        assert!(resource_api_info("rs").is_some());
    }

    #[cfg(feature = "kubernetes")]
    #[test]
    fn test_resource_api_info_unknown() {
        assert!(resource_api_info("foobar").is_none());
    }
}
