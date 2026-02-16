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
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::borrow::Cow;
use tools::SpiceModelTool;
use tracing::Span;
use tracing_futures::Instrument;

use crate::tools::utils::parameters;

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct TeamsToolParams {
    /// The text message to post to Microsoft Teams.
    message: String,

    /// Optional title for the message card.
    title: Option<String>,

    /// Optional structured sections for the adaptive card. Each section has a "title" and "text" field.
    card_sections: Option<Vec<TeamsCardSection>>,
}

#[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
pub struct TeamsCardSection {
    /// Section title.
    title: String,

    /// Section body text (supports markdown).
    text: String,
}

#[derive(Debug)]
pub struct TeamsTool {
    name: String,
    description: String,
    webhook_url: String,
    client: reqwest::Client,
}

impl TeamsTool {
    /// Create a new `TeamsTool`.
    ///
    /// # Errors
    ///
    /// Returns an error if the webhook URL cannot be parsed or points to a
    /// host that is not an allowed Microsoft Teams / Azure webhook domain.
    pub fn try_new(
        name: Option<&str>,
        description: Option<&str>,
        webhook_url: String,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let parsed = url::Url::parse(&webhook_url)?;

        let host = parsed
            .host_str()
            .ok_or("Webhook URL does not contain a host")?;

        let allowed = host.ends_with(".webhook.office.com")
            || host.ends_with(".logic.azure.com")
            || host.ends_with(".office365.com");

        if !allowed {
            return Err(format!(
                "Webhook URL host '{host}' is not an allowed Microsoft Teams domain. \
                 Allowed domains: *.webhook.office.com, *.logic.azure.com, *.office365.com"
            )
            .into());
        }

        Ok(Self {
            name: name.unwrap_or("ms_teams").to_string(),
            description: description
                .unwrap_or("Post messages and reports to Microsoft Teams channels")
                .to_string(),
            webhook_url,
            client: reqwest::Client::new(),
        })
    }

    /// Build a Teams Adaptive Card payload from the given parameters.
    fn build_adaptive_card(&self, params: &TeamsToolParams) -> Value {
        let mut body_elements: Vec<Value> = Vec::new();

        // If a title is present, insert a bold TextBlock at the beginning.
        if let Some(ref title) = params.title {
            body_elements.push(json!({
                "type": "TextBlock",
                "text": title,
                "weight": "Bolder",
                "size": "Medium"
            }));
        }

        // Always include the main message text.
        body_elements.push(json!({
            "type": "TextBlock",
            "text": params.message,
            "wrap": true
        }));

        // Append any additional card sections.
        if let Some(ref sections) = params.card_sections {
            for section in sections {
                body_elements.push(json!({
                    "type": "TextBlock",
                    "text": section.title,
                    "weight": "Bolder",
                    "separator": true
                }));
                body_elements.push(json!({
                    "type": "TextBlock",
                    "text": section.text,
                    "wrap": true
                }));
            }
        }

        json!({
            "type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": null,
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": body_elements
                }
            }]
        })
    }
}

#[async_trait]
impl SpiceModelTool for TeamsTool {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn description(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(&self.description))
    }

    fn parameters(&self) -> Option<Value> {
        parameters::<TeamsToolParams>()
    }

    async fn call(&self, arg: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let span: Span = tracing::span!(target: "task_history", tracing::Level::INFO, "tool_use::ms_teams", tool = self.name().to_string(), input = arg);

        let tool_use_result: Result<Value, Box<dyn std::error::Error + Send + Sync>> = async {
            let req: TeamsToolParams = serde_json::from_str(arg)?;

            let payload = self.build_adaptive_card(&req);

            let response = self
                .client
                .post(&self.webhook_url)
                .header("Content-Type", "application/json")
                .json(&payload)
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                return Err(format!(
                    "Microsoft Teams webhook returned HTTP {status}: {body}"
                )
                .into());
            }

            let sections_count = req
                .card_sections
                .as_ref()
                .map_or(0, std::vec::Vec::len);

            Ok(json!({
                "status": "sent",
                "title": req.title,
                "message_length": req.message.len(),
                "sections_count": sections_count
            }))
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

    #[test]
    fn test_reject_invalid_webhook_url() {
        let result = TeamsTool::try_new(None, None, "https://evil.example.com/webhook".to_string());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not an allowed Microsoft Teams domain"),
            "Expected domain validation error, got: {err}"
        );
    }

    #[test]
    fn test_accept_valid_webhook_url() {
        let result = TeamsTool::try_new(
            None,
            None,
            "https://myorg.webhook.office.com/webhookb2/some-guid/IncomingWebhook/some-id/some-token".to_string(),
        );
        assert!(result.is_ok());
        let tool = result.unwrap();
        assert_eq!(tool.name, "ms_teams");
        assert_eq!(
            tool.description,
            "Post messages and reports to Microsoft Teams channels"
        );
    }

    #[test]
    fn test_accept_logic_azure_url() {
        let result = TeamsTool::try_new(
            None,
            None,
            "https://prod-00.logic.azure.com/workflows/abc123/triggers/manual/paths/invoke?api-version=2016-06-01".to_string(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_adaptive_card_simple() {
        let tool = TeamsTool::try_new(
            None,
            None,
            "https://myorg.webhook.office.com/webhookb2/guid".to_string(),
        )
        .unwrap();

        let params = TeamsToolParams {
            message: "Hello, Teams!".to_string(),
            title: None,
            card_sections: None,
        };

        let card = tool.build_adaptive_card(&params);

        // Verify top-level structure.
        assert_eq!(card["type"], "message");

        let attachments = card["attachments"].as_array().unwrap();
        assert_eq!(attachments.len(), 1);
        assert_eq!(
            attachments[0]["contentType"],
            "application/vnd.microsoft.card.adaptive"
        );
        assert!(attachments[0]["contentUrl"].is_null());

        let content = &attachments[0]["content"];
        assert_eq!(content["type"], "AdaptiveCard");
        assert_eq!(content["version"], "1.4");
        assert_eq!(
            content["$schema"],
            "http://adaptivecards.io/schemas/adaptive-card.json"
        );

        // With no title and no sections, body should have exactly 1 element (the message).
        let body = content["body"].as_array().unwrap();
        assert_eq!(body.len(), 1);
        assert_eq!(body[0]["type"], "TextBlock");
        assert_eq!(body[0]["text"], "Hello, Teams!");
        assert_eq!(body[0]["wrap"], true);
    }

    #[test]
    fn test_build_adaptive_card_with_title_and_sections() {
        let tool = TeamsTool::try_new(
            None,
            None,
            "https://myorg.webhook.office.com/webhookb2/guid".to_string(),
        )
        .unwrap();

        let params = TeamsToolParams {
            message: "Summary of the deployment.".to_string(),
            title: Some("Deployment Report".to_string()),
            card_sections: Some(vec![
                TeamsCardSection {
                    title: "Services".to_string(),
                    text: "All services healthy.".to_string(),
                },
                TeamsCardSection {
                    title: "Metrics".to_string(),
                    text: "Latency p99: 120ms".to_string(),
                },
            ]),
        };

        let card = tool.build_adaptive_card(&params);
        let body = card["attachments"][0]["content"]["body"]
            .as_array()
            .unwrap();

        // Expected body elements:
        //   1. Title TextBlock (weight Bolder, size Medium)
        //   2. Message TextBlock (wrap true)
        //   3. Section 1 title TextBlock (weight Bolder, separator true)
        //   4. Section 1 text TextBlock
        //   5. Section 2 title TextBlock (weight Bolder, separator true)
        //   6. Section 2 text TextBlock
        assert_eq!(body.len(), 6);

        // Verify title element.
        assert_eq!(body[0]["text"], "Deployment Report");
        assert_eq!(body[0]["weight"], "Bolder");
        assert_eq!(body[0]["size"], "Medium");

        // Verify main message element.
        assert_eq!(body[1]["text"], "Summary of the deployment.");
        assert_eq!(body[1]["wrap"], true);

        // Verify first section.
        assert_eq!(body[2]["text"], "Services");
        assert_eq!(body[2]["weight"], "Bolder");
        assert_eq!(body[2]["separator"], true);
        assert_eq!(body[3]["text"], "All services healthy.");

        // Verify second section.
        assert_eq!(body[4]["text"], "Metrics");
        assert_eq!(body[4]["weight"], "Bolder");
        assert_eq!(body[4]["separator"], true);
        assert_eq!(body[5]["text"], "Latency p99: 120ms");
    }
}
